# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MdqCriticNetworks"""

import functools
import math
import numpy as np

import torch
import torch.nn.functional as f
import torch.nn as nn

import alf
import alf.layers as layers
import alf.nest as nest
from alf.networks import Network, EncodingNetwork, ParallelEncodingNetwork
from alf.initializers import variance_scaling_init
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, spec_utils, tensor_utils
import alf.utils.math_ops as math_ops
from alf.utils.action_quantizer import ActionQuantizer


@alf.configurable
class MdqCriticNetwork(Network):
    """Create an instance of MdqCriticNetwork for estimating action-value
    of continuous actions and action sampling used in the MDQ algorithm."""

    def __init__(self,
                 input_tensor_spec,
                 action_qt: ActionQuantizer = None,
                 num_critic_replicas=2,
                 obs_encoding_layer_params=None,
                 pre_encoding_layer_params=None,
                 mid_encoding_layer_params=None,
                 post_encoding_layer_params=None,
                 free_form_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 debug_summaries=False,
                 name="MdqCriticNetwork"):
        """Creates an instance of `MdqCriticNetwork` for estimating action-value
        of continuous actions and action sampling.

        Currently there are two branches of networks:
            - free-form branch: a plain MLP for Q-learning
            - adv-form branch: an advantage form of network for action
                generation. It is trained by a target from the free-form net.

        The adv-form branch has the following structures for flexibility:
            obs -> [obs_encoding_net] -> encoded_obs
            encoded_obs, action ->
                                [pre_encoding_nets] ->
                                [mid_shared_encoding_nets] ->
                                [post_encoding_nets] -> outputs
            where the pre_encoding_nets and post_encoding_nets do not share
            parameters across action dimensions while mid_shared_encoding_nets
            shares parameters across action dimensions.
            If the encoding_layer_params for a sub-net is None, that sub-net is
            effectively neglected.

        Furthermore, to enable parallel computation across action dimension in
        the case of value computation, we have both parallel and individual
        versions for the nets without parameter sharing. For example, for
        post_encoding_nets, we also have post_encoding_parallel_net, which is
        essentially the equivalent form of post_encoding_nets but supports
        parallel forwarding. The parameters of the two versions are synced.
        The partial actions (a[0:i]) are zero-padded for both parallel and
        individual networks to enable parallel computation.


        For conciseness purpose, the following notations will be used when
        convenient:
            - B: batch size
            - d: dimensionality of feature
            - n: number of network replica
            - action_dim: the dimensionality of actions
            - action_bin: number of discrete bins for each action dim

        Args:
            input_tensor_spec: A tuple of TensorSpecs (observation_spec, action_spec)
                representing the inputs.
            action_qt (ActionQuantizer): action quantization module
            num_critic_replicas (int): number of critic networks
            obs_encoding_layer_params (tuple[int]): a tuple of integers
                representing hidden FC layer sizes for encoding observations.
            pre_encoding_layer_params (tuple[int]): a tuple of integers
                representing hidden FC layer sizes for encoding concatenated
                [encoded_observation, actions]. Parameters are not shared across
                action dimensions
            mid_encoding_layer_params (tuple[int]): a tuple of integers
                representing hidden FC layer for further encoding the outputs
                from pre_encoding_net. The parameters are shared across action
                dimensions.
            post_encoding_layer_params (tuple[int]): a tuple of integers
                representing hidden FC layer for further encoding the outputs
                from mid_encoding_net. The parameters are not shared across
                action dimensions.
            free_form_fc_layer_params (tuple[int]): a tuple of integers
                representing hidden FC layer for Q-learning. We refer it as
                the free form to differentiate it from the mdq-form of network
                which is structured.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a variance_scaling_initializer
                with uniform distribution will be used.
            name (str):
        """

        super().__init__(input_tensor_spec, name=name)

        observation_spec, action_spec = input_tensor_spec

        flat_action_spec = nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network')

        self._single_action_spec = flat_action_spec[0]

        if action_qt is None:
            action_qt = ActionQuantizer(action_spec, "uniform", 15)
        self._action_qt = action_qt
        self._action_bins = self._action_qt._action_bins

        # the logpi of the uniform prior used for KL computation
        self._log_pi_uniform_prior = -np.log(self._action_bins)

        self._action_dim = action_spec.shape[0]  # control vector dim
        self._num_critic_replicas = num_critic_replicas

        self._obs_encoding_net = ParallelEncodingNetwork(
            observation_spec,
            self._num_critic_replicas,
            fc_layer_params=obs_encoding_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        last_activation = math_ops.identity
        last_kernel_initializer = functools.partial(torch.nn.init.uniform_, \
                                a=-0.003, b=0.003)

        in_size = self._action_dim

        self._pre_encoding_nets = []
        for i in range(self._action_dim):
            # output_spec.shape: [n, d]
            self._pre_encoding_nets.append(
                ParallelEncodingNetwork(
                    TensorSpec((self._obs_encoding_net.output_spec.shape[-1] +
                                in_size, )),
                    self._num_critic_replicas,
                    fc_layer_params=pre_encoding_layer_params,
                    activation=activation,
                    kernel_initializer=kernel_initializer))

        # parallel along both critic and action dims without sharing parameters
        # for each action dimension.
        # input: [B, action_dim*n, d]: need to stack over dim1
        # output: [B, action_dim*n, d']: need to unstack over dim1 for
        # splitting over networks
        self._pre_encoding_parallel_net = ParallelEncodingNetwork(
            TensorSpec(
                (self._obs_encoding_net.output_spec.shape[-1] + in_size, )),
            self._num_critic_replicas * self._action_dim,
            fc_layer_params=pre_encoding_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        # parallel along both critic and action dims with sharing parameters
        # for each action dimension.
        # input: [action_dim*B, n, d]: need to stack over dim0
        # output: [action_dim*B, n, d']: need to unstack over dim0 for
        # splitting over networks
        self._mid_shared_encoding_nets = ParallelEncodingNetwork(
            TensorSpec(
                (self._pre_encoding_parallel_net.output_spec.shape[-1], )),
            self._num_critic_replicas,
            fc_layer_params=mid_encoding_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)
        out_size = self._mid_shared_encoding_nets.output_spec.shape[-1]

        post_enc_out_size = self._action_qt.action_bins

        self._post_encoding_nets = []
        for i in range(self._action_dim):
            self._post_encoding_nets.append(
                ParallelEncodingNetwork(
                    TensorSpec((out_size, )),
                    self._num_critic_replicas,
                    fc_layer_params=post_encoding_layer_params,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    last_layer_size=post_enc_out_size,
                    last_activation=last_activation,
                    last_kernel_initializer=last_kernel_initializer))

        # parallel along both critic and action dims without sharing parameters
        # for each action dimension.
        # input: [B, action_dim*n, d]: need to stack over dim1
        # output: [B, action_dim*n, d']: need to unstack over dim1 for
        # splitting over networks
        self._post_encoding_parallel_net = ParallelEncodingNetwork(
            TensorSpec((out_size, )),
            self._num_critic_replicas * self._action_dim,
            fc_layer_params=post_encoding_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            last_layer_size=post_enc_out_size,
            last_activation=last_activation,
            last_kernel_initializer=last_kernel_initializer)

        assert free_form_fc_layer_params is not None

        self._free_form_q_net = ParallelEncodingNetwork(
            TensorSpec((observation_spec.shape[-1] + self._action_dim, )),
            self._num_critic_replicas,
            fc_layer_params=free_form_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            last_layer_size=1,
            last_activation=math_ops.identity,
            last_kernel_initializer=last_kernel_initializer)

        MdqCriticNetwork._parallel_to_individual_network_sync(
            self._pre_encoding_parallel_net,
            self._pre_encoding_nets,
            step=self._num_critic_replicas)

        MdqCriticNetwork._parallel_to_individual_network_sync(
            self._post_encoding_parallel_net,
            self._post_encoding_nets,
            step=self._num_critic_replicas)

        self._output_spec = TensorSpec(())

        self._debug_summaries = debug_summaries

    @torch.no_grad()
    def get_action(self, inputs, alpha, greedy):
        """Sample action from the distribution induced by the mdq-net.

        Args:
            inputs: A tuple of Tensors consistent with `input_tensor_spec`
            alpha: the temperature used for the advantage computation
            greedy (bool): If True, do greedy sampling by taking the mode of
                the distribution. If False, do direct sampling from the
                distribution.
        Returns:
            actions (torch.Tensor): a tensor of the shape [B, n, action_dim]
            log_pi_per_dim (torch.Tensor): a tensor of the shape
                [B, n, action_dim] representing the log_pi for each dimension
                of the sampled multi-dimensional action
        """

        observations = inputs

        # [B, n, d]
        t_shape = (observations.shape[0], self._num_critic_replicas,
                   self._action_dim)

        actions = torch.zeros(t_shape)
        log_pi_per_dim = torch.zeros(t_shape)

        # [B, n, d]
        encoded_obs, _ = self._obs_encoding_net(observations)

        if actions.ndim == 2:
            actions = tensor_utils.tensor_extend_new_dim(
                actions, dim=1, n=self._num_critic_replicas)

        action_padded = torch.zeros(t_shape)

        for i in range(self._action_dim):
            action_padded[..., 0:i] = actions[..., 0:i]
            joint = torch.cat((encoded_obs, action_padded.detach()), -1)

            action_values_i, _ = self._net_forward_individual(joint, alpha, i)

            trans_action_values_i = self._transform_action_value(
                action_values_i, alpha)
            sampled_indices, sampled_log_pi = self._sample_action_from_value(
                trans_action_values_i / alpha, alpha, greedy)
            # convert index to action
            actions[..., i] = self._action_qt.ind_to_action(sampled_indices)
            log_pi_per_dim[..., i] = sampled_log_pi

        return actions, log_pi_per_dim

    def forward(self, inputs, alpha, state=(), free_form=False):
        """Computes action-value given an observation.

        Args:
            inputs:  A tuple of Tensors consistent with `input_tensor_spec`
            alpha: the temperature used for the advantage computation
            state: empty for API consistenty
            free_form (bool): use the free-form branch for computation if True;
                default value is False

        Returns:
            Q_values (torch.Tensor):
                - if free_form is True, its shape is [B, n]
                - if free_form is False, its shape is [B, n, action_dim]
            state: empty
        """

        if free_form:
            Q_values, state = self._free_form_q_net(inputs)
            Q_values = Q_values.squeeze(2)
            return Q_values, state

        observations, actions = inputs

        # observations: [B, d]
        # encoded_obs: [B, n, d']
        # Note that when obs_encoding_net is a dummy network
        # (i.e., layer_params is None), d' is the same as d.
        encoded_obs, _ = self._obs_encoding_net(observations)

        if actions.ndim == 2:
            # [B, action_dim] -> [B, n, action_dim]
            actions = tensor_utils.tensor_extend_new_dim(
                actions, dim=1, n=self._num_critic_replicas)

        # [B, n, action_dim]
        t_shape = (observations.shape[0], self._num_critic_replicas,
                   self._action_dim)

        # [action_dim, B, n, 1]
        Q_values = torch.zeros(self._action_dim, observations.shape[0],
                               self._num_critic_replicas, 1)

        joint = torch.empty(0)
        action_padded = torch.zeros(t_shape)

        # prepare parallel-forwarding inputs
        inputs_per_dim = []
        for i in range(self._action_dim):
            action_padded[..., 0:i] = actions[..., 0:i]
            # concat (obs, action) for each action dimension
            inputs_per_dim.append(
                torch.cat((encoded_obs, action_padded.detach()), dim=-1))

        # concat per dim input batch to a joint batch along dim1
        # [B, action_dim*n, d]
        joint = torch.cat(inputs_per_dim, dim=1)

        # forward the joint batch
        # action_values_per_dim: [action_dim, B, n, action_bin]
        action_values_per_dim, _ = self._net_forward_parallel(
            joint, alpha, batch_size=observations.shape[0])

        trans_action_values_per_dim = self._transform_action_value(
            action_values_per_dim, alpha)

        for i in range(self._action_dim):
            action_ind = self._action_qt.action_to_ind(actions[..., i])
            if i == 0:
                action_value_i = self._batched_index_select(
                    action_values_per_dim[i], -1, action_ind.long())
                Q_values[i] = action_value_i
                # KL-divergence
                Q_values[i] = Q_values[i] - alpha * self._log_pi_uniform_prior
            else:
                selected_trans_action_value_i = self._batched_index_select(
                    trans_action_values_per_dim[i], -1, action_ind.long())
                Q_values[i] = Q_values[i - 1] + selected_trans_action_value_i
                # KL-divergence
                Q_values[i] = Q_values[i] - alpha * self._log_pi_uniform_prior

        # [action_dim, B, n, 1] -> [B, n, action_dim]
        Q_values = Q_values.squeeze(3).permute(1, 2, 0)
        return Q_values, state

    def _net_forward_individual(self, inputs, alpha, i, state=()):
        """Individual forwarding for a specified action dims for value computation.
        Args:
            inputs (torch.Tensor): a tensor of the shape [B, n, d]
            alpha: the temperature used for the advantage computation
            i (int): the specified action dim to perform forwarding
        Returns:
            action_values_i (torch.Tensor): a tensor of the shape [B, n, action_bin]
            state: empty
        """
        inputs, _ = self._pre_encoding_nets[i](inputs)
        action_values_i, state = self._mid_shared_encoding_nets(inputs)
        action_values_i, state = self._post_encoding_nets[i](action_values_i)
        return action_values_i, state

    def _net_forward_parallel(self, inputs, alpha, batch_size, state=()):
        """Parallel forwarding across action dims for value computation.
        Args:
            inputs (torch.Tensor): a tensor of the shape [B, action_dim*n, d]
                with the data for each action dimension concanated along the
                dim1 for parallel computation
            alpha: the temperature used for the advantage computation
            batch_size: the size of the original batch without stacking
                all action dimensions
        Returns:
            action_values (torch.Tensor): a tensor of the shape
                [action_dim, B, n, action_bin]
            state: empty
        """
        # [B, action_dim*n, d]
        action_values_pre, _ = self._pre_encoding_parallel_net(inputs)
        # [B, action_dim*n, d] -> [action_dim*B, n, d]
        action_values_pre = self._reshape_from_ensemble_to_batch(
            action_values_pre, batch_size)
        action_values_mid, state = self._mid_shared_encoding_nets(
            action_values_pre)
        # [action_dim*B, n, d] -> [B, action_dim*n, d]
        action_values_mid = self._reshape_from_batch_to_ensemble(
            action_values_mid, batch_size)
        action_values_final, _ = self._post_encoding_parallel_net(
            action_values_mid)
        # [B, action_dim*n, d]->  [B, action_dim, n, d] -> [action_dim, B, n, d]
        action_values = action_values_final.view(batch_size, self._action_dim,
                                                 self._num_critic_replicas,
                                                 -1).transpose(0, 1)

        return action_values, state

    def _reshape_from_batch_to_ensemble(self, joint_batch, batch_size):
        """Reshape the joint batch of the shape [action_dim*B, n, d] to
            [B, action_dim*n, d], i.e., separate and move the action dimension
            axis from the batch dimension (dim0) to the ensemble dimension (dim1)
        Args:
            joint_batch (torch.Tensor): a tensor of the shape [action_dim*B, n, d]
                with the data for each action dimension concanated along the
                batch dimension (dim0)
            batch_size: the size of the original batch without stacking
                all action dimensions
        Returns:
            reshaped_batch (torch.Tensor): a tensor of the shape
                [B, action_dim*n, d]
        """
        assert len(joint_batch.shape) == 3 and joint_batch.shape[:-1] == \
                (self._action_dim * batch_size, self._num_critic_replicas)

        d = joint_batch.shape[-1]
        # [action_dim*B, n, d] -> [action_dim, B, n, d]
        reshaped_batch = joint_batch.view(self._action_dim, batch_size,
                                          self._num_critic_replicas, d)

        #  [action_dim, B, n, d] ->  [B, action_dim, n, d] -> [B, action_dim*n, d]
        reshaped_batch = reshaped_batch.transpose(0, 1).reshape(
            batch_size, -1, d)
        return reshaped_batch

    def _reshape_from_ensemble_to_batch(self, joint_batch, batch_size):
        """Reshape the joint batch of the shape [B, action_dim*n, d] to
            [action_dim*B, n, d], i.e., separate and move the action dimension
            axis from the ensemble dimension (dim1) to the batch dimension (dim0)
        Args:
            joint_batch (torch.Tensor): a tensor of the shape [B, action_dim*n, d]
                with the data for each action dimension concanated along the
                ensemble dimension (dim1)
            batch_size: the size of the original batch without stacking
                all action dimensions
        Returns:
            reshaped_batch (torch.Tensor): a tensor of the shape
                [action_dim*B, n, d]
        """
        assert len(joint_batch.shape) == 3 and joint_batch.shape[:-1] == \
                (batch_size, self._action_dim * self._num_critic_replicas)

        d = joint_batch.shape[-1]
        # [B, action_dim*n, d] -> [B, action_dim, n, d]
        reshaped_batch = joint_batch.view(batch_size, self._action_dim,
                                          self._num_critic_replicas, d)

        # [B, action_dim, n, d] -> [action_dim, B, n, d] -> [action_dim*B, n, d]
        reshaped_batch = reshaped_batch.transpose(0, 1).reshape(
            -1, self._num_critic_replicas, d)
        return reshaped_batch

    def _transform_action_value(self, action_values, alpha):
        """Transform raw action values to valid alpha * log_pi
        Args:
            action_values (torch.Tensor): raw action values computed from a
                network, with the last dim as the distribution dimension
            alpha: the temperature used for the transformation

        Returns:
            transformed_value (torch.Tensor): a tensor with value equals
                alpha * log_pi computed from input action_values
        """
        v_value = alpha * torch.logsumexp(
            action_values / alpha, dim=-1, keepdim=True)
        transformed_value = action_values - v_value
        return transformed_value

    def _sample_action_from_value(self, logits, alpha, greedy=False):
        """Sample discrete action from given logits
        Args:
            logits (torch.Tensor): log pi of the discrete distribution with
                the last dim as the distribution dimension
            alpha: the temperature used for the transformation
            greedy (bool): if True, do greedy sampling by taking the mode
                of the distribution; otherwise, sample according
                to the probability of the distribution
        Returns:
            sampled_ind (torch.Tensor): the indices of the sampled action
            sampled_log_pi (torch.Tensor): the log prob of the sampled action
        """
        if greedy:
            sampled_log_pi, sampled_ind = torch.max(logits, dim=-1)
        else:
            batch_size = logits.shape[0]

            # logits [B, n, d] -> [B*n, d]
            batched_logits = logits.reshape(-1, self._action_bins)
            dist = torch.distributions.categorical.Categorical(
                logits=batched_logits)

            # [1, B*n] -> [B, n]
            sampled_ind = dist.sample((1, ))
            sampled_log_pi = dist.log_prob(sampled_ind)

            sampled_ind = sampled_ind.view(batch_size, -1)
            sampled_log_pi = sampled_log_pi.view(batch_size, -1)

        return sampled_ind, sampled_log_pi

    def _batched_index_select(self, t, dim, inds):
        expanded_ind = inds.unsqueeze(-1)
        out = t.gather(dim, expanded_ind)
        return out

    @staticmethod
    def _parallel_to_individual_network_sync(p_net, np_net, step):
        """Sync parameters from parallel version to individual version
        Args:
            p_net (ParallelNetwork): the parallel version of network
            np_net (list[Network|ParallelNetwork]): a list of the individual
                networks. Note that each individual network can also be an
                instance of ParallelNetwork.
            step (int): the replica contained in the individual network.
                For example:
                 - if the individual net is a plain network, step=1
                 - if the individual net is a parallel network, step = replica
                    of the individual net
        """
        split_num = len(np_net)
        for i in range(split_num):
            for ws, wt in zip(p_net.parameters(), np_net[i].parameters()):
                wt.data.copy_(ws[i * step:(i + 1) * step])

    def get_uniform_prior_logpi(self):
        return self._log_pi_uniform_prior

    def sync_net(self):
        MdqCriticNetwork._parallel_to_individual_network_sync(
            self._pre_encoding_parallel_net, self._pre_encoding_nets,
            self._num_critic_replicas)

        MdqCriticNetwork._parallel_to_individual_network_sync(
            self._post_encoding_parallel_net, self._post_encoding_nets,
            self._num_critic_replicas)
