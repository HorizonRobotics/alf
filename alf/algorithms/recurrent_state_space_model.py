# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
r"""Learning a latent state representation and a dynamics model in the latent
space. The overall model architecture follows `PlaNet
<https://arxiv.org/abs/1811.04551>`_ (Equation 4 in the paper):

    .. code-block:: text

        "Learning Latent Dynamics for Planning from Pixels", Hafner et al., ICML 2019
"""

import gin

import torch

import alf
from alf.networks import LSTMEncodingNetwork, ImageDecodingNetwork, EncodingNetwork
from alf.algorithms.algorithm import Algorithm
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops, losses
from alf.data_structures import namedtuple, AlgStep, LossInfo

RSSMState = namedtuple("RSSMState", ["s", "h"], default_value=())


@gin.configurable
class RecurrentStateSpaceModel(Algorithm):
    r"""A recurrent state-space model consists of several networks:

    * recurrent state network :math:`h_t=f(h_{t-1}, s_{t-1}, a_{t-1})`
    * state prior network :math:`s_t \sim p(\cdot|h_t)`
    * observation network :math:`o_t \sim p(\cdot|h_t,s_t)`
    * reward network :math:`r_t \sim p(\cdot|h_t,s_t)`
    * state posterior network (encoder) :math:`s_t \sim q(\cdot|h_t,o_t)`

    where :math:`s` is the latent state representation and :math:`a` is the
    action. From a VAE perspective, the state prior network models the prior of
    :math:`s_t` (imagined state transition) without seeing the new observation
    :math:`o_t`, and the state posterior network is the variational encoder that
    encodes the new observation into a latent representation to reconstruct
    :math:`o_t` and :math:`r_t` while maintaining the posterior close to the
    prior belief by minimizing:

    .. math::

        \text{KL}\left[q(s_t|h_t,o_t) || p(s_t|h_t)\right]

    .. note::

        A small difference between this implementation and the formulation above
        is that we will use the output of the recurrent state network instead of
        :math:`h_t` as the input to other networks. If there are post FC layers,
        then the output is different from :math:`h_t`.

    This model is a specific kind of VAE in the context of RL.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 state_dim,
                 recurrent_state_network_ctor=LSTMEncodingNetwork,
                 state_prior_network_ctor=EncodingNetwork,
                 observation_network_ctor=ImageDecodingNetwork,
                 reward_network_ctor=EncodingNetwork,
                 state_posterior_network_ctor=EncodingNetwork,
                 beta=1.0,
                 optimizer=None,
                 name="RecurrentStateSpaceModel"):
        r"""
        Args:
            observation_spec (nested TensorSpec): the observation spec of
                :math:`o_t`.
            action_spec (nested TensorSpec): the action spec of :math:`a_t`
            state_dim (int): dimension of the latent state vector :math:`s_t`.
            recurrent_state_network_ctor (Callable): a function that creates the
                recurrent state network :math:`h_t=f(h_{t-1}, s_{t-1}, a_{t-1})`
                that takes inputs in the form of :math:`(s_{t-1},a_{t-1})`. The
                ``input_tensor_spec`` of this network will be automatically set.
            state_prior_network_ctor (Callable): a function that creates the
                state prior network :math:`s_t \sim p(\cdot|h_t)`. The network
                will always output a concatenation of ``(mean, log_var)`` representing
                a diagonal Gaussian from which the state will be sampled. The
                ``input_tensor_spec``, ``last_layer_size``, and ``last_activation``
                of this network will be automatically set.
            observation_network_ctor (Callable): a function that creates the
                observation network :math:`o_t \sim p(\cdot|h_t,s_t)`. Because
                we always assume an identity covariance matrix for the observation
                variable, this network only outputs the mean vector. The
                ``input_tensor_spec`` of this network will be automatically set.
            reward_network_ctor (Callable): a function that creates the reward
                network :math:`r_t \sim p(\cdot|h_t,s_t)`. Because we always
                assume unit variance for the reward variable, this network only
                outputs the mean. The ``input_tensor_spec``, ``last_layer_size``,
                and ``last_activation`` of this network will be automatically set.
            state_posterior_network_ctor (Callable): a function that creates the
                state posterior network :math:`s_t \sim q(\cdot|h_t,o_t)` that
                takes inputs in the form of :math:`(h_t,o_t)`. Like the state
                prior network, this network will always output a concatenation
                of ``(delta_mean, delta_log_var)``. This delta vector will be
                added to the output of ``state_prior_network`` to represent a
                diagonal Gaussian from which the posterior state will be sampled.
                The ``input_tensor_spec``, ``last_layer_size``, and ``last_activation``
                of this network will be automatically set.
            beta (float): the coefficient of KL divergence cost.
            optimizer (Optimizer): optional optimizer for training this model.
            name (str): name of the model
        """
        state_spec = TensorSpec(shape=[state_dim])
        recurrent_state_network = recurrent_state_network_ctor(
            input_tensor_spec=(state_spec, action_spec))

        super(RecurrentStateSpaceModel, self).__init__(
            # maintain a history of ``s_t`` and ``h_t``
            train_state_spec=RSSMState(
                s=state_spec, h=recurrent_state_network.state_spec),
            optimizer=optimizer,
            name=name)

        self._beta = float(beta)
        self._state_spec = state_spec
        self._state_dim = state_dim

        self._recurrent_state_network = recurrent_state_network
        self._state_prior_network = state_prior_network_ctor(
            input_tensor_spec=recurrent_state_network.output_spec,
            last_layer_size=state_dim * 2,
            last_activation=math_ops.identity)
        assert len(recurrent_state_network.output_spec.shape) == 1, (
            "Only allow a flat output vector from the recurrent state"
            " network! Got spec {}".format(
                recurrent_state_network.output_spec))
        recurrent_out_dim = recurrent_state_network.output_spec.shape[0]
        self._observation_network = observation_network_ctor(
            input_tensor_spec=TensorSpec(
                shape=[recurrent_out_dim + state_dim]))
        assert self._observation_network.output_spec == observation_spec, (
            "Ouput spec of the observation network is wrong! "
            "{} Should be {}".format(self._observation_network.output_spec,
                                     observation_spec))
        self._reward_network = reward_network_ctor(
            input_tensor_spec=TensorSpec(
                shape=[recurrent_out_dim + state_dim]),
            last_layer_size=1,
            last_activation=math_ops.identity)
        self._state_posterior_network = state_posterior_network_ctor(
            input_tensor_spec=(recurrent_state_network.output_spec,
                               observation_spec),
            last_layer_size=state_dim * 2,
            last_activation=math_ops.identity)

    @property
    def state_spec(self):
        return self._state_spec

    def _split_s_mean_and_log_var(self, s_mean_and_log_var):
        return (s_mean_and_log_var[..., :self._state_dim],
                s_mean_and_log_var[..., self._state_dim:])

    def _sample_state(self, s_mean_and_log_var):
        """Sample a state using the reparameterization trick given a concatenation
        of the mean and log variance of a diagonal Gaussian.
        """
        s_mean, s_log_var = self._split_s_mean_and_log_var(s_mean_and_log_var)
        # reparameterization sampling: z = u + var ** 0.5 * eps
        eps = torch.randn(s_mean.shape)
        s = s_mean + torch.exp(s_log_var * 0.5) * eps
        return s

    def predict_step(self, inputs, state: RSSMState):
        r"""Predict the latent state :math:`s_t` given the previous action
        :math:`a_{t-1}`, the previous latent state :math:`s_{t-1}`, and the
        observation :math:`o_t`. Two possible cases:

        1. The current observation :math:`o_t` is provided as an input, then the
           prediction is a posterior latent state.
        2. Otherwise the prediction is a prior latent state in the absence of
           the observation. This scenario can be imagination of a rollout
           trajectory wihout interacting with the environment.

        Args:
            inputs (nested Tensor): a tuple of ``(prev_action, observation)``.
                The prediction is prior or posterior depending on whether
                ``observation`` is ``None`` or not.
            state (RSSMState): short-term memory of the model.

        Returns:
            AlgStep:
            - output: the predicted latent state :math:`s_t`.
            - state: the updated model state
            - info: empty ()
        """
        prev_action, observation = inputs
        out, h = self._recurrent_state_network((state.s, prev_action),
                                               state=state.h)
        s_mean_and_log_var, _ = self._state_prior_network(out)
        if observation is not None:
            delta_s_mean_and_log_var, _ = self._state_posterior_network(
                (out, observation))
            s_mean_and_log_var += delta_s_mean_and_log_var
        s = self._sample_state(s_mean_and_log_var)
        return AlgStep(output=s, state=RSSMState(s=s, h=h), info=())

    def rollout_step(self, inputs, state: RSSMState):
        r"""Behaves the same with ``predict_step()``."""
        return self.predict_step(inputs, state)

    def train_step(self, inputs, state: RSSMState):
        r"""Train the model by reconstructing observations and rewards with MSE
        losses.

        Args:
            inputs (nested Tensor): a tuple of ``(prev_action, observation, reward)``
            state (RSSMState): short-term memory of the model

        Returns:
            AlgStep:
            - output: the predicted posterior latent state given the observation
            - state: the updated model state
            - info (LossInfo):
                - loss: sum of reconstruction losses and KLD loss
                - extra: a dictionary of observation reconstruction loss, reward
                  reconstruction loss, and the KLD loss
        """
        prev_action, observation, reward = inputs

        # compute the kl divergen loss
        out, h = self._recurrent_state_network((state.s, prev_action),
                                               state=state.h)
        prior_s_mean_and_log_var, _ = self._state_prior_network(out)
        delta_s_mean_and_log_var, _ = self._state_posterior_network(
            (out, observation))
        delta_s_mean, delta_s_log_var = self._split_s_mean_and_log_var(
            delta_s_mean_and_log_var)
        _, prior_s_log_var = self._split_s_mean_and_log_var(
            prior_s_mean_and_log_var)
        kl_div_loss = (
            math_ops.square(delta_s_mean) / torch.exp(prior_s_log_var) +
            torch.exp(delta_s_log_var) - delta_s_log_var - 1.0)
        kl_div_loss = 0.5 * torch.sum(kl_div_loss, dim=-1)

        # sample the latent state
        s_mean_and_log_var = prior_s_mean_and_log_var + delta_s_mean_and_log_var
        s = self._sample_state(s_mean_and_log_var)
        new_s = torch.cat([out, s], dim=-1)

        # reconstruct observation
        rec_observation, _ = self._observation_network(new_s)
        rec_obs_loss = losses.element_wise_squared_loss(
            x=observation, y=rec_observation)
        rec_obs_loss = torch.sum(
            rec_obs_loss.reshape(rec_obs_loss.shape[0], -1), dim=-1)

        # reconstruct reward
        rec_reward, _ = self._reward_network(new_s)
        rec_reward_loss = losses.element_wise_squared_loss(
            x=reward, y=rec_reward)

        return AlgStep(
            output=s,
            state=RSSMState(s=s, h=h),
            info=LossInfo(
                loss=(
                    self._beta * kl_div_loss + rec_obs_loss + rec_reward_loss),
                extra=dict(
                    kld_loss=kl_div_loss,
                    observation_loss=rec_obs_loss,
                    reward_loss=rec_reward_loss)))
