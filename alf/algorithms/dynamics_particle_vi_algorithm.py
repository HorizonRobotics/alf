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

from collections import namedtuple
import gin
import numpy as np
import torch

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.dynamics_learning_algorithm import DynamicsState, DynamicsInfo
# from alf.algorithms.dynamics_learning_algorithm import DynamicsLearningAlgorithm
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.algorithms.hypernetwork_algorithm import HyperNetwork
from alf.data_structures import (AlgStep, LossInfo, namedtuple, StepType,
                                 TimeStep)
from alf.nest import nest
from alf.nest.utils import NestConcat, get_outer_rank
from alf.networks import Network, EncodingNetwork
from alf.networks.param_dynamics_networks import ParamDynamicsNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import losses, math_ops, spec_utils, tensor_utils


@gin.configurable
class DynamicsParVIAlgorithm(FuncParVIAlgorithm):
    """Base Dynamics Learning Module

    This module trys to learn the dynamics of environment.
    """

    def __init__(self,
                 action_spec,
                 feature_spec,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 num_particles=10,
                 entropy_regularization=1.,
                 loss_type="regression",
                 voting="soft",
                 par_vi="svgd",
                 prediction_mode='mean',
                 function_vi=False,
                 optimizer=None,
                 debug_summaries=False,
                 name="DynamicsParVIAlgorithm"):
        """Create a DynamicsParVIAlgorithm.

        Args:
            action_spec (BoundedTensorSpec): representing the actions.
            feature_spec (BoundedTensorSpec): representing the features.
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where 
                ``use_bias`` is optional.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            num_particles (int): number of sampling particles
            entropy_regularization (float): weight of entropy regularization
            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            par_vi (str): types of particle-based methods for variational inference,
                types are [``svgd``, ``gfsf``]
            prediction_mode (str): concensus method for predicting next state,
                types are [``mean``, ``sample``], where ``mean`` outputs the mean
                output of the ensemble of dynamic models, ``sample`` outputs the
                output of a randomly selected menber of the ensemble.
            function_vi (bool): whether to use function value based par_vi.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            name (str)
        """

        if num_particles > 1:
            ens_feature_spec = TensorSpec(
                (num_particles, feature_spec.shape[0]), dtype=torch.float32)
        else:
            ens_feature_spec = feature_spec
        # ens_feature_spec = TensorSpec((num_particles, feature_spec.shape[0]),
        #                               dtype=torch.float32)
        train_state_spec = DynamicsState(feature=ens_feature_spec, network=())

        flat_action_spec = nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, "doesn't support nested action_spec"

        flat_feature_spec = nest.flatten(feature_spec)
        assert len(
            flat_feature_spec) == 1, "doesn't support nested feature_spec"

        action_spec = flat_action_spec[0]

        if action_spec.is_discrete:
            self._num_actions = action_spec.maximum - action_spec.minimum + 1
        else:
            self._num_actions = action_spec.shape[-1]

        self._action_spec = action_spec
        self._feature_spec = feature_spec
        assert prediction_mode in [
            'mean', 'sample'
        ], ('Only prediction modes of "mean" and "sample" are supported.')
        self._prediction_mode = prediction_mode

        # feature_dim = flat_feature_spec[0].shape[-1]

        encoded_action_spec = TensorSpec((self._num_actions, ),
                                         dtype=torch.float32)

        param_net = ParamDynamicsNetwork(
            input_tensor_spec=(feature_spec, encoded_action_spec),
            output_tensor_spec=flat_feature_spec[0],
            fc_layer_params=fc_layer_params,
            activation=activation)

        super().__init__(
            param_net=param_net,
            train_state_spec=train_state_spec,
            num_particles=num_particles,
            entropy_regularization=entropy_regularization,
            loss_type=loss_type,
            voting=voting,
            par_vi=par_vi,
            function_vi=function_vi,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

    @property
    def num_predictives(self):
        return self._num_particles

    def _encode_action(self, action):
        if self._action_spec.is_discrete:
            return torch.nn.functional.one_hot(
                action, num_classes=self._num_actions)
        else:
            return action

    def update_state(self, time_step: TimeStep, state: DynamicsState):
        """Update the state based on TimeStep data. This function is
            mainly used during rollout together with a planner.
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (DynamicsState): state for DynamicsLearningAlgorithm
                (previous observation)
        Returns:
            state (DynamicsState): updated dynamics state
        """
        feature = time_step.observation
        if feature.shape == state.feature.shape:
            updated_state = state._replace(feature=feature)
        else:
            # feature [B, d], state.feature: [B, n, d]
            updated_state = state._replace(
                feature=self._expand_to_replica(feature, self._feature_spec))
        return updated_state

    # def get_state_specs(self):
    #     """Get the state specs of the current module.
    #     This function is mainly used for constructing the nested state specs
    #     by the upper-level module.
    #     """
    #     raise NotImplementedError

    def _expand_to_replica(self, inputs, spec):
        """Expand the inputs of shape [B, ...] to [B, n, ...] if n > 1,
            where n is the number of particles. When n = 1, the unexpanded
            inputs will be returned.
        Args:
            inputs (Tensor): the input tensor to be expanded
            spec (TensorSpec): the spec of the unexpanded inputs. It is used to
                determine whether the inputs is already an expanded one. If it
                is already expanded, inputs will be returned without any
                further processing.
        Returns:
            Tensor: the expaneded inputs or the original inputs.
        """
        outer_rank = get_outer_rank(inputs, spec)
        if outer_rank == 1 and self.num_particles > 1:
            return inputs.unsqueeze(1).expand(-1, self.num_particles,
                                              *inputs.shape[1:])
        else:
            return inputs

    def predict_step(self, time_step: TimeStep, state: DynamicsState):
        """Predict the current observation using ``time_step.prev_action``
            and the feature of the previous observation from ``state``.
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (DynamicsState): state for dynamics learning
        Returns:
            AlgStep:
                output:
                state (DynamicsState):
                info (DynamicsInfo):
        """
        obs = state.feature
        action = self._encode_action(time_step.prev_action)

        # perform preprocessing
        observations = self._expand_to_replica(obs, self._feature_spec)
        actions = self._expand_to_replica(action, self._action_spec)

        forward_step = super().predict_step((observations, actions),
                                            state=state.network)
        forward_deltas = forward_step.output
        forward_pred = observations + forward_deltas

        if self._debug_summaries:
            preds_cov = tensor_utils.cov(forward_deltas).cpu()
            alf.summary.scalar("cov_norm_dynamics_predictions",
                               preds_cov.norm(dim=(1, 2)).mean())

        network_state = forward_step.state
        state = state._replace(feature=forward_pred, network=network_state)
        return AlgStep(output=forward_pred, state=state, info=actions)

    def train_step(self,
                   time_step: TimeStep,
                   state: DynamicsState,
                   entropy_regularization=None):
        """
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (DynamicsState): state for dynamics learning (previous observation)
        Returns:
            AlgStep:
                output:
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        obs = state.feature
        action = self._encode_action(time_step.prev_action)
        # perform preprocessing
        observations = self._expand_to_replica(obs, self._feature_spec)
        actions = self._expand_to_replica(action, self._action_spec)

        next_obs = time_step.observation
        next_obs = self._expand_to_replica(next_obs, self._feature_spec)

        target = next_obs - observations

        # we mask out FIRST as its state is invalid
        loss_masks = (time_step.step_type != StepType.FIRST).to(torch.float32)
        global_steps = alf.summary.get_global_counter()
        # entropy_regularization = 1. / np.sqrt(global_steps)
        entropy_regularization = 1. / global_steps
        # entropy_regularization = 1. / (global_steps * global_steps)
        dynamics_step = super().train_step(
            ((observations, actions), target),
            entropy_regularization=entropy_regularization,
            loss_mask=loss_masks)

        # loss_info, _ = self.update_with_gradient(dynamics_step.info)

        forward_loss = dynamics_step.info.loss
        forward_loss = forward_loss.sum().repeat(obs.shape[0])

        info = DynamicsInfo(
            loss=LossInfo(
                loss=forward_loss, extra=dict(forward_loss=forward_loss)))

        state = state._replace(feature=target)

        return AlgStep(output=(), state=state, info=info)

    def calc_loss(self, info: DynamicsInfo):
        # Here we take mean over the loss to avoid undesired additional
        # masking from base algorithm's ``update_with_gradient``.
        # scalar_loss = nest.map_structure(torch.mean, info.loss)
        scalar_loss = info.loss.loss
        while scalar_loss.ndim > 1:
            scalar_loss = scalar_loss[0]

        return LossInfo(scalar_loss=scalar_loss[0])
        # return LossInfo(scalar_loss=())
