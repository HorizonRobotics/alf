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
import torch

from alf.algorithms.algorithm import Algorithm
from alf.data_structures import (AlgStep, Experience, LossInfo, namedtuple,
                                 StepType, TimeStep)
from alf.nest import nest
from alf.nest.utils import NestConcat
from alf.networks import Network, EncodingNetwork, DynamicsNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import dist_utils, losses, math_ops, spec_utils, tensor_utils

DynamicsState = namedtuple(
    "DynamicsState", ["feature", "network"], default_value=())
DynamicsInfo = namedtuple("DynamicsInfo", ["loss"])

MultiStepInfo = namedtuple(
    'MultiStepInfo',
    [
        # actual actions taken in the next unroll_steps + 1 steps
        # [B, unroll_steps + 1, ...]
        'action',

        # The flag to indicate whether to include this target into loss
        # [B, unroll_steps + 1]
        'mask',

        # nest for targets
        # [B, unroll_steps + 1, ...]
        'target'
    ])


@gin.configurable
class DynamicsLearningAlgorithm(Algorithm):
    """Base Dynamics Learning Module

    This module trys to learn the dynamics of environment.
    """

    def __init__(self,
                 train_state_spec,
                 action_spec,
                 feature_spec,
                 hidden_size=256,
                 dynamics_network: DynamicsNetwork = None,
                 name="DynamicsLearningAlgorithm"):
        """Create a DynamicsLearningAlgorithm.

        Args:
            hidden_size (int|tuple): size of hidden layer(s)
            dynamics_network (Network): network for predicting the change of
                the next feature based on the previous feature and action.
                It should accept input with spec of the format
                [feature_spec, encoded_action_spec] and output a tensor of the
                shape feature_spec. For discrete action case, encoded_action
                is a one-hot representation of the action. For continuous
                action, encoded action is the original action.
        """
        super().__init__(train_state_spec=train_state_spec, name=name)

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

        if isinstance(hidden_size, int):
            hidden_size = (hidden_size, )

        if dynamics_network is None:
            encoded_action_spec = TensorSpec((self._num_actions, ),
                                             dtype=torch.float32)
            dynamics_network = DynamicsNetwork(
                name="dynamics_net",
                input_tensor_spec=(feature_spec, encoded_action_spec),
                preprocessing_combiner=NestConcat(),
                fc_layer_params=hidden_size,
                output_tensor_spec=flat_feature_spec[0])

        self._dynamics_network = dynamics_network

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
        pass

    def get_state_specs(self):
        """Get the state specs of the current module.
        This function is mainly used for constructing the nested state specs
        by the upper-level module.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def train_step(self, time_step: TimeStep, state: DynamicsState):
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
        raise NotImplementedError

    def calc_loss(self, info: DynamicsInfo):
        # Here we take mean over the loss to avoid undesired additional
        # masking from base algorithm's ``update_with_gradient``.
        scalar_loss = nest.map_structure(torch.mean, info.loss)
        return LossInfo(scalar_loss=scalar_loss, extra=loss.extra)


@gin.configurable
class DeterministicDynamicsAlgorithm(DynamicsLearningAlgorithm):
    """Deterministic Dynamics Learning Module

    This module trys to learn the dynamics of environment with a
    determinstic model.
    """

    def __init__(self,
                 action_spec,
                 feature_spec,
                 hidden_size=256,
                 num_replicas=1,
                 dynamics_network: DynamicsNetwork = None,
                 name="DeterministicDynamicsAlgorithm"):
        """Create a DeterministicDynamicsAlgorithm.

        Args:
            hidden_size (int|tuple): size of hidden layer(s)
            num_replicas (int): number of dynamics network replicas
            dynamics_network (Network): network for predicting the change of
                the next feature based on the previous feature and action.
                It should accept input with spec of the format
                [feature_spec, encoded_action_spec] and output a tensor of the
                shape feature_spec. For discrete action case, encoded_action
                is a one-hot representation of the action. For continuous
                action, encoded action is the original action.
        """
        if dynamics_network is not None:
            dynamics_network_state_spec = dynamics_network.state_spec

        ens_feature_spec = TensorSpec((num_replicas, feature_spec.shape[0]),
                                      dtype=torch.float32)

        super().__init__(
            train_state_spec=DynamicsState(
                feature=ens_feature_spec, network=dynamics_network_state_spec),
            action_spec=action_spec,
            feature_spec=feature_spec,
            hidden_size=hidden_size,
            dynamics_network=dynamics_network,
            name=name)
        self._num_replicas = num_replicas

    def _expand_to_replica(self, inputs):
        # inputs [B n l]
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0).expand(self._num_replicas,
                                                *inputs.shape)
            inputs = inputs.transpose(0, 1)  # [B n l]
        return inputs

    def predict_step(self, time_step: TimeStep, state: DynamicsState):
        """Predict the current observation using ``time_step.prev_action``
            and the feature of the previous observation from ``state``.
            Note that time_step.observation is not used for the prediction.
        """
        action = self._encode_action(time_step.prev_action)
        obs = state.feature

        # perform preprocessing
        observations = self._expand_to_replica(obs)
        actions = self._expand_to_replica(action)

        forward_deltas, network_state = self._dynamics_network(
            (observations, actions), state=state.network)
        forward_pred = observations + forward_deltas

        state = state._replace(feature=forward_pred, network=network_state)
        return AlgStep(output=forward_pred, state=state, info=())

    def update_state(self, time_step: TimeStep, state: DynamicsState):
        """Update the state based on TimeStep data. This function is
            mainly used during rollout together with a planner. This
            function is necessary as we need to update the feature in
            DynamicsState with those of the current observation, after
            each step of rollout.
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (DynamicsState): state for DeterministicDynamicsAlgorithm
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
                feature=self._expand_to_replica(feature))
        return updated_state

    def train_step(self, time_step: TimeStep, state: DynamicsState):
        """
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (DynamicsState): state for dynamics learning (previous observation)
        Returns:
            AlgStep:
                output: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        feature = time_step.observation
        feature = self._expand_to_replica(feature)
        dynamics_step = self.predict_step(time_step, state)
        forward_pred = dynamics_step.output
        forward_loss = (feature - forward_pred)**2

        # [B, n, obs_dim]
        forward_loss = 0.5 * forward_loss.mean(
            list(range(2, forward_loss.ndim))).sum(1)

        # we mask out FIRST as its state is invalid
        valid_masks = (time_step.step_type != StepType.FIRST).to(torch.float32)
        forward_loss = forward_loss * valid_masks

        info = DynamicsInfo(
            loss=LossInfo(
                loss=forward_loss, extra=dict(forward_loss=forward_loss)))

        state = state._replace(feature=feature)

        return AlgStep(output=(), state=state, info=info)


@gin.configurable
class StochasticDynamicsAlgorithm(DeterministicDynamicsAlgorithm):
    """Stochastic Dynamics Learning Module

    This module trys to learn the dynamics of environment with a
    stochastic model.
    """

    def __init__(self,
                 action_spec,
                 feature_spec,
                 hidden_size=256,
                 num_replicas=1,
                 dynamics_network: DynamicsNetwork = None,
                 name="StochasticDynamicsAlgorithm"):
        """Create a StochasticDynamicsAlgorithm.

        Args:
            hidden_size (int|tuple): size of hidden layer(s)
            dynamics_network (Network): network for predicting next feature
                based on the previous feature and action. It should accept
                input with spec [feature_spec, encoded_action_spec] and output
                a tensor of shape feature_spec. For discrete action,
                encoded_action is an one-hot representation of the action.
                For continuous action, encoded action is the original action.
            # particles_per_replica (int): number of particles used for each replica
        """

        assert dynamics_network._prob, "should use probabistic network"
        super().__init__(
            action_spec=action_spec,
            feature_spec=feature_spec,
            hidden_size=hidden_size,
            num_replicas=num_replicas,
            dynamics_network=dynamics_network)

    def predict_step(self, time_step: TimeStep, state: DynamicsState):
        """Predict the next observation given the current time_step.
                The next step is predicted using the prev_action from time_step
                and the feature from state.
        """
        action = self._encode_action(time_step.prev_action)
        obs = state.feature

        # perform preprocessing
        observations = self._expand_to_replica(obs)
        actions = self._expand_to_replica(action)

        out, network_states = self._dynamics_network((observations, actions),
                                                     state=state.network)

        # assume Diagonal Multidimensional Normal distribution
        assert isinstance(out, dist_utils.DiagMultivariateNormal), \
            "only DiagMultivariateNormal distribution is supported"
        forward_deltas_mean = out.mean
        forward_deltas = forward_deltas_mean + torch.randn_like(
            forward_deltas_mean) * out.stddev

        logvar = (out.stddev**2).log()

        forward_preds = observations + forward_deltas
        state = state._replace(feature=forward_preds, network=network_states)
        return AlgStep(output=forward_preds, state=state, info=logvar)

    def train_step(self, time_step: TimeStep, state: DynamicsState):
        """
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """

        feature = time_step.observation
        feature = self._expand_to_replica(feature)
        dynamics_step = self.predict_step(time_step, state)

        forward_pred = dynamics_step.output
        logvar = dynamics_step.info
        inv_var = torch.exp(-logvar)

        forward_loss = (feature - forward_pred)**2 * inv_var + logvar

        # [B, n, obs_dim]
        forward_loss = forward_loss.mean(list(range(2,
                                                    forward_loss.ndim))).sum(1)

        valid_masks = (time_step.step_type != StepType.FIRST).to(torch.float32)

        forward_loss = forward_loss * valid_masks

        info = DynamicsInfo(
            loss=LossInfo(
                loss=forward_loss, extra=dict(forward_loss=forward_loss)))
        state = state._replace(feature=feature)

        return AlgStep(output=(), state=state, info=info)

    @torch.no_grad()
    def preprocess_experience(self, experience: Experience):
        """Fill experience.rollout_info with PredictiveRepresentationLearnerInfo

        Note that the shape of experience is [B, T, ...]
        """
        assert experience.batch_info != ()
        batch_info: BatchInfo = experience.batch_info
        replay_buffer: ReplayBuffer = experience.replay_buffer
        mini_batch_length = experience.step_type.shape[1]

        with alf.device(replay_buffer.device):
            # [B, 1]
            positions = convert_device(batch_info.positions).unsqueeze(-1)
            # [B, 1]
            env_ids = convert_device(batch_info.env_ids).unsqueeze(-1)

            # [B, T]
            positions = positions + torch.arange(mini_batch_length)

            # [B, T]
            steps_to_episode_end = replay_buffer.steps_to_episode_end(
                positions, env_ids)
            # [B, T]
            episode_end_positions = positions + steps_to_episode_end

            # [B, T, unroll_steps+1]
            positions = positions.unsqueeze(-1) + torch.arange(
                self._num_unroll_steps + 1)
            # [B, 1, 1]
            env_ids = env_ids.unsqueeze(-1)
            # [B, T, 1]
            episode_end_positions = episode_end_positions.unsqueeze(-1)

            # [B, T, unroll_steps+1]
            mask = positions <= episode_end_positions

            # mask out first step
            mask = positions != positions[..., 0:1]

            # [B, T, unroll_steps+1]
            positions = torch.min(positions, episode_end_positions)
            # print("===mask")
            # print(mask)

            # [B, T, unroll_steps+1]
            target = replay_buffer.get_field(self._target_fields, env_ids,
                                             positions)

            # [B, T, unroll_steps+1]
            action = replay_buffer.get_field('action', env_ids, positions)

            rollout_info = MultiStepInfo(
                action=action, mask=mask, target=target)

        rollout_info = convert_device(rollout_info)

        return experience._replace(rollout_info=rollout_info)
