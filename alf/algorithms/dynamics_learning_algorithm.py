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

import gin.tf
import tensorflow as tf

from tf_agents.networks.network import Network
import tf_agents.specs.tensor_spec as tensor_spec

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.utils.encoding_network import EncodingNetwork
from alf.data_structures import ActionTimeStep, namedtuple

DynamicsState = namedtuple(
    "DynamicsState", ["feature", "network"], default_value=())
DynamicsInfo = namedtuple("DynamicsInfo", ["loss"])


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
                 dynamics_network: Network = None,
                 name="DynamicsLearningAlgorithm"):
        """Create a DynamicsLearningAlgorithm.

        Args:
            hidden_size (int|tuple): size of hidden layer(s)
            dynamics_network (Network): network for predicting next feature
                based on the previous feature and action. It should accept
                input with spec [feature_spec, encoded_action_spec] and output
                a tensor of shape feature_spec. For discrete action,
                encoded_action is an one-hot representation of the action.
                For continuous action, encoded action is the original action.
        """
        super().__init__(train_state_spec=train_state_spec, name=name)

        flat_action_spec = tf.nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, "doesn't support nested action_spec"

        flat_feature_spec = tf.nest.flatten(feature_spec)
        assert len(
            flat_feature_spec) == 1, "doesn't support nested feature_spec"

        action_spec = flat_action_spec[0]

        if tensor_spec.is_discrete(action_spec):
            self._num_actions = action_spec.maximum - action_spec.minimum + 1
        else:
            self._num_actions = action_spec.shape[-1]

        self._action_spec = action_spec
        self._feature_spec = feature_spec

        feature_dim = flat_feature_spec[0].shape[-1]

        if isinstance(hidden_size, int):
            hidden_size = (hidden_size, )

        if dynamics_network is None:
            encoded_action_spec = tensor_spec.TensorSpec((self._num_actions, ),
                                                         dtype=tf.float32)
            dynamics_network = EncodingNetwork(
                name="dynamics_net",
                input_tensor_spec=[feature_spec, encoded_action_spec],
                fc_layer_params=hidden_size,
                last_layer_size=feature_dim)

        self._dynamics_network = dynamics_network

    def _encode_action(self, action):
        if tensor_spec.is_discrete(self._action_spec):
            return tf.one_hot(indices=action, depth=self._num_actions)
        else:
            return action

    def update_state(self, time_step: ActionTimeStep, state: DynamicsState):
        """Update the state based on ActionTimeStep data. This function is
            mainly used during rollout together with a planner
        Args:
            time_step (ActionTimeStep): input data for dynamics learning
            state (Tensor): state for DML (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state: feature
                info (DynamicsInfo):
        """
        pass

    def get_state_specs(self):
        """Get the state specs of the current module.
        This function is mainly used for constructing the nested state specs
        by the upper-level module.
        """
        pass

    def train_step(self, time_step: ActionTimeStep, state: DynamicsState):
        """
        Args:
            time_step (ActionTimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        pass

    def calc_loss(self, info: DynamicsInfo):
        loss = tf.nest.map_structure(tf.reduce_mean, info.loss)
        return LossInfo(
            loss=info.loss, scalar_loss=loss.loss, extra=loss.extra)


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
                 dynamics_network: Network = None,
                 name="DeterministicDynamicsAlgorithm"):
        """Create a DeterministicDynamicsAlgorithm.

        Args:
            hidden_size (int|tuple): size of hidden layer(s)
            dynamics_network (Network): network for predicting next feature
                based on the previous feature and action. It should accept
                input with spec [feature_spec, encoded_action_spec] and output
                a tensor of shape feature_spec. For discrete action,
                encoded_action is an one-hot representation of the action.
                For continuous action, encoded action is the original action.
        """
        if dynamics_network is not None:
            dynamics_network_state_spec = dynamics_network.state_spec
        else:
            dynamics_network_state_spec = ()

        super().__init__(
            train_state_spec=DynamicsState(
                feature=feature_spec, network=dynamics_network_state_spec),
            action_spec=action_spec,
            feature_spec=feature_spec,
            hidden_size=hidden_size,
            dynamics_network=dynamics_network,
            name=name)

    def predict(self, time_step: ActionTimeStep, state: DynamicsState):
        """Predict the next observation given the current time_step.
                The next step is predicted using the prev_action from time_step
                and the feature from state.
        """
        action = self._encode_action(time_step.prev_action)
        obs = state.feature
        forward_delta, network_state = self._dynamics_network(
            inputs=[obs, action], network_state=state.network)
        forward_pred = obs + forward_delta
        state = state._replace(feature=forward_pred, network=network_state)
        return AlgorithmStep(outputs=forward_pred, state=state, info=())

    def update_state(self, time_step: ActionTimeStep, state: DynamicsState):
        """Update the state based on ActionTimeStep data. This function is
            mainly used during rollout together with a planner
        Args:
            time_step (ActionTimeStep): input data for dynamics learning
            state (Tensor): state for DML (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state: feature
                info (DynamicsInfo):
        """
        feature = time_step.observation
        return state._replace(feature=feature)

    def train_step(self, time_step: ActionTimeStep, state: DynamicsState):
        """
        Args:
            time_step (ActionTimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        feature = time_step.observation
        dynamics_step = self.predict(time_step, state)
        forward_pred = dynamics_step.outputs
        forward_loss = 0.5 * tf.reduce_mean(
            tf.square(feature - forward_pred), axis=-1)

        info = DynamicsInfo(
            loss=LossInfo(
                loss=forward_loss, extra=dict(forward_loss=forward_loss)))
        state = DynamicsState(feature=feature)

        return AlgorithmStep(outputs=(), state=state, info=info)
