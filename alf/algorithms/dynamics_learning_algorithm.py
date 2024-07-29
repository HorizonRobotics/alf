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

from collections import namedtuple
from typing import Callable, Optional, Any
import torch

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import (AlgStep, Experience, LossInfo, namedtuple,
                                 StepType, TimeStep)
from alf.nest import nest
from alf.nest.utils import NestConcat, get_outer_rank
from alf.networks import Network, EncodingNetwork, DynamicsNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import dist_utils, losses, math_ops, spec_utils, tensor_utils

DynamicsState = namedtuple(
    "DynamicsState", ["feature", "network"], default_value=())
DynamicsInfo = namedtuple("DynamicsInfo", ["loss", "dist"], default_value=())


@alf.configurable
class DynamicsLearningAlgorithm(Algorithm):
    """Base Dynamics Learning Module

    This module learns the dynamics of environment with a deterministic model.
    """

    def __init__(self,
                 train_state_spec,
                 action_spec,
                 feature_spec,
                 hidden_size=256,
                 num_replicas=1,
                 dynamics_network: DynamicsNetwork = None,
                 checkpoint=None,
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
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
        """
        super().__init__(
            train_state_spec=train_state_spec,
            checkpoint=checkpoint,
            name=name)

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
        self._num_replicas = num_replicas

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

        if num_replicas > 1:
            self._dynamics_network = dynamics_network.make_parallel(
                num_replicas)
        else:
            self._dynamics_network = dynamics_network

    @property
    def num_replicas(self):
        return self._num_replicas

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
        return LossInfo(scalar_loss=scalar_loss.loss, extra=scalar_loss.loss)


@alf.configurable
class DeterministicDynamicsAlgorithm(DynamicsLearningAlgorithm):
    """Deterministic Dynamics Learning Module

    This module tries to learn the dynamics of environment with a
    deterministic model.
    """

    def __init__(self,
                 action_spec,
                 feature_spec,
                 hidden_size=256,
                 num_replicas=1,
                 dynamics_network_ctor: Optional[
                     Callable[[Any, Any], DynamicsNetwork]] = None,
                 name="DeterministicDynamicsAlgorithm"):
        """Create a DeterministicDynamicsAlgorithm.

        Args:
            hidden_size (int|tuple): size of hidden layer(s)
            num_replicas (int): number of network replicas to be used
                in the ensemble for dynamics learning
            dynamics_network_ctor: Used to construct a network for predicting the change of
                the next feature based on the previous feature and action.
                It should accept input with spec of the format
                [feature_spec, encoded_action_spec] and output a tensor of the
                shape feature_spec. For discrete action case, encoded_action
                is a one-hot representation of the action. For continuous
                action, encoded action is the original action.

        """
        dynamics_network = None
        if dynamics_network_ctor is not None:
            dynamics_network = dynamics_network_ctor(
                input_tensor_spec=(feature_spec, action_spec),
                output_tensor_spec=feature_spec)

        if dynamics_network is not None:
            dynamics_network_state_spec = dynamics_network.state_spec

        if num_replicas > 1:
            ens_feature_spec = TensorSpec(
                (num_replicas, feature_spec.shape[0]), dtype=torch.float32)
        else:
            ens_feature_spec = feature_spec

        super().__init__(
            train_state_spec=DynamicsState(
                feature=ens_feature_spec, network=dynamics_network_state_spec),
            action_spec=action_spec,
            feature_spec=feature_spec,
            num_replicas=num_replicas,
            hidden_size=hidden_size,
            dynamics_network=dynamics_network,
            name=name)

    def _expand_to_replica(self, inputs, spec):
        """Expand the inputs of shape [B, ...] to [B, n, ...] if n > 1,
            where n is the number of replicas. When n = 1, the unexpanded
            inputs will be returned.
        Args:
            inputs (Tensor): the input tensor to be expanded
            spec (TensorSpec): the spec of the unexpanded inputs. It is used to
                determine whether the inputs is already an expanded one. If it
                is already expanded, inputs will be returned without any
                further processing.
        Returns:
            Tensor: the expanded inputs or the original inputs.
        """
        outer_rank = get_outer_rank(inputs, spec)
        if outer_rank == 1 and self._num_replicas > 1:
            return inputs.unsqueeze(1).expand(-1, self._num_replicas,
                                              *inputs.shape[1:])
        else:
            return inputs

    def predict_step(self, time_step: TimeStep, state: DynamicsState):
        """Predict the next observation given the current time_step.
                The next step is predicted using the ``prev_action`` from
                time_step and the ``feature`` from state.
        Args:
            time_step (TimeStep): time step structure. The ``prev_action`` from
                time_step will be used for predicting feature of the next step.
                It should be a Tensor of the shape [B, ...], or [B, n, ...] when
                n > 1, where n denotes the number of dynamics network replicas.
                When the input tensor has the shape of [B, ...] and n > 1,
                it will be first expanded to [B, n, ...] to match the number of
                dynamics network replicas.
            state (DynamicsState): state for dynamics learning with the
                following fields:
                - feature (Tensor): features of the previous observation of the
                    shape [B, ...], or [B, n, ...] when n > 1. When
                    ``state.feature`` has the shape of [B, ...] and n > 1,
                    it will be first expanded to [B, n, ...] to match the
                    number of dynamics network replicas.
                    It is used for predicting the feature of the next step
                    together with ``time_step.prev_action``.
                - network: the input state of the dynamics network
        Returns:
            AlgStep:
                outputs (Tensor): predicted feature of the next step, of the
                    shape [B, ...], or [B, n, ...] when n > 1.
                state (DynamicsState): with the following fields
                    - feature (Tensor): [B, n, ...] (or [B, n, ...] when n > 1)
                        shape tensor representing
                        the predicted feature of the next step
                    - network: the updated state of the dynamics network
                info: empty tuple ()
        """
        action = self._encode_action(time_step.prev_action)
        obs = state.feature

        # perform preprocessing
        observations = self._expand_to_replica(obs, self._feature_spec)
        actions = self._expand_to_replica(action, self._action_spec)

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
                feature=self._expand_to_replica(feature, self._feature_spec))
        return updated_state

    def train_step(self, time_step: TimeStep, state: DynamicsState):
        """
        Args:
            time_step (TimeStep): time step structure. The ``prev_action`` from
                time_step will be used for predicting feature of the next step.
                It should be a Tensor of the shape [B, ...] or [B, n, ...] when
                n > 1, where n denotes the number of dynamics network replicas.
                When the input tensor has the shape of [B, ...] and n > 1, it
                will be first expanded to [B, n, ...] to match the number of
                dynamics network replicas.
            state (DynamicsState): state for dynamics learning with the
                following fields:
                - feature (Tensor): features of the previous observation of the
                    shape [B, ...] or [B, n, ...] when n > 1. When
                    ``state.feature`` has the shape of [B, ...] and n > 1, it
                    will be first expanded to [B, n, ...] to match the number
                    of dynamics network replicas.
                    It is used for predicting the feature of the next step
                    together with ``time_step.prev_action``.
                - network: the input state of the dynamics network
        Returns:
            AlgStep:
                outputs: empty tuple ()
                state (DynamicsState): with the following fields
                    - feature (Tensor): [B, ...] (or [B, n, ...] when n > 1)
                        shape tensor representing the predicted feature of
                        the next step
                    - network: the updated state of the dynamics network
                info (DynamicsInfo): with the following fields being updated:
                    - loss (LossInfo):
        """
        feature = time_step.observation
        feature = self._expand_to_replica(feature, self._feature_spec)
        dynamics_step = self.predict_step(time_step, state)
        forward_pred = dynamics_step.output
        forward_loss = (feature - forward_pred)**2

        if forward_loss.ndim > 2:
            # [B, n, ...] -> [B, ...]
            forward_loss = forward_loss.sum(1)
        if forward_loss.ndim > 1:
            forward_loss = 0.5 * forward_loss.mean(
                list(range(1, forward_loss.ndim)))

        # we mask out FIRST as its state is invalid
        valid_masks = (time_step.step_type != StepType.FIRST).to(torch.float32)
        forward_loss = forward_loss * valid_masks

        info = DynamicsInfo(
            loss=LossInfo(
                loss=forward_loss, extra=dict(forward_loss=forward_loss)))

        state = state._replace(feature=feature)

        return AlgStep(output=(), state=state, info=info)


@alf.configurable
class StochasticDynamicsAlgorithm(DeterministicDynamicsAlgorithm):
    """Stochastic Dynamics Learning Module

    This module learns the dynamics of environment with a
    stochastic model.
    """

    def __init__(self,
                 action_spec,
                 feature_spec,
                 hidden_size=256,
                 num_replicas=1,
                 dynamics_network_ctor: Optional[
                     Callable[[Any, Any], DynamicsNetwork]] = None,
                 name="StochasticDynamicsAlgorithm"):
        """Create a StochasticDynamicsAlgorithm.

        Args:
            hidden_size (int|tuple): size of hidden layer(s)
            num_replicas (int): number of network replicas to be used
                in the ensemble for dynamics learning
            dynamics_network_ctor: used to construct network for predicting next
                feature based on the previous feature and action. It should
                accept input with spec [feature_spec, encoded_action_spec] and
                output a tensor of shape feature_spec. For discrete action,
                encoded_action is an one-hot representation of the action. For
                continuous action, encoded action is the original action.
        """
        super().__init__(
            action_spec=action_spec,
            feature_spec=feature_spec,
            hidden_size=hidden_size,
            num_replicas=num_replicas,
            dynamics_network_ctor=dynamics_network_ctor)
        assert self._dynamics_network._prob, "should use probabilistic network"

    def predict_step(self, time_step: TimeStep, state: DynamicsState):
        """Predict the next observation given the current time_step.
                The next step is predicted using the ``prev_action`` from
                time_step and the ``feature`` from state.
        Args:
            time_step (TimeStep): time step structure. The ``prev_action`` from
                time_step will be used for predicting feature of the next step.
                It should be a Tensor of the shape [B, ...], or [B, n, ...] when
                n > 1, where n denotes the number of dynamics network replicas.
                When the input tensor has the shape of [B, ...] and n > 1,
                it will be first expanded to [B, n, ...] to match the number of
                dynamics network replicas.
            state (DynamicsState): state for dynamics learning with the
                following fields:
                - feature (Tensor): features of the previous observation of the
                    shape [B, ...], or [B, n, ...] when n > 1. When
                    ``state.feature`` has the shape of [B, ...] and n > 1,
                    it will be first expanded to [B, n, ...] to match the
                    number of dynamics network replicas.
                    It is used for predicting the feature of the next step
                    together with ``time_step.prev_action``.
                - network: the input state of the dynamics network
        Returns:
            AlgStep:
                outputs (Tensor): predicted feature of the next step, of the
                    shape [B, ...], or [B, n, ...] when n > 1.
                state (DynamicsState): with the following fields
                    - feature (Tensor): [B, n, ...] (or [B, n, ...] when n > 1)
                        shape tensor representing
                        the predicted feature of the next step
                    - network: the updated state of the dynamics network
                info (DynamicsInfo): with the following fields being updated:
                    - dist (td.Distribution): the predictive distribution which
                        can be used for further calculation or summarization.
        """
        action = self._encode_action(time_step.prev_action)
        obs = state.feature

        # perform preprocessing
        observations = self._expand_to_replica(obs, self._feature_spec)
        actions = self._expand_to_replica(action, self._action_spec)

        dist, network_states = self._dynamics_network((observations, actions),
                                                      state=state.network)

        forward_deltas = dist.sample()

        forward_preds = observations + forward_deltas
        state = state._replace(feature=forward_preds, network=network_states)
        return AlgStep(
            output=forward_preds, state=state, info=DynamicsInfo(dist=dist))

    def train_step(self, time_step: TimeStep, state: DynamicsState):
        """
        Args:
            time_step (TimeStep): time step structure. The ``prev_action`` from
                time_step will be used for predicting feature of the next step.
                It should be a Tensor of the shape [B, ...] or [B, n, ...] when
                n > 1, where n denotes the number of dynamics network replicas.
                When the input tensor has the shape of [B, ...] and n > 1, it
                will be first expanded to [B, n, ...] to match the number of
                dynamics network replicas.
            state (DynamicsState): state for dynamics learning with the
                following fields:
                - feature (Tensor): features of the previous observation of the
                    shape [B, ...] or [B, n, ...] when n > 1. When
                    ``state.feature`` has the shape of [B, ...] and n > 1, it
                    will be first expanded to [B, n, ...] to match the number
                    of dynamics network replicas.
                    It is used for predicting the feature of the next step
                    together with ``time_step.prev_action``.
                - network: the input state of the dynamics network
        Returns:
            AlgStep:
                outputs: empty tuple ()
                state (DynamicsState): with the following fields
                    - feature (Tensor): [B, ...] (or [B, n, ...] when n > 1)
                        shape tensor representing the predicted feature of
                        the next step
                    - network: the updated state of the dynamics network
                info (DynamicsInfo): with the following fields being updated:
                    - loss (LossInfo):
                    - dist (td.Distribution): the predictive distribution which
                        can be used for further calculation or summarization.
        """

        feature = time_step.observation
        feature = self._expand_to_replica(feature, self._feature_spec)
        dynamics_step = self.predict_step(time_step, state)

        dist = dynamics_step.info.dist
        forward_loss = -dist.log_prob(feature - state.feature)

        if forward_loss.ndim > 2:
            # [B, n, ...] -> [B, ...]
            forward_loss = forward_loss.sum(1)
        if forward_loss.ndim > 1:
            forward_loss = forward_loss.mean(list(range(1, forward_loss.ndim)))

        valid_masks = (time_step.step_type != StepType.FIRST).to(torch.float32)

        forward_loss = forward_loss * valid_masks

        info = DynamicsInfo(
            loss=LossInfo(
                loss=forward_loss, extra=dict(forward_loss=forward_loss)),
            dist=dist)
        state = state._replace(feature=feature)

        return AlgStep(output=(), state=state, info=info)
