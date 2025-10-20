# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Simple Concurrent RL Algorithm.

Routes batch elements to independent copies of a base algorithm.
"""

from typing import Callable, Optional
from collections import namedtuple
import torch
import torch.nn as nn

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import TimeStep, AlgStep, LossInfo
from alf.tensor_specs import TensorSpec


SimpleConcurrentState = namedtuple('SimpleConcurrentState',
                                    ['algorithm_states'])


@alf.configurable
class SimpleConcurrentAlgorithm(OffPolicyAlgorithm):
    """SimpleConcurrent Algorithm.

    Creates K independent copies of a base algorithm and routes batch element i
    to algorithm copy (i % K). Each copy maintains independent:
    - Parameters
    - Optimizer
    - Replay buffer
    - Training state

    This is useful for:
    - Training diverse ensemble of agents
    - Parallel exploration with different policies
    - Load balancing across different tasks
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 algorithm_ctor: Callable,
                 num_copies: int = 2,
                 reward_spec=TensorSpec(()),
                 env=None,
                 config: Optional[TrainerConfig] = None,
                 checkpoint: Optional[str] = None,
                 debug_summaries: bool = False,
                 name: str = "SimpleConcurrentAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            algorithm_ctor (Callable): Function to construct the base algorithm.
                Will be called as ``algorithm_ctor(observation_spec, action_spec, ...)``.
                Should return an RLAlgorithm (typically OffPolicyAlgorithm).
            num_copies (int): Number of independent algorithm copies (K).
            reward_spec (TensorSpec): representing the reward(s).
            env (Environment): The batched environment to interact with.
            config (TrainerConfig): config for training.
            checkpoint (str): checkpoint path in format "prefix@path".
            debug_summaries (bool): whether to create debug summaries.
            name (str): name of this algorithm.
        """
        # Infer on_policy from first algorithm copy
        # Create a temporary instance to check on_policy property
        temp_alg = algorithm_ctor(
            observation_spec=observation_spec, action_spec=action_spec)
        is_on_policy = temp_alg.on_policy

        # Collect state specs from temporary algorithm
        train_state_spec = []
        rollout_state_spec = []
        predict_state_spec = []

        for i in range(num_copies):
            train_state_spec.append(temp_alg.train_state_spec)
            rollout_state_spec.append(temp_alg.rollout_state_spec)
            predict_state_spec.append(temp_alg.predict_state_spec)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec,
            predict_state_spec=predict_state_spec,
            is_on_policy=is_on_policy,
            env=env,
            config=config,
            checkpoint=checkpoint,
            optimizer=None,  # Each sub-algorithm has its own optimizer
            debug_summaries=debug_summaries,
            name=name)

        self._num_copies = num_copies

        # Create K independent algorithm copies
        self._algorithms = nn.ModuleList([
            algorithm_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                reward_spec=reward_spec,
                env=None,  # Only root algorithm gets env
                config=config,
                debug_summaries=debug_summaries,
                name=f'{name}_copy_{i}')
            for i in range(num_copies)
        ])

        # Setup replay buffers for off-policy algorithms
        if not is_on_policy and config:
            for alg in self._algorithms:
                if hasattr(alg, 'set_replay_buffer'):
                    alg.set_replay_buffer(
                        num_envs=env.batch_size if env else 1,
                        max_length=config.replay_buffer_length,
                        prioritized_sampling=config.priority_replay)

    def _trainable_attributes_to_ignore(self):
        """Prevent parent optimizer from managing sub-algorithm parameters."""
        return ['_algorithms']

    def _route_batch_to_algorithms(self, time_step, state):
        """Route each batch element to the appropriate algorithm copy.

        Returns:
            dict: mapping algorithm index -> (sliced_time_step, sliced_state, batch_indices)
        """
        batch_size = alf.nest.get_nest_batch_size(time_step.observation)

        routing = {}
        for i in range(self._num_copies):
            # Find batch elements for this algorithm: i % K == algorithm_index
            batch_indices = torch.tensor(
                [j for j in range(batch_size) if j % self._num_copies == i],
                dtype=torch.int64)

            if len(batch_indices) == 0:
                continue

            # Slice time_step for this algorithm
            sliced_time_step = alf.nest.map_structure(
                lambda x: x[batch_indices], time_step)

            # Slice state for this algorithm
            sliced_state = state[i] if isinstance(state, list) else state

            routing[i] = (sliced_time_step, sliced_state, batch_indices)

        return routing

    def _scatter_outputs(self, outputs_by_alg, batch_size):
        """Scatter algorithm outputs back to full batch dimension.

        Args:
            outputs_by_alg: dict mapping algorithm index -> (output, batch_indices)
            batch_size: target batch size

        Returns:
            nested Tensor with shape [batch_size, ...]
        """
        # Get structure from first output
        first_output = next(iter(outputs_by_alg.values()))[0]

        def _scatter_single_tensor(tensor_by_alg):
            """Scatter a single tensor from all algorithms."""
            # Determine output shape
            sample_tensor = next(iter(tensor_by_alg.values()))
            out_shape = [batch_size] + list(sample_tensor.shape[1:])
            result = torch.zeros(
                out_shape,
                dtype=sample_tensor.dtype,
                device=sample_tensor.device)

            # Scatter each algorithm's outputs
            for alg_idx, (tensor, batch_indices) in tensor_by_alg.items():
                result[batch_indices] = tensor

            return result

        # Create dict: {alg_idx: (tensor, batch_indices)} for each leaf
        def _extract_leaf_with_indices(path_to_leaf):
            return {
                alg_idx: (alf.nest.get_field(output, path_to_leaf),
                          batch_indices)
                for alg_idx, (output, batch_indices) in outputs_by_alg.items()
            }

        # Get all paths in the nest
        flat_structure = alf.nest.flatten(first_output)
        result_flat = []

        for leaf_idx in range(len(flat_structure)):
            # For each leaf, gather from all algorithms and scatter
            leaf_by_alg = {
                alg_idx: (alf.nest.flatten(output)[leaf_idx], batch_indices)
                for alg_idx, (output, batch_indices) in outputs_by_alg.items()
            }
            result_flat.append(_scatter_single_tensor(leaf_by_alg))

        return alf.nest.pack_sequence_as(first_output, result_flat)

    def rollout_step(self, inputs: TimeStep, state) -> AlgStep:
        """Route batch elements to algorithm copies for rollout.

        Args:
            inputs: TimeStep with shape [B, ...]
            state: List of states, one per algorithm copy

        Returns:
            AlgStep with output shape [B, ...] and updated states
        """
        batch_size = alf.nest.get_nest_batch_size(inputs.observation)
        routing = self._route_batch_to_algorithms(inputs, state)

        # Collect outputs
        outputs_dict = {}  # {alg_idx: (alg_step.output, batch_indices)}
        new_states = [None] * self._num_copies
        infos_dict = {}  # {alg_idx: (alg_step.info, batch_indices)}

        for alg_idx, (sliced_time_step, sliced_state,
                      batch_indices) in routing.items():
            alg_step = self._algorithms[alg_idx].rollout_step(
                sliced_time_step, sliced_state)

            outputs_dict[alg_idx] = (alg_step.output, batch_indices)
            new_states[alg_idx] = alg_step.state
            infos_dict[alg_idx] = (alg_step.info, batch_indices)

        # Scatter outputs back to full batch
        output = self._scatter_outputs(outputs_dict, batch_size)
        info = self._scatter_outputs(infos_dict, batch_size)

        return AlgStep(
            output=output, state=new_states, info=info)

    def train_step(self, inputs: TimeStep, state, rollout_info) -> AlgStep:
        """Route batch elements to algorithm copies for training.

        Args:
            inputs: TimeStep from replay buffer, shape [B, ...]
            state: List of states, one per algorithm copy
            rollout_info: nested Tensor from rollout_step

        Returns:
            AlgStep with training info
        """
        batch_size = alf.nest.get_nest_batch_size(inputs.observation)
        routing = self._route_batch_to_algorithms(inputs, state)

        outputs_dict = {}
        new_states = [None] * self._num_copies
        infos_dict = {}

        for alg_idx, (sliced_time_step, sliced_state,
                      batch_indices) in routing.items():
            # Slice rollout_info
            sliced_rollout_info = alf.nest.map_structure(
                lambda x: x[batch_indices], rollout_info)

            alg_step = self._algorithms[alg_idx].train_step(
                sliced_time_step, sliced_state, sliced_rollout_info)

            outputs_dict[alg_idx] = (alg_step.output, batch_indices)
            new_states[alg_idx] = alg_step.state
            infos_dict[alg_idx] = (alg_step.info, batch_indices)

        output = self._scatter_outputs(outputs_dict, batch_size)
        info = self._scatter_outputs(infos_dict, batch_size)

        return AlgStep(output=output, state=new_states, info=info)

    def calc_loss(self, info) -> LossInfo:
        """Compute and aggregate losses from all algorithm copies.

        Args:
            info: nested Tensor with shape [T, B, ...]

        Returns:
            LossInfo with aggregated loss
        """
        # Route info to each algorithm
        batch_size = alf.nest.get_nest_batch_size(info)

        total_loss = ()
        total_priority = ()
        extra_dict = {}

        for alg_idx in range(self._num_copies):
            # Find batch elements for this algorithm
            batch_indices = torch.tensor(
                [j for j in range(batch_size) if j % self._num_copies == alg_idx
                 ],
                dtype=torch.int64)

            if len(batch_indices) == 0:
                continue

            # Slice info for this algorithm
            sliced_info = alf.nest.map_structure(lambda x: x[:, batch_indices],
                                                 info)

            # Compute loss for this algorithm
            loss_info = self._algorithms[alg_idx].calc_loss(sliced_info)

            # Accumulate losses using add_ignore_empty to handle () gracefully
            total_loss = alf.utils.math_ops.add_ignore_empty(
                total_loss, loss_info.loss)
            total_priority = alf.utils.math_ops.add_ignore_empty(
                total_priority, loss_info.priority)

            extra_dict[f'alg_{alg_idx}'] = loss_info.extra

        return LossInfo(
            loss=total_loss, priority=total_priority, extra=extra_dict)

    def predict_step(self, inputs: TimeStep, state) -> AlgStep:
        """Route batch elements to algorithm copies for prediction.

        Args:
            inputs: TimeStep with shape [B, ...]
            state: List of states, one per algorithm copy

        Returns:
            AlgStep with predictions
        """
        batch_size = alf.nest.get_nest_batch_size(inputs.observation)
        routing = self._route_batch_to_algorithms(inputs, state)

        outputs_dict = {}
        new_states = [None] * self._num_copies
        infos_dict = {}

        for alg_idx, (sliced_time_step, sliced_state,
                      batch_indices) in routing.items():
            alg_step = self._algorithms[alg_idx].predict_step(
                sliced_time_step, sliced_state)

            outputs_dict[alg_idx] = (alg_step.output, batch_indices)
            new_states[alg_idx] = alg_step.state
            infos_dict[alg_idx] = (alg_step.info, batch_indices)

        output = self._scatter_outputs(outputs_dict, batch_size)
        info = self._scatter_outputs(infos_dict, batch_size)

        return AlgStep(output=output, state=new_states, info=info)
