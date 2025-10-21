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

import torch
import torch.nn as nn

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import AlgStep, LossInfo, TimeStep
from alf.tensor_specs import TensorSpec


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

    def __init__(
        self,
        observation_spec,
        action_spec,
        algorithm_ctor: Callable,
        num_copies: int = 2,
        reward_spec=TensorSpec(()),
        env=None,
        config: Optional[TrainerConfig] = None,
        checkpoint: Optional[str] = None,
        debug_summaries: bool = False,
        name: str = "SimpleConcurrentAlgorithm",
    ):
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
            observation_spec=observation_spec, action_spec=action_spec
        )
        is_on_policy = temp_alg.on_policy

        # Collect state specs from temporary algorithm
        train_state_spec = [temp_alg.train_state_spec for _ in range(num_copies)]
        rollout_state_spec = [temp_alg.rollout_state_spec for _ in range(num_copies)]
        predict_state_spec = [temp_alg.predict_state_spec for _ in range(num_copies)]

        # Clean up temporary algorithm
        del temp_alg

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
            name=name,
        )

        self._num_copies = num_copies

        # Validate that batch size will be compatible with num_copies
        if env and hasattr(env, "batch_size"):
            assert (
                env.batch_size % num_copies == 0
            ), f"Environment batch size {env.batch_size} must be a multiple of num_copies {num_copies}"

        # Create K independent algorithm copies
        self._algorithms = nn.ModuleList(
            [
                algorithm_ctor(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    reward_spec=reward_spec,
                    env=None,  # Only root algorithm gets env
                    config=config,
                    debug_summaries=debug_summaries,
                    name=f"{name}_copy_{i}",
                )
                for i in range(num_copies)
            ]
        )

    def get_initial_predict_state(self, batch_size):
        """Get initial predict state for all algorithm copies."""
        # For single environment evaluation, use only the first algorithm copy
        if batch_size == 1:
            return self._algorithms[0].get_initial_predict_state(batch_size)
        else:
            return [
                alg.get_initial_predict_state(batch_size) for alg in self._algorithms
            ]

    def get_initial_rollout_state(self, batch_size):
        """Get initial rollout state for all algorithm copies."""
        return [alg.get_initial_rollout_state(batch_size) for alg in self._algorithms]

    def get_initial_train_state(self, batch_size):
        """Get initial train state for all algorithm copies."""
        return [alg.get_initial_train_state(batch_size) for alg in self._algorithms]

        # Setup replay buffers for off-policy algorithms
        if not is_on_policy and config:
            for alg in self._algorithms:
                if hasattr(alg, "set_replay_buffer"):
                    alg.set_replay_buffer(
                        num_envs=env.batch_size if env else 1,
                        max_length=config.replay_buffer_length,
                        prioritized_sampling=config.priority_replay,
                    )

    def _trainable_attributes_to_ignore(self):
        """Prevent parent optimizer from managing sub-algorithm parameters."""
        return ["_algorithms"]

    def _route_batch_to_algorithms(self, time_step, state):
        """Route each batch element to the appropriate algorithm copy.

        Returns:
            dict: mapping algorithm index -> (sliced_time_step, sliced_state, batch_indices)
        """
        batch_size = alf.nest.get_nest_batch_size(time_step.observation)
        device = next(iter(alf.nest.flatten(time_step.observation))).device

        routing = {}
        for i in range(self._num_copies):
            # Use efficient slicing: elements where j % K == i
            batch_indices = torch.arange(i, batch_size, self._num_copies, device=device)

            if len(batch_indices) == 0:
                continue

            # Slice time_step for this algorithm
            sliced_time_step = alf.nest.map_structure(
                lambda x: x[batch_indices], time_step
            )

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
            # Get first tensor to determine structure
            first_tensor = next(iter(tensor_by_alg.values()))[0]

            # Handle non-tensor values (e.g., empty tuples)
            if not isinstance(first_tensor, torch.Tensor):
                return first_tensor

            # Determine output shape
            out_shape = [batch_size] + list(first_tensor.shape[1:])
            result = torch.zeros(
                out_shape, dtype=first_tensor.dtype, device=first_tensor.device
            )

            # Scatter each algorithm's outputs
            for tensor, batch_indices in tensor_by_alg.values():
                result[batch_indices] = tensor

            return result

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

        for alg_idx, (sliced_time_step, sliced_state, batch_indices) in routing.items():
            alg_step = self._algorithms[alg_idx].rollout_step(
                sliced_time_step, sliced_state
            )

            outputs_dict[alg_idx] = (alg_step.output, batch_indices)
            new_states[alg_idx] = alg_step.state
            infos_dict[alg_idx] = (alg_step.info, batch_indices)

        # Scatter outputs back to full batch
        output = self._scatter_outputs(outputs_dict, batch_size)
        info = self._scatter_outputs(infos_dict, batch_size)

        return AlgStep(output=output, state=new_states, info=info)

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

        for alg_idx, (sliced_time_step, sliced_state, batch_indices) in routing.items():
            # Slice rollout_info
            sliced_rollout_info = alf.nest.map_structure(
                lambda x: x[batch_indices], rollout_info
            )

            alg_step = self._algorithms[alg_idx].train_step(
                sliced_time_step, sliced_state, sliced_rollout_info
            )

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

        device = next(iter(alf.nest.flatten(info))).device

        for alg_idx in range(self._num_copies):
            # Use efficient slicing: elements where j % K == alg_idx
            batch_indices = torch.arange(
                alg_idx, batch_size, self._num_copies, device=device
            )

            if len(batch_indices) == 0:
                continue

            # Slice info for this algorithm
            sliced_info = alf.nest.map_structure(lambda x: x[:, batch_indices], info)

            # Compute loss for this algorithm
            loss_info = self._algorithms[alg_idx].calc_loss(sliced_info)

            # Accumulate losses using add_ignore_empty to handle () gracefully
            total_loss = alf.utils.math_ops.add_ignore_empty(total_loss, loss_info.loss)
            total_priority = alf.utils.math_ops.add_ignore_empty(
                total_priority, loss_info.priority
            )

            extra_dict[f"alg_{alg_idx}"] = loss_info.extra

        return LossInfo(loss=total_loss, priority=total_priority, extra=extra_dict)

    def predict_step(self, inputs: TimeStep, state) -> AlgStep:
        """Route batch elements to algorithm copies for prediction.

        Args:
            inputs: TimeStep with shape [B, ...]
            state: List of states, one per algorithm copy, or single state for evaluation

        Returns:
            AlgStep with predictions
        """
        batch_size = alf.nest.get_nest_batch_size(inputs.observation)

        # Handle single environment evaluation (batch_size=1, single state)
        if batch_size == 1 and not isinstance(state, list):
            alg_step = self._algorithms[0].predict_step(inputs, state)
            return AlgStep(
                output=alg_step.output, state=alg_step.state, info=alg_step.info
            )

        # Handle multi-environment training (batch_size > 1, list of states)
        routing = self._route_batch_to_algorithms(inputs, state)

        outputs_dict = {}
        new_states = [None] * self._num_copies
        infos_dict = {}

        for alg_idx, (sliced_time_step, sliced_state, batch_indices) in routing.items():
            alg_step = self._algorithms[alg_idx].predict_step(
                sliced_time_step, sliced_state
            )

            outputs_dict[alg_idx] = (alg_step.output, batch_indices)
            new_states[alg_idx] = alg_step.state
            infos_dict[alg_idx] = (alg_step.info, batch_indices)

        output = self._scatter_outputs(outputs_dict, batch_size)
        info = self._scatter_outputs(infos_dict, batch_size)

        return AlgStep(output=output, state=new_states, info=info)
