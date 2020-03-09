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
"""Runs multiple environments in parallel processes and steps them in batch.

Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/parallel_py_environment.py
"""

from absl import logging
import gin
import torch

from alf.environments import torch_environment
from alf.environments.process_environment import ProcessEnvironment
import alf.nest as nest


@gin.configurable
class ParallelTorchEnvironment(torch_environment.TorchEnvironment):
    """Batch together environments and simulate them in external processes.

    The environments are created in external processes by calling the provided
    callables. This can be an environment class, or a function creating the
    environment and potentially wrapping it. The returned environment should not
    access global variables.
    """

    def __init__(self,
                 env_constructors,
                 start_serially=True,
                 blocking=False,
                 flatten=False):
        """Batch together environments and simulate them in external processes.

        The environments can be different but must use the same action and
        observation specs.

        Args:
            env_constructors (list[Callable]): a list of callable environment creators.
            start_serially (bool): whether to start environments serially or in parallel.
            blocking (bool): whether to step environments one after another.
            flatten (bool): whether to use flatten action and time_steps during
                communication to reduce overhead.

        Raises:
            ValueError: If the action or observation specs don't match.
        """
        super(ParallelTorchEnvironment, self).__init__()
        self._envs = []
        self._env_ids = []
        for env_id, ctor in enumerate(env_constructors):
            env = ProcessEnvironment(ctor, env_id=env_id, flatten=flatten)
            self._envs.append(env)
            self._env_ids.append(env_id)
        self._num_envs = len(env_constructors)
        self._blocking = blocking
        self._start_serially = start_serially
        self.start()
        self._action_spec = self._envs[0].action_spec()
        self._observation_spec = self._envs[0].observation_spec()
        self._time_step_spec = self._envs[0].time_step_spec()
        self._parallel_execution = True
        if any(env.action_spec() != self._action_spec for env in self._envs):
            raise ValueError(
                'All environments must have the same action spec.')
        if any(env.time_step_spec() != self._time_step_spec
               for env in self._envs):
            raise ValueError(
                'All environments must have the same time_step_spec.')
        self._flatten = flatten

    def start(self):
        logging.info('Spawning all processes.')
        for env in self._envs:
            env.start(wait_to_start=self._start_serially)
        if not self._start_serially:
            logging.info('Waiting for all processes to start.')
            for env in self._envs:
                env.wait_start()
        logging.info('All processes started.')

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._num_envs

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def time_step_spec(self):
        return self._time_step_spec

    def _reset(self):
        """Reset all environments and combine the resulting observation.

        Returns:
            Time step with batch dimension.
        """
        time_steps = [env.reset(self._blocking) for env in self._envs]
        if not self._blocking:
            time_steps = [promise() for promise in time_steps]
        return self._stack_time_steps(time_steps)

    def _step(self, actions):
        """Forward a batch of actions to the wrapped environments.

        Args:
            actions: Batched action, possibly nested, to apply to the environment.

        Raises:
            ValueError: Invalid actions.

        Returns:
            Batch of observations, rewards, and done flags.
        """
        time_steps = [
            env.step(action, self._blocking)
            for env, action in zip(self._envs, self._unstack_actions(actions))
        ]
        # When blocking is False we get promises that need to be called.
        if not self._blocking:
            time_steps = [promise() for promise in time_steps]
        return self._stack_time_steps(time_steps)

    def close(self):
        """Close all external process."""
        logging.info('Closing all processes.')
        for env in self._envs:
            env.close()
        logging.info('All processes closed.')

    def _stack_time_steps(self, time_steps):
        """Given a list of TimeStep, combine to one with a batch dimension."""
        if self._flatten:
            return nest.fast_map_structure_flatten(
                lambda *arrays: torch.stack(arrays), self._time_step_spec,
                *time_steps)
        else:
            return nest.fast_map_structure(lambda *arrays: torch.stack(arrays),
                                           *time_steps)

    def _unstack_actions(self, batched_actions):
        """Returns a list of actions from potentially nested batch of actions."""
        flattened_actions = nest.flatten(batched_actions)
        if self._flatten:
            unstacked_actions = zip(*flattened_actions)
        else:
            unstacked_actions = [
                nest.pack_sequence_as(batched_actions, actions)
                for actions in zip(*flattened_actions)
            ]
        return unstacked_actions

    def seed(self, seeds):
        """Seeds the parallel environments."""
        if len(seeds) != len(self._envs):
            raise ValueError(
                'Number of seeds should match the number of parallel_envs.')

        promises = [
            env.call('seed', seed) for seed, env in zip(seeds, self._envs)
        ]
        # Block until all envs are seeded.
        return [promise() for promise in promises]
