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
"""Runs multiple environments in parallel processes and steps them in batch.

Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/parallel_py_environment.py
"""

from absl import logging
import numpy
import torch

import alf
from alf.environments import alf_environment
from alf.environments.process_environment import ProcessEnvironment
import alf.nest as nest


@alf.configurable
class ParallelAlfEnvironment(alf_environment.AlfEnvironment):
    """Batch together environments and simulate them in external processes.

    The environments are created in external processes by calling the provided
    callables. This can be an environment class, or a function creating the
    environment and potentially wrapping it. The environments can be different
    but must use the same action and observation specs.

    The returned environment should not access global variables.
    """

    def __init__(self,
                 env_constructors,
                 start_serially=True,
                 blocking=False,
                 flatten=True,
                 num_spare_envs_for_reload=0,
                 torch_num_threads_per_env=1):
        """
        Args:
            env_constructors (list[Callable]): a list of callable environment creators.
            start_serially (bool): whether to start environments serially or in parallel.
            blocking (bool): whether to step environments one after another.
            flatten (bool): whether to use flatten action and time_steps during
                communication to reduce overhead.
            num_spare_envs_for_reload (int): if positive, these environments will be
                maintained in a separate queue and be used to handle slow env resets.
                The batch_size is ``len(env_constructors) - num_spare_envs_for_reload``
            torch_num_threads_per_env (int): how many threads torch will use for each
                env proc. Note that if you have lots of parallel envs, it's best
                to set this number as 1. Leave this as 'None' to skip the change.

        Raises:
            ValueError: If the action or observation specs don't match.
        """
        super(ParallelAlfEnvironment, self).__init__()
        if num_spare_envs_for_reload > 0:
            assert not blocking, "Spare envs only allowed in non-blocking mode"
        self._envs = []
        self._env_ids = []
        self._num_envs = len(env_constructors) - num_spare_envs_for_reload
        self._spare_queue = []
        self._spare_promises = []
        for env_id, ctor in enumerate(env_constructors):
            env = ProcessEnvironment(
                ctor,
                env_id=env_id,
                flatten=flatten,
                torch_num_threads_per_env=torch_num_threads_per_env)
            if env_id < self._num_envs:
                self._envs.append(env)
                self._env_ids.append(env_id)
            else:
                self._spare_queue.append(env)
        self._blocking = blocking
        # When non blocking, we reset immediately whenever LAST step is encountered,
        # and the reset promises are stored in this array:
        self._reset_ts = [None] * self._num_envs
        self._start_serially = start_serially
        self.start()
        self._action_spec = self._envs[0].action_spec()
        self._observation_spec = self._envs[0].observation_spec()
        self._reward_spec = self._envs[0].reward_spec()
        self._time_step_spec = self._envs[0].time_step_spec()
        self._env_info_spec = self._envs[0].env_info_spec()
        self._num_tasks = self._envs[0].num_tasks
        self._task_names = self._envs[0].task_names
        self._time_step_with_env_info_spec = self._time_step_spec._replace(
            env_info=self._env_info_spec)
        if any(env.is_tensor_based for env in self._envs):
            raise ValueError(
                'All environments must be array-based environments.')
        if any(env.action_spec() != self._action_spec for env in self._envs):
            raise ValueError(
                'All environments must have the same action spec.')
        if any(env.time_step_spec() != self._time_step_spec
               for env in self._envs):
            raise ValueError(
                'All environments must have the same time_step_spec.')
        self._flatten = flatten
        self._closed = False

    @property
    def envs(self):
        """The list of individual environment."""
        return self._envs

    @property
    def num_spare_envs_for_reload(self):
        return self._num_spare_envs_for_reload

    def start(self):
        logging.info('Spawning all processes.')
        for env in self._envs:
            env.start(wait_to_start=self._start_serially)
        for env in self._spare_queue:
            env.start(wait_to_start=self._start_serially)
        if not self._start_serially:
            logging.info('Waiting for all processes to start.')
            for env in self._envs:
                env.wait_start()
            for env in self._spare_queue:
                env.wait_start()
        logging.info('All processes started.')

    @property
    def is_tensor_based(self):
        return True

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._num_envs

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def task_names(self):
        return self._task_names

    def call_envs_with_same_args(self, func_name, *args, **kwargs):
        """Call each environment's named function with the same args.

        Args:
            func_name (str): name of the function to call
            *args: args to pass to the function
            **kwargs: kwargs to pass to the function

        return:
            list: list of results from each environment
        """
        return [env.call(func_name, *args, **kwargs)() for env in self._envs]

    def env_info_spec(self):
        return self._env_info_spec

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def time_step_spec(self):
        return self._time_step_spec

    def render(self, mode="rgb_array"):
        return self._envs[0].render(mode)

    @property
    def metadata(self):
        return self._envs[0].metadata

    def _reset(self):
        """Reset all environments and combine the resulting observation.

        Returns:
            Time step with batch dimension.
        """
        time_steps = [env.reset(self._blocking) for env in self._envs]

        # handle spare promises
        if not self._blocking:
            [p() for p in self._reset_ts if p is not None]
            [p() for p in self._spare_promises]
            self._reset_ts = [None] * self._num_envs
            self._spare_promises = [
                env.reset(self._blocking) for env in self._spare_queue
            ]
            time_steps = [promise() for promise in time_steps]

        time_steps = self._stack_time_steps(time_steps)

        if not self._blocking:
            self._handle_done(time_steps)

        return time_steps

    def _step(self, actions):
        """Forward a batch of actions to the wrapped environments.

        Args:
            actions: Batched action, possibly nested, to apply to the environment.

        Raises:
            ValueError: Invalid actions.

        Returns:
            Batch of observations, rewards, and done flags.
        """
        if not self._blocking:
            time_steps = self._step_or_handle_last_done(
                self._unstack_actions(actions))
        else:
            time_steps = [
                env.step(action, self._blocking) for env, action in zip(
                    self._envs, self._unstack_actions(actions))
            ]

        # When blocking is False we get promises that need to be called.
        if not self._blocking:
            time_steps = [promise() for promise in time_steps]

        time_steps = self._stack_time_steps(time_steps)

        if not self._blocking:
            self._handle_done(time_steps)

        return time_steps

    def _step_or_handle_last_done(self, actions):
        time_steps = []
        for i, (env, action) in enumerate(zip(self._envs, actions)):
            reset_ts = self._reset_ts[i]
            if reset_ts is not None:
                # reset was already called, simply use the result from previous stepping
                time_steps.append(reset_ts)
                self._reset_ts[i] = None
            else:
                time_steps.append(env.step(action, self._blocking))
        return time_steps

    def _handle_done(self, time_steps):
        for i, reset_ts in enumerate(self._reset_ts):
            env = self._envs[i]
            if time_steps.step_type[i] == alf.data_structures.StepType.LAST:
                reset_ts = env.reset(blocking=False)
                # If this episode is done, env should immediately reset and enter queue
                if self._spare_queue:
                    spare_env = self._spare_queue.pop(0)
                    spare_env_id = spare_env._env_id
                    # spare_env is becoming an active env
                    spare_env._env_id = env._env_id
                    self._envs[i] = spare_env
                    # env is becoming a spare env.
                    # save env_id of spare env, just in case.
                    # Random seeds are handled separately.
                    env._env_id = spare_env_id
                    self._spare_queue.append(env)
                    old_reset_ts = self._spare_promises.pop(0)
                    self._spare_promises.append(reset_ts)
                else:
                    # Even when not using spare envs, we still allow
                    # early call of (non-blocking) reset to speed up rollout.
                    old_reset_ts = reset_ts
                # Handle promises of the reset()
                self._reset_ts[i] = old_reset_ts
            else:
                self._reset_ts[i] = None
        return time_steps

    def sync_progress(self):
        """Sync the progress of all environments.

        ALF schedulers need to have progress information to calculate scheduled
        values. For parallel environments, the environments are in different
        processes and the progress information needs to be synced with the main
        in order to use schedulers in the environment.
        """
        [env.sync_progress() for env in self._envs]

    def close(self):
        """Close all external process."""
        if self._closed:
            return
        logging.info('Closing all processes.')
        [p() for p in self._reset_ts if p is not None]
        [p() for p in self._spare_promises]
        self._reset_ts = []
        self._spare_promises = []
        for env in self._envs:
            env.close()
        for env in self._spare_queue:
            env.close()
        self._closed = True
        logging.info('All processes closed.')

    def _stack_time_steps(self, time_steps):
        """Given a list of TimeStep, combine to one with a batch dimension."""
        if self._flatten:
            stacked = nest.fast_map_structure_flatten(
                lambda *arrays: numpy.stack(arrays),
                self._time_step_with_env_info_spec, *time_steps)
        else:
            stacked = nest.fast_map_structure(
                lambda *arrays: numpy.stack(arrays), *time_steps)
        if self._spare_queue:
            env_ids = numpy.array([e._env_id for e in self._envs])
            stacked = stacked._replace(env_id=env_ids)
        stacked = nest.map_structure(
            lambda x: torch.as_tensor(x, device='cpu'), stacked)
        if alf.get_default_device() == "cuda":
            cpu = stacked
            stacked = nest.map_structure(lambda x: x.cuda(), cpu)
            stacked._cpu = cpu
        return stacked

    def _unstack_actions(self, batched_actions):
        """Returns a list of actions from potentially nested batch of actions."""
        batched_actions = nest.map_structure(lambda x: x.cpu().numpy(),
                                             batched_actions)
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
        envs = self._envs + self._spare_queue
        if len(seeds) != len(envs):
            raise ValueError(
                'Number of seeds should match the number of parallel_envs.')
        promises = [env.call('seed', seed) for seed, env in zip(seeds, envs)]
        # Block until all envs are seeded.
        return [promise() for promise in promises]
