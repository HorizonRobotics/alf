# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import numpy as np
import torch

from absl import logging
import alf
from alf.environments import alf_environment
from alf.environments.process_environment import ProcessEnvironment
import alf.nest as nest
import os
import time
from . import _penv


@alf.configurable
class FastParallelEnvironment(alf_environment.AlfEnvironment):
    """Batch together environments and simulate them in external processes.

    The environments are created in external processes by calling the provided
    callables. This can be an environment class, or a function creating the
    environment and potentially wrapping it. The environments can be different
    but must use the same action and observation specs.

    Different from ``parallel_environment.ParallelAlfEnvironment``, ``FastParallelEnvironment``
    uses shared memory to transfer ``TimeStep`` from each process environment
    to the main process.

    Terminology:

    - main process: the process where ParallelEnvironment is created
    - client process: the process running the actual individual environment created
        using env_constructors

    Design:

    ``FastParallelEnvironment`` uses ``_penv.ParallelEnvironment`` (implemented in C++)
    to coordinate step() and reset().
    Each ``ProcessEnvironment`` maintains one ``_penv.ProcessEnvironmentCaller``
    in the main process and one ``_penv.ProcessEnvironment`` in the client process.

    In the client process, ``_penv.ProcessEnvironment.worker()`` runs in a loop to
    wait for jobs from either ``_penv.ParallelEnvironment`` or ``_penv.ProcessEnvironmentCaller``.

    There are 4 types of job:

    - step: step the environment. Sent from ``_penv.ParallelEnvironment``. The
        result is communicated back using shared memory.
    - reset: reset the environment. Sent from ``_penv.ParallelEnvironment``.
        The result is communicated back using shared memory.
    - close: close the environment. Sent from ``_penv.ProcessEnvironmentCaller``.
        This will cause the worker to finish and quit the process.
    - call: access other methods of the environment. Sent from ``_penv.ProcessEnvironmentCaller``.
        This takes advantage of the pipe mechanism used by  the ``ParallelAlfEnvironment``.
        This is achieved by calling ``call_handler`` to do communication using
        python pipe. The reason of using the original pipe mechanism for other
        types of communication is that it is not easy to handle communication of
        unknown size using shared memory.

    Args:
        env_constructors (list[Callable]): a list of callable environment creators.
            Each callable should accept env_id as the only argument and return
            the created environment.
        start_serially (bool): whether to start environments serially or in parallel.
        blocking (bool): not used. Kept for the same interface as ``ParallelAlfEnvironment``.
        flatten (bool): not used. Kept for the same interface as ``ParallelAlfEnvironment``.
        num_spare_envs_for_reload (int): if positive, these environments will be
            maintained in a separate queue and be used to handle slow env resets.
            The batch_size is ``len(env_constructors) - num_spare_envs_for_reload``
        torch_num_threads_per_env (int): how many threads torch will use for each
            env proc. Note that if you have lots of parallel envs, it's best
            to set this number as 1. Leave this as 'None' to skip the change.
            Note that this option also affect the thread num for numpy.
        start_method: either "fork" or "spawn". This specifies how the
            subprocesses for the ``ProcessEnvironment`` is created. Normally you
            should use "fork" because it is faster. There are cases when "fork"
            is too bloated or not safe (e.g. inherit shared resources that may
            cause contention), and using "spawn" can resolve that at the price
            of the process creation (and only the creation) speed.

    Raises:
        ValueError: If the action or observation specs don't match.

    """

    def __init__(
            self,
            env_constructors,
            start_serially=True,
            blocking=False,  # unused
            flatten=True,  # unused
            num_spare_envs_for_reload=0,
            torch_num_threads_per_env=1,
            start_method: str = "fork"):
        super().__init__()
        num_envs = len(env_constructors) - num_spare_envs_for_reload
        name = f"alf_penv_{os.getpid()}_{time.time()}"
        self._envs = []
        self._spare_envs = []
        self._start_method = start_method
        for env_id, ctor in enumerate(env_constructors):
            env = ProcessEnvironment(
                ctor,
                env_id=env_id,
                fast=True,
                num_envs=num_envs,
                torch_num_threads_per_env=torch_num_threads_per_env,
                start_method=start_method,
                name=name)
            if env_id < num_envs:
                self._envs.append(env)
            else:
                self._spare_envs.append(env)
        self._num_envs = len(env_constructors)
        self._num_spare_envs_for_reload = num_spare_envs_for_reload
        self._start_serially = start_serially
        self.start()
        self._action_spec = self._envs[0].action_spec()
        self._observation_spec = self._envs[0].observation_spec()
        self._reward_spec = self._envs[0].reward_spec()
        self._time_step_spec = self._envs[0].time_step_spec()
        self._env_info_spec = self._envs[0].env_info_spec()
        self._num_tasks = self._envs[0].num_tasks
        self._task_names = self._envs[0].task_names
        self._batch_size = self._envs[0].batch_size * num_envs
        time_step_with_env_info_spec = self._time_step_spec._replace(
            env_info=self._env_info_spec)
        batch_size_per_env = self._envs[0].batch_size
        batched = self._envs[0].batched
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
        if any(env.env_info_spec() != self._env_info_spec
               for env in self._envs):
            raise ValueError(
                'All environments must have the same env_info_spec.')
        if any(env.batch_size != batch_size_per_env for env in self._envs):
            raise ValueError('All environments must have the same batch_size.')
        if any(env.batched != batched for env in self._envs):
            raise ValueError('All environments must have the same batched.')
        self._closed = False
        self._penv = _penv.ParallelEnvironment(
            num_envs, num_spare_envs_for_reload, batch_size_per_env, batched,
            self._action_spec, time_step_with_env_info_spec, name)

    @property
    def envs(self):
        """The list of individual environment."""
        return self._envs

    @property
    def num_spare_envs_for_reload(self):
        return self._num_spare_envs_for_reload

    def start(self):
        acting_text = {
            "fork": "Forking",
            "spawn": "Spawning"
        }[self._start_method]
        logging.info(f"{acting_text} all {len(self._envs)} processes.")
        for env in self._envs:
            env.start(wait_to_start=self._start_serially)
        for env in self._spare_envs:
            env.start(wait_to_start=self._start_serially)
        if not self._start_serially:
            logging.info('Waiting for all processes to start.')
            for env in self._envs:
                env.wait_start()
            for env in self._spare_envs:
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
        return self._batch_size

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

    def _to_tensor(self, stacked):
        # we need to do np.copy because the result from _penv.step() or
        # _penv.reset() reuses the same internal buffer.
        stacked = nest.map_structure(
            lambda x: torch.as_tensor(np.copy(x), device='cpu'), stacked)
        if alf.get_default_device() == "cuda":
            cpu = stacked
            stacked = nest.map_structure(lambda x: x.cuda(), cpu)
            stacked._cpu = cpu
        return stacked

    def _step(self, action):
        def _to_numpy(x):
            # When AMP is enabled, the action.dtype can be torch.float16. We
            # need to convert it to torch.float32 to match the dtype from
            # action_spec
            if x.dtype == torch.float16:
                x = x.float()
            x = x.cpu().numpy()
            # parallel_environment.cpp requires the arrays to be contiguous. If
            # x is already contiguous, ascontiguousarray() will simply return x.
            x = np.ascontiguousarray(x)
            return x

        action = nest.map_structure(_to_numpy, action)
        return self._to_tensor(self._penv.step(action))

    def _reset(self):
        return self._to_tensor(self._penv.reset())

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
        i = 0
        for env in self._envs:
            env.close()
            i += 1
            if i % 100 == 0:
                logging.info(f"Closed {i} processes")
        for env in self._spare_envs:
            env.close()
            i += 1
            if i % 100 == 0:
                logging.info(f"Closed {i} processes")
        self._closed = True

    def seed(self, seeds):
        """Seeds the parallel environments."""
        envs = self._envs + self._spare_envs
        if len(seeds) != len(envs):
            raise ValueError(
                'Number of seeds should match the number of parallel_envs.')
        promises = [env.call('seed', seed) for seed, env in zip(seeds, envs)]
        # Block until all envs are seeded.
        return [promise() for promise in promises]
