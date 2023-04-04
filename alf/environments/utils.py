# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

import functools
import numpy as np
import random

import alf
from alf.environments import suite_gym
from alf.environments import thread_environment, parallel_environment, fast_parallel_environment
from alf.environments import alf_wrappers


class UnwrappedEnvChecker(object):
    """
    A class for checking if there is already an unwrapped env in the current
    process. For some games, if the check is True, then we should stop creating
    more envs (multiple envs cannot coexist in a process).

    See ``suite_socialbot.py`` for an example usage of this class.
    """

    def __init__(self):
        self._unwrapped_env_in_process = False

    def check(self):
        assert not self._unwrapped_env_in_process, \
            "You cannot create more envs once there has been an env in the main process!"

    def update(self, wrap_with_process):
        """
        Update the flag.

        Args:
            wrap_with_process (bool): if False, an env is being created without
                being wrapped by a subprocess.
        """
        self._unwrapped_env_in_process |= not wrap_with_process

    def check_and_update(self, wrap_with_process):
        """
        Combine self.check() and self.update()
        """
        self.check()
        self.update(wrap_with_process)


def _env_constructor(env_load_fn, env_name, batch_size_per_env, env_id):
    if batch_size_per_env == 1:
        return env_load_fn(env_name, env_id)
    envs = [
        env_load_fn(env_name, env_id * batch_size_per_env + i)
        for i in range(batch_size_per_env)
    ]
    return alf_wrappers.BatchEnvironmentWrapper(envs)


@alf.configurable
def create_environment(env_name='CartPole-v0',
                       env_load_fn=suite_gym.load,
                       num_parallel_environments=30,
                       batch_size_per_env=1,
                       nonparallel=False,
                       flatten=True,
                       start_serially=True,
                       num_spare_envs=0,
                       parallel_environment_ctor=fast_parallel_environment.
                       FastParallelEnvironment,
                       seed=None,
                       batched_wrappers=()):
    """Create a batched environment.

    Args:
        env_name (str|list[str]): env name. If it is a list, ``MultitaskWrapper``
            will be used to create multi-task environments. Each one of them
            consists of the environments listed in ``env_name``.
        env_load_fn (Callable) : callable that create an environment
            If env_load_fn has attribute ``batched`` and it is True,
            ``evn_load_fn(env_name, batch_size=num_parallel_environments)``
            will be used to create the batched environment. Otherwise, a
            ``ParallAlfEnvironment`` will be created.
        num_parallel_environments (int): num of parallel environments
        batch_size_per_env (int): if >1, will create ``num_parallel_environments/batch_size_per_env``
            ``ProcessEnvironment``. Each of these ``ProcessEnvironment`` holds
            ``batch_size_per_env`` environments. The potential benefit of using
            ``batch_size_per_env>1`` is to reduce the number of processes being
            used.
        num_spare_envs (int): num of spare parallel envs for speed up reset.
        nonparallel (bool): force to create a single env in the current
            process. Used for correctly exposing game gin confs to tensorboard.
        start_serially (bool): start environments serially or in parallel.
        flatten (bool): whether to use flatten action and time_steps during
            communication to reduce overhead.
        num_spare_envs (int): number of spare parallel environments to speed
            up reset.  Useful when a reset is much slower than a regular step.
        parallel_environment_ctor (Callable): used to contruct parallel environment.
            Available constructors are: ``fast_parallel_environment.FastParallelEnvironment``
            and ``parallel_environment.ParallelAlfEnvironment``.
        seed (None|int): random number seed for environment.  A random seed is
            used if None.
        batched_wrappers (Iterable): a list of wrappers which can wrap batched
            AlfEnvironment.
    Returns:
        AlfEnvironment:
    """
    assert num_parallel_environments % batch_size_per_env == 0, (
        f"num_parallel_environments ({num_parallel_environments}) cannot be"
        f"divided by batch_size_per_env ({batch_size_per_env})")
    num_envs = num_parallel_environments // batch_size_per_env
    if batch_size_per_env > 1:
        assert num_spare_envs == 0, "Do not support spare environments for batch_size_per_env > 1"
        assert parallel_environment_ctor == fast_parallel_environment.FastParallelEnvironment
    if isinstance(env_name, (list, tuple)):
        env_load_fn = functools.partial(alf_wrappers.MultitaskWrapper.load,
                                        env_load_fn)

    if hasattr(env_load_fn, 'batched') and env_load_fn.batched:
        if nonparallel:
            alf_env = env_load_fn(env_name, batch_size=1)
        else:
            alf_env = env_load_fn(
                env_name, batch_size=num_parallel_environments)
    elif nonparallel:
        # Each time we can only create one unwrapped env at most
        if getattr(env_load_fn, 'no_thread_env', False):
            # In this case the environment is marked as "not compatible with
            # thread environment", and we will create it in the main thread.
            # BatchedTensorWrapper is applied to make sure the I/O is batched
            # torch tensor based.
            alf_env = alf_wrappers.BatchedTensorWrapper(env_load_fn(env_name))
        else:
            # Create and step the env in a separate thread. env `step` and
            #   `reset` must run in the same thread which the env is created in
            #   for some simulation environments such as social_bot(gazebo)
            alf_env = thread_environment.ThreadEnvironment(lambda: env_load_fn(
                env_name))

        if seed is None:
            alf_env.seed(np.random.randint(0, np.iinfo(np.int32).max))
        else:
            alf_env.seed(seed)
    else:
        # flatten=True will use flattened action and time_step in
        #   process environments to reduce communication overhead.
        alf_env = parallel_environment_ctor(
            [
                functools.partial(_env_constructor, env_load_fn, env_name,
                                  batch_size_per_env)
            ] * num_envs,
            flatten=flatten,
            start_serially=start_serially,
            num_spare_envs_for_reload=num_spare_envs)

        if seed is None:
            alf_env.seed([
                np.random.randint(0,
                                  np.iinfo(np.int32).max)
                for i in range(num_envs)
            ])
            if num_spare_envs > 0:
                alf_env.seed_spare([
                    np.random.randint(0,
                                      np.iinfo(np.int32).max)
                    for i in range(num_spare_envs)
                ])
        else:
            # We want deterministic behaviors for each environment, but different
            # behaviors among different individual environments (to increase the
            # diversity of environment data)!
            alf_env.seed([seed + i for i in range(num_envs)])
            if num_spare_envs > 0:
                alf_env.seed_spare([
                    seed + i + num_parallel_environments
                    for i in range(num_spare_envs)
                ])

    for wrapper in batched_wrappers:
        alf_env = wrapper(alf_env)

    return alf_env


@alf.configurable
def load_with_random_max_episode_steps(env_name,
                                       env_load_fn=suite_gym.load,
                                       min_steps=200,
                                       max_steps=250):
    """Create environment with random max_episode_steps in range
    ``[min_steps, max_steps]``.

    Args:
        env_name (str): env name
        env_load_fn (Callable) : callable that create an environment
        min_steps (int): represent min value of the random range
        max_steps (int): represent max value of the random range
    Returns:
        AlfEnvironment:
    """
    return env_load_fn(
        env_name, max_episode_steps=random.randint(min_steps, max_steps))
