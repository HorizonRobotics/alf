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
import inspect
import numpy as np
import random
import torch

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


def _get_wrapped_fn(fn):
    """Get the function that is wrapped by ``functools.partial``"""
    while isinstance(fn, functools.partial):
        fn = fn.func
    return fn


def _env_constructor(env_load_fn, env_name, batch_size_per_env, seed, env_id):
    # We need to set random seed before env_load_fn because some environment
    # perform use random numbers in its constructor, so we need to randomize
    # the seed for it.
    alf.utils.common.set_random_seed(seed)

    # In this case, the environment loader is already batched. Just use it to
    # create an environment with the specified batch size.
    #
    # NOTE: here it ASSUMES that the created batched environment will take the
    # following env IDs: env_id, env_id + 1, ... ,env_id + batch_size - 1
    batched = getattr(_get_wrapped_fn(env_load_fn), 'batched', False)
    if batched:
        return env_load_fn(
            env_name, env_id=env_id, batch_size=batch_size_per_env)
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
                       eval_env_load_fn=None,
                       for_evaluation=False,
                       num_parallel_environments=30,
                       batch_size_per_env=None,
                       eval_batch_size_per_env=None,
                       nonparallel=False,
                       flatten=True,
                       start_serially=True,
                       num_spare_envs=0,
                       torch_num_threads_per_env=1,
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
            ``evn_load_fn(env_name, env_id=env_id, batch_size=batch_size_per_env)``
            will be used to create the batched environment. Otherwise,
            ``env_load_fn(env_name, env_id)`` will be used to create the environment.
            env_id is the index of the environment in the batch in the range of
            ``[0, num_parallel_enviroments / batch_size_per_env)``.
            And if "num_parallel_environments" is in the signature of ``env_load_fn``,
            num_parallel_environments will be provided as a keyword argument.
        eval_env_load_fn (Callable) : callable that create an environment for
            evaluation. If None, use ``env_load_fn``. This argument is useful
            for cases when the evaluation environment is different from the
            training environment.
        for_evaluation (bool): whether to create an environment for evaluation
            (if True) or for training (if False). If True, ``eval_env_load_fn``
            will be used for creating the environment if provided. Otherwise,
            ``env_load_fn`` will be used.
        num_parallel_environments (int): num of parallel environments
        batch_size_per_env (Optional[int]): if >1, will create
            ``num_parallel_environments/batch_size_per_env``
            ``ProcessEnvironment``. Each of these ``ProcessEnvironment`` holds
            ``batch_size_per_env`` environments. If each underlying environment
            of ``ProcessEnvironment`` is itself batched, ``batch_size_per_env``
            will be used as the batch size for them. Otherwise
            ``BatchEnvironmentWrapper`` will be sused to instruct each process
            to run the underlying environments sequentially on operations such
            as ``step()``. The potential benefit of using
            ``batch_size_per_env>1`` is to reduce the number of processes being
            used, or to take advantages of the batched nature of the underlying
            environment.
            If None, it will be `num_parallel_envrironments` if ``env_load_fn``
                is batched and 1 otherwise.
        eval_batch_size_per_env (int): if provided, it will be used as the
            batch size for evaluation environment. Otherwise, use
            ``batch_size_per_env``.
        num_spare_envs (int): num of spare parallel envs for speed up reset.
        nonparallel (bool): force to create a single env in the current
            process. Used for correctly exposing game gin confs to tensorboard.
            If True, ``num_parallel_environments`` will be asserted to be 1, and
            ``batch_size_per_env`` has to be None or 1, to avoid potential mistakes
            in run configuration.
        start_serially (bool): start environments serially or in parallel.
        flatten (bool): whether to use flatten action and time_steps during
            communication to reduce overhead.
        num_spare_envs (int): number of spare parallel environments to speed
            up reset.  Useful when a reset is much slower than a regular step.
        torch_num_threads_per_env (int): how many threads torch will use for each
            env proc. Note that if you have lots of parallel envs, it's best
            to set this number as 1. Leave this as 'None' to skip the change.
            Only used if the env is not batched and ``nonparallel==False``.
        parallel_environment_ctor (Callable): used to construct parallel environment.
            Available constructors are: ``fast_parallel_environment.FastParallelEnvironment``
            and ``parallel_environment.ParallelAlfEnvironment``.
        seed (None|int): random number seed for environment.  A random seed is
            used if None.
        batched_wrappers (Iterable): a list of wrappers which can wrap batched
            AlfEnvironment.
    Returns:
        AlfEnvironment:

    """

    # Some environment may take long time to load. So we use GPU before loading
    # environments so that other people knows that this GPU is being used.
    if torch.cuda.is_available():
        tmp = torch.zeros((32, ))
        torch.cuda.synchronize()

    if for_evaluation:
        # for creating an evaluation environment, use ``eval_env_load_fn`` if
        # provided and fall back to ``env_load_fn`` otherwise
        env_load_fn = eval_env_load_fn or env_load_fn
        batch_size_per_env = eval_batch_size_per_env or batch_size_per_env

    # env_load_fn may be a functools.partial, so we need to get the wrapped
    # function to get its attributes
    batched = getattr(_get_wrapped_fn(env_load_fn), 'batched', False)
    no_thread_env = getattr(
        _get_wrapped_fn(env_load_fn), 'no_thread_env', False)

    if nonparallel:
        assert num_parallel_environments == 1, "nonparallel is True"
        if batch_size_per_env is not None:
            assert batch_size_per_env == 1, "nonparallel is True"

    if batch_size_per_env is None:
        if batched:
            batch_size_per_env = num_parallel_environments
        else:
            batch_size_per_env = 1

    assert num_parallel_environments % batch_size_per_env == 0, (
        f"num_parallel_environments ({num_parallel_environments}) cannot be"
        f"divided by batch_size_per_env ({batch_size_per_env})")
    num_envs = num_parallel_environments // batch_size_per_env
    if batch_size_per_env > 1:
        assert num_spare_envs == 0, "Do not support spare environments for batch_size_per_env > 1"
        assert parallel_environment_ctor == fast_parallel_environment.FastParallelEnvironment

    if 'num_parallel_environments' in inspect.signature(
            env_load_fn).parameters:
        env_load_fn = functools.partial(
            env_load_fn, num_parallel_environments=num_parallel_environments)

    if isinstance(env_name, (list, tuple)):
        env_load_fn = functools.partial(alf_wrappers.MultitaskWrapper.load,
                                        env_load_fn)

    if batched and batch_size_per_env == num_parallel_environments:
        alf_env = env_load_fn(env_name, batch_size=num_parallel_environments)
        if not alf_env.is_tensor_based:
            alf_env = alf_wrappers.TensorWrapper(alf_env)
    elif nonparallel:
        # Each time we can only create one unwrapped env at most
        if no_thread_env:
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
        if seed is None:
            seeds = list(
                map(
                    int,
                    np.random.randint(0,
                                      np.iinfo(np.int32).max,
                                      num_envs + num_spare_envs)))
        else:
            seeds = [seed + i for i in range(num_envs + num_spare_envs)]
        ctors = [
            functools.partial(_env_constructor, env_load_fn, env_name,
                              batch_size_per_env, seed) for seed in seeds
        ]
        # flatten=True will use flattened action and time_step in
        #   process environments to reduce communication overhead.
        alf_env = parallel_environment_ctor(
            ctors,
            flatten=flatten,
            start_serially=start_serially,
            num_spare_envs_for_reload=num_spare_envs,
            torch_num_threads_per_env=torch_num_threads_per_env)
        alf_env.seed(seeds)

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
