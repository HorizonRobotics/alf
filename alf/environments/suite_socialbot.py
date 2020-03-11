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

try:
    import social_bot
    # The following import is to allow gin config of environments take effects
    import social_bot.envs
except ImportError:
    social_bot = None

import contextlib
from fasteners.process_lock import InterProcessLock
import functools
import gin
import gym
import socket

from alf.environments import suite_gym, torch_wrappers, process_environment
from alf.environments.utils import UnwrappedEnvChecker

DEFAULT_SOCIALBOT_PORT = 11345

_unwrapped_env_checker_ = UnwrappedEnvChecker()


def is_available():
    return social_bot is not None


@gin.configurable
def load(environment_name,
         env_id=None,
         port=None,
         wrap_with_process=False,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         torch_env_wrappers=()):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name (str): Name for the environment to load.
        env_id (int): (optional) ID of the environment.
        port (int): Port used for the environment
        wrap_with_process (bool): Whether wrap environment in a new process
        discount (float): Discount to use for the environment.
        max_episode_steps (int): If None the max_episode_steps will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no timestep_limit set in the environment's spec.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers, 
            classes to use directly on the gym environment.
        torch_env_wrappers (Iterable): Iterable with references to torch_wrappers 
            classes to use on the torch environment.

    Returns:
        A TorchEnvironmentBase instance.
    """
    _unwrapped_env_checker_.check_and_update(wrap_with_process)
    if gym_env_wrappers is None:
        gym_env_wrappers = ()
    if torch_env_wrappers is None:
        torch_env_wrappers = ()

    gym_spec = gym.spec(environment_name)
    if max_episode_steps is None:
        if gym_spec.timestep_limit is not None:
            max_episode_steps = gym_spec.max_episode_steps
        else:
            max_episode_steps = 0

    def env_ctor(port, env_id=None):
        gym_env = gym_spec.make(port=port)
        return suite_gym.wrap_env(
            gym_env,
            env_id=env_id,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=gym_env_wrappers,
            torch_env_wrappers=torch_env_wrappers)

    port_range = [port, port + 1] if port else [DEFAULT_SOCIALBOT_PORT]
    with _get_unused_port(*port_range) as port:
        if wrap_with_process:
            process_env = process_environment.ProcessEnvironment(
                functools.partial(env_ctor, port))
            process_env.start()
            torch_env = torch_wrappers.TorchEnvironmentBaseWrapper(process_env)
        else:
            torch_env = env_ctor(port=port, env_id=env_id)
    return torch_env


@contextlib.contextmanager
def _get_unused_port(start, end=65536):
    """Get an unused port in the range [start, end) .

    Args:
        start (int) : port range start
        end (int): port range end
    """
    process_lock = None
    unused_port = None
    try:
        for port in range(start, end):
            process_lock = InterProcessLock(
                path='/tmp/socialbot/{}.lock'.format(port))
            if not process_lock.acquire(blocking=False):
                process_lock.lockfile.close()
                process_lock = None
                continue
            try:
                with contextlib.closing(socket.socket()) as sock:
                    sock.bind(('', port))
                    unused_port = port
                    break
            except socket.error:
                process_lock.release()
                process_lock = None
        if unused_port is None:
            raise socket.error("No unused port in [{}, {})".format(start, end))
        yield unused_port
    finally:
        if process_lock is not None:
            process_lock.release()
