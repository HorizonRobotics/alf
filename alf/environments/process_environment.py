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
"""Step a single env in a separate process for lock free parallelism.

Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/parallel_py_environment.py
"""

from absl import flags, logging
import atexit
from enum import Enum
from functools import partial
import multiprocessing
import sys
import threadpoolctl
import torch
import traceback
from typing import Dict, Any, Callable, List, Tuple

import alf
import alf.nest as nest
from alf.utils import common
from alf.utils.per_process_context import PerProcessContext
from alf.utils.schedulers import update_all_progresses, get_all_progresses, disallow_scheduler
from alf.utils.spawned_process_utils import SpawnedProcessContext, get_spawned_process_context, set_spawned_process_context
from . import _penv

FLAGS = flags.FLAGS


def _init_after_spawn(context: SpawnedProcessContext):
    """Perform necessary initialization of flags and configurations when a new
    subprocess is "spawn"-ed for ``ProcessEnvironment``.

    This function is not needed if the subprocess is created via "fork".
    However, if it is created via "spawn", the subprocess will not automatically
    inherit resources such as the ALF configurations, and this function needs to
    be called to ensure the ALF configurations are initialized.

    Args:
        pre_configs: Specifies the set of pre configs that the parent process uses.

    """
    if context.ddp_rank >= 0:
        PerProcessContext().set_distributed(context.ddp_rank,
                                            context.ddp_num_procs)

    # 0. Update the global context for this spawned process. This will
    #    alter the behavior of ``get_env()``.
    set_spawned_process_context(context)

    # 1. Parse the relevant flags for the current subprocess. The set of
    #    relevant flags are defined below. Note that the command line arguments
    #    and options are inherited from the parent process via ``sys.argv``.
    flags.DEFINE_string(
        "root_dir", None,
        'Root directory for writing logs/summaries/checkpoints.')
    flags.DEFINE_string("conf", None, "Path to the alf config file.")
    flags.DEFINE_multi_string("conf_param", None, "Config binding parameters.")
    FLAGS(sys.argv, known_only=True)
    FLAGS.mark_as_parsed()
    FLAGS.alsologtostderr = True
    conf_file = common.get_conf_file()

    # 2. Configure the logging
    logging.set_verbosity(logging.INFO)

    # 3. Load the configuration
    alf.pre_config(dict(context.pre_configs))
    common.parse_conf_file(conf_file)


class _MessageType(Enum):
    """Message types for communication via the pipe.

    The ProcessEnvironment uses pipe to perform IPC, where each of the message
    has a message type. This Enum provides all the available message types.
    """
    READY = 1
    ACCESS = 2
    CALL = 3
    RESULT = 4
    EXCEPTION = 5
    CLOSE = 6
    SYNC_PROGRESS = 7


def _worker(conn: multiprocessing.connection,
            env_constructor: Callable,
            start_method: str,
            pre_configs: List[Tuple[str, Any]],
            env_id: int = None,
            flatten: bool = False,
            fast: bool = False,
            num_envs: int = 0,
            torch_num_threads_per_env: int = 1,
            ddp_num_procs: int = 1,
            ddp_rank: int = -1,
            name: str = ''):
    """The process waits for actions and sends back environment results.

    Args:
        conn: Connection for communication to the main process.
        env_constructor: callable environment creator.
        start_method: whether this subprocess is created via "fork" or "spawn".
        pre_configs: pre configs that need to be inherited if created via "spawn".
        env_id: the id of the env
        flatten: whether to assume flattened actions and time_steps
          during communication to avoid overhead.
        fast: whether created by ``FastParallelEnvironment`` or not.
        num_envs: number of environments in the ``FastParallelEnvironment``.
            Only used if ``fast`` is True.
        torch_num_threads_per_env: how many threads torch will use for each
            env proc. Note that if you have lots of parallel envs, it's best to
            set this number as 1. Leave this as 'None' to skip the change.
            Note that this option also affect the thread num for numpy.
        name: name of the FastParallelEnvironment. Only used if ``fast`` is True.

    Raises:
        KeyError: When receiving a message of unknown type.
    """
    try:
        alf.set_default_device("cpu")
        if torch_num_threads_per_env is not None:
            torch.set_num_threads(torch_num_threads_per_env)
            # recommended way of changing num_threads for numpy:
            # https://numpy.org/doc/stable/reference/global_state.html#number-of-threads-used-for-linear-algebra
            threadpoolctl.threadpool_limits(torch_num_threads_per_env)
        if start_method == "spawn":
            _init_after_spawn(
                SpawnedProcessContext(
                    ddp_num_procs=ddp_num_procs,
                    ddp_rank=ddp_rank,
                    env_id=env_id,
                    env_ctor=env_constructor,
                    pre_configs=pre_configs))
            # env may have been created during parse_conf_file called by _init_after_spawn
            # so we should not create it again using env_constructor
            env = alf.get_env()
        else:
            env = env_constructor(env_id=env_id)
        if not alf.get_config_value("TrainerConfig.sync_progress_to_envs"):
            disallow_scheduler()
        action_spec = env.action_spec()
        if fast:
            penv = _penv.ProcessEnvironment(
                env, partial(process_call, conn, env, flatten,
                             action_spec), env_id, num_envs, env.batch_size,
                env.batched, env.action_spec(),
                env.time_step_spec()._replace(env_info=env.env_info_spec()),
                name)
            conn.send(_MessageType.READY)  # Ready.
            try:
                penv.worker()
            except KeyboardInterrupt:
                penv.quit()
            except Exception:
                traceback.print_exc()
                penv.quit()
        else:
            conn.send(_MessageType.READY)  # Ready.
            while True:
                if not process_call(conn, env, flatten, action_spec):
                    break
    except KeyboardInterrupt:
        # When worker receives interruption from keyboard (i.e. Ctrl-C), notify
        # the parent process to shut down quietly by sending the CLOSE message.
        #
        # This is to avoid sometimes tens of environment processes panicking
        # simultaneously.
        conn.send((_MessageType.CLOSE, None))
    except Exception:  # pylint: disable=broad-except
        etype, evalue, tb = sys.exc_info()
        stacktrace = ''.join(traceback.format_exception(etype, evalue, tb))
        message = 'Error in environment process: {}'.format(stacktrace)
        logging.error(message)
        conn.send((_MessageType.EXCEPTION, stacktrace))
    finally:
        conn.close()


def process_call(conn, env, flatten, action_spec):
    """
    Returns:
        True: continue to work
        False: end the worker
    """
    try:
        # Only block for short times to have keyboard exceptions be raised.
        while True:
            if conn.poll(0.1):
                break
        message, payload = conn.recv()
    except (EOFError, KeyboardInterrupt):
        return False
    if message == _MessageType.ACCESS:
        name = payload
        result = getattr(env, name)
        conn.send((_MessageType.RESULT, result))
    elif message == _MessageType.CALL:
        name, args, kwargs = payload
        if flatten and name == 'step':
            args = [nest.pack_sequence_as(action_spec, args[0])]
        result = getattr(env, name)(*args, **kwargs)
        if flatten and name in ['step', 'reset']:
            result = nest.flatten(result)
            assert all([not isinstance(x, torch.Tensor) for x in result
                        ]), ("Tensor result is not allowed: %s" % name)
        conn.send((_MessageType.RESULT, result))
    elif message == _MessageType.SYNC_PROGRESS:
        update_all_progresses(payload)
    elif message == _MessageType.CLOSE:
        assert payload is None
        env.close()
        return False
    else:
        raise KeyError('Received message of unknown type {}'.format(message))
    return True


class ProcessEnvironment(object):
    def __init__(self,
                 env_constructor: Callable,
                 env_id: int = None,
                 flatten: bool = False,
                 fast: bool = False,
                 num_envs: int = 0,
                 torch_num_threads_per_env: int = 1,
                 start_method: str = "fork",
                 name: str = ""):
        """Step environment in a separate process for lock free parallelism.

        The environment is created in an external process by calling the provided
        callable. This can be an environment class, or a function creating the
        environment and potentially wrapping it. The returned environment should
        not access global variables.

        Args:
            env_constructor: callable environment creator.
            env_id: ID of the the env
            flatten: whether to assume flattened actions and time_steps
                during communication to avoid overhead.
            fast: whether created by ``FastParallelEnvironment`` or not.
            num_envs: number of environments in the ``FastParallelEnvironment``.
                Only used if ``fast`` is True.
            torch_num_threads_per_env: how many threads torch will use for each
                env proc. Note that if you have lots of parallel envs, it's best
                to set this number as 1. Leave this as 'None' to skip the change.
            start_method: whether this subprocess is created via "fork" or "spawn".
            name: name of the FastParallelEnvironment. Only used if ``fast``
                is True.

        Attributes:
            observation_spec: The cached observation spec of the environment.
            action_spec: The cached action spec of the environment.
            time_step_spec: The cached time step spec of the environment.
        """
        self._env_constructor = env_constructor
        self._flatten = flatten
        self._env_id = env_id
        self._observation_spec = None
        self._action_spec = None
        self._reward_spec = None
        self._time_step_spec = None
        self._env_info_spec = None
        self._conn = None
        self._fast = fast
        self._num_envs = num_envs
        self._torch_num_threads = torch_num_threads_per_env
        assert start_method in [
            "fork", "spawn"
        ], (f"Unrecognized start method '{start_method}' specified for "
            "ProcessEnvironment. It should be either 'fork' or 'spawn'.")
        self._start_method = start_method
        self._name = name
        if fast:
            self._penv = _penv.ProcessEnvironmentCaller(env_id, name)

    def start(self, wait_to_start=True):
        """Start the process.

        Args:
            wait_to_start (bool): Whether the call should wait for an env initialization.
        """
        assert not self._conn, "Cannot start() ProcessEnvironment multiple times"
        mp_ctx = multiprocessing.get_context(self._start_method)
        self._conn, conn = mp_ctx.Pipe()

        ddp_num_procs = PerProcessContext().num_processes
        ddp_rank = PerProcessContext().ddp_rank

        self._process = mp_ctx.Process(
            target=_worker,
            args=(conn, self._env_constructor, self._start_method,
                  alf.get_handled_pre_configs(), self._env_id, self._flatten,
                  self._fast, self._num_envs, self._torch_num_threads,
                  ddp_num_procs, ddp_rank, self._name),
            name=f"ProcessEnvironment-{self._env_id}")
        atexit.register(self.close)
        self._process.start()
        if wait_to_start:
            self.wait_start()

    def wait_start(self):
        """Wait for the started process to finish initialization."""
        assert self._conn, "Run ProcessEnvironment.start() first"
        result = self._conn.recv()
        if isinstance(result, Exception):
            self._conn.close()
            self._process.join(5)
            raise result
        assert result == _MessageType.READY, result

    def env_info_spec(self):
        if not self._env_info_spec:
            self._env_info_spec = self.call('env_info_spec')()
        return self._env_info_spec

    def observation_spec(self):
        if not self._observation_spec:
            self._observation_spec = self.call('observation_spec')()
        return self._observation_spec

    def action_spec(self):
        if not self._action_spec:
            self._action_spec = self.call('action_spec')()
        return self._action_spec

    def reward_spec(self):
        if not self._reward_spec:
            self._reward_spec = self.call('reward_spec')()
        return self._reward_spec

    def time_step_spec(self):
        if not self._time_step_spec:
            self._time_step_spec = self.call('time_step_spec')()
        return self._time_step_spec

    def __getattr__(self, name):
        """Request an attribute from the environment.

        Note that this involves communication with the external process, so it can
        be slow.

        Args:
            name (str): Attribute to access.

        Returns:
            Value of the attribute.
        """
        assert self._conn, "Run ProcessEnvironment.start() first"
        if self._fast:
            self._penv.call()
        self._conn.send((_MessageType.ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """Asynchronously call a method of the external environment.

        Args:
            name (str): Name of the method to call.
            *args: Positional arguments to forward to the method.
            **kwargs: Keyword arguments to forward to the method.

        Returns:
            Promise object that blocks and provides the return value when called.
        """
        assert self._conn, "Run ProcessEnvironment.start() first"
        if self._fast:
            self._penv.call()
        payload = name, args, kwargs
        self._conn.send((_MessageType.CALL, payload))
        return self._receive

    def close(self):
        """Send a close message to the external process and join it."""
        try:
            if self._fast:
                self._penv.close()
            else:
                self._conn.send((_MessageType.CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join()

    def step(self, action, blocking=True):
        """Step the environment.

        Args:
            action (nested tensors): The action to apply to the environment.
            blocking (bool): Whether to wait for the result.

        Returns:
            time step when blocking, otherwise callable that returns the time step.
        """
        promise = self.call('step', action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=True):
        """Reset the environment.

        Args:
            blocking (bool): Whether to wait for the result.

        Returns:
            New observation when blocking, otherwise callable that returns the new
            observation.
        """
        promise = self.call('reset')
        if blocking:
            return promise()
        else:
            return promise

    def sync_progress(self):
        """Sync the progress of the environment.
        """
        if self._fast:
            self._penv.call()
        self._conn.send((_MessageType.SYNC_PROGRESS, get_all_progresses()))

    def _receive(self):
        """Wait for a message from the worker process and return its payload.

        Raises:
            Exception: An exception was raised inside the worker process.
            KeyError: The received message is of an unknown type.

        Returns:
            Payload object of the message.
        """
        assert self._conn, "Run ProcessEnvironment.start() first"
        message, payload = self._conn.recv()

        # Re-raise exceptions in the main process.
        if message == _MessageType.EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        elif message == _MessageType.RESULT:
            return payload
        elif message == _MessageType.CLOSE:
            # When notified that the child process is going to shut down, do not
            # panic and handle it quietly.
            return None
        self.close()
        raise KeyError(
            'Received message of unexpected type {}'.format(message))

    def render(self, mode='human'):
        """Render the environment.

        Args:
            mode (str): One of ['rgb_array', 'human']. Renders to an numpy array, or brings
                up a window where the environment can be visualized.
        Returns:
            An ndarray of shape [width, height, 3] denoting an RGB image if mode is
            `rgb_array`. Otherwise return nothing and render directly to a display
            window.
        Raises:
            NotImplementedError: If the environment does not support rendering.
        """
        return self.call('render', mode)()
