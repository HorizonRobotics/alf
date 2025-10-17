# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from absl import logging
from typing import Callable
import time
import io
import queue
import random
import threading
import subprocess
import zmq

import torch
import torch.multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Manager

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.environments.alf_environment import AlfEnvironment
from alf.experience_replayers.replay_buffer import ReplayBuffer
from alf.data_structures import Experience, make_experience, StepType
from alf.utils.common import allow_child_to_ptrace
from alf.utils.per_process_context import PerProcessContext
from alf.utils import dist_utils
from alf.utils.summary_utils import record_time
from alf.utils.common_test import _test_tensor_sharing


class UnrollerMessage(object):
    # confirmation
    OK = 'unroller: ok'


def get_local_ip():
    """Get the ip address of the local machine."""
    return subprocess.check_output(["hostname",
                                    "-I"]).decode().strip().split()[0]


@alf.configurable
class TrainerAddrConfig(object):
    """A simple class for configuring the address of the trainer."""

    def __init__(self, ip: str = 'localhost', port: int = 50000):
        """
        Args:
            ip: ip address of the trainer.
            port: port number used by the trainer.
        """
        self.ip = ip
        self.port = port


_trainer_addr_config = TrainerAddrConfig()
_params_port_offset = 100
_unroller_port_offset = 1000


def create_zmq_socket(type: int, ip: str, port: int, id: str = None):
    """A helper function for creating a ZMQ socket.

    Args:
        type: type of the ZMQ socket, e.g., zmq.DEALER, zmq.PUB, etc. See
            https://sachabarbs.wordpress.com/2014/08/21/zeromq-2-the-socket-types-2/
        ip: ip address. If it's '*', then `socket.bind()` will be used.
        port: port number.
        id: identity of the socket (optional). Only required for DEALER
            sockets.

    Returns:
        tuple:
        - socket: used for sending/receiving messages
        - ZMQ context
    """
    cxt = zmq.Context()
    socket = cxt.socket(type)
    if id is not None:
        socket.identity = id.encode('utf-8')
    addr = 'tcp://' + ':'.join([ip, str(port)])
    if ip == '*':
        socket.bind(addr)
    else:
        socket.connect(addr)
    return socket, cxt


class DistributedOffPolicyAlgorithm(OffPolicyAlgorithm):

    def __init__(self,
                 core_alg_ctor: Callable,
                 *args,
                 port: int = 50000,
                 env: AlfEnvironment = None,
                 config: TrainerConfig = None,
                 optimizer: alf.optimizers.Optimizer = None,
                 debug_summaries: bool = False,
                 name: str = "DistributedOffPolicyAlgorithm",
                 **kwargs):
        """
        Args:
            core_alg_ctor: creates the algorithm to be wrapped by this class.
            config: the global ``TrainerConfig`` instance. The user is required
                to always specify this argument.
            port: port number for communication on the *current* machine.
            env: The environment to interact with. Its batch size must be 1.
            optimizer: optimizer for the training the core algorithm.
            debug_summaries: True if debug summaries should be created.
            name: the name of this algorithm.
            *args: args to pass to ``core_alg_ctor``.
            **kwargs: kwargs to pass to ``core_alg_ctor``.
        """
        # Need to pass ``config`` to core alg following the standard algorithm interface.
        #``env`` is set to None to avoid the creation of replay buffer in the ``core_alg``.
        core_alg = core_alg_ctor(*args,
                                 config=config,
                                 env=None,
                                 debug_summaries=debug_summaries,
                                 **kwargs)
        assert not core_alg.on_policy, (
            "The core algorithm must be off-policy!")
        assert env.batch_size == 1, (
            "DistributedOffPolicyAlgorithm currently only supports batch_size=1"
        )
        super().__init__(
            observation_spec=core_alg.observation_spec,
            action_spec=core_alg.action_spec,
            reward_spec=core_alg._reward_spec,
            train_state_spec=core_alg.train_state_spec,
            rollout_state_spec=core_alg.rollout_state_spec,
            predict_state_spec=core_alg.predict_state_spec,
            env=env,
            config=config,
            optimizer=optimizer,
            # Prevent in-alg ckpt since there is no such a use case.
            checkpoint=None,
            debug_summaries=debug_summaries,
            name=name)

        self._core_alg = core_alg
        self._port = port
        self._ddp_rank = max(0, PerProcessContext().ddp_rank)
        self._num_ranks = PerProcessContext().num_processes

    def _distributed_state_dict(self) -> dict:
        """Return `self._core_alg` state dict for distributed training.

        This dict will be used for param syncing between a trainer and an unroller.
        Sometimes optimizers have large state vectors which we want to exclude.
        Also we should exclude other parameters such as those of pretrained models.
        """
        # Note that self._core_alg won't create a relay buffer so we don't have
        # to worry about including it in the state dict.
        return {
            k: v
            for k, v in self._core_alg.state_dict().items()
            if (('_optimizers.' not in k) and (
                not isinstance(v, torch.nn.Parameter) or v.requires_grad))
        }

    ###############################
    ######### Forward calls #######
    ###############################
    def predict_step(self, inputs, state):
        return self._core_alg.predict_step(inputs, state)

    def rollout_step(self, inputs, state):
        return self._core_alg.rollout_step(inputs, state)

    def train_step(self, inputs, state, rollout_info):
        return self._core_alg.train_step(inputs, state, rollout_info)

    def calc_loss(self, info):
        return self._core_alg.calc_loss(info)

    def preprocess_experience(self, root_inputs, rollout_info, batch_info):
        return self._core_alg.preprocess_experience(root_inputs, rollout_info,
                                                    batch_info)

    def transform_experience(self, experience: Experience):
        # Global data transformer
        experience = super().transform_experience(experience)
        # In the case where core_alg has in-alg data transformer
        experience = self._core_alg.transform_experience(experience)
        return experience

    def after_update(self, root_inputs, info):
        return self._core_alg.after_update(root_inputs, info)

    def after_train_iter(self, root_inputs, rollout_info):
        return self._core_alg.after_train_iter(root_inputs, rollout_info)


def receive_experience_data(replay_buffer: ReplayBuffer,
                            new_unroller_ips_and_ports: 'Manager.Queue',
                            worker_id: int) -> None:
    """A worker function for consistently receiving experience data from
    unrollers.

    It will be called in a child process. Each worker creates a ZMQ DEALER
    socket and listen for experience data from the unrollers.

    This function has to be in a process instead of a thread, because the
    ``replay_buffer.add_batch`` will modify the global device, which causes
    conflicts with the training code.

    Args:
        replay_buffer: an instance of ``ReplayBuffer`` to store the received
            experience data.  It must allow multi-processing.
        new_unroller_ips_and_ports: a queue to store the ip and port of
            new unrollers.
        worker_id: the id of the worker; used by each unroller to route the
            experience data.
    """
    # A temporary buffer for each unroller to store exp data. Because multiple
    # unrollers might send exps to the same DDP rank at the same time, we need
    # to differentiate the sources. When a complete segment of exp data is ready,
    # we will add it to the replay buffer.
    unroller_exps_buffer = {}
    socket = None
    # Listen for experience data forever
    while True:
        try:
            unroller_ip, unroller_port = new_unroller_ips_and_ports.get_nowait(
            )
            # A new unroller has connected to the trainer
            if socket is None:
                socket, _ = create_zmq_socket(zmq.DEALER, unroller_ip,
                                              unroller_port,
                                              f'worker-{worker_id}')
            else:
                addr = 'tcp://' + ':'.join([unroller_ip, str(unroller_port)])
                # Connect to an additional ROUTER
                socket.connect(addr)
        except queue.Empty:
            pass

        if socket is not None:
            # Receive data from any router
            unroller_id, message = socket.recv_multipart()

            buffer = io.BytesIO(message)
            exp_params = torch.load(buffer, map_location='cpu')
            # we prune env_info according to the replay buffer for the following reasons:
            # 1) avoid env_info mismatch and allow the distributed unroller to have
            # a customized env_info for tb summarization,
            # 2) reduce memory usage for the replay buffer
            exp_params = alf.utils.common.prune_exp_replay_env_info(
                exp_params,
                env_info_spec=replay_buffer.data_spec.time_step.env_info)

            # Use a temp buffer to store the received exps
            if unroller_id not in unroller_exps_buffer:
                unroller_exps_buffer[unroller_id] = []
            unroller_exps_buffer[unroller_id].append(exp_params)

            if int(exp_params.step_type) == StepType.LAST:
                # Add the temp exp buffer to the replay buffer
                # ``DistributedOffPolicyAlgorithm`` assumes batch_size=1
                exp_params.env_id.zero_()
                for i, exp_params in enumerate(
                        unroller_exps_buffer[unroller_id]):
                    replay_buffer.add_batch(exp_params, exp_params.env_id)
                unroller_exps_buffer[unroller_id] = []
        else:
            time.sleep(0.1)


def pull_params_from_trainer(memory_name: str, memory_lock: mp.Lock,
                             unroller_id: str, params_socket_rank: int):
    """ Once new params arrive, we put it in the shared memory and mark updated.
    Later after the current unroll finishes, the unroller can load the
    new params.

    Args:
        memory_name: the name of the shared memory which is used to store the
            updated params for the main process.
        memory_lock: the lock for the shared memory write/read.
        unroller_id: the id of the unroller.
        params_socket_rank: which DDP rank will be syncing params with this unroller.
    """
    socket, _ = create_zmq_socket(
        zmq.DEALER, _trainer_addr_config.ip,
        _trainer_addr_config.port + _params_port_offset + params_socket_rank,
        unroller_id + "_params")
    params = SharedMemory(name=memory_name)
    # signifies that this unroller is ready to receive params
    socket.send_string(UnrollerMessage.OK)
    while True:
        data = socket.recv()
        with memory_lock:
            params.buf[0] = 1
            params.buf[1:] = data
        socket.send_string(UnrollerMessage.OK)


@alf.configurable(whitelist=[
    'max_utd_ratio', 'push_params_every_n_grad_updates', 'name', 'optimizer'
])
class DistributedTrainer(DistributedOffPolicyAlgorithm):

    def __init__(self,
                 core_alg_ctor: Callable,
                 *args,
                 max_utd_ratio: float = 10.,
                 push_params_every_n_grad_updates: int = 1,
                 env: AlfEnvironment = None,
                 config: TrainerConfig = None,
                 optimizer: alf.optimizers.Optimizer = None,
                 debug_summaries: bool = False,
                 name: str = "DistributedTrainer",
                 **kwargs):
        """
        Args:
            core_alg_ctor: creates the algorithm to be wrapped by this class.
                This algorithm's ``train_step()`` will be used for training.
            max_utd_ratio: max update-to-data ratio, defined as the ratio between
                the number of gradient updates and the number of exp samples
                put in the replay buffer. If the current ratio is higher than
                this value, the trainer will pause training until more experience
                samples are sent from unrollers.
                NOTE: When using DDP, if there is any subprocess exceeding this
                value, the overall training will be paused, because DDP needs to
                sync gradients among subprocesses after each backward.
                A larger value will make the trainer more likely overfit to the
                replay buffer data, while a smaller value will lead to data wastage.
            push_params_every_n_grad_updates: push model parameters to the unroller
                every this number of gradient updates.
            *args: additional args to pass to ``core_alg_ctor``.
            **kwargs: additional kwargs to pass to ``core_alg_ctor``.
        """
        super().__init__(core_alg_ctor,
                         *args,
                         port=_trainer_addr_config.port,
                         env=env,
                         config=config,
                         optimizer=optimizer,
                         debug_summaries=debug_summaries,
                         name=name,
                         **kwargs)

        self._push_params_every_n_grad_updates = push_params_every_n_grad_updates

        # Ports:
        # 1. registration port: self._port + self._ddp_rank
        # 2. params port: self._port + _params_port_offset + self._ddp_rank

        self._max_utd_ratio = max_utd_ratio

        # overwrite ``observe_for_replay`` to make sure it is never called
        # by the parent ``RLAlgorithm``
        self.observe_for_replay = self._observe_for_replay

        self._params_socket, _ = create_zmq_socket(
            zmq.ROUTER, '*', self._port + _params_port_offset + self._ddp_rank)

        assert config.unroll_length == -1, (
            'unroll_length must be -1 (no unrolling)')
        # Total number of gradient updates so far
        self._total_updates = 0
        # How many times ``train_iter()`` has been called.
        # Cannot directly use ``alf.summary.get_global_counter()`` because it
        # may be incremented every mini-batch
        self._num_train_iters = 0

        # respect core_alg's replay buffer setting
        self._num_earliest_frames_ignored = self._core_alg._num_earliest_frames_ignored

        # We always test tensor sharing among processes, because
        # we rely on undocumented features of PyTorch:
        # 1. tensors will automatically be moved to shared memory, even without
        #    ``Module.share_memory()`` or ``Tensor.share_memory_()`` being called.
        # 2. only a 'spawned' subprocess is reliable for tensor sharing.
        _test_tensor_sharing()

    def _observe_for_replay(self, exp: Experience):
        raise RuntimeError(
            'observe_for_replay should not be called for trainer')

    @property
    def is_main_ddp_rank(self):
        return self._ddp_rank == 0

    def _send_params_to_unroller(self,
                                 unroller_id: str,
                                 first_time: bool = False) -> bool:
        """Send model params to a specified unroller.

        Args:
            unroller_id: id (bytes str) of the unroller.
            first_time: whether this is the first time this function gets called.
                For the first time, we need to wait for the unroller's socket ready
                signal.

        Returns:
            bool: True if the unroller is still alive.
        """
        unroller_id1 = unroller_id + b'_params'
        if first_time:
            # Block until the unroller is ready to receive params
            # If we don't do so, the outgoing params might get lost before
            # the receiving socket is actually created.
            unroller_id_, message = self._params_socket.recv_multipart()
            assert unroller_id_ == unroller_id1
            assert message == UnrollerMessage.OK.encode()

        # Get all parameters/buffers in a state dict and send them out
        buffer = io.BytesIO()
        torch.save(self._distributed_state_dict(), buffer)
        self._params_socket.send_multipart([unroller_id1, buffer.getvalue()])
        # 3 sec timeout for receiving unroller's acknowledgement
        # In case some unrollers might die, we don't want to block forever
        for _ in range(30):
            try:
                _, message = self._params_socket.recv_multipart(
                    flags=zmq.NOBLOCK)
                assert message == UnrollerMessage.OK.encode()
                logging.debug(
                    f"[worker-{self._ddp_rank}] Params sent to unroller"
                    f" {unroller_id.decode()}.")
                return True
            except zmq.Again:
                time.sleep(0.1)
        return False

    def _create_unroller_registration_thread(self):
        manager = Manager()
        self._new_unroller_ips_and_ports = manager.Queue()
        self._unrollers_to_update_params = set()
        registered_unrollers = set()

        def _wait_unroller_registration():
            """Wait for new registration from a unroller.
            """
            total_unrollers = 0
            # Each rank has its own port number and a registration socket to
            # handle new unrollers.
            register_socket, _ = create_zmq_socket(zmq.ROUTER, '*',
                                                   self._port + self._ddp_rank)
            while True:
                unroller_id, message = register_socket.recv_multipart()
                if unroller_id not in registered_unrollers:
                    # A new unroller has connected to the trainer
                    # The init message should always be: 'init'
                    assert message.decode() == 'init'
                    _, unroller_ip, unroller_port = unroller_id.decode().split(
                        '-')
                    # Store the new unroller ip and port so that later each rank
                    # can connect to it for experience data.
                    self._new_unroller_ips_and_ports.put(
                        (unroller_ip, int(unroller_port)))
                    registered_unrollers.add(unroller_id)
                    logging.info(
                        f"Rank {self._ddp_rank} registered {unroller_ip} {unroller_port}"
                    )

                    if self.is_main_ddp_rank:
                        # Send the number of workers to the new unroller,
                        # so that it is able to know other workers.
                        # Also send the DDP rank that's responsible for the unroller's
                        # params syncing. See ``_train_iter_off_policy``
                        # where the params sending tasks are distributed.
                        k = total_unrollers % self._num_ranks
                        register_socket.send_multipart([
                            unroller_id,
                            (f'worker-0: {self._num_ranks} {k}').encode()
                        ])

                    # Then we check if its params socket communicates with the
                    # current rank.
                    if total_unrollers % self._num_ranks == self._ddp_rank:
                        self._unrollers_to_update_params.add(unroller_id)
                        # Always first sync the params with a new unroller.
                        assert self._send_params_to_unroller(unroller_id,
                                                             first_time=True)

                    total_unrollers += 1

        thread = threading.Thread(target=_wait_unroller_registration)
        thread.daemon = True
        thread.start()

    def _create_data_receiver_subprocess(self):
        """Create a process to receive experience data from unrollers.
        """
        # First create the replay buffer in the main process. For this, we need
        # to create a dummy experience to set up the replay buffer.
        time_step = self._env.current_time_step()
        rollout_state = self.get_initial_rollout_state(self._env.batch_size)
        alg_step = self.rollout_step(time_step, rollout_state)
        exp = make_experience(time_step, alg_step, rollout_state)
        exp = alf.utils.common.prune_exp_replay_state(exp,
                                                      self._use_rollout_state,
                                                      self.rollout_state_spec,
                                                      self.train_state_spec)

        # enable multi_processing in replay_buffer here, because we need to
        # receive data in a subprocess and process the data in the main process.
        ctx = mp.get_context('spawn')
        self._set_replay_buffer(exp, mp_context=ctx)
        assert self._replay_buffer._allow_multiprocess, (
            "The replay buffer must allow multi-processing.")

        # start the data receiver subprocess
        # Need to create the subprocess with 'spawn' so that we can pass a Module
        # object to subprocess with tensors in shared memory.
        process = ctx.Process(target=receive_experience_data,
                              args=(self._replay_buffer,
                                    self._new_unroller_ips_and_ports,
                                    self._ddp_rank),
                              daemon=True)
        process.start()
        allow_child_to_ptrace(process.pid)

    def utd(self):
        total_exps = int(self._replay_buffer.get_current_position().sum())
        if total_exps == 0:
            return 0
        return self._total_updates / total_exps

    def _train_iter_off_policy(self):
        if self._num_train_iters == 0:
            # First time will be called by ``Trainer._restore_checkpoint()``
            # where the ckpt (if any) will be loaded after this function.
            self._num_train_iters += 1
            return super()._train_iter_off_policy()

        if self._num_train_iters == 1:
            # Only open the unroller registration after we are sure that
            # the trainer's ckpt (if any) has been loaded, so that the trainer
            # will send correct params to any newly added unroller.
            self._create_unroller_registration_thread()
            # Because unroll_length=-1, ``observe_for_replay`` will never be called.
            # Instead, we call a separate data receiver process that consistently
            # pulls data from unrollers.
            self._create_data_receiver_subprocess()

        # A worker will pause when either happens:
        # 1. replay buffer is not ready (initial collect steps not reached)
        # 2. utd ratio is too high (training is too fast; wait for more data)
        while True:
            replay_buffer_not_ready = (self._replay_buffer.total_size
                                       < self._config.initial_collect_steps)
            utd_exceeded = self.utd() > self._max_utd_ratio
            if not (replay_buffer_not_ready or utd_exceeded):
                break
            time.sleep(0.01)

        steps = super()._train_iter_off_policy()
        self._total_updates += self._config.num_updates_per_train_iter

        with record_time("time/trainer_send_params_to_unroller"):
            if (self._total_updates %
                    self._push_params_every_n_grad_updates == 0):
                # Sending params to all the connected unrollers.
                dead_unrollers = []
                logging.debug(
                    f"Rank {self._ddp_rank} sends params to unrollers "
                    f"{self._unrollers_to_update_params}")
                for unroller_id in self._unrollers_to_update_params:
                    if not self._send_params_to_unroller(unroller_id):
                        dead_unrollers.append(unroller_id)
                # remove dead unrollers
                for unroller_id in dead_unrollers:
                    self._unrollers_to_update_params.remove(unroller_id)

        self._num_train_iters += 1

        return steps


@alf.configurable(
    whitelist=['episode_length', 'name', 'optimizer', 'unroller_only'])
class DistributedUnroller(DistributedOffPolicyAlgorithm):

    def __init__(self,
                 core_alg_ctor: Callable,
                 *args,
                 episode_length: int = 0,
                 env: AlfEnvironment = None,
                 config: TrainerConfig = None,
                 unroller_only: bool = False,
                 debug_summaries: bool = False,
                 name: str = "DistributedUnroller",
                 **kwargs):
        """
        Args:
            core_alg_ctor: creates the algorithm to be wrapped by this class.
                This algorithm's ``predict_step()`` and ``rollout_step()`` will
                be used for evaluation and rollout.
            episode_length: the maximum number of experiences sent to one training
                worker before switching to the next one. If this arg<=0, the unroller
                will wait for ``StepType.LAST`` from the env before switching. It is the
                user's responsibility to make sure that the env returns ``StepType.LAST``.
                Otherwise, for every so many experiences, it will set the last exp
                step type to an artificial ``StepType.LAST``, and switching.
                For the RL in real setting where the env never ends an episode,
                the rl_algorithm needs to use preprocess_experience_for_replay to
                correctly override the step_type from the env with the one from
                rollout_info, only then can we set this value to 0.
            unroller_only: if True, will skip the training related steps (including
                registering to all trainer workers, _create_pull_params_subprocess,
                _check_params_update observe_for_replay) and do unroll only. Therefore,
                it won't be blocked by the trainer and is useful for visualizing the
                unroll behaviors without training.
            *args: additional args to pass to ``core_alg_ctor``.
            **kwargs: additional kwargs to pass to ``core_alg_ctor``.
        """
        super().__init__(
            core_alg_ctor,
            *args,
            # Each unroller gets a random port number. If two or more unrollers
            # exist on the same machine but get the same port number, there will
            # be a port error.
            port=(_trainer_addr_config.port + random.randint(
                _unroller_port_offset, 2 * _unroller_port_offset)),
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name,
            **kwargs)

        self._episode_length = episode_length
        self._num_exps = 0
        self._is_first_step = True
        self._unroller_only = unroller_only

        ip = get_local_ip()
        self._id = f"unroller-{ip}-{self._port}"

        # For sending experience data
        # We will create it in a lazy way
        self._exp_socket = None

        # Record the current worker the data is being sent to
        # To maintain load balance, we want to cycle through the workers
        # in a round-robin fashion.
        self._current_worker = 0

        # Whether this unroller has registered to all trainer workers
        self._registered = False

    def _register_to_trainer(self):
        """Create a REQ socket and query the number of workers, ip address, and
        port number from the trainer.
        """
        # First register to the main rank
        register_socket, cxt = create_zmq_socket(zmq.DEALER,
                                                 _trainer_addr_config.ip,
                                                 _trainer_addr_config.port,
                                                 self._id)

        register_socket.send_string('init')
        message = register_socket.recv_string()
        assert message.startswith('worker-0:')
        # message format: "worker-0: N k"
        num_trainer_workers, params_socket_rank = message.split(' ')[1:]
        self._num_trainer_workers = int(num_trainer_workers)
        self._params_socket_rank = int(params_socket_rank)
        logging.info(
            f'Found {self._num_trainer_workers} workers on the trainer. ')
        # Randomly select a worker as the cycle start so that multiple unrollers
        # won't contribute to data imbalance on the trainer side.
        self._current_worker = random.randint(0, self._num_trainer_workers - 1)

        for i in range(1, self._num_trainer_workers):
            addr = 'tcp://' + ':'.join(
                [_trainer_addr_config.ip,
                 str(_trainer_addr_config.port + i)])
            register_socket.connect(addr)

        # Broadcast to all trainer workers
        for i in range(self._num_trainer_workers):
            register_socket.send_string('init')

        # Sleep to prevent closing the socket too early to send the messages
        time.sleep(1.)
        register_socket.close()
        cxt.term()

    def _create_pull_params_subprocess(self):
        # Compute the total size of the params
        buffer = io.BytesIO()
        torch.save(self._distributed_state_dict(), buffer)
        size = len(buffer.getvalue())
        # Create a shared memory object to store the new params
        # The first char indicates whether the params have been updated
        self._shared_alg_params = SharedMemory(create=True,
                                               size=size + 1,
                                               name='params_' + self._id)
        # Initialize the update char to False (not updated)
        self._shared_alg_params.buf[0] = 0

        self._shared_mem_lock = mp.Lock()

        mp.set_start_method('fork', force=True)
        process = mp.Process(target=pull_params_from_trainer,
                             args=(self._shared_alg_params.name,
                                   self._shared_mem_lock, self._id,
                                   self._params_socket_rank),
                             daemon=True)
        process.start()
        allow_child_to_ptrace(process.pid)

    def observe_for_replay(self, exp: Experience):
        """Send experience data to the trainer.

        Every time we make sure a full episode is sent to the same DDP rank, if
        multi-gpu training is enabled on the trainer.
        """
        if self._unroller_only:
            return

        # Populate information from algorithm (rollout_info) to time_step.
        # Used to overwrite env reward etc. with task reward generator's output
        # during async unroll.
        exp = self._core_alg.preprocess_experience_for_replay(exp)

        # Get the current worker id to send the exp to
        worker_id = f'worker-{self._current_worker}'
        self._num_exps += 1
        episode_end = ((self._episode_length <= 0 and bool(exp.is_last()))
                       or (self._num_exps % self._episode_length == 0))

        if self._is_first_step:
            # When the unroller has a ``max_episode_length``, we need to correctly
            # set first time step type.
            if not episode_end:
                # In rare cases, the first step is also the last step, we don't
                # overwrite LAST to FIRST
                exp = alf.nest.set_field(
                    exp, 'time_step.step_type',
                    torch.tensor([StepType.FIRST], dtype=torch.int32))
            self._is_first_step = False

        if episode_end:
            # One episode finishes; move to the next worker
            # We need to make sure a whole episode is always sent to the same
            # worker so that the temporal information is preserved.
            exp = alf.nest.set_field(
                exp, 'time_step.step_type',
                torch.tensor([StepType.LAST], dtype=torch.int32))
            # Ask the trainer to dump to the replay buffer
            self._is_first_step = True
            self._current_worker = (self._current_worker +
                                    1) % self._num_trainer_workers

        # First prune exp's replay state to save communication overhead
        exp = alf.utils.common.prune_exp_replay_state(exp,
                                                      self._use_rollout_state,
                                                      self.rollout_state_spec,
                                                      self.train_state_spec)
        # Need to convert the experience to params because it might contain distributions.
        exp_params = dist_utils.distributions_to_params(exp)
        # Use torch's save to serialize
        buffer = io.BytesIO()
        torch.save(exp_params, buffer)

        if self._exp_socket is None:
            self._exp_socket, _ = create_zmq_socket(zmq.ROUTER, '*',
                                                    self._port, self._id)

        try:
            self._exp_socket.send_multipart([
                worker_id.encode(), self._exp_socket.identity,
                buffer.getvalue()
            ])
        except zmq.error.ZMQError:
            # Trainer is down.
            # We might want to keep running the unroller but restart a trainer later.
            logging.warning(
                f"Trainer {worker_id} is not reachable. Skip sending exp to it."
            )

    def _check_params_update(self) -> bool:
        """Returns True if params have been updated.
        """
        # Check if the params have been updated
        buffer = None
        with self._shared_mem_lock:
            if self._shared_alg_params.buf[0] == 1:
                with record_time(
                        "time/dist_unroller_params_update/1_buffer_from_io"):
                    buffer = io.BytesIO(self._shared_alg_params.buf[1:])
        if buffer is not None:
            with record_time("time/dist_unroller_params_update/2_load_to_cpu"):
                state_dict = torch.load(buffer, map_location='cpu')
            # We might only update part of the params
            with record_time(
                    "time/dist_unroller_params_update/3_load_state_dict"):
                self._core_alg.load_state_dict(state_dict, strict=False)
            logging.debug("Params updated from the trainer.")
            with self._shared_mem_lock:
                self._shared_alg_params.buf[0] = 0
            return True
        return False

    def train_iter(self):
        """Perform one training iteration of the unroller.

        There is actually no training happening in this function. But the unroller
        will check if there are updated params available.
        """
        # Copied from super().train_iter()
        if self._config.empty_cache:
            with record_time("time/dist_unroller_train_iter/0_empty_cache"):
                torch.cuda.empty_cache()

        if self._unroller_only:
            self._unroll_iter_off_policy()
            return 0

        if not self._registered:
            # We need lazy registration so that trainer's params has a higher
            # priority than the unroller's loaded params (if enabled).
            self._register_to_trainer()
            # Wait until the unroller receives the first params update from trainer
            # We don't want to do this in ``__init__`` because the params might
            # get overwritten by a checkpointer.
            self._create_pull_params_subprocess()
            while True:
                if self._check_params_update():
                    break
                time.sleep(0.01)
            self._registered = True

        # Experience will be sent to the trainer in this function
        with record_time(
                "time/dist_unroller_train_iter/1_unroll_iter_off_policy"):
            self._unroll_iter_off_policy()
        with record_time(
                "time/dist_unroller_train_iter/2_check_params_update"):
            self._check_params_update()
        return 0
