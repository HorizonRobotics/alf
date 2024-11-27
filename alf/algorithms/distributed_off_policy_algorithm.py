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

from typing import Callable
import threading
import multiprocessing as mp
import zmq
import time
import io
import subprocess
from absl import logging

import torch

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.environments.alf_environment import AlfEnvironment
from alf.experience_replayers.replay_buffer import ReplayBuffer
from alf.data_structures import Experience, make_experience
from alf.trainers import policy_trainer
from alf.utils.per_process_context import PerProcessContext
from alf.utils import dist_utils


def get_local_ip():
    """Get the ip address of the local machine."""
    return subprocess.check_output(["hostname",
                                    "-I"]).decode().strip().split()[0]


@alf.configurable
class UnrollerAddrConfig(object):
    """A simple class for configuring the address of the unroller."""

    def __init__(self, ip: str = 'localhost', port: int = 50000):
        """
        Args:
            ip: ip address of the unroller.
            port: port number used by the unroller.
        """
        self.ip = ip
        self.port = port


_unroller_addr_config = UnrollerAddrConfig()


def create_zmq_socket(type: int, ip: str, port: int, id: str = None):
    """A helper function for creating a ZMQ socket.

    Args:
        type: type of the socket.
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
                 *args,
                 port: int = 50000,
                 core_alg_ctor: Callable = OffPolicyAlgorithm,
                 env: AlfEnvironment = None,
                 config: TrainerConfig = None,
                 optimizer: alf.optimizers.Optimizer = None,
                 checkpoint: str = None,
                 debug_summaries: bool = False,
                 name: str = "DistributedOffPolicyAlgorithm",
                 **kwargs):
        """
        Args:
            config: the global ``TrainerConfig`` instance. The user is required
                to always specify this argument.
            port: port number for communication on the *current* machine.
            core_alg_ctor: creates the algorithm to be wrapped by this class.
            env: The environment to interact with. Its batch size must be 1.
            optimizer: optimizer for the training the core algorithm.
            checkpoint: a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries: True if debug summaries should be created.
            name: the name of this algorithm.
            *args: args to pass to ``core_alg_ctor``.
            **kwargs: kwargs to pass to ``core_alg_ctor``.
        """
        # No need to pass ``config`` or ``env`` to core alg
        core_alg = core_alg_ctor(
            *args,
            config=None,
            env=None,
            debug_summaries=debug_summaries,
            **kwargs)
        assert isinstance(
            core_alg,
            OffPolicyAlgorithm), ("The core algorithm must be off-policy!")
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
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._core_alg = core_alg
        self._port = port

    ###############################
    ######### Forward calls #######
    ###############################
    @alf.utils.common.mark_eval
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

    def after_update(self, root_inputs, info):
        return self._core_alg.after_update(root_inputs, info)

    def after_train_iter(self, root_inputs, rollout_info):
        return self._core_alg.after_train_iter(root_inputs, rollout_info)


def receive_experience_data(replay_buffer: ReplayBuffer, worker_id: int,
                            unroller_ip: str, unroller_port: int,
                            total_exps_received: mp.Value) -> None:
    """A worker function for consistently receiving experience data from the
    unroller.

    It will be called in a child process. Each worker creates a ZMQ DEALER
    socket and listen for experience data from the unroller.

    Args:
        replay_buffer: an instance of ``RelayBuffer`` to store the received
            experience data. It must have the flag ``allow_multiprocess=True``.
        worker_id: the id of the worker; used by the unroller to route the
            experience data.
        unroller_ip: ip address of the unroller.
        unroller_port: port number used by the unroller.
        total_exps_received: a shared variable to store the total number of
            experience data received.
    """

    socket, _ = create_zmq_socket(zmq.DEALER, unroller_ip, unroller_port,
                                  f'worker-{worker_id}')
    # Listen for experience data forever
    while True:
        buffer = io.BytesIO(socket.recv())
        exp_params = torch.load(buffer, map_location='cpu')
        replay_buffer.add_batch(exp_params, exp_params.env_id)
        # ``exp_params`` is per-step exp, so no time dimension
        B = exp_params.step_type.shape[0]
        with total_exps_received.get_lock():
            total_exps_received.value += B


def send_params_to_unroller(m: torch.nn.Module, port: int,
                            lock: threading.Lock):
    """A worker function for responding to the unroller's request for updated params.

    Args:
        m: a torch module whose parameters will be sent to the unroller.
        port: port number used by the trainer.
        lock: a lock to prevent concurrent read/write on the params.
    """
    socket, _ = create_zmq_socket(zmq.REP, '*', port)
    while True:
        try:
            message = socket.recv_string(flags=zmq.NOBLOCK)
            if message == "unroller: update":
                # Get all parameters/buffers in a state dict and send them out
                buffer = io.BytesIO()
                with lock:
                    torch.save(m.state_dict(), buffer)
                socket.send(buffer.getvalue())
                logging.debug("[worker-0] Params sent to the unroller.")
        except zmq.Again:
            time.sleep(0.1)


@alf.configurable(whitelist=[
    'max_utd_ratio', 'core_alg_ctor', 'checkpoint', 'name', 'optimizer'
])
class DistributedTrainer(DistributedOffPolicyAlgorithm):
    def __init__(self,
                 *args,
                 max_utd_ratio: float = 10.,
                 core_alg_ctor: Callable = OffPolicyAlgorithm,
                 env: AlfEnvironment = None,
                 config: TrainerConfig = None,
                 optimizer: alf.optimizers.Optimizer = None,
                 checkpoint: str = None,
                 debug_summaries: bool = False,
                 name: str = "DistributedTrainer",
                 **kwargs):
        """
        Args:
            max_utd_ratio: max update-to-data ratio, defined as the ratio between
                the number of gradient updates and the number of exp samples
                put in the replay buffer. If the current ratio is higher than
                this value, the trainer will pause training until more experience
                samples are sent from the unroller.
                NOTE: When using DDP, if there is any subprocess exceeding this
                value, the overall training will be paused, because DDP needs to
                sync gradients among subprocesses after each backward.
                A larger value will make the trainer more likely overfit to the
                replay buffer data, while a smaller value will lead to data wastage.
            core_alg_ctor: creates the algorithm to be wrapped by this class.
                This algorithm's ``train_step()`` will be used for training.
            *args: additional args to pass to ``core_alg_ctor``.
            **kwargs: additional kwargs to pass to ``core_alg_ctor``.
        """
        super().__init__(
            *args,
            port=_unroller_addr_config.port + 1,
            core_alg_ctor=core_alg_ctor,
            env=env,
            config=config,
            optimizer=optimizer,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name,
            **kwargs)

        self._max_utd_ratio = max_utd_ratio
        self._unroller_ip = _unroller_addr_config.ip
        self._unroller_port = _unroller_addr_config.port

        # overwrite ``observe_for_replay`` to make sure it is never called
        # by the parent ``RLAlgorithm``
        self.observe_for_replay = self._observe_for_replay

        if self.is_main_ddp_rank:
            self._send_trainer_info_to_unroller()
            self._create_params_sender_thread()

        assert config.unroll_length == -1, (
            'unroll_length must be -1 (no unrolling)')
        # Because unroll_length=-1, ``observe_for_replay`` will never be called
        # in ``unroll()``. Instead, we override it to be called by a separate
        # data receiver process that consistently pulls data from the unroller.
        self._create_data_receiver_subprocess()
        self._total_updates = 0

    def _observe_for_replay(self, exp: Experience):
        raise RuntimeError(
            'observe_for_replay should not be called for trainer')

    @property
    def is_main_ddp_rank(self):
        return PerProcessContext().ddp_rank <= 0

    def _create_params_sender_thread(self):
        """Create a process to send the params to the unroller.
        """
        self._params_lock = threading.Lock()
        # start the params sending subprocess
        th = threading.Thread(
            target=send_params_to_unroller,
            args=(self._core_alg, self._port, self._params_lock))
        th.daemon = True
        th.start()

    def _create_data_receiver_subprocess(self):
        """Create a proc to receive experience data from the unroller.
        """
        # First create the replay buffer in the main process. For this, we need
        # to create a dummy experience to set up the replay buffer.
        time_step = self._env.current_time_step()
        rollout_state = self.get_initial_rollout_state(self._env.batch_size)
        alg_step = self.rollout_step(time_step, rollout_state)
        exp = make_experience(time_step, alg_step, rollout_state)
        exp = alf.utils.common.prune_exp_replay_state(
            exp, self._use_rollout_state, self.rollout_state_spec,
            self.train_state_spec)
        alf.config('ReplayBuffer', allow_multiprocess=True)
        self._set_replay_buffer(exp)

        # Create a shared value to record the total number of experience samples
        # received from the unroller
        self._total_exps_received = mp.Value('i', 0)

        # In the case of DDP, each subprocess is spawned. By default, if we create
        # a new subprocess, the default start method inherited is spawn. In this case,
        # we need to explicitly set the start method to fork, so that the daemon
        # subprocess can share torch modules.
        mp.set_start_method('fork', force=True)
        # start the data receiver subprocess
        process = mp.Process(
            target=receive_experience_data,
            args=(self._replay_buffer, max(0,
                                           PerProcessContext().ddp_rank),
                  self._unroller_ip, self._unroller_port,
                  self._total_exps_received),
            daemon=True)
        process.start()

    def _send_trainer_info_to_unroller(self):
        """Create a REP socket and send the number of workers to the unroller,
        so that the unroller is able to know the worker ids

        'worker-0', 'worker-1', ...

        to route experience data to.

        Also send the trainer's ip and port to the unroller.
        """
        socket, cxt = create_zmq_socket(zmq.REQ, self._unroller_ip,
                                        self._unroller_port)
        logging.info(
            '[worker-0] Waiting to send the unroller the number of workers...')
        trainer_ip = get_local_ip()
        socket.send_string(
            f'worker-0: {PerProcessContext().num_processes}, {trainer_ip}, {self._port}'
        )
        message = socket.recv_string()
        assert message == 'unroller: ok'
        socket.close()
        cxt.term()

    @property
    def utd(self):
        with self._total_exps_received.get_lock():
            exps_received = self._total_exps_received.value
        if exps_received == 0:
            return 0
        return self._total_updates / exps_received

    def _train_iter_off_policy(self):
        # A worker will pause when either happens:
        # 1. replay buffer is not ready (initial collect steps not reached)
        # 2. utd ratio is too high (training is too fast; wait for more data)
        while True:
            replay_buffer_not_ready = (self._replay_buffer.total_size <
                                       self._config.initial_collect_steps)
            utd_exceeded = self.utd > self._max_utd_ratio
            if not (replay_buffer_not_ready or utd_exceeded):
                break
            if replay_buffer_not_ready:
                logging.debug(
                    f"[worker-{PerProcessContext().ddp_rank}] Pause: replay buffer is not ready yet"
                )
            if utd_exceeded:
                logging.debug(
                    f"[worker-{PerProcessContext().ddp_rank}] Pause: UTD exceeded"
                )
            time.sleep(0.1)

        steps = super()._train_iter_off_policy()
        self._total_updates += self._config.num_updates_per_train_iter
        return steps

    def _backward_and_gradient_update(self, loss):
        if self.is_main_ddp_rank:
            # Lock the params to avoid sending them to the unroller while being updated.
            # We only need to lock for the main DDP rank because the other DDP ranks
            # will sync with it.
            with self._params_lock:
                return super()._backward_and_gradient_update(loss)
        else:
            return super()._backward_and_gradient_update(loss)


@alf.configurable(whitelist=[
    'core_alg_ctor', 'pull_params_every_n_iters', 'checkpoint', 'name',
    'optimizer'
])
class DistributedUnroller(DistributedOffPolicyAlgorithm):
    def __init__(self,
                 *args,
                 core_alg_ctor: Callable = OffPolicyAlgorithm,
                 pull_params_every_n_iters: int = 1,
                 env: AlfEnvironment = None,
                 config: TrainerConfig = None,
                 checkpoint: str = None,
                 debug_summaries: bool = False,
                 name: str = "DistributedUnroller",
                 **kwargs):
        """
        Args:
            core_alg_ctor: creates the algorithm to be wrapped by this class.
                This algorithm's ``predict_step()`` and ``rollout_step()`` will
                be used for evaluation and rollout.
            pull_params_every_n_iters: pull model parameters from the trainer
                every ``pull_params_every_n_iters`` iterations.
            *args: additional args to pass to ``core_alg_ctor``.
            **kwargs: additional kwargs to pass to ``core_alg_ctor``.
        """
        super().__init__(
            *args,
            port=_unroller_addr_config.port,
            core_alg_ctor=core_alg_ctor,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name,
            **kwargs)
        self._pull_params_every_n_iters = pull_params_every_n_iters

        # Get some critical information from the trainer
        self._query_trainer_info()
        # For sending experience data
        self._socket, _ = create_zmq_socket(zmq.ROUTER, '*', self._port)
        # For requesting new model params
        self._param_socket, _ = create_zmq_socket(zmq.REQ, self._trainer_ip,
                                                  self._trainer_port)
        # First need to sync params with the trainer
        self._pull_params_from_trainer()

        # Record the current worker the data is being sent to
        # To maintain load balance, we want to cycle through the workers
        # in a round-robin fashion.
        self._current_worker = 0

    @property
    def has_offline(self):
        """Hardcode this flag to train without creating an online replay buffer.

        Because the unroller never creates an online replay buffer, we need this
        hacked flag for to perform training in ``_train_iter_off_policy()``.
        """
        return True

    def _query_trainer_info(self):
        """Create a REQ socket and query the number of workers, ip address, and
        port number from the trainer.
        """
        socket, cxt = create_zmq_socket(zmq.REP, '*', self._port)
        logging.info('Waiting for the number of workers from the trainer...')
        message = socket.recv_string()
        assert message.startswith('worker-0: ')
        # message format: "worker-0: N, x.x.x.x, P"
        num_trainer_workers, trainer_ip, trainer_port = message.split(
            ':')[1].split(',')
        self._num_trainer_workers = int(num_trainer_workers)
        self._trainer_ip = trainer_ip.strip()
        self._trainer_port = int(trainer_port)
        logging.info(
            f'Found {self._num_trainer_workers} workers on the trainer. '
            f'Trainer ip: {self._trainer_ip} port: {self._trainer_port}')
        socket.send_string('unroller: ok')
        socket.close()
        cxt.term()

    def observe_for_replay(self, exp: Experience):
        """Send experience data to the trainer.

        Every time we make sure a full episode is sent to the same DDP rank, if
        multi-gpu training is enabled on the trainer.
        """
        # First prune exp's replay state to save communication overhead
        exp = alf.utils.common.prune_exp_replay_state(
            exp, self._use_rollout_state, self.rollout_state_spec,
            self.train_state_spec)
        # Need to convert the experience to params because it might contain distributions.
        exp_params = dist_utils.distributions_to_params(exp)
        # Use torch's save to serialize
        buffer = io.BytesIO()
        torch.save(exp_params, buffer)

        worker_id = f'worker-{self._current_worker}'
        self._socket.send_multipart([worker_id.encode(), buffer.getvalue()])

        if bool(exp.is_last()):
            # One episode finishes; move to the next worker
            # We need to make sure a whole episode is always sent to the same
            # worker so that the temporal information is preserved in its replay
            # buffer.
            self._current_worker = (
                self._current_worker + 1) % self._num_trainer_workers

    def _pull_params_from_trainer(self):
        """Send a request to the trainer to let it send back the updated params for
        ``self._core_alg``.
        """
        self._param_socket.send_string('unroller: update')
        # Start receiving params; will get blocked if the trainer is not running
        buffer = io.BytesIO(self._param_socket.recv())
        state_dict = torch.load(buffer, map_location='cpu')
        self._core_alg.load_state_dict(state_dict)
        logging.debug("Params updated from the trainer.")

    def train_from_replay_buffer(self, update_global_counter=False):
        """Pull model parameters from the trainer.

        Right now always pull whenever this function is called.
        """
        if alf.summary.get_global_counter(
        ) % self._pull_params_every_n_iters == 0:
            self._pull_params_from_trainer()
        return 0
