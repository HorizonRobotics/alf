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
import random
import threading
import subprocess
import zmq

import torch
import torch.multiprocessing as mp

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.environments.alf_environment import AlfEnvironment
from alf.experience_replayers.replay_buffer import ReplayBuffer
from alf.data_structures import Experience, make_experience
from alf.utils.per_process_context import PerProcessContext
from alf.utils import dist_utils


class UnrollerMessage(object):
    # unroller indicates end of experience for the current segment
    EXP_SEG_END = 'unroller: last_seg_exp'


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
                 core_alg_ctor: Callable,
                 *args,
                 port: int = 50000,
                 env: AlfEnvironment = None,
                 config: TrainerConfig = None,
                 optimizer: alf.optimizers.Optimizer = None,
                 checkpoint: str = None,
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
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._core_alg = core_alg
        self._port = port
        self._ddp_rank = max(0, PerProcessContext().ddp_rank)

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


def receive_experience_data(replay_buffer: ReplayBuffer,
                            new_unroller_ips_and_ports: mp.Queue,
                            worker_id: int) -> None:
    """A worker function for consistently receiving experience data from
    unrollers.

    It will be called in a child process. Each worker creates a ZMQ DEALER
    socket and listen for experience data from the unrollers.

    This function has to be in a process instead of a thread, because the
    ``replay_buffer.add_batch`` will modify the global device, which causes
    conflicts with the training code.

    Args:
        replay_buffer: an instance of ``RelayBuffer`` to store the received
            experience data. It must have the flag ``allow_multiprocess=True``.
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
        while not new_unroller_ips_and_ports.empty():
            unroller_ip, unroller_port = new_unroller_ips_and_ports.get()
            # A new unroller has connected to the trainer
            if socket is None:
                socket, _ = create_zmq_socket(zmq.DEALER, unroller_ip,
                                              unroller_port,
                                              f'worker-{worker_id}')
            else:
                addr = 'tcp://' + ':'.join([unroller_ip, str(unroller_port)])
                # Connect to an additional ROUTER
                socket.connect(addr)
        if socket is not None:
            # Receive data from any router
            unroller_id, message = socket.recv_multipart()
            if message == UnrollerMessage.EXP_SEG_END.encode():
                # Add the temp exp buffer to the replay buffer
                for exp_params in unroller_exps_buffer[unroller_id]:
                    replay_buffer.add_batch(exp_params, exp_params.env_id)
                unroller_exps_buffer[unroller_id] = []
            else:
                buffer = io.BytesIO(message)
                exp_params = torch.load(buffer, map_location='cpu')
                # Use a temp buffer to store the received exps
                if unroller_id not in unroller_exps_buffer:
                    unroller_exps_buffer[unroller_id] = []
                unroller_exps_buffer[unroller_id].append(exp_params)
        else:
            time.sleep(0.1)


def pull_params_from_trainer(shared_dict: dict, unroller_id: str):
    """ Once new params arrive, we put it in the shared dict and mark the
    ``params_updated`` as True. Later after the current unroll finishes,
    the unroller can load the new params.
    """
    socket, _ = create_zmq_socket(
        zmq.DEALER, _trainer_addr_config.ip,
        _trainer_addr_config.port + _params_port_offset,
        unroller_id + "_params")
    while True:
        shared_dict['params'] = socket.recv()
        shared_dict['params_updated'] = True


@alf.configurable(whitelist=[
    'max_utd_ratio', 'push_params_every_n_iters', 'checkpoint', 'name',
    'optimizer'
])
class DistributedTrainer(DistributedOffPolicyAlgorithm):
    def __init__(self,
                 core_alg_ctor: Callable,
                 *args,
                 max_utd_ratio: float = 10.,
                 push_params_every_n_iters: int = 1,
                 env: AlfEnvironment = None,
                 config: TrainerConfig = None,
                 optimizer: alf.optimizers.Optimizer = None,
                 checkpoint: str = None,
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
            push_params_every_n_iters: push model parameters to the unroller
                every this number of iterations.
            *args: additional args to pass to ``core_alg_ctor``.
            **kwargs: additional kwargs to pass to ``core_alg_ctor``.
        """
        super().__init__(
            core_alg_ctor,
            *args,
            port=_trainer_addr_config.port,
            env=env,
            config=config,
            optimizer=optimizer,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name,
            **kwargs)

        self._push_params_every_n_iters = push_params_every_n_iters

        # Ports:
        # 1. registration port: self._port + self._ddp_rank
        # 2. params port: self._port + _params_port_offset

        self._max_utd_ratio = max_utd_ratio

        # overwrite ``observe_for_replay`` to make sure it is never called
        # by the parent ``RLAlgorithm``
        self.observe_for_replay = self._observe_for_replay

        if self.is_main_ddp_rank:
            self._params_socket, _ = create_zmq_socket(
                zmq.ROUTER, '*', self._port + _params_port_offset)

        assert config.unroll_length == -1, (
            'unroll_length must be -1 (no unrolling)')
        # Total number of gradient updates so far
        self._total_updates = 0
        self._daemons_started = False

    def _observe_for_replay(self, exp: Experience):
        raise RuntimeError(
            'observe_for_replay should not be called for trainer')

    @property
    def is_main_ddp_rank(self):
        return self._ddp_rank == 0

    def _send_params_to_unroller(self, unroller_id: str):
        # Need to first receive a message from the unroller so that
        # send_multipart
        # Get all parameters/buffers in a state dict and send them out
        buffer = io.BytesIO()
        torch.save(self._core_alg.state_dict(), buffer)
        self._params_socket.send_multipart(
            [unroller_id + b'_params',
             buffer.getvalue()])
        logging.debug(
            f"[worker-0] Params sent to unroller {unroller_id.decode()}.")

    def _create_unroller_registration_thread(self):
        self._new_unroller_ips_and_ports = mp.Queue()
        self._connected_unrollers = set()

        def _wait_unroller_registration():
            """Wait for new registration from a unroller.
            """
            # Each rank has its own port number and a registration socket to
            # handle new unrollers.
            register_socket, _ = create_zmq_socket(zmq.ROUTER, '*',
                                                   self._port + self._ddp_rank)
            while True:
                unroller_id, message = register_socket.recv_multipart()
                if unroller_id not in self._connected_unrollers:
                    # A new unroller has connected to the trainer
                    self._connected_unrollers.add(unroller_id)
                    # The init message should always be: 'init'
                    assert message.decode() == 'init'
                    _, unroller_ip, unroller_port = unroller_id.decode().split(
                        '-')
                    logging.info(
                        f"Rank {self._ddp_rank} registered {unroller_ip} {unroller_port}"
                    )
                    # Store the new unroller ip and port so that later each rank
                    # can connect to it for experience data.
                    self._new_unroller_ips_and_ports.put((unroller_ip,
                                                          int(unroller_port)))
                    if self.is_main_ddp_rank:
                        # Send the number of workers to the new unroller,
                        # so that it is able to know other workers.
                        register_socket.send_multipart([
                            unroller_id,
                            (f'worker-0: {PerProcessContext().num_processes}'
                             ).encode()
                        ])
                        # Always first sync the params with a new unroller.
                        self._send_params_to_unroller(unroller_id)

        thread = threading.Thread(target=_wait_unroller_registration)
        thread.daemon = True
        thread.start()

    def _create_data_receiver_subprocess(self):
        """Create a proc to receive experience data from unrollers.
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

        # In the case of DDP, each subprocess is spawned. By default, if we create
        # a new subprocess, the default start method inherited is spawn. In this case,
        # we need to explicitly set the start method to fork, so that the daemon
        # subprocess can share torch modules.
        mp.set_start_method('fork', force=True)
        # start the data receiver subprocess
        process = mp.Process(
            target=receive_experience_data,
            args=(self._replay_buffer, self._new_unroller_ips_and_ports,
                  self._ddp_rank),
            daemon=True)
        process.start()

    def utd(self):
        total_exps = int(self._replay_buffer.get_current_position().sum())
        if total_exps == 0:
            return 0
        return self._total_updates / total_exps

    def _train_iter_off_policy(self):
        if not self._daemons_started:
            # Only open the unroller registration after we are sure that
            # the trainer's ckpt (if any) has been loaded, so that the trainer
            # will send correct params to any newly added unroller.
            self._create_unroller_registration_thread()
            # Because unroll_length=-1, ``observe_for_replay`` will never be called.
            # Instead, we call a separate data receiver process that consistently
            # pulls data from unrollers.
            self._create_data_receiver_subprocess()
            self._daemons_started = True

        # A worker will pause when either happens:
        # 1. replay buffer is not ready (initial collect steps not reached)
        # 2. utd ratio is too high (training is too fast; wait for more data)
        while True:
            replay_buffer_not_ready = (self._replay_buffer.total_size <
                                       self._config.initial_collect_steps)
            utd_exceeded = self.utd() > self._max_utd_ratio
            if not (replay_buffer_not_ready or utd_exceeded):
                break
            time.sleep(0.01)

        steps = super()._train_iter_off_policy()
        self._total_updates += self._config.num_updates_per_train_iter

        if (self.is_main_ddp_rank and alf.summary.get_global_counter() %
                self._push_params_every_n_iters == 0):
            # Sending params to all the connected unrollers.
            for unroller_id in self._connected_unrollers:
                self._send_params_to_unroller(unroller_id)

        return steps


@alf.configurable(whitelist=['deploy_mode', 'checkpoint', 'name', 'optimizer'])
class DistributedUnroller(DistributedOffPolicyAlgorithm):
    def __init__(self,
                 core_alg_ctor: Callable,
                 *args,
                 deploy_mode: bool = False,
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
            deploy_mode: True if this unroller is used for deployment. In this
                case, the unroller will not communicate with a trainer.
            checkpoint: this in-alg ckpt will be ignored if ``deploy_mode==False``.
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
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name,
            **kwargs)

        ip = get_local_ip()
        self._id = f"unroller-{ip}-{self._port}"

        # For sending experience data
        if not deploy_mode:
            self._exp_socket, _ = create_zmq_socket(zmq.ROUTER, '*',
                                                    self._port)
            self._create_pull_params_subprocess()

        # Record the current worker the data is being sent to
        # To maintain load balance, we want to cycle through the workers
        # in a round-robin fashion.
        self._current_worker = 0

        self._deploy_mode = deploy_mode
        # Whether this unroller has registered to all trainer workers
        self._registered = False

    def _register_to_trainer(self):
        """Create a REQ socket and query the number of workers, ip address, and
        port number from the trainer.
        """
        # First register to the main rank
        register_socket, cxt = create_zmq_socket(
            zmq.DEALER, _trainer_addr_config.ip, _trainer_addr_config.port,
            self._id)

        register_socket.send_string('init')
        message = register_socket.recv_string()
        assert message.startswith('worker-0:')
        # message format: "worker-0: N"
        num_trainer_workers = message.split(':')[1]
        self._num_trainer_workers = int(num_trainer_workers)
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

        register_socket.close()
        cxt.term()

    def _create_pull_params_subprocess(self):
        # Compute the total size of the params
        buffer = io.BytesIO()
        torch.save(self._core_alg.state_dict(), buffer)
        size = len(buffer.getvalue())
        # Create a shared dict
        self._shared_dict = mp.Manager().dict()
        self._shared_dict['params_updated'] = False
        self._shared_dict['params'] = bytes(size)

        mp.set_start_method('fork', force=True)
        process = mp.Process(
            target=pull_params_from_trainer,
            args=(self._shared_dict, self._id),
            daemon=True)
        process.start()

    def observe_for_replay(self, exp: Experience):
        """Send experience data to the trainer.

        Every time we make sure a full episode is sent to the same DDP rank, if
        multi-gpu training is enabled on the trainer.
        """
        if self._deploy_mode:
            return
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
        try:
            self._exp_socket.send_multipart([
                worker_id.encode(), self._exp_socket.identity,
                buffer.getvalue()
            ])
        except zmq.error.ZMQError:  # trainer is down
            pass

        if bool(exp.is_last()):
            # One episode finishes; move to the next worker
            # We need to make sure a whole episode is always sent to the same
            # worker so that the temporal information is preserved in its replay
            # buffer.
            self._exp_socket.send_multipart([
                worker_id.encode(), self._exp_socket.identity,
                UnrollerMessage.EXP_SEG_END.encode()
            ])
            self._current_worker = (
                self._current_worker + 1) % self._num_trainer_workers

    def _check_paramss_update(self) -> bool:
        """Returns True if params have been updated.
        """
        # Check if the params have been updated
        if self._shared_dict['params_updated']:
            buffer = io.BytesIO(self._shared_dict['params'])
            state_dict = torch.load(buffer, map_location='cpu')
            self._core_alg.load_state_dict(state_dict)
            logging.debug("Params updated from the trainer.")
            self._shared_dict['params_updated'] = False
            return True
        return False

    def _train_iter_off_policy(self):
        if not self._registered and not self._deploy_mode:
            # We need lazy registration so that trainer's params has a higher
            # priority than the unroller's loaded params (if enabled).
            self._register_to_trainer()
            # Wait until the unroller receives the first params update from trainer
            # We don't want to do this in ``__init__`` because the params might
            # get overwritten by a checkpointer.
            while True:
                if self._check_paramss_update():
                    break
                time.sleep(0.01)
            self._registered = True

        # Experience will be sent to the trainer in this function
        self._unroll_iter_off_policy()
        if not self._deploy_mode:
            self._check_paramss_update()
        return 0
