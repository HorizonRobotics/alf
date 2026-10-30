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
import os
import queue
import random
import threading
import subprocess
import uuid
from contextlib import nullcontext
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
from alf.algorithms.distributed_off_policy_transport import (
    ExperienceSender, ExperienceReceiver, RegistrationServer, ParameterServer,
    LatestParameters, register_unroller, publish_shared_parameters,
    take_shared_parameters, shared_parameter_version, PARAM_HEADER_SIZE,
    GET_PARAMS, PARAMS)


def get_local_ip():
    """Get the ip address of the local machine."""
    return subprocess.check_output(["hostname",
                                    "-I"]).decode().strip().split()[0]


def _memory_debug_info() -> str:
    """Return a compact snapshot of process and cgroup memory state."""
    rss_mb = None
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    rss_mb = rss_kb / 1024
                    break
    except Exception:
        pass

    cgroup_paths = [
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ]
    cgroup_current_mb = None
    for path in cgroup_paths:
        try:
            with open(path, "r") as f:
                cgroup_current_mb = int(f.read().strip()) / 1024 / 1024
                break
        except Exception:
            continue

    limit_paths = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    cgroup_limit_mb = None
    for path in limit_paths:
        try:
            with open(path, "r") as f:
                value = f.read().strip()
                if value != "max":
                    cgroup_limit_mb = int(value) / 1024 / 1024
                break
        except Exception:
            continue

    shm_mb = None
    try:
        stat = os.statvfs("/dev/shm")
        shm_mb = (stat.f_bavail * stat.f_frsize) / 1024 / 1024
    except Exception:
        pass

    parts = []
    if rss_mb is not None:
        parts.append(f"rss={rss_mb:.2f}MB")
    if cgroup_current_mb is not None:
        parts.append(f"cgroup_current={cgroup_current_mb:.2f}MB")
    if cgroup_limit_mb is not None:
        parts.append(f"cgroup_limit={cgroup_limit_mb:.2f}MB")
    if shm_mb is not None:
        parts.append(f"shm_free={shm_mb:.2f}MB")
    return ", ".join(parts) if parts else "memory info unavailable"


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
                 debug_summaries: bool = False,
                 name: str = "DistributedOffPolicyAlgorithm",
                 **kwargs):
        """
            Note that optimizers should be configured on the core algorithm
            created by ``core_alg_ctor``. This wrapper does not own optimizers
            so its checkpoint format stays the same as the core algorithm.

        Args:
            core_alg_ctor: creates the algorithm to be wrapped by this class.
            config: the global ``TrainerConfig`` instance. The user is required
                to always specify this argument.
            port: port number for communication on the *current* machine.
            env: The environment to interact with. Its batch size must be 1.
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
            # Prevent in-alg ckpt since there is no such a use case.
            checkpoint=None,
            debug_summaries=debug_summaries,
            name=name)

        self._core_alg = core_alg
        self._port = port
        self._ddp_rank = max(0, PerProcessContext().ddp_rank)
        self._num_ranks = PerProcessContext().num_processes

    def state_dict(self, *args, **kwargs):
        return self._core_alg.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return self._core_alg.load_state_dict(state_dict,
                                              strict=strict,
                                              **kwargs)

    def _distributed_state_dict(self) -> dict:
        """Return `self._core_alg` state dict for distributed training.

        This dict will be used for param syncing between a trainer and an unroller.
        Sometimes optimizers have large state vectors which we want to exclude.
        Also we should exclude other parameters such as those of pretrained models.
        """
        # Note that self._core_alg won't create a relay buffer so we don't have
        # to worry about including it in the state dict.
        logging.info(f"Rank {self._ddp_rank} building distributed state dict "
                     f"({ _memory_debug_info() }).")
        state = {
            k: v
            for k, v in self._core_alg.state_dict().items()
            if (('_optimizers.' not in k) and (
                not isinstance(v, torch.nn.Parameter) or v.requires_grad))
        }
        total_bytes = 0
        tensor_entries = 0
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total_bytes += v.numel() * v.element_size()
                tensor_entries += 1
        logging.info(
            f"Rank {self._ddp_rank} built distributed state dict with "
            f"{len(state)} entries, {tensor_entries} tensor entries, "
            f"estimated tensor bytes={total_bytes / 1024 / 1024:.2f} MB "
            f"({ _memory_debug_info() }).")
        return state

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

    def summarize_metrics(self):
        self._core_alg.summarize_metrics()


def receive_experience_data(replay_buffer: ReplayBuffer,
                            new_unroller_ips_and_ports: 'Manager.Queue',
                            worker_id: int,
                            stop_event=None) -> None:
    """Receive and acknowledge every step without blocking new connections.

    Incomplete episodes remain buffered on the trainer. Completed episodes are
    committed under one replay lock, so MC-return training cannot sample them
    before their LAST step has backfilled the returns. Pending experience is
    not durable across a trainer process crash.
    """
    pending_episodes = {}

    def accept(unroller_id, payload):
        exp = torch.load(io.BytesIO(payload),
                         map_location='cpu',
                         weights_only=False)
        exp = alf.utils.common.prune_exp_replay_env_info(
            exp, env_info_spec=replay_buffer.data_spec.time_step.env_info)
        exp.env_id.zero_()
        episode = pending_episodes.setdefault(unroller_id, [])
        episode.append(exp)
        if int(exp.step_type) == StepType.LAST:
            # add_batch() takes the same re-entrant multiprocessing lock.
            lock = replay_buffer._lock
            with lock if lock is not None else nullcontext():
                for step in episode:
                    replay_buffer.add_batch(step, step.env_id)
            del pending_episodes[unroller_id]

    receiver = ExperienceReceiver(
        new_unroller_ips_and_ports,
        worker_id,
        accept,
        discard=lambda identity: pending_episodes.pop(identity, None))
    try:
        while stop_event is None or not stop_event.is_set():
            receiver.run_once()
    finally:
        receiver.close()


def pull_params_from_trainer(memory_name, memory_lock, unroller_id,
                             params_socket_rank, trainer_ip, trainer_port):
    """Request the latest version and replace older unread snapshots."""
    socket, context = create_zmq_socket(
        zmq.DEALER, trainer_ip,
        trainer_port + _params_port_offset + params_socket_rank,
        unroller_id + '_params')
    socket.setsockopt(zmq.LINGER, 0)
    params = SharedMemory(name=memory_name)
    received_version = -1
    try:
        while True:
            socket.send_multipart([GET_PARAMS, str(received_version).encode()])
            if not socket.poll(500):
                continue
            frames = socket.recv_multipart()
            if len(frames) != 3 or frames[0] != PARAMS:
                raise ValueError('Invalid parameter snapshot message')
            _, version, payload = frames
            version = int(version)
            publish_shared_parameters(params, memory_lock, version, payload)
            received_version = max(received_version, version)
    finally:
        params.close()
        socket.close()
        context.term()


@alf.configurable(
    whitelist=['max_utd_ratio', 'push_params_every_n_grad_updates', 'name'])
class DistributedTrainer(DistributedOffPolicyAlgorithm):

    def __init__(self,
                 core_alg_ctor: Callable,
                 *args,
                 max_utd_ratio: float = 10.,
                 push_params_every_n_grad_updates: int = 1,
                 env: AlfEnvironment = None,
                 config: TrainerConfig = None,
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
            Note that optimizers should be configured on the core algorithm
                created by ``core_alg_ctor``. This wrapper does not own
                optimizers so checkpoints stay wrapper-agnostic.
            *args: additional args to pass to ``core_alg_ctor``.
            **kwargs: additional kwargs to pass to ``core_alg_ctor``.
        """
        super().__init__(core_alg_ctor,
                         *args,
                         port=_trainer_addr_config.port,
                         env=env,
                         config=config,
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

        self._latest_parameters = LatestParameters()
        self._parameter_registrations = queue.Queue()
        self._communication_stop = threading.Event()
        self._communication_errors = queue.Queue()
        self._communication_threads = []
        self._data_receiver_process = None
        self._data_receiver_stop = None
        self._communication_manager = None

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

    def _publish_parameters(self, force=False):
        """Serialize in the training thread, outside any optimizer update."""
        version, _ = self._latest_parameters.get()
        if version == self._total_updates or (version >= 0 and not force):
            return
        buffer = io.BytesIO()
        torch.save(self._distributed_state_dict(), buffer)
        self._latest_parameters.publish(self._total_updates, buffer.getvalue())

    def _start_communication_thread(self, make_server):

        def run():
            server = None
            try:
                # Socket construction and all I/O belong to this thread.
                server = make_server()
                while not self._communication_stop.is_set():
                    server.run_once()
            except Exception as exc:
                self._communication_errors.put(exc)
            finally:
                if server is not None:
                    server.close()

        thread = threading.Thread(target=run, daemon=True)
        self._communication_threads.append(thread)
        thread.start()

    def _create_unroller_registration_thread(self):
        self._communication_manager = Manager()
        self._new_unroller_ips_and_ports = self._communication_manager.Queue()

        def registered(identity, ip, port, parameter_rank):
            self._new_unroller_ips_and_ports.put((identity, ip, port))
            if parameter_rank == self._ddp_rank:
                self._parameter_registrations.put(identity)

        self._start_communication_thread(lambda: RegistrationServer(
            self._port + self._ddp_rank, self._ddp_rank, self._num_ranks,
            registered))
        self._start_communication_thread(lambda: ParameterServer(
            self._port + _params_port_offset + self._ddp_rank, self.
            _parameter_registrations, self._latest_parameters))

    def _check_communication_workers(self):
        try:
            error = self._communication_errors.get_nowait()
        except queue.Empty:
            error = None
        if error is not None:
            raise RuntimeError(
                'Distributed communication thread failed') from error
        if (self._data_receiver_process is not None
                and not self._data_receiver_process.is_alive()):
            raise RuntimeError(
                'Distributed experience receiver exited with code '
                f'{self._data_receiver_process.exitcode}')

    def finish_train(self):
        self._communication_stop.set()
        for thread in self._communication_threads:
            thread.join(timeout=2)
        if self._data_receiver_process is not None:
            self._data_receiver_stop.set()
            self._data_receiver_process.join(timeout=2)
            if self._data_receiver_process.is_alive():
                self._data_receiver_process.terminate()
            self._data_receiver_process.join(timeout=2)
        if self._communication_manager is not None:
            self._communication_manager.shutdown()
        super().finish_train()

    def _create_replay_buffer_sample_experience(self):
        """Create a sample experience used to initialize the replay buffer."""
        time_step = self._env.current_time_step()
        rollout_state = self.get_initial_rollout_state(self._env.batch_size)
        alg_step = self.rollout_step(time_step, rollout_state)
        exp = make_experience(time_step, alg_step, rollout_state)
        return alf.utils.common.prune_exp_replay_state(exp,
                                                       self._use_rollout_state,
                                                       self.rollout_state_spec,
                                                       self.train_state_spec)

    def _create_multiprocessing_replay_buffer(self):
        """Create the replay buffer in shared memory."""
        ctx = mp.get_context('spawn')
        assert self._replay_buffer is None
        self._set_replay_buffer(self._create_replay_buffer_sample_experience(),
                                mp_context=ctx)
        assert self._replay_buffer._allow_multiprocess, (
            "The replay buffer must allow multi-processing.")

    def _create_data_receiver_subprocess(self):
        """Create a process to receive experience data from unrollers.
        """
        # The warm-up train_iter() creates the normal trainer replay buffer in
        # multiprocessing form before checkpoint restore, so restored shards
        # load directly into the buffer that the receiver subprocess will use.
        assert self._replay_buffer is not None
        assert self._replay_buffer._allow_multiprocess
        ctx = mp.get_context('spawn')
        self._data_receiver_stop = ctx.Event()

        # start the data receiver subprocess
        # Need to create the subprocess with 'spawn' so that we can pass a Module
        # object to subprocess with tensors in shared memory.
        process = ctx.Process(target=receive_experience_data,
                              args=(self._replay_buffer,
                                    self._new_unroller_ips_and_ports,
                                    self._ddp_rank, self._data_receiver_stop),
                              daemon=True)
        process.start()
        self._data_receiver_process = process
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
            # Create the normal trainer replay buffer in multiprocessing form
            # before checkpoint load so replay data is restored directly into
            # it. Do not start the receiver subprocess yet; it should only
            # consume unroller data after checkpoint restore has completed.
            self._create_multiprocessing_replay_buffer()
            self._num_train_iters += 1
            return 0

        if self._num_train_iters == 1:
            # Only open the unroller registration after we are sure that
            # the trainer's ckpt (if any) has been loaded, so that the trainer
            # will send correct params to any newly added unroller.
            # Publish before any DDP update can block waiting for another rank's
            # replay data. Initial sync must remain available independently of
            # progress in the training thread.
            self._publish_parameters(force=True)
            self._create_unroller_registration_thread()
            # Because unroll_length=-1, ``observe_for_replay`` will never be called.
            # Instead, we call a separate data receiver process that consistently
            # pulls data from unrollers.
            self._create_data_receiver_subprocess()

        # A worker will pause when either happens:
        # 1. replay buffer is not ready (initial collect steps not reached)
        # 2. utd ratio is too high (training is too fast; wait for more data)
        next_replay_buffer_log_time = time.time()
        utd_throttle_start_time = None
        utd_throttle_total_time = 0.
        while True:
            self._check_communication_workers()
            replay_buffer_size = self._replay_buffer.total_size
            replay_buffer_not_ready = (replay_buffer_size
                                       < self._config.initial_collect_steps)
            utd = self.utd()
            utd_exceeded = utd > self._max_utd_ratio
            now = time.time()
            if utd_exceeded and utd_throttle_start_time is None:
                utd_throttle_start_time = now
            elif not utd_exceeded and utd_throttle_start_time is not None:
                wait_time = now - utd_throttle_start_time
                utd_throttle_total_time += wait_time
                utd_throttle_start_time = None
            if now >= next_replay_buffer_log_time:
                logging.info(
                    f"Rank {self._ddp_rank} replay buffer steps="
                    f"{replay_buffer_size}, initial_collect_steps="
                    f"{self._config.initial_collect_steps}, utd={utd:.4f}, "
                    f"waiting_for_data={replay_buffer_not_ready}, "
                    f"waiting_for_utd={utd_exceeded}.")
                next_replay_buffer_log_time = now + 5.0
            if not (replay_buffer_not_ready or utd_exceeded):
                break
            time.sleep(0.01)

        if utd_throttle_total_time > 0:
            alf.summary.scalar("time/trainer_wait_for_utd",
                               utd_throttle_total_time)

        steps = super()._train_iter_off_policy()
        self._total_updates += self._config.num_updates_per_train_iter

        with record_time("time/trainer_send_params_to_unroller"):
            if (self._total_updates %
                    self._push_params_every_n_grad_updates == 0):
                self._publish_parameters(force=True)

        self._num_train_iters += 1

        return steps


@alf.configurable(whitelist=['episode_length', 'name', 'unroller_only'])
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
        # A fresh process/session must not inherit a killed unroller's sequence
        # numbers, even when the random port happens to be reused.
        self._id = f"unroller-{ip}-{self._port}-{uuid.uuid4().hex}"

        self._experience_sender = None
        self._pull_params_process = None
        self._shared_alg_params = None
        self._loaded_params_version = -1

        # Record the current worker the data is being sent to
        # To maintain load balance, we want to cycle through the workers
        # in a round-robin fashion.
        self._current_worker = 0

        # Whether this unroller has registered to all trainer workers
        self._registered = False

    def _register_to_trainer(self):
        """Bind experience first, then register explicitly with every rank."""
        self._experience_sender = ExperienceSender(self._port,
                                                   self._id.encode())
        address = _trainer_addr_config
        self._num_trainer_workers, self._params_socket_rank = register_unroller(
            address.ip, address.port, self._id.encode())
        for rank in range(1, self._num_trainer_workers):
            register_unroller(address.ip, address.port + rank,
                              self._id.encode(), self._params_socket_rank)
        # Do not collect the first rollout until all receivers have connected.
        self._experience_sender.wait_for_workers(self._num_trainer_workers)
        self._current_worker = random.randint(0, self._num_trainer_workers - 1)

    def _create_pull_params_subprocess(self):
        buffer = io.BytesIO()
        torch.save(self._distributed_state_dict(), buffer)
        # Leave headroom for serialization metadata; wire payload size is stored
        # explicitly rather than assuming identical byte lengths on both hosts.
        capacity = len(buffer.getvalue()) + 65536
        self._shared_alg_params = SharedMemory(create=True,
                                               size=capacity +
                                               PARAM_HEADER_SIZE,
                                               name='params_' + self._id)
        self._shared_alg_params.buf[:PARAM_HEADER_SIZE] = bytes(
            PARAM_HEADER_SIZE)
        ctx = mp.get_context('spawn')
        self._shared_mem_lock = ctx.Lock()
        process = ctx.Process(
            target=pull_params_from_trainer,
            args=(self._shared_alg_params.name, self._shared_mem_lock,
                  self._id, self._params_socket_rank, _trainer_addr_config.ip,
                  _trainer_addr_config.port),
            daemon=True)
        process.start()
        self._pull_params_process = process
        allow_child_to_ptrace(process.pid)

    def finish_train(self):
        if self._pull_params_process is not None:
            self._pull_params_process.terminate()
            self._pull_params_process.join(timeout=2)
        if self._shared_alg_params is not None:
            self._shared_alg_params.close()
            self._shared_alg_params.unlink()
        if self._experience_sender is not None:
            self._experience_sender.close()
        super().finish_train()

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
        if self._episode_length <= 0:
            episode_end = bool(exp.is_last())
        else:
            episode_end = (self._num_exps % self._episode_length == 0)

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

        # Backpressure is intentional: the next environment step must not
        # overtake experience that the trainer has not accepted yet.
        self._experience_sender.send(worker_id.encode(), buffer.getvalue())

    def _check_params_update(self) -> bool:
        """Apply the newest available version without erasing later arrivals."""
        updated = False
        while True:
            snapshot = take_shared_parameters(self._shared_alg_params,
                                              self._shared_mem_lock)
            if snapshot is None:
                return updated
            version, payload = snapshot
            if version <= self._loaded_params_version:
                continue
            state_dict = torch.load(io.BytesIO(payload),
                                    map_location='cpu',
                                    weights_only=False)
            # A newer snapshot may have arrived during deserialization. Skip
            # this stale one. Arrivals during model loading are applied on the
            # next pass, before returning to rollout.
            if shared_parameter_version(self._shared_alg_params,
                                        self._shared_mem_lock) > version:
                continue
            self._core_alg.load_state_dict(state_dict, strict=False)
            self._loaded_params_version = version
            updated = True

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

        if (self._pull_params_process is not None
                and not self._pull_params_process.is_alive()):
            raise RuntimeError(
                'Distributed parameter receiver exited with code '
                f'{self._pull_params_process.exitcode}')

        if not self._registered:
            # We need lazy registration so that trainer's params has a higher
            # priority than the unroller's loaded params (if enabled).
            self._register_to_trainer()
            # Wait until the unroller receives the first params update from trainer
            # We don't want to do this in ``__init__`` because the params might
            # get overwritten by a checkpointer.
            self._create_pull_params_subprocess()
            while True:
                if not self._pull_params_process.is_alive():
                    raise RuntimeError(
                        'Distributed parameter receiver exited with code '
                        f'{self._pull_params_process.exitcode}')
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
