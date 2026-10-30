# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""CPU tests with real ZeroMQ sockets for remote off-policy training."""

import io
import queue
import socket
import threading
import time
import unittest
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest import mock

import torch
import torch.multiprocessing as mp
import zmq

import alf
from alf.algorithms import distributed_off_policy_transport as transport
from alf.algorithms.distributed_off_policy_algorithm import (
    DistributedUnroller, receive_experience_data, pull_params_from_trainer)
from alf.data_structures import Experience, TimeStep, StepType
from alf.experience_replayers.replay_buffer import ReplayBuffer
from alf.utils import dist_utils


def _port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _identity(port, session='session'):
    return f'unroller-127.0.0.1-{port}-{session}'.encode()


def _experience(kind, value, reward=0.):
    return Experience(time_step=TimeStep(
        step_type=torch.tensor([kind], dtype=torch.int32),
        reward=torch.tensor([reward]),
        discount=torch.ones(1),
        observation=torch.tensor([[float(value)]]),
        prev_action=torch.zeros(1, 1),
        env_id=torch.zeros(1, dtype=torch.int64),
        env_info={}),
                      action=torch.zeros(1, 1))


def _serialize(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _killed_unroller(port, payload, accepted):
    sender = transport.ExperienceSender(port, _identity(port, 'killed'))
    sender.wait_for_workers(1)
    sender.send(b'worker-0', payload)
    accepted.set()
    threading.Event().wait()


class _ServerThread:
    """Create/use/close each server socket in its owning thread."""

    def __init__(self, factory):
        self._stop = threading.Event()
        self._started = threading.Event()
        self.errors = []

        def run():
            server = None
            try:
                server = factory()
                self.server = server
                self._started.set()
                while not self._stop.is_set():
                    server.run_once(20)
            except Exception as exc:
                self.errors.append(exc)
                self._started.set()
            finally:
                if server is not None:
                    server.close()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        if not self._started.wait(5):
            raise AssertionError('Server failed to start')
        if self.errors:
            raise self.errors[0]

    def close(self):
        self._stop.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise AssertionError('Server failed to stop')
        if self.errors:
            raise self.errors[0]


class _TransportTestCase(unittest.TestCase):

    def _server(self, factory):
        worker = _ServerThread(factory)
        self.addCleanup(worker.close)
        return worker

    def _send(self, port, values, num_workers=1):
        ready = threading.Event()
        done = threading.Event()
        bound = threading.Event()
        errors = []

        def run():
            sender = None
            try:
                sender = transport.ExperienceSender(port, _identity(port))
                bound.set()
                sender.wait_for_workers(num_workers)
                ready.set()
                for rank, value in values:
                    sender.send(f'worker-{rank}'.encode(), value)
            except Exception as exc:
                errors.append(exc)
            finally:
                if sender is not None:
                    sender.close()
                done.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.assertTrue(bound.wait(5))
        return ready, done, errors


class DistributedTransportTest(_TransportTestCase):

    def test_stale_session_ready_and_ack_are_ignored(self):
        port = _port()
        ready, done, errors = self._send(port, [(0, b'payload')])
        context = zmq.Context()
        peer = context.socket(zmq.DEALER)
        peer.identity = b'worker-0'
        peer.connect(f'tcp://127.0.0.1:{port}')
        self.addCleanup(context.term)
        self.addCleanup(lambda: peer.close(linger=0))
        peer.send_multipart([transport.READY, b'old-session'])
        self.assertFalse(ready.wait(.1))
        peer.send_multipart([transport.READY, _identity(port)])
        self.assertTrue(peer.poll(5000))
        self.assertEqual(
            peer.recv_multipart(),
            [transport.EXPERIENCE,
             _identity(port), b'0', b'payload'])
        peer.send_multipart([transport.ACK, b'old-session', b'0'])
        self.assertFalse(done.wait(.1))
        peer.send_multipart([transport.ACK, _identity(port), b'0'])
        self.assertTrue(done.wait(5))
        self.assertEqual(errors, [])

    def test_ack_from_another_rank_does_not_release_sender(self):
        port = _port()
        _, done, errors = self._send(port, [(0, b'payload')], num_workers=2)
        context = zmq.Context()
        self.addCleanup(context.term)
        peers = []
        for rank in range(2):
            peer = context.socket(zmq.DEALER)
            peer.identity = f'worker-{rank}'.encode()
            peer.connect(f'tcp://127.0.0.1:{port}')
            peer.send_multipart([transport.READY, _identity(port)])
            self.addCleanup(lambda peer=peer: peer.close(linger=0))
            peers.append(peer)
        self.assertTrue(peers[0].poll(5000))
        peers[0].recv_multipart()
        peers[1].send_multipart([transport.ACK, _identity(port), b'0'])
        self.assertFalse(done.wait(.1))
        peers[0].send_multipart([transport.ACK, _identity(port), b'0'])
        self.assertTrue(done.wait(5))
        self.assertEqual(errors, [])

    def test_waits_for_receiver_and_delivers_every_step_in_order(self):
        port = _port()
        values = [str(i).encode() for i in range(20)]
        ready, done, errors = self._send(port, [(0, v) for v in values])
        self.assertFalse(ready.wait(.1))
        registrations = queue.Queue()
        registrations.put((_identity(port), '127.0.0.1', port))
        received = []
        self._server(lambda: transport.ExperienceReceiver(
            registrations, 0, lambda identity, payload: received.append(payload
                                                                        )))
        self.assertTrue(done.wait(10))
        self.assertEqual(errors, [])
        self.assertEqual(received, values)

    def test_lost_ack_retries_without_duplicate_insertion(self):
        port = _port()
        registrations = queue.Queue()
        registrations.put((_identity(port), '127.0.0.1', port))
        received = []
        original_send = transport._send
        dropped = []

        def drop_first_ack(socket, frames):
            if frames[0] == transport.ACK and not dropped:
                dropped.append(True)
                return False
            return original_send(socket, frames)

        with mock.patch.object(transport, '_send', side_effect=drop_first_ack):
            self._server(lambda: transport.ExperienceReceiver(
                registrations, 0, lambda identity, payload: received.append(
                    payload)))
            _, done, errors = self._send(port, [(0, b'first'), (0, b'second')])
            self.assertTrue(done.wait(10))
        self.assertEqual(errors, [])
        self.assertEqual(dropped, [True])
        self.assertEqual(received, [b'first', b'second'])

    def test_sender_waits_until_accept_callback_finishes(self):
        port = _port()
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        registrations = queue.Queue()
        registrations.put((_identity(port), '127.0.0.1', port))

        def accept(identity, payload):
            entered.set()
            if not release.wait(5):
                raise AssertionError('Test did not release receiver')

        self._server(lambda: transport.ExperienceReceiver(
            registrations, 0, accept))
        _, done, errors = self._send(port, [(0, b'experience')])
        self.assertTrue(entered.wait(5))
        self.assertFalse(done.wait(.1))
        release.set()
        self.assertTrue(done.wait(5))
        self.assertEqual(errors, [])

    def test_idle_peer_does_not_block_replacement(self):
        registrations = queue.Queue()
        old_port, new_port = _port(), _port()
        # No process binds the old port: this peer can never send anything.
        registrations.put((_identity(old_port), '127.0.0.1', old_port))
        received = []
        self._server(lambda: transport.ExperienceReceiver(
            registrations, 0, lambda identity, payload: received.append(payload
                                                                        )))
        registrations.put((_identity(new_port), '127.0.0.1', new_port))
        _, done, errors = self._send(new_port, [(0, b'replacement')])
        self.assertTrue(done.wait(10))
        self.assertEqual(errors, [])
        self.assertEqual(received, [b'replacement'])

    def test_replacement_can_reuse_dead_unroller_port(self):
        port = _port()
        registrations = queue.Queue()
        old_id = _identity(port, 'old')
        registrations.put((old_id, '127.0.0.1', port))
        received, discarded = [], []
        self._server(lambda: transport.ExperienceReceiver(
            registrations, 0, lambda identity, payload: received.append(
                payload), discarded.append))
        registrations.put((_identity(port), '127.0.0.1', port))
        _, done, errors = self._send(port, [(0, b'new-session')])
        self.assertTrue(done.wait(10))
        self.assertEqual(errors, [])
        self.assertEqual(discarded, [old_id])
        self.assertEqual(received, [b'new-session'])

    def test_rank_routing_uses_independent_sequences(self):
        port = _port()
        received = [[], []]
        for rank in range(2):
            registrations = queue.Queue()
            registrations.put((_identity(port), '127.0.0.1', port))
            self._server(lambda rank=rank, registrations=registrations:
                         transport.ExperienceReceiver(
                             registrations, rank, lambda identity, payload:
                             received[rank].append(payload)))
        _, done, errors = self._send(port, [(0, b'a'), (0, b'b'), (1, b'c'),
                                            (0, b'd'), (1, b'e')], 2)
        self.assertTrue(done.wait(10))
        self.assertEqual(errors, [])
        self.assertEqual(received, [[b'a', b'b', b'd'], [b'c', b'e']])

    def test_assignment_is_independent_of_registration_order(self):
        ports = [_port(), _port()]
        registered = [[], []]
        for rank in range(2):
            self._server(lambda rank=rank: transport.RegistrationServer(
                ports[rank], rank, 2, lambda identity, ip, port, assigned:
                registered[rank].append((identity, assigned))))
        first, second = _identity(51001), _identity(51002)
        self.assertEqual(
            transport.register_unroller('127.0.0.1', ports[0], first), (2, 0))
        self.assertEqual(
            transport.register_unroller('127.0.0.1', ports[0], second), (2, 1))
        # Other ranks deliberately see the opposite arrival order.
        transport.register_unroller('127.0.0.1', ports[1], second, 1)
        transport.register_unroller('127.0.0.1', ports[1], first, 0)
        self.assertEqual(registered, [[(first, 0),
                                       (second, 1)], [(second, 1),
                                                      (first, 0)]])
        # Retries must return the same assignment without registering twice.
        self.assertEqual(
            transport.register_unroller('127.0.0.1', ports[0], first), (2, 0))
        self.assertEqual(len(registered[0]), 2)

    def test_parameter_server_skips_old_versions_and_dead_clients(self):
        port = _port()
        registrations = queue.Queue()
        registrations.put(b'dead-unroller')
        latest = transport.LatestParameters()
        latest.publish(1, b'old')
        latest.publish(3, b'newest')
        latest.publish(2, b'out-of-order')
        self._server(lambda: transport.ParameterServer(port, registrations,
                                                       latest))
        context = zmq.Context()
        client = context.socket(zmq.DEALER)
        client.identity = b'live-unroller_params'
        client.connect(f'tcp://127.0.0.1:{port}')
        self.addCleanup(context.term)
        self.addCleanup(lambda: client.close(linger=0))
        # A request may arrive before registration has been processed.
        client.send_multipart([transport.GET_PARAMS, b'-1'])
        self.assertFalse(client.poll(100))
        registrations.put(b'live-unroller')
        self.assertTrue(client.poll(5000))
        self.assertEqual(client.recv_multipart(),
                         [transport.PARAMS, b'3', b'newest'])
        client.send_multipart([transport.GET_PARAMS, b'3'])
        self.assertFalse(client.poll(100))
        latest.publish(4, b'four')
        self.assertTrue(client.poll(5000))
        self.assertEqual(client.recv_multipart(),
                         [transport.PARAMS, b'4', b'four'])


class LatestParameterTest(unittest.TestCase):

    def setUp(self):
        self.memory = SharedMemory(create=True, size=65536)
        self.memory.buf[:transport.PARAM_HEADER_SIZE] = bytes(
            transport.PARAM_HEADER_SIZE)
        self.lock = threading.Lock()
        self.addCleanup(self.memory.unlink)
        self.addCleanup(self.memory.close)

    def _publish(self, version):
        buffer = io.BytesIO()
        torch.save({'weight': torch.tensor(version)}, buffer)
        transport.publish_shared_parameters(self.memory, self.lock, version,
                                            buffer.getvalue())

    def _unroller(self, load):
        return SimpleNamespace(_shared_alg_params=self.memory,
                               _shared_mem_lock=self.lock,
                               _loaded_params_version=-1,
                               _core_alg=SimpleNamespace(load_state_dict=load))

    def test_latest_unread_snapshot_replaces_old_and_rejects_stale(self):
        self._publish(1)
        self._publish(3)
        self._publish(2)
        version, payload = transport.take_shared_parameters(
            self.memory, self.lock)
        self.assertEqual(version, 3)
        self.assertEqual(torch.load(io.BytesIO(payload))['weight'].item(), 3)
        self.assertIsNone(
            transport.take_shared_parameters(self.memory, self.lock))

    def test_update_during_model_load_is_applied_before_returning(self):
        applied = []

        def load(state, strict):
            applied.append(state['weight'].item())
            if len(applied) == 1:
                self._publish(2)

        self._publish(1)
        unroller = self._unroller(load)
        self.assertTrue(DistributedUnroller._check_params_update(unroller))
        self.assertEqual(applied, [1, 2])
        self.assertEqual(unroller._loaded_params_version, 2)
        self.assertFalse(DistributedUnroller._check_params_update(unroller))

    def test_superseded_snapshot_is_skipped_before_model_load(self):
        self._publish(1)
        applied = []
        unroller = self._unroller(lambda state, strict: applied.append(state[
            'weight'].item()))
        original_load = torch.load

        def load(*args, **kwargs):
            state = original_load(*args, **kwargs)
            if state['weight'].item() == 1:
                self._publish(2)
            return state

        with mock.patch.object(torch, 'load', side_effect=load):
            self.assertTrue(DistributedUnroller._check_params_update(unroller))
        self.assertEqual(applied, [2])


class RemoteProcessTest(_TransportTestCase):
    """Exercise the actual spawned replay/parameter receiver entry points."""

    def _receiver_process(self):
        alf.set_default_device('cpu')
        context = mp.get_context('spawn')
        replay = ReplayBuffer(data_spec=dist_utils.extract_spec(_experience(
            StepType.FIRST, 0),
                                                                from_dim=1),
                              num_environments=1,
                              max_length=32,
                              keep_episodic_info=True,
                              record_episodic_return=True,
                              compute_episodic_return_on_last_step=True,
                              default_return=0.,
                              mp_context=context)
        registrations = context.Queue()
        stop = context.Event()
        process = context.Process(target=receive_experience_data,
                                  args=(replay, registrations, 0, stop))
        process.start()

        def close():
            stop.set()
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
            registrations.close()
            registrations.join_thread()

        self.addCleanup(close)
        return replay, registrations, process

    def test_spawned_receiver_commits_complete_episode_and_returns(self):
        replay, registrations, process = self._receiver_process()
        port = _port()
        registrations.put((_identity(port), '127.0.0.1', port))
        first_steps_accepted = threading.Event()
        finish = threading.Event()
        self.addCleanup(finish.set)

        def values():
            yield 0, _serialize(_experience(StepType.FIRST, 10))
            yield 0, _serialize(_experience(StepType.MID, 11))
            first_steps_accepted.set()
            if not finish.wait(10):
                raise AssertionError('Test did not finish episode')
            yield 0, _serialize(_experience(StepType.LAST, 12, reward=1.))

        _, done, errors = self._send(port, values())
        self.assertTrue(first_steps_accepted.wait(15))
        self.assertEqual(replay.total_size, 0)
        finish.set()
        self.assertTrue(done.wait(10))
        self.assertEqual(errors, [])
        self.assertTrue(process.is_alive())
        self.assertEqual(replay.total_size, 3)
        returns = replay.get_discounted_return(torch.tensor([0, 0, 0]),
                                               torch.tensor([0, 1, 2]))
        self.assertTrue(torch.all(returns[:2] > 0))
        self.assertEqual(returns[-1].item(), 0)

    def test_killed_unroller_is_replaced_without_merging_partial_episode(self):
        replay, registrations, receiver = self._receiver_process()
        port = _port()
        registrations.put((_identity(port, 'killed'), '127.0.0.1', port))
        context = mp.get_context('spawn')
        accepted = context.Event()
        killed = context.Process(
            target=_killed_unroller,
            args=(port, _serialize(_experience(StepType.FIRST,
                                               -100)), accepted))
        killed.start()

        def close_unroller():
            if killed.is_alive():
                killed.terminate()
            killed.join(5)

        self.addCleanup(close_unroller)
        self.assertTrue(accepted.wait(20))
        self.assertEqual(replay.total_size, 0)
        close_unroller()
        registrations.put((_identity(port), '127.0.0.1', port))
        _, done, errors = self._send(
            port, [(0, _serialize(_experience(StepType.FIRST, 20))),
                   (0, _serialize(_experience(StepType.MID, 21))),
                   (0, _serialize(_experience(StepType.LAST, 22, reward=1.)))])
        self.assertTrue(done.wait(10))
        self.assertEqual(errors, [])
        self.assertTrue(receiver.is_alive())
        self.assertEqual(replay.total_size, 3)
        data, _ = replay.gather_all()
        self.assertEqual(data.observation.flatten().tolist(), [20., 21., 22.])

    def test_spawned_parameter_receiver_keeps_latest_snapshot(self):
        port = _port()
        registrations = queue.Queue()
        registrations.put(b'unroller-test')
        latest = transport.LatestParameters()
        latest.publish(1, _serialize({'weight': torch.tensor(1)}))
        self._server(lambda: transport.ParameterServer(port, registrations,
                                                       latest))
        memory = SharedMemory(create=True, size=65536)
        memory.buf[:transport.PARAM_HEADER_SIZE] = bytes(
            transport.PARAM_HEADER_SIZE)
        self.addCleanup(memory.unlink)
        self.addCleanup(memory.close)
        context = mp.get_context('spawn')
        lock = context.Lock()
        process = context.Process(target=pull_params_from_trainer,
                                  args=(memory.name, lock, 'unroller-test', 0,
                                        '127.0.0.1', port - 100))
        process.start()

        def close():
            process.terminate()
            process.join(5)

        self.addCleanup(close)
        latest.publish(2, _serialize({'weight': torch.tensor(2)}))
        latest.publish(3, _serialize({'weight': torch.tensor(3)}))
        deadline = time.monotonic() + 15
        while (transport.shared_parameter_version(memory, lock) < 3
               and time.monotonic() < deadline and process.is_alive()):
            time.sleep(.02)
        self.assertTrue(process.is_alive())
        version, payload = transport.take_shared_parameters(memory, lock)
        self.assertEqual(version, 3)
        self.assertEqual(torch.load(io.BytesIO(payload))['weight'].item(), 3)


if __name__ == '__main__':
    unittest.main()
