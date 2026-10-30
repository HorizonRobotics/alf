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
"""Acknowledged experience transport and latest-version parameter transport.

Experience is retried until its receiver acknowledges acceptance. A killed
unroller's in-flight data are not durable. Parameter snapshots, in contrast,
are replaceable: slow unrollers request only the newest available version.
Every ZeroMQ socket is created, used, and closed by one thread/process.
"""

import errno
import queue
import struct
import threading
import time

import zmq

READY = b'ready'
EXPERIENCE = b'experience'
ACK = b'ack'
REGISTER = b'register-v2'
REGISTERED = b'registered-v2'
GET_PARAMS = b'get-params'
PARAMS = b'params'
_RETRY_SECONDS = 0.5
_PARAM_HEADER = struct.Struct('!BQQ')  # unread, version, payload size
PARAM_HEADER_SIZE = _PARAM_HEADER.size


def _send(socket, frames):
    """Try a control/data send without blocking other connections."""
    try:
        socket.send_multipart(frames, flags=zmq.NOBLOCK)
        return True
    except zmq.ZMQError as exc:
        if exc.errno not in (zmq.EAGAIN, errno.EHOSTUNREACH):
            raise
        return False


def _socket(context, kind, identity=None):
    socket = context.socket(kind)
    socket.setsockopt(zmq.LINGER, 0)
    if identity is not None:
        socket.identity = identity
    if kind == zmq.ROUTER:
        socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
    else:
        socket.setsockopt(zmq.IMMEDIATE, 1)
    return socket


class ExperienceSender:
    """A stop-and-wait sender; observe cannot return before acceptance.

    Sequence numbers are per trainer rank and per unroller session. Lost ACKs
    cause retransmission, which the receiver deduplicates before insertion.
    """

    def __init__(self, port, identity):
        self._identity = identity
        self._context = zmq.Context()
        self._socket = _socket(self._context, zmq.ROUTER, identity)
        self._socket.bind(f'tcp://*:{port}')
        self._ready = set()
        self._sequences = {}

    def _receive(self, timeout_ms):
        if not self._socket.poll(timeout_ms):
            return None
        frames = self._socket.recv_multipart()
        if (len(frames) == 3 and frames[1:] == [READY, self._identity]):
            self._ready.add(frames[0])
        return frames

    def wait_for_workers(self, num_workers):
        expected = {f'worker-{rank}'.encode() for rank in range(num_workers)}
        while not expected.issubset(self._ready):
            self._receive(100)

    def send(self, worker_id, payload):
        sequence = self._sequences.get(worker_id, 0)
        sequence_bytes = str(sequence).encode()
        retry_at = 0.
        while True:
            now = time.monotonic()
            if worker_id in self._ready and now >= retry_at:
                _send(self._socket, [
                    worker_id, EXPERIENCE, self._identity, sequence_bytes,
                    payload
                ])
                retry_at = now + _RETRY_SECONDS
            frames = self._receive(100)
            if frames == [worker_id, ACK, self._identity, sequence_bytes]:
                self._sequences[worker_id] = sequence + 1
                return

    def close(self):
        self._socket.close()
        self._context.term()


class ExperienceReceiver:
    """Poll independent unroller connections and keep accepting new ones.

    ``accept`` must finish storing the experience before returning; only then
    is the corresponding sequence acknowledged. Idle/dead peers never block
    registration or traffic from another peer.
    """

    def __init__(self, registrations, rank, accept, discard=None):
        self._registrations = registrations
        self._rank = rank
        self._accept = accept
        self._discard = discard
        self._context = zmq.Context()
        self._poller = zmq.Poller()
        self._connections = {}
        self._peers = {}
        self._endpoints = {}
        self._last_sequence = {}
        self._next_ready = 0.

    def run_once(self, timeout_ms=100):
        while True:
            try:
                unroller_id, ip, port = self._registrations.get_nowait()
            except queue.Empty:
                break
            if unroller_id in self._connections:
                continue
            endpoint = (ip, port)
            old_id = self._endpoints.get(endpoint)
            if old_id is not None:
                # A restarted process can reuse a dead unroller's port. Its
                # old DEALER must not reconnect with the same worker identity.
                old_socket = self._connections.pop(old_id)
                self._poller.unregister(old_socket)
                del self._peers[old_socket]
                old_socket.close()
                self._last_sequence.pop(old_id, None)
                if self._discard is not None:
                    self._discard(old_id)
            socket = _socket(self._context, zmq.DEALER,
                             f'worker-{self._rank}'.encode())
            socket.connect(f'tcp://{ip}:{port}')
            self._connections[unroller_id] = socket
            self._endpoints[endpoint] = unroller_id
            self._peers[socket] = unroller_id
            self._poller.register(socket, zmq.POLLIN)
            self._next_ready = 0.

        if time.monotonic() >= self._next_ready:
            for socket, unroller_id in self._peers.items():
                _send(socket, [READY, unroller_id])
            self._next_ready = time.monotonic() + _RETRY_SECONDS

        for socket, _ in self._poller.poll(timeout_ms):
            frames = socket.recv_multipart()
            if len(frames) != 4 or frames[0] != EXPERIENCE:
                raise ValueError('Invalid distributed experience message')
            _, session, sequence_bytes, payload = frames
            sequence = int(sequence_bytes)
            unroller_id = self._peers[socket]
            if session != unroller_id:
                # Ignore stale connections during a port's session handover.
                continue
            previous = self._last_sequence.get(unroller_id, -1)
            if sequence > previous + 1:
                raise ValueError('Gap in distributed experience sequence')
            if sequence == previous + 1:
                self._accept(unroller_id, payload)
                self._last_sequence[unroller_id] = sequence
            # A duplicate means the previous ACK was lost. Never insert twice.
            _send(socket, [ACK, unroller_id, sequence_bytes])

    def close(self):
        for socket in self._peers:
            socket.close()
        self._context.term()


class RegistrationServer:
    """Rank 0 assigns a parameter rank; all other ranks use that assignment."""

    def __init__(self, port, rank, num_ranks, on_register):
        self._rank = rank
        self._num_ranks = num_ranks
        self._on_register = on_register
        self._assignments = {}
        self._context = zmq.Context()
        self._socket = _socket(self._context, zmq.ROUTER)
        self._socket.bind(f'tcp://*:{port}')

    def run_once(self, timeout_ms=100):
        if not self._socket.poll(timeout_ms):
            return
        identity, *frames = self._socket.recv_multipart()
        if not frames or frames[0] != REGISTER:
            _send(self._socket, [identity, b'error', b'Expected register-v2'])
            return
        if self._rank == 0:
            assigned = self._assignments.get(
                identity,
                len(self._assignments) % self._num_ranks)
        else:
            if len(frames) != 2:
                raise ValueError('Registration requires rank 0 assignment')
            assigned = int(frames[1])
        if not 0 <= assigned < self._num_ranks:
            raise ValueError('Invalid parameter-serving rank')
        if identity not in self._assignments:
            _, ip, port, _ = identity.decode().split('-')
            self._assignments[identity] = assigned
            self._on_register(identity, ip, int(port), assigned)
        elif self._assignments[identity] != assigned:
            raise ValueError('Conflicting parameter-serving rank')
        _send(self._socket, [
            identity, REGISTERED,
            str(self._num_ranks).encode(),
            str(assigned).encode()
        ])

    def close(self):
        self._socket.close()
        self._context.term()


def register_unroller(ip, port, identity, assigned=None):
    """Retry one rank's registration until its explicit reply is received."""
    context = zmq.Context()
    socket = _socket(context, zmq.DEALER, identity)
    socket.connect(f'tcp://{ip}:{port}')
    request = [REGISTER]
    if assigned is not None:
        request.append(str(assigned).encode())
    try:
        while True:
            _send(socket, request)
            if not socket.poll(int(_RETRY_SECONDS * 1000)):
                continue
            frames = socket.recv_multipart()
            if len(frames) != 3 or frames[0] != REGISTERED:
                raise RuntimeError(
                    f'Invalid trainer registration reply: {frames}')
            num_ranks, parameter_rank = map(int, frames[1:])
            if assigned is not None and parameter_rank != assigned:
                raise RuntimeError(
                    'Trainer changed rank 0 parameter assignment')
            return num_ranks, parameter_rank
    finally:
        socket.close()
        context.term()


class LatestParameters:
    """An immutable serialized snapshot shared between trainer threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = (-1, None)

    def publish(self, version, payload):
        with self._lock:
            if version > self._value[0]:
                self._value = (version, payload)

    def get(self):
        with self._lock:
            return self._value


class ParameterServer:
    """Own the parameter socket and answer requests with the latest snapshot.

    Requests include the last received version, acting as an acknowledgement
    belonging to that socket identity. No client can consume another's ACK,
    and a slow or dead client never blocks another client's initial sync.
    """

    def __init__(self, port, registrations, latest):
        self._registrations = registrations
        self._latest = latest
        self._registered = set()
        self._requested = {}
        self._context = zmq.Context()
        self._socket = _socket(self._context, zmq.ROUTER)
        self._socket.bind(f'tcp://*:{port}')

    def run_once(self, timeout_ms=100):
        while True:
            try:
                self._registered.add(self._registrations.get_nowait() +
                                     b'_params')
            except queue.Empty:
                break
        if self._socket.poll(timeout_ms):
            identity, *frames = self._socket.recv_multipart()
            if len(frames) == 2 and frames[0] == GET_PARAMS:
                self._requested[identity] = int(frames[1])
        version, payload = self._latest.get()
        if payload is None:
            return
        for identity, received_version in list(self._requested.items()):
            if identity in self._registered and received_version < version:
                if _send(self._socket,
                         [identity, PARAMS,
                          str(version).encode(), payload]):
                    del self._requested[identity]

    def close(self):
        self._socket.close()
        self._context.term()


def publish_shared_parameters(memory, lock, version, payload):
    """Overwrite an unread snapshot only with a newer version."""
    if len(payload) > len(memory.buf) - PARAM_HEADER_SIZE:
        raise ValueError('Parameter snapshot exceeds shared-memory capacity')
    with lock:
        _, previous_version, previous_size = _PARAM_HEADER.unpack_from(
            memory.buf)
        if previous_size and version <= previous_version:
            return
        memory.buf[PARAM_HEADER_SIZE:PARAM_HEADER_SIZE +
                   len(payload)] = payload
        _PARAM_HEADER.pack_into(memory.buf, 0, 1, version, len(payload))


def take_shared_parameters(memory, lock):
    """Consume the notification while copying, so newer arrivals stay unread."""
    with lock:
        unread, version, size = _PARAM_HEADER.unpack_from(memory.buf)
        if not unread:
            return None
        payload = bytes(memory.buf[PARAM_HEADER_SIZE:PARAM_HEADER_SIZE + size])
        _PARAM_HEADER.pack_into(memory.buf, 0, 0, version, size)
        return version, payload


def shared_parameter_version(memory, lock):
    with lock:
        return _PARAM_HEADER.unpack_from(memory.buf)[1]
