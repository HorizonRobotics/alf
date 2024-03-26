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
import multiprocessing as mp
from collections import namedtuple
import os
import tempfile
from time import sleep

import torch

from absl.testing import parameterized

import alf
from alf.tensor_specs import TensorSpec
from alf.utils.data_buffer import RingBuffer, DataBuffer
from alf.utils.checkpoint_utils import Checkpointer

DataItem = alf.data_structures.namedtuple(
    "DataItem", [
        "env_id", "x", "o", "reward", "step_type", "batch_info",
        "replay_buffer", "rollout_info_field"
    ],
    default_value=())


# Using cpu tensors are needed for running on cuda enabled devices,
# as we are not using the spawn method to start subprocesses.
def get_batch(env_ids, dim, t, x):
    batch_size = len(env_ids)
    x = torch.as_tensor(x, dtype=torch.float32, device="cpu")
    t = torch.as_tensor(t, dtype=torch.int32, device="cpu")
    ox = (x * torch.arange(
        batch_size, dtype=torch.float32, requires_grad=True,
        device="cpu").unsqueeze(1) * torch.arange(
            dim, dtype=torch.float32, requires_grad=True,
            device="cpu").unsqueeze(0))
    a = x * torch.ones(batch_size, dtype=torch.float32, device="cpu")
    g = torch.zeros(batch_size, dtype=torch.float32, device="cpu")
    # reward function adapted from ReplayBuffer: default_reward_fn
    r = torch.where(
        torch.abs(a - g) < .05,
        torch.zeros(batch_size, dtype=torch.float32, device="cpu"),
        -torch.ones(batch_size, dtype=torch.float32, device="cpu"))
    return DataItem(
        env_id=torch.tensor(env_ids, dtype=torch.int64, device="cpu"),
        x=ox,
        step_type=t * torch.ones(batch_size, dtype=torch.int32, device="cpu"),
        o=dict({
            "a": a,
            "g": g
        }),
        reward=r)


class RingBufferTest(parameterized.TestCase, alf.test.TestCase):
    dim = 20
    max_length = 4
    num_envs = 8

    def __init__(self, *args):
        super().__init__(*args)
        alf.set_default_device("cpu")  # spawn forking is required to use cuda.
        self.data_spec = DataItem(
            env_id=alf.TensorSpec(shape=(), dtype=torch.int64),
            x=alf.TensorSpec(shape=(self.dim, ), dtype=torch.float32),
            step_type=alf.TensorSpec(shape=(), dtype=torch.int32),
            o=dict({
                "a": alf.TensorSpec(shape=(), dtype=torch.float32),
                "g": alf.TensorSpec(shape=(), dtype=torch.float32)
            }),
            reward=alf.TensorSpec(shape=(), dtype=torch.float32))

    @parameterized.named_parameters([
        ('test_sync', False),
        ('test_async', True),
    ])
    def test_ring_buffer(self, allow_multiprocess):
        ring_buffer = RingBuffer(
            data_spec=self.data_spec,
            num_environments=self.num_envs,
            max_length=self.max_length,
            allow_multiprocess=allow_multiprocess)

        batch1 = get_batch([1, 2, 3, 5, 6], self.dim, t=1, x=0.4)
        if not allow_multiprocess:
            # enqueue: blocking mode only available under allow_multiprocess
            self.assertRaises(
                AssertionError,
                ring_buffer.enqueue,
                batch1,
                env_ids=batch1.env_id,
                blocking=True)

        # Test dequeque()
        for t in range(2, 10):
            batch1 = get_batch([1, 2, 3, 5, 6], self.dim, t=t, x=0.4)
            # test that the created batch has gradients
            self.assertTrue(batch1.x.requires_grad)
            ring_buffer.enqueue(batch1, batch1.env_id)
        if not allow_multiprocess:
            # dequeue: blocking mode only available under allow_multiprocess
            self.assertRaises(
                AssertionError,
                ring_buffer.dequeue,
                env_ids=batch1.env_id,
                blocking=True)
        # Exception because some environments do not have data
        self.assertRaises(AssertionError, ring_buffer.dequeue)
        batch = ring_buffer.dequeue(env_ids=batch1.env_id)
        self.assertEqual(batch.step_type, torch.tensor([[6]] * 5))
        # test that RingBuffer detaches gradients of inputs
        self.assertFalse(batch.x.requires_grad)
        batch = ring_buffer.dequeue(env_ids=batch1.env_id)
        self.assertEqual(batch.step_type, torch.tensor([[7]] * 5))
        batch = ring_buffer.dequeue(env_ids=torch.tensor([1, 2]))
        self.assertEqual(batch.step_type, torch.tensor([[8]] * 2))
        batch = ring_buffer.dequeue(env_ids=batch1.env_id)
        self.assertEqual(batch.step_type,
                         torch.tensor([[9], [9], [8], [8], [8]]))
        # Exception because some environments do not have data
        self.assertRaises(
            AssertionError, ring_buffer.dequeue, env_ids=batch1.env_id)

        # Test dequeue multiple
        ring_buffer.clear()
        for t in range(5, 10):
            batch1 = get_batch([1, 2, 3, 5, 6], self.dim, t=t, x=0.4)
            # test that the created batch has gradients
            ring_buffer.enqueue(batch1, batch1.env_id)
        # Normal dequeue in the middle of the ring buffer
        batch = ring_buffer.dequeue(env_ids=batch1.env_id, n=2)
        self.assertEqual(batch.step_type, torch.tensor([[6, 7]] * 5))
        # This dequeue crosses the end of the ring buffer
        batch = ring_buffer.dequeue(env_ids=batch1.env_id, n=2)
        self.assertEqual(batch.step_type, torch.tensor([[8, 9]] * 5))

        # Test remove_up_to
        ring_buffer.remove_up_to(4)
        for t in range(6, 10):
            batch2 = get_batch(range(0, 8), self.dim, t=t, x=0.4)
            ring_buffer.enqueue(batch2)
        prev_size = ring_buffer._current_size.clone()
        prev_pos = ring_buffer._current_pos.clone()
        ring_buffer.remove_up_to(2)
        self.assertEqual(prev_size - 2, ring_buffer._current_size)
        # shouldn't change last data pos
        self.assertEqual(prev_pos, ring_buffer._current_pos)
        # remove_up_to more than there are elements shouldn't raise error
        ring_buffer.remove_up_to(3)
        self.assertEqual(ring_buffer._current_size, torch.tensor([0] * 8))

        if allow_multiprocess:
            # Test block on dequeue without enough data
            def delayed_enqueue(ring_buffer, batch):
                alf.set_default_device("cpu")
                sleep(0.04)
                ring_buffer.enqueue(batch, batch.env_id)

            p = mp.Process(
                target=delayed_enqueue,
                args=(ring_buffer,
                      alf.nest.map_structure(lambda x: x.cpu(), batch1)))
            p.start()
            batch = ring_buffer.dequeue(env_ids=batch1.env_id, blocking=True)
            self.assertEqual(batch.step_type, torch.tensor([[9]] * 5))

            # Test block on enqueue without free space
            ring_buffer.clear()
            for t in range(6, 10):
                batch2 = get_batch(range(0, 8), self.dim, t=t, x=0.4)
                ring_buffer.enqueue(batch2)

            def delayed_dequeue():
                # cpu tensor on subprocess.  Otherwise, spawn method is needed.
                alf.set_default_device("cpu")
                sleep(0.04)
                ring_buffer.dequeue()  # 6(deleted), 7, 8, 9
                sleep(0.04)  # 10, 7, 8, 9
                ring_buffer.dequeue()  # 10, 7(deleted), 8, 9

            p = mp.Process(target=delayed_dequeue)
            p.start()
            batch2 = get_batch(range(0, 8), self.dim, t=10, x=0.4)
            ring_buffer.enqueue(batch2, blocking=True)
            p.join()
            self.assertEqual(ring_buffer._current_size[0], torch.tensor(3))

            # Test stop queue event
            def blocking_dequeue(ring_buffer):
                ring_buffer.dequeue(blocking=True)

            p = mp.Process(target=blocking_dequeue, args=(ring_buffer, ))
            ring_buffer.clear()
            p.start()
            sleep(0.02)  # for subprocess to enter while loop
            ring_buffer.stop()
            p.join()
            self.assertEqual(
                ring_buffer.dequeue(env_ids=batch1.env_id, blocking=True),
                None)

            ring_buffer.revive()
            for t in range(6, 10):
                batch2 = get_batch(range(0, 8), self.dim, t=t, x=0.4)
                self.assertEqual(
                    ring_buffer.enqueue(batch2, blocking=True), True)

            ring_buffer.stop()
            self.assertEqual(ring_buffer.enqueue(batch2, blocking=True), False)


class DataBufferTest(alf.test.TestCase):
    def test_data_buffer(self):
        dim = 20
        capacity = 256
        data_spec = (TensorSpec(shape=()), TensorSpec(shape=(dim // 3 - 1, )),
                     TensorSpec(shape=(dim - dim // 3, )))

        data_buffer = DataBuffer(data_spec=data_spec, capacity=capacity)

        def _get_batch(batch_size):
            x = torch.randn(batch_size, dim, requires_grad=True)
            x = (x[:, 0], x[:, 1:dim // 3], x[..., dim // 3:])
            return x

        data_buffer.add_batch(_get_batch(100))
        self.assertEqual(int(data_buffer.current_size), 100)
        batch = _get_batch(1000)
        # test that the created batch has gradients
        self.assertTrue(batch[0].requires_grad)
        data_buffer.add_batch(batch)
        ret = data_buffer.get_batch(2)
        # test that DataBuffer detaches gradients of inputs
        self.assertFalse(ret[0].requires_grad)
        self.assertEqual(int(data_buffer.current_size), capacity)
        ret = data_buffer.get_batch_by_indices(torch.arange(capacity))
        self.assertEqual(ret[0], batch[0][-capacity:])
        self.assertEqual(ret[1], batch[1][-capacity:])
        self.assertEqual(ret[2], batch[2][-capacity:])
        batch = _get_batch(100)
        data_buffer.add_batch(batch)
        ret = data_buffer.get_batch_by_indices(
            torch.arange(data_buffer.current_size - 100,
                         data_buffer.current_size))
        self.assertEqual(ret[0], batch[0])
        self.assertEqual(ret[1], batch[1])
        self.assertEqual(ret[2], batch[2][-capacity:])

        # Test checkpoint working
        with tempfile.TemporaryDirectory() as checkpoint_directory:
            checkpoint = Checkpointer(
                checkpoint_directory, data_buffer=data_buffer)
            checkpoint.save(10)
            data_buffer = DataBuffer(data_spec=data_spec, capacity=capacity)
            checkpoint = Checkpointer(
                checkpoint_directory, data_buffer=data_buffer)
            global_step = checkpoint.load()
            self.assertEqual(global_step, 10)

        ret = data_buffer.get_batch_by_indices(
            torch.arange(data_buffer.current_size - 100,
                         data_buffer.current_size))
        self.assertEqual(ret[0], batch[0])
        self.assertEqual(ret[1], batch[1])
        self.assertEqual(ret[2], batch[2][-capacity:])

        data_buffer.clear()
        self.assertEqual(int(data_buffer.current_size), 0)


if __name__ == '__main__':
    alf.test.main()
