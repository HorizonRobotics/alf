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

from collections import namedtuple
import torch

import alf
from alf.experience_replayers.replay_buffer import RingBuffer, ReplayBuffer

DataItem = namedtuple("DataItem", ["env_id", "x", "t"])


def _get_batch(env_ids, dim, t, x):
    batch_size = len(env_ids)
    x = (x * torch.arange(batch_size, dtype=torch.float32).unsqueeze(1) *
         torch.arange(dim, dtype=torch.float32).unsqueeze(0))
    return DataItem(
        env_id=torch.tensor(env_ids, dtype=torch.int64),
        x=x,
        t=t * torch.ones(batch_size, dtype=torch.int32))


class RingBufferTest(alf.test.TestCase):
    dim = 20
    max_length = 4
    num_envs = 8

    def __init__(self, *args):
        super().__init__(*args)
        self.data_spec = DataItem(
            env_id=alf.TensorSpec(shape=(), dtype=torch.int64),
            x=alf.TensorSpec(shape=(self.dim, ), dtype=torch.float32),
            t=alf.TensorSpec(shape=(), dtype=torch.int32))

    def test_ring_buffer(self):
        ring_buffer = RingBuffer(
            data_spec=self.data_spec,
            num_environments=self.num_envs,
            max_length=self.max_length)
        # Test dequeque()
        for t in range(2, 10):
            batch1 = _get_batch([1, 2, 3, 5, 6], self.dim, t=t, x=0.4)
            ring_buffer.enqueue(batch1, batch1.env_id)
        # Exception because some environments do not have data
        self.assertRaises(AssertionError, ring_buffer.dequeue)
        batch = ring_buffer.dequeue(env_ids=batch1.env_id)
        self.assertEqual(batch.t, torch.tensor([6] * 5))
        batch = ring_buffer.dequeue(env_ids=batch1.env_id)
        self.assertEqual(batch.t, torch.tensor([7] * 5))
        batch = ring_buffer.dequeue(env_ids=torch.tensor([1, 2]))
        self.assertEqual(batch.t, torch.tensor([8] * 2))
        batch = ring_buffer.dequeue(env_ids=batch1.env_id)
        self.assertEqual(batch.t, torch.tensor([[9], [9], [8], [8], [8]]))
        # Exception because some environments do not have data
        self.assertRaises(
            AssertionError, ring_buffer.dequeue, env_ids=batch1.env_id)


class ReplayBufferTest(RingBufferTest):
    def test_replay_buffer(self):
        replay_buffer = ReplayBuffer(
            data_spec=self.data_spec,
            num_environments=self.num_envs,
            max_length=self.max_length)

        batch1 = _get_batch([0, 4, 7], self.dim, t=0, x=0.1)
        replay_buffer.add_batch(batch1, batch1.env_id)
        self.assertEqual(replay_buffer._current_size,
                         torch.tensor([1, 0, 0, 0, 1, 0, 0, 1]))
        self.assertEqual(replay_buffer._current_pos,
                         torch.tensor([1, 0, 0, 0, 1, 0, 0, 1]))
        self.assertRaises(AssertionError, replay_buffer.get_batch, 8, 1)

        batch2 = _get_batch([1, 2, 3, 5, 6], self.dim, t=0, x=0.2)
        replay_buffer.add_batch(batch2, batch2.env_id)
        self.assertEqual(replay_buffer._current_size,
                         torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]))
        self.assertEqual(replay_buffer._current_pos,
                         torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]))

        batch = replay_buffer.gather_all()
        self.assertEqual(list(batch.t.shape), [8, 1])

        self.assertRaises(AssertionError, replay_buffer.get_batch, 8, 2)
        replay_buffer.get_batch(13, 1)

        batch = replay_buffer.get_batch(8, 1)
        # squeeze the time dimension
        batch = alf.nest.map_structure(lambda bat: bat.squeeze(1), batch)
        bat1 = alf.nest.map_structure(lambda bat: bat[batch1.env_id], batch)
        bat2 = alf.nest.map_structure(lambda bat: bat[batch2.env_id], batch)
        self.assertEqual(bat1.env_id, batch1.env_id)
        self.assertEqual(bat1.x, batch1.x)
        self.assertEqual(bat1.t, batch1.t)
        self.assertEqual(bat2.env_id, batch2.env_id)
        self.assertEqual(bat2.x, batch2.x)
        self.assertEqual(bat2.t, batch2.t)

        for t in range(1, 10):
            batch3 = _get_batch([0, 4, 7], self.dim, t=t, x=0.3)
            j = (t + 1) % self.max_length
            s = min(t + 1, self.max_length)
            replay_buffer.add_batch(batch3, batch3.env_id)
            self.assertEqual(replay_buffer._current_size,
                             torch.tensor([s, 1, 1, 1, s, 1, 1, s]))
            self.assertEqual(replay_buffer._current_pos,
                             torch.tensor([j, 1, 1, 1, j, 1, 1, j]))

        batch2 = _get_batch([1, 2, 3, 5, 6], self.dim, t=1, x=0.2)
        replay_buffer.add_batch(batch2, batch2.env_id)
        batch = replay_buffer.get_batch(8, 1)
        # squeeze the time dimension
        batch = alf.nest.map_structure(lambda bat: bat.squeeze(1), batch)
        bat3 = alf.nest.map_structure(lambda bat: bat[batch3.env_id], batch)
        bat2 = alf.nest.map_structure(lambda bat: bat[batch2.env_id], batch)
        self.assertEqual(bat3.env_id, batch3.env_id)
        self.assertEqual(bat3.x, batch3.x)
        self.assertEqual(bat2.env_id, batch2.env_id)
        self.assertEqual(bat2.x, batch2.x)

        batch = replay_buffer.get_batch(8, 2)
        t2 = []
        t3 = []
        for t in range(2):
            batch_t = alf.nest.map_structure(lambda b: b[:, t], batch)
            bat3 = alf.nest.map_structure(lambda bat: bat[batch3.env_id],
                                          batch_t)
            bat2 = alf.nest.map_structure(lambda bat: bat[batch2.env_id],
                                          batch_t)
            t2.append(bat2.t)
            self.assertEqual(bat3.env_id, batch3.env_id)
            self.assertEqual(bat3.x, batch3.x)
            self.assertEqual(bat2.env_id, batch2.env_id)
            self.assertEqual(bat2.x, batch2.x)
            t3.append(bat3.t)

        # Test time consistency
        self.assertEqual(t2[0] + 1, t2[1])
        self.assertEqual(t3[0] + 1, t3[1])

        batch = replay_buffer.get_batch(128, 2)
        self.assertEqual(batch.t[:, 0] + 1, batch.t[:, 1])
        self.assertEqual(list(batch.t.shape), [128, 2])

        batch = replay_buffer.get_batch(10, 2)
        self.assertEqual(batch.t[:, 0] + 1, batch.t[:, 1])
        self.assertEqual(list(batch.t.shape), [10, 2])

        batch = replay_buffer.get_batch(4, 2)
        self.assertEqual(batch.t[:, 0] + 1, batch.t[:, 1])
        self.assertEqual(list(batch.t.shape), [4, 2])

        # Test gather_all()
        # Exception because the size of all the environments are not same
        self.assertRaises(AssertionError, replay_buffer.gather_all)

        for t in range(2, 10):
            batch4 = _get_batch([1, 2, 3, 5, 6], self.dim, t=t, x=0.4)
            replay_buffer.add_batch(batch4, batch4.env_id)
        batch = replay_buffer.gather_all()
        self.assertEqual(list(batch.t.shape), [8, 4])

        # Test clear()
        replay_buffer.clear()
        self.assertEqual(replay_buffer.total_size, 0)


if __name__ == '__main__':
    alf.test.main()
