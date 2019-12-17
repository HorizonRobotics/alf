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
import tensorflow as tf

from alf.experience_replayers.replay_buffer import ReplayBuffer

DataItem = namedtuple("DataItem", ["env_id", "x", "t"])


class ReplayBufferTest(tf.test.TestCase):
    def assertArrayEqual(self, x, y):
        if isinstance(x, list):
            x = tf.constant(x)
        if isinstance(y, list):
            y = tf.constant(y)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(float(tf.reduce_max(abs(x - y))), 0)

    def test_replay_buffer(self):
        dim = 20
        max_length = 4
        num_envs = 8
        data_spec = DataItem(
            env_id=tf.TensorSpec(shape=(), dtype=tf.int32),
            x=tf.TensorSpec(shape=(dim, ), dtype=tf.float32),
            t=tf.TensorSpec(shape=(), dtype=tf.int32))

        replay_buffer = ReplayBuffer(
            data_spec=data_spec,
            num_environments=num_envs,
            max_length=max_length)

        def _get_batch(env_ids, t, x):
            batch_size = len(env_ids)
            x = (x * tf.expand_dims(tf.range(batch_size, dtype=tf.float32), 1)
                 * tf.expand_dims(tf.range(dim, dtype=tf.float32), 0))
            return DataItem(
                env_id=tf.constant(env_ids),
                x=x,
                t=t * tf.ones((batch_size, ), tf.int32))

        batch1 = _get_batch([0, 4, 7], t=0, x=0.1)
        replay_buffer.add_batch(batch1, batch1.env_id)
        self.assertArrayEqual(replay_buffer._current_size,
                              [1, 0, 0, 0, 1, 0, 0, 1])
        self.assertArrayEqual(replay_buffer._current_pos,
                              [1, 0, 0, 0, 1, 0, 0, 1])
        with self.assertRaises(tf.errors.InvalidArgumentError):
            replay_buffer.get_batch(8, 1)

        batch2 = _get_batch([1, 2, 3, 5, 6], t=0, x=0.2)
        replay_buffer.add_batch(batch2, batch2.env_id)
        self.assertArrayEqual(replay_buffer._current_size,
                              [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertArrayEqual(replay_buffer._current_pos,
                              [1, 1, 1, 1, 1, 1, 1, 1])
        with self.assertRaises(tf.errors.InvalidArgumentError):
            replay_buffer.get_batch(8, 2)
            replay_buffer.get_batch(13, 1)
        batch = replay_buffer.get_batch(8, 1)
        # squeeze the time dimension
        batch = tf.nest.map_structure(lambda bat: tf.squeeze(bat, axis=1),
                                      batch)
        bat1 = tf.nest.map_structure(
            lambda bat: tf.gather(bat, batch1.env_id, axis=0), batch)
        bat2 = tf.nest.map_structure(
            lambda bat: tf.gather(bat, batch2.env_id, axis=0), batch)
        self.assertArrayEqual(bat1.env_id, batch1.env_id)
        self.assertArrayEqual(bat1.x, batch1.x)
        self.assertArrayEqual(bat1.t, batch1.t)
        self.assertArrayEqual(bat2.env_id, batch2.env_id)
        self.assertArrayEqual(bat2.x, batch2.x)
        self.assertArrayEqual(bat2.t, batch2.t)

        for t in range(1, 10):
            batch3 = _get_batch([0, 4, 7], t=t, x=0.3)
            j = (t + 1) % max_length
            s = min(t + 1, max_length)
            replay_buffer.add_batch(batch3, batch3.env_id)
            self.assertArrayEqual(replay_buffer._current_size,
                                  [s, 1, 1, 1, s, 1, 1, s])
            self.assertArrayEqual(replay_buffer._current_pos,
                                  [j, 1, 1, 1, j, 1, 1, j])

        batch2 = _get_batch([1, 2, 3, 5, 6], t=1, x=0.2)
        replay_buffer.add_batch(batch2, batch2.env_id)
        batch = replay_buffer.get_batch(8, 1)
        # squeeze the time dimension
        batch = tf.nest.map_structure(lambda bat: tf.squeeze(bat, axis=1),
                                      batch)
        bat3 = tf.nest.map_structure(
            lambda bat: tf.gather(bat, batch3.env_id, axis=0), batch)
        bat2 = tf.nest.map_structure(
            lambda bat: tf.gather(bat, batch2.env_id, axis=0), batch)
        self.assertArrayEqual(bat3.env_id, batch3.env_id)
        self.assertArrayEqual(bat3.x, batch3.x)
        self.assertArrayEqual(bat2.env_id, batch2.env_id)
        self.assertArrayEqual(bat2.x, batch2.x)

        batch = replay_buffer.get_batch(8, 2)
        t2 = []
        t3 = []
        for t in range(2):
            batch_t = tf.nest.map_structure(lambda b: b[:, t], batch)
            bat3 = tf.nest.map_structure(
                lambda bat: tf.gather(bat, batch3.env_id, axis=0), batch_t)
            bat2 = tf.nest.map_structure(
                lambda bat: tf.gather(bat, batch2.env_id, axis=0), batch_t)
            t2.append(bat2.t)
            self.assertArrayEqual(bat3.env_id, batch3.env_id)
            self.assertArrayEqual(bat3.x, batch3.x)
            self.assertArrayEqual(bat2.env_id, batch2.env_id)
            self.assertArrayEqual(bat2.x, batch2.x)
            t3.append(bat3.t)
        self.assertArrayEqual(t2[0] + 1, t2[1])
        self.assertArrayEqual(t3[0] + 1, t3[1])

        batch = replay_buffer.get_batch(128, 2)
        self.assertArrayEqual(batch.t[:, 0] + 1, batch.t[:, 1])
        self.assertEqual(batch.t.shape, [128, 2])


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    tf.test.main()
