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

import itertools
import torch

from absl.testing import parameterized

import alf
from alf import data_structures as ds
from alf.utils.data_buffer import RingBuffer
from alf.utils.data_buffer_test import get_batch, DataItem, RingBufferTest
from alf.experience_replayers.replay_buffer import ReplayBuffer
from alf.algorithms.data_transformer import HindsightExperienceTransformer


class ReplayBufferTest(RingBufferTest):
    def tearDown(self):
        super().tearDown()

    def test_replay_with_hindsight_relabel(self):
        self.max_length = 8
        torch.manual_seed(0)

        replay_buffer = ReplayBuffer(
            data_spec=self.data_spec,
            num_environments=2,
            max_length=self.max_length,
            keep_episodic_info=True,
            step_type_field="t",
            with_replacement=True)

        transform = HindsightExperienceTransformer(
            self.data_spec,
            her_proportion=0.8,
            achieved_goal_field="o.a",
            desired_goal_field="o.g")

        steps = [
            [
                ds.StepType.FIRST,  # will be overwritten
                ds.StepType.MID,  # idx == 1 in buffer
                ds.StepType.LAST,
                ds.StepType.FIRST,
                ds.StepType.MID,
                ds.StepType.MID,
                ds.StepType.LAST,
                ds.StepType.FIRST,
                ds.StepType.MID  # idx == 0
            ],
            [
                ds.StepType.FIRST,  # will be overwritten in RingBuffer
                ds.StepType.LAST,  # idx == 1 in RingBuffer
                ds.StepType.FIRST,
                ds.StepType.MID,
                ds.StepType.MID,
                ds.StepType.LAST,
                ds.StepType.FIRST,
                ds.StepType.MID,
                ds.StepType.MID  # idx == 0
            ]
        ]
        # insert data that will be overwritten later
        for b, t in list(itertools.product(range(2), range(8))):
            batch = get_batch([b], self.dim, t=steps[b][t], x=0.1 * t + b)
            replay_buffer.add_batch(batch, batch.env_id)
        # insert data
        for b, t in list(itertools.product(range(2), range(9))):
            batch = get_batch([b], self.dim, t=steps[b][t], x=0.1 * t + b)
            replay_buffer.add_batch(batch, batch.env_id)

        # Test padding
        idx = torch.tensor([[7, 0, 0, 6, 3, 3, 3, 0], [6, 0, 5, 2, 2, 2, 0,
                                                       6]])
        pos = replay_buffer._pad(idx, torch.tensor([[0] * 8, [1] * 8]))
        self.assertTrue(
            torch.equal(
                pos,
                torch.tensor([[15, 16, 16, 14, 11, 11, 11, 16],
                              [14, 16, 13, 10, 10, 10, 16, 14]],
                             dtype=torch.int64)))

        # Verify _index is built correctly.
        # Note, the _index_pos 8 represents headless timesteps, which are
        # outdated and not the same as the result of padding: 16.
        pos = torch.tensor([[15, 8, 8, 14, 11, 11, 11, 16],
                            [14, 8, 13, 10, 10, 10, 16, 14]])

        self.assertTrue(torch.equal(replay_buffer._indexed_pos, pos))
        self.assertTrue(
            torch.equal(replay_buffer._headless_indexed_pos,
                        torch.tensor([10, 9])))

        # Save original exp for later testing.
        g_orig = replay_buffer._buffer.o["g"].clone()
        r_orig = replay_buffer._buffer.reward.clone()

        # HER selects indices [0, 2, 3, 4] to relabel, from all 5:
        # env_ids: [[0, 0], [1, 1], [0, 0], [1, 1], [0, 0]]
        # pos:     [[6, 7], [1, 2], [1, 2], [3, 4], [5, 6]] + 8
        # selected:    x               x       x       x
        # future:  [   7       2       2       4       6  ] + 8
        # g        [[.7,.7],[0, 0], [.2,.2],[1.4,1.4],[.6,.6]]  # 0.1 * t + b with default 0
        # reward:  [[-1,0], [-1,-1],[-1,0], [-1,0], [-1,0]]  # recomputed with default -1
        env_ids = torch.tensor([0, 0, 1, 0])
        dist = replay_buffer.steps_to_episode_end(
            replay_buffer._pad(torch.tensor([7, 2, 4, 6]), env_ids), env_ids)
        self.assertEqual(list(dist), [1, 0, 1, 0])

        # Test HER relabeled experiences
        res, info = replay_buffer.get_batch(5, 2)
        res = res._replace(batch_info=info)
        res = transform.transform_experience(res)

        self.assertEqual(list(res.o["g"].shape), [5, 2])

        # Test relabeling doesn't change original experience
        self.assertTrue(torch.allclose(r_orig, replay_buffer._buffer.reward))
        self.assertTrue(torch.allclose(g_orig, replay_buffer._buffer.o["g"]))

        # test relabeled goals
        g = torch.tensor([0.7, 0., .2, 1.4, .6]).unsqueeze(1).expand(5, 2)
        self.assertTrue(torch.allclose(res.o["g"], g))

        # test relabeled rewards
        r = torch.tensor([[-1., 0.], [-1., -1.], [-1., 0.], [-1., 0.],
                          [-1., 0.]])
        self.assertTrue(torch.allclose(res.reward, r))

    # Gold standard functions to test HER.
    def episode_end_indices(self, b):
        """Compute episode ending indices in RingBuffer b.

        Args:
            b (ReplayBuffer): HER ReplayBuffer object.
        Returns:
            epi_ends (tensor): shape ``(2, E)``, ``epi_ends[0]`` are the
                ``env_ids``, ``epi_ends[1]`` are the ending positions of the
                episode ending/LAST steps.
                We assume every possible ``env_id`` is present.
        """
        step_types = alf.nest.get_field(b._buffer, b._step_type_field)
        epi_ends = torch.where(step_types == ds.StepType.LAST)
        epi_ends = alf.nest.map_structure(lambda d: d.type(torch.int64),
                                          epi_ends)
        # if an env has no LAST step, populate with pos - 1
        last_step_pos = b.circular(b._current_pos - 1)
        all_envs = torch.arange(b._num_envs)
        non_last_step_envs = torch.where(
            step_types[(all_envs, last_step_pos)] != ds.StepType.LAST)[0]
        epi_ends = (torch.cat([epi_ends[0], non_last_step_envs]),
                    torch.cat([epi_ends[1],
                               last_step_pos[non_last_step_envs]]))
        return epi_ends

    # Another gold standard function
    def steps_to_episode_end(self, b, env_ids, idx):
        """Compute the distance to the closest episode end in future.

        Args:
            b (ReplayBuffer): HER ReplayBuffer object.
            env_ids (tensor): shape ``L``.
            idx (tensor): shape ``L``, indexes of the current timesteps in
                the replay buffer.
        Returns:
            tensor of shape ``L``.
        """
        epi_ends = self.episode_end_indices(b)
        MAX_INT = 1000000000
        pos = b._pad(idx, env_ids)
        padded_ends = b._pad(epi_ends[1], epi_ends[0])
        min_dist = torch.ones_like(pos)
        # Using a loop over envs reduces memory by num_envs^3.
        # Due to the small memory footprint, speed is also much faster.
        for env_id in range(b._num_envs):
            (pos_env_index, ) = torch.where(env_ids == env_id)
            (end_env_index, ) = torch.where(epi_ends[0] == env_id)
            _pos = torch.gather(pos, dim=0, index=pos_env_index)
            _ends = torch.gather(padded_ends, dim=0, index=end_env_index)
            L = _pos.shape[0]
            E = _ends.shape[0]
            dist = _ends.unsqueeze(0).expand(L, E) - _pos.unsqueeze(1).expand(
                L, E)
            positive_dist = torch.where(
                dist < 0, torch.tensor(MAX_INT, dtype=torch.int64), dist)
            _min_dist, _ = torch.min(positive_dist, dim=1)
            min_dist.scatter_(dim=0, index=pos_env_index, src=_min_dist)
        return min_dist

    def generate_step_types(self, num_envs, max_steps, end_prob):
        steps = torch.tensor([ds.StepType.MID] * max_steps * num_envs)
        # start with FIRST
        env_firsts = torch.arange(num_envs)
        steps[env_firsts * max_steps] = torch.tensor([ds.StepType.FIRST])
        # randomly insert episode ends (no overlapping positions)
        segs = int(max_steps * num_envs * end_prob)
        ends = (torch.arange(segs) * (1. / end_prob)).type(torch.int64)
        ends += (torch.rand(segs) * (1. / end_prob - 1) + 1).type(torch.int64)
        steps[ends] = torch.tensor([ds.StepType.LAST]).expand(segs)
        valid_starts, = torch.where(
            ends +
            1 != torch.arange(max_steps, num_envs * max_steps, max_steps))
        steps[(ends + 1)[valid_starts]] = torch.tensor(
            [ds.StepType.FIRST]).expand(valid_starts.shape[0])
        return steps

    @parameterized.parameters([
        (False, False),
        (False, True),
        (True, False),
    ])
    def test_replay_buffer(self, allow_multiprocess, with_replacement):
        replay_buffer = ReplayBuffer(
            data_spec=self.data_spec,
            num_environments=self.num_envs,
            max_length=self.max_length,
            allow_multiprocess=allow_multiprocess)

        batch1 = get_batch([0, 4, 7], self.dim, t=0, x=0.1)
        replay_buffer.add_batch(batch1, batch1.env_id)
        self.assertEqual(replay_buffer._current_size,
                         torch.tensor([1, 0, 0, 0, 1, 0, 0, 1]))
        self.assertEqual(replay_buffer._current_pos,
                         torch.tensor([1, 0, 0, 0, 1, 0, 0, 1]))
        self.assertRaises(AssertionError, replay_buffer.get_batch, 8, 1)

        batch2 = get_batch([1, 2, 3, 5, 6], self.dim, t=0, x=0.2)
        replay_buffer.add_batch(batch2, batch2.env_id)
        self.assertEqual(replay_buffer._current_size,
                         torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]))
        self.assertEqual(replay_buffer._current_pos,
                         torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]))

        batch = replay_buffer.gather_all()
        self.assertEqual(list(batch.t.shape), [8, 1])
        # test that RingBuffer detaches gradients of inputs
        self.assertFalse(batch.x.requires_grad)

        self.assertRaises(AssertionError, replay_buffer.get_batch, 8, 2)
        replay_buffer.get_batch(13, 1)[0]

        batch = replay_buffer.get_batch(8, 1)[0]
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
            batch3 = get_batch([0, 4, 7], self.dim, t=t, x=0.3)
            j = t + 1
            s = min(t + 1, self.max_length)
            replay_buffer.add_batch(batch3, batch3.env_id)
            self.assertEqual(replay_buffer._current_size,
                             torch.tensor([s, 1, 1, 1, s, 1, 1, s]))
            self.assertEqual(replay_buffer._current_pos,
                             torch.tensor([j, 1, 1, 1, j, 1, 1, j]))

        batch2 = get_batch([1, 2, 3, 5, 6], self.dim, t=1, x=0.2)
        replay_buffer.add_batch(batch2, batch2.env_id)
        batch = replay_buffer.get_batch(8, 1)[0]
        # squeeze the time dimension
        batch = alf.nest.map_structure(lambda bat: bat.squeeze(1), batch)
        bat3 = alf.nest.map_structure(lambda bat: bat[batch3.env_id], batch)
        bat2 = alf.nest.map_structure(lambda bat: bat[batch2.env_id], batch)
        self.assertEqual(bat3.env_id, batch3.env_id)
        self.assertEqual(bat3.x, batch3.x)
        self.assertEqual(bat2.env_id, batch2.env_id)
        self.assertEqual(bat2.x, batch2.x)

        batch = replay_buffer.get_batch(8, 2)[0]
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

        batch = replay_buffer.get_batch(128, 2)[0]
        self.assertEqual(batch.t[:, 0] + 1, batch.t[:, 1])
        self.assertEqual(list(batch.t.shape), [128, 2])

        batch = replay_buffer.get_batch(10, 2)[0]
        self.assertEqual(batch.t[:, 0] + 1, batch.t[:, 1])
        self.assertEqual(list(batch.t.shape), [10, 2])

        batch = replay_buffer.get_batch(4, 2)[0]
        self.assertEqual(batch.t[:, 0] + 1, batch.t[:, 1])
        self.assertEqual(list(batch.t.shape), [4, 2])

        # Test gather_all()
        # Exception because the size of all the environments are not same
        self.assertRaises(AssertionError, replay_buffer.gather_all)

        for t in range(2, 10):
            batch4 = get_batch([1, 2, 3, 5, 6], self.dim, t=t, x=0.4)
            replay_buffer.add_batch(batch4, batch4.env_id)
        batch = replay_buffer.gather_all()
        self.assertEqual(list(batch.t.shape), [8, 4])

        # Test clear()
        replay_buffer.clear()
        self.assertEqual(replay_buffer.total_size, 0)

    def test_recent_data_and_without_replacement(self):
        num_envs = 4
        max_length = 100
        replay_buffer = ReplayBuffer(
            data_spec=self.data_spec,
            num_environments=num_envs,
            max_length=max_length,
            with_replacement=False,
            recent_data_ratio=0.5,
            recent_data_steps=4)
        replay_buffer.add_batch(get_batch([0, 1, 2, 3], self.dim, t=0, x=0.))
        batch, info = replay_buffer.get_batch(4, 1)
        self.assertEqual(info.env_ids, torch.tensor([0, 1, 2, 3]))

        replay_buffer.add_batch(get_batch([0, 1, 2, 3], self.dim, t=1, x=1.0))
        batch, info = replay_buffer.get_batch(8, 1)
        self.assertEqual(info.env_ids, torch.tensor([0, 1, 2, 3] * 2))

        for t in range(2, 32):
            replay_buffer.add_batch(
                get_batch([0, 1, 2, 3], self.dim, t=t, x=t))
        batch, info = replay_buffer.get_batch(32, 1)
        self.assertEqual(info.env_ids[16:], torch.tensor([0, 1, 2, 3] * 4))
        # The first half is from recent data
        self.assertEqual(info.env_ids[:16], torch.tensor([0, 1, 2, 3] * 4))
        self.assertEqual(
            info.positions[:16],
            torch.tensor([28] * 4 + [29] * 4 + [30] * 4 + [31] * 4))

    def test_num_earliest_frames_ignored_uniform(self):
        num_envs = 4
        max_length = 100
        replay_buffer = ReplayBuffer(
            data_spec=self.data_spec,
            num_environments=num_envs,
            max_length=max_length,
            keep_episodic_info=False,
            num_earliest_frames_ignored=2)

        replay_buffer.add_batch(get_batch([0, 1, 2, 3], self.dim, t=0, x=0.))
        # not enough data
        self.assertRaises(AssertionError, replay_buffer.get_batch, 1, 1)

        replay_buffer.add_batch(get_batch([0, 1, 2, 3], self.dim, t=1, x=0.))
        # not enough data
        self.assertRaises(AssertionError, replay_buffer.get_batch, 1, 1)

        replay_buffer.add_batch(get_batch([0, 1, 2, 3], self.dim, t=2, x=0.))
        for _ in range(10):
            batch, batch_info = replay_buffer.get_batch(1, 1)
            self.assertEqual(batch.t, torch.tensor([[2]]))

    def test_num_earliest_frames_ignored_priortized(self):
        replay_buffer = ReplayBuffer(
            data_spec=self.data_spec,
            num_environments=self.num_envs,
            max_length=self.max_length,
            num_earliest_frames_ignored=2,
            keep_episodic_info=False,
            prioritized_sampling=True)

        batch1 = get_batch([1], self.dim, x=0.25, t=0)
        replay_buffer.add_batch(batch1, batch1.env_id)
        # not enough data
        self.assertRaises(AssertionError, replay_buffer.get_batch, 1, 1)

        batch2 = get_batch([1], self.dim, x=0.25, t=1)
        replay_buffer.add_batch(batch2, batch1.env_id)
        # not enough data
        self.assertRaises(AssertionError, replay_buffer.get_batch, 1, 1)

        batch3 = get_batch([1], self.dim, x=0.25, t=2)
        replay_buffer.add_batch(batch3, batch1.env_id)
        for _ in range(10):
            batch, batch_info = replay_buffer.get_batch(1, 1)
            self.assertEqual(batch_info.env_ids,
                             torch.tensor([1], dtype=torch.int64))
            self.assertEqual(batch_info.importance_weights, 1.)
            self.assertEqual(batch_info.importance_weights, torch.tensor([1.]))
            self.assertEqual(batch.t, torch.tensor([[2]]))

    def test_prioritized_replay(self):
        replay_buffer = ReplayBuffer(
            data_spec=self.data_spec,
            num_environments=self.num_envs,
            max_length=self.max_length,
            prioritized_sampling=True)
        self.assertRaises(AssertionError, replay_buffer.get_batch, 1, 1)

        batch1 = get_batch([1], self.dim, x=0.25, t=0)
        replay_buffer.add_batch(batch1, batch1.env_id)

        batch, batch_info = replay_buffer.get_batch(1, 1)
        self.assertEqual(batch_info.env_ids,
                         torch.tensor([1], dtype=torch.int64))
        self.assertEqual(batch_info.importance_weights, 1.)
        self.assertEqual(batch_info.importance_weights, torch.tensor([1.]))
        self.assertRaises(AssertionError, replay_buffer.get_batch, 1, 2)

        batch2 = get_batch([1], self.dim, x=0.5, t=1)
        replay_buffer.add_batch(batch1, batch1.env_id)

        batch, batch_info = replay_buffer.get_batch(4, 2)
        self.assertEqual(batch_info.env_ids,
                         torch.tensor([1], dtype=torch.int64))
        self.assertEqual(batch_info.importance_weights, torch.tensor([1.]))
        self.assertEqual(batch_info.importance_weights, torch.tensor([1.] * 4))

        batch, batch_info = replay_buffer.get_batch(1000, 1)
        n0 = (replay_buffer.circular(batch_info.positions) == 0).sum()
        n1 = (replay_buffer.circular(batch_info.positions) == 1).sum()
        self.assertEqual(n0, 500)
        self.assertEqual(n1, 500)
        replay_buffer.update_priority(
            env_ids=torch.tensor([1, 1], dtype=torch.int64),
            positions=torch.tensor([0, 1], dtype=torch.int64),
            priorities=torch.tensor([0.5, 1.5]))
        batch, batch_info = replay_buffer.get_batch(1000, 1)
        n0 = (replay_buffer.circular(batch_info.positions) == 0).sum()
        n1 = (replay_buffer.circular(batch_info.positions) == 1).sum()
        self.assertEqual(n0, 250)
        self.assertEqual(n1, 750)

        batch2 = get_batch([0, 2], self.dim, x=0.5, t=1)
        replay_buffer.add_batch(batch2, batch2.env_id)
        batch, batch_info = replay_buffer.get_batch(1000, 1)

        def _get(env_id, pos):
            flag = ((batch_info.env_ids == env_id) *
                    (batch_info.positions == replay_buffer._pad(pos, env_id)))
            w = batch_info.importance_weights[torch.nonzero(
                flag, as_tuple=True)[0]]
            return flag.sum(), w

        n0, w0 = _get(0, 0)
        n1, w1 = _get(1, 0)
        n2, w2 = _get(1, 1)
        n3, w3 = _get(2, 0)
        self.assertEqual(n0, 300)
        self.assertEqual(n1, 100)
        self.assertEqual(n2, 300)
        self.assertEqual(n3, 300)
        self.assertTrue(torch.all(w0 == 1.2))
        self.assertTrue(torch.all(w1 == 0.4))
        self.assertTrue(torch.all(w2 == 1.2))
        self.assertTrue(torch.all(w3 == 1.2))

        replay_buffer.update_priority(
            env_ids=torch.tensor([1, 2], dtype=torch.int64),
            positions=torch.tensor([1, 0], dtype=torch.int64),
            priorities=torch.tensor([1.0, 1.0]))
        batch, batch_info = replay_buffer.get_batch(1000, 1)

        n0, w0 = _get(0, 0)
        n1, w1 = _get(1, 0)
        n2, w2 = _get(1, 1)
        n3, w3 = _get(2, 0)
        self.assertEqual(n0, 375)
        self.assertEqual(n1, 125)
        self.assertEqual(n2, 250)
        self.assertEqual(n3, 250)
        self.assertTrue(torch.all(w0 == 1.5))
        self.assertTrue(torch.all(w1 == 0.5))
        self.assertTrue(torch.all(w2 == 1.0))
        self.assertTrue(torch.all(w3 == 1.0))

    def test_clear_with_keep_last_exp(self):
        replay_buffer = ReplayBuffer(
            data_spec=self.data_spec,
            num_environments=self.num_envs,
            max_length=self.max_length)

        batch1 = get_batch([0, 1, 2, 3, 4, 5, 6, 7], self.dim, t=0, x=0.1)
        batch2 = get_batch([0, 1, 2, 3, 4, 5, 6, 7], self.dim, t=1, x=0.3)
        batch3 = get_batch([0, 1, 2, 3, 4, 5, 6, 7], self.dim, t=2, x=0.5)
        batch4 = get_batch([0, 1, 2, 3, 4, 5, 6, 7], self.dim, t=2, x=0.8)
        batch5 = get_batch([0, 1, 2, 3, 4, 5, 6, 7], self.dim, t=2, x=1.9)
        batch6 = get_batch([0, 1, 2, 3, 4, 5, 6, 7], self.dim, t=2, x=2.9)

        # The buffer max length is 4, therefore after the 6 add_batch() batch 3,
        # 4, 5 6 will be in the buffer.
        replay_buffer.add_batch(batch1)
        replay_buffer.add_batch(batch2)
        replay_buffer.add_batch(batch3)
        replay_buffer.add_batch(batch4)
        replay_buffer.add_batch(batch5)
        replay_buffer.add_batch(batch6)

        self.assertEqual(8 * 4, replay_buffer.total_size)
        replay_buffer.clear(keep_last_exp=True)
        self.assertEqual(8 * 1, replay_buffer.total_size)

        remaining_batch = replay_buffer.gather_all()

        # Check that after clear(), the last experience of the
        # previous buffer is kept.
        self.assertEqual(
            torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]]),
            remaining_batch.env_id)
        self.assertEqual(torch.tensor([[2.9]] * 8), remaining_batch.o['a'])


if __name__ == '__main__':
    alf.test.main()
