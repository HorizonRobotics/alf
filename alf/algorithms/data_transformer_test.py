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

from absl.testing import parameterized
import gin
import torch

import alf
from alf.data_structures import namedtuple, StepType
from alf.experience_replayers.replay_buffer import ReplayBuffer, BatchInfo
from alf.experience_replayers.replay_buffer_test import ReplayBufferTest
from alf.algorithms.data_transformer import FrameStacker, ImageScaleTransformer, HindsightExperienceTransformer
from alf.utils import common
from alf.utils.data_buffer_test import get_batch

DataItem = namedtuple(
    'DataItem', ['step_type', 'observation', 'batch_info', 'replay_buffer'],
    default_value=())


class FrameStackerTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(-1, 0)
    def test_frame_stacker(self, stack_axis=0):
        data_spec = DataItem(
            step_type=alf.TensorSpec((), dtype=torch.int32),
            observation=dict(
                scalar=alf.TensorSpec(()),
                vector=alf.TensorSpec((7, )),
                matrix=alf.TensorSpec((5, 6)),
                tensor=alf.TensorSpec((2, 3, 4))))
        replay_buffer = ReplayBuffer(
            data_spec=data_spec,
            num_environments=2,
            max_length=1024,
            num_earliest_frames_ignored=2)
        frame_stacker = FrameStacker(
            data_spec.observation,
            stack_size=3,
            stack_axis=stack_axis,
            fields=['scalar', 'vector', 'matrix', 'tensor'])

        new_spec = frame_stacker.transformed_observation_spec
        self.assertEqual(new_spec['scalar'].shape, (3, ))
        self.assertEqual(new_spec['vector'].shape, (21, ))
        if stack_axis == -1:
            self.assertEqual(new_spec['matrix'].shape, (5, 18))
            self.assertEqual(new_spec['tensor'].shape, (2, 3, 12))
        elif stack_axis == 0:
            self.assertEqual(new_spec['matrix'].shape, (15, 6))
            self.assertEqual(new_spec['tensor'].shape, (6, 3, 4))

        def _step_type(t, period):
            if t % period == 0:
                return StepType.FIRST
            if t % period == period - 1:
                return StepType.LAST
            return StepType.MID

        observation = alf.nest.map_structure(
            lambda spec: spec.randn((1000, 2)), data_spec.observation)
        state = common.zero_tensor_from_nested_spec(frame_stacker.state_spec,
                                                    2)

        def _get_stacked_data(t, b):
            if stack_axis == -1:
                return dict(
                    scalar=observation['scalar'][t, b],
                    vector=observation['vector'][t, b].reshape(-1),
                    matrix=observation['matrix'][t, b].transpose(0, 1).reshape(
                        5, 18),
                    tensor=observation['tensor'][t, b].permute(1, 2, 0,
                                                               3).reshape(
                                                                   2, 3, 12))
            elif stack_axis == 0:
                return dict(
                    scalar=observation['scalar'][t, b],
                    vector=observation['vector'][t, b].reshape(-1),
                    matrix=observation['matrix'][t, b].reshape(15, 6),
                    tensor=observation['tensor'][t, b].reshape(6, 3, 4))

        def _check_equal(stacked, expected, b):
            self.assertEqual(stacked['scalar'][b], expected['scalar'])
            self.assertEqual(stacked['vector'][b], expected['vector'])
            self.assertEqual(stacked['matrix'][b], expected['matrix'])
            self.assertEqual(stacked['tensor'][b], expected['tensor'])

        for t in range(1000):
            batch = DataItem(
                step_type=torch.tensor([_step_type(t, 17),
                                        _step_type(t, 22)]),
                observation=alf.nest.map_structure(lambda x: x[t],
                                                   observation))
            replay_buffer.add_batch(batch)
            timestep, state = frame_stacker.transform_timestep(batch, state)
            if t == 0:
                for b in (0, 1):
                    expected = _get_stacked_data([0, 0, 0], b)
                    _check_equal(timestep.observation, expected, b)
            if t == 1:
                for b in (0, 1):
                    expected = _get_stacked_data([0, 0, 1], b)
                    _check_equal(timestep.observation, expected, b)
            if t == 2:
                for b in (0, 1):
                    expected = _get_stacked_data([0, 1, 2], b)
                    _check_equal(timestep.observation, expected, b)
            if t == 16:
                for b in (0, 1):
                    expected = _get_stacked_data([14, 15, 16], b)
                    _check_equal(timestep.observation, expected, b)
            if t == 17:
                for b, t in ((0, [17, 17, 17]), (1, [15, 16, 17])):
                    expected = _get_stacked_data(t, b)
                    _check_equal(timestep.observation, expected, b)
            if t == 18:
                for b, t in ((0, [17, 17, 18]), (1, [16, 17, 18])):
                    expected = _get_stacked_data(t, b)
                    _check_equal(timestep.observation, expected, b)
            if t == 22:
                for b, t in ((0, [20, 21, 22]), (1, [22, 22, 22])):
                    expected = _get_stacked_data(t, b)
                    _check_equal(timestep.observation, expected, b)

        batch_info = BatchInfo(
            env_ids=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            positions=torch.tensor([0, 1, 18, 22], dtype=torch.int64))

        # [4, 2, ...]
        experience = replay_buffer.get_field(
            '', batch_info.env_ids.unsqueeze(-1),
            batch_info.positions.unsqueeze(-1) + torch.arange(2))
        experience = experience._replace(
            batch_info=batch_info, replay_buffer=replay_buffer)
        experience = frame_stacker.transform_experience(experience)
        expected = _get_stacked_data([0, 0, 0], 0)
        _check_equal(experience.observation, expected, (0, 0))
        expected = _get_stacked_data([0, 0, 1], 0)
        _check_equal(experience.observation, expected, (0, 1))

        expected = _get_stacked_data([0, 0, 1], 1)
        _check_equal(experience.observation, expected, (1, 0))
        expected = _get_stacked_data([0, 1, 2], 1)
        _check_equal(experience.observation, expected, (1, 1))

        expected = _get_stacked_data([17, 17, 18], 0)
        _check_equal(experience.observation, expected, (2, 0))
        expected = _get_stacked_data([17, 18, 19], 0)
        _check_equal(experience.observation, expected, (2, 1))

        expected = _get_stacked_data([22, 22, 22], 1)
        _check_equal(experience.observation, expected, (3, 0))
        expected = _get_stacked_data([22, 22, 23], 1)
        _check_equal(experience.observation, expected, (3, 1))


class ImageScaleTransformerTest(alf.test.TestCase):
    def test_image_scale_transformer(self):
        spec = alf.TensorSpec((3, 16, 16), dtype=torch.uint8)
        transformer = ImageScaleTransformer(spec, min=0.)
        new_spec = transformer.transformed_observation_spec
        self.assertEqual(new_spec.dtype, torch.float32)
        self.assertEqual(new_spec.minimum, 0.)
        self.assertEqual(new_spec.maximum, 1.)
        timestep = DataItem(
            observation=torch.randint(256, (3, 16, 16)).to(torch.uint8))
        transformed = transformer.transform_timestep(timestep, ())[0]
        self.assertLess(
            (transformed.observation * 255 - timestep.observation).abs().max(),
            1e-4)
        transformed = transformer.transform_experience(timestep)
        self.assertLess(
            (transformed.observation * 255 - timestep.observation).abs().max(),
            1e-4)

        spec = dict(
            img=alf.TensorSpec((3, 16, 16), dtype=torch.uint8),
            other=alf.TensorSpec(()))
        self.assertRaises(AssertionError, ImageScaleTransformer, spec, min=0.)
        self.assertRaises(
            AssertionError,
            ImageScaleTransformer,
            spec,
            min=0.,
            fields=['other'])
        transformer = ImageScaleTransformer(spec, min=0., fields=['img'])


class HindsightExperienceTransformerTest(ReplayBufferTest):
    @parameterized.named_parameters([
        ('test_dense_epi_ends', 0.1),
        ('test_sparse_epi_ends', 0.004),
    ])
    def test_compute_her_future_step_distance(self, end_prob):
        num_envs = 2
        max_length = 100
        torch.manual_seed(0)
        configs = [
            "HindsightExperienceTransformer.her_proportion=0.8",
            'HindsightExperienceTransformer.achieved_goal_field="o.a"',
            'HindsightExperienceTransformer.desired_goal_field="o.g"'
        ]
        gin.parse_config_files_and_bindings("", configs)

        replay_buffer = ReplayBuffer(
            data_spec=self.data_spec,
            num_environments=num_envs,
            max_length=max_length,
            keep_episodic_info=True,
            step_type_field="t")

        transform = HindsightExperienceTransformer(self.data_spec)

        # insert data
        max_steps = 1000
        # generate step_types with certain density of episode ends
        steps = self.generate_step_types(
            num_envs, max_steps, end_prob=end_prob)
        for t in range(max_steps):
            for b in range(num_envs):
                batch = get_batch([b],
                                  self.dim,
                                  t=steps[b * max_steps + t],
                                  x=1. / max_steps * t + b)
                replay_buffer.add_batch(batch, batch.env_id)
            if t > 1:
                sample_steps = min(t, max_length)
                env_ids = torch.tensor([0] * sample_steps + [1] * sample_steps)
                idx = torch.tensor(
                    list(range(sample_steps)) + list(range(sample_steps)))
                gd = self.steps_to_episode_end(replay_buffer, env_ids, idx)
                idx_orig = replay_buffer._indexed_pos.clone()
                idx_headless_orig = replay_buffer._headless_indexed_pos.clone()
                d = replay_buffer.steps_to_episode_end(
                    replay_buffer._pad(idx, env_ids), env_ids)
                # Test distance to end computation
                if not torch.equal(gd, d):
                    outs = [
                        "t: ", t, "\nenvids:\n", env_ids, "\nidx:\n", idx,
                        "\npos:\n",
                        replay_buffer._pad(idx, env_ids), "\nNot Equal: a:\n",
                        gd, "\nb:\n", d, "\nsteps:\n", replay_buffer._buffer.t,
                        "\nindexed_pos:\n", replay_buffer._indexed_pos,
                        "\nheadless_indexed_pos:\n",
                        replay_buffer._headless_indexed_pos
                    ]
                    outs = [str(out) for out in outs]
                    assert False, "".join(outs)

                # Save original exp for later testing.
                g_orig = replay_buffer._buffer.o["g"].clone()
                r_orig = replay_buffer._buffer.reward.clone()

                # HER relabel experience
                res, info = replay_buffer.get_batch(sample_steps, 2)
                res = res._replace(batch_info=info)
                res = transform.transform_experience(res)

                self.assertEqual(list(res.o["g"].shape), [sample_steps, 2])

                # Test relabeling doesn't change original experience
                self.assertTrue(
                    torch.allclose(r_orig, replay_buffer._buffer.reward))
                self.assertTrue(
                    torch.allclose(g_orig, replay_buffer._buffer.o["g"]))
                self.assertTrue(
                    torch.all(idx_orig == replay_buffer._indexed_pos))
                self.assertTrue(
                    torch.all(idx_headless_orig == replay_buffer.
                              _headless_indexed_pos))


if __name__ == '__main__':
    alf.test.main()
