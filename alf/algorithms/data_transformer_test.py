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
import torch

import alf
from alf.data_structures import namedtuple, StepType
from alf.experience_replayers.replay_buffer import ReplayBuffer, BatchInfo
from alf.algorithms.data_transformer import FrameStacker, ImageScaleTransformer
from alf.utils import common

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


if __name__ == '__main__':
    alf.test.main()
