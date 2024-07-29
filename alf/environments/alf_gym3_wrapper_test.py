# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import numpy as np
import torch
from procgen import ProcgenGym3Env
import gym3

import alf
from alf import TensorSpec, BoundedTensorSpec
from alf.environments.alf_gym3_wrapper import AlfGym3Wrapper
from alf.utils.common import zero_tensor_from_nested_spec
from alf.data_structures import StepType


class GymWrapperOnProcgenTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        env = ProcgenGym3Env(num=4, env_name='bossfight')
        self._env = AlfGym3Wrapper(env, support_force_reset=True)

    def test_batch_size(self):
        self.assertTrue(self._env.batched)
        self.assertEqual(4, self._env.batch_size)

    def test_overriden_specs(self):
        self.assertEqual({
            'level_seed': TensorSpec(shape=(), dtype=torch.int32),
            'prev_level_complete': TensorSpec(shape=(), dtype=torch.uint8),
            'prev_level_seed': TensorSpec(shape=(), dtype=torch.int32)
        }, self._env.env_info_spec())

        self.assertEqual({
            'rgb':
                BoundedTensorSpec(
                    shape=(3, 64, 64),
                    dtype=torch.uint8,
                    minimum=np.array(0),
                    maximum=np.array(255))
        }, self._env.observation_spec())

        self.assertEqual(
            BoundedTensorSpec(
                shape=(),
                dtype=torch.int32,
                minimum=np.array(0),
                maximum=np.array(14)), self._env.action_spec())

    def test_step(self):
        action = zero_tensor_from_nested_spec(self._env.action_spec(),
                                              self._env.batch_size)
        time_step = self._env.step(action)
        self.assertEqual(
            torch.tensor([StepType.FIRST] * 4), time_step.step_type)
        self.assertEqual((4, 3, 64, 64), time_step.observation['rgb'].shape)
        time_step = self._env.step(action)
        self.assertEqual(torch.tensor([StepType.MID] * 4), time_step.step_type)
        self.assertEqual((4, 3, 64, 64), time_step.observation['rgb'].shape)

        # Now send -1 to the 1st and 3rd environments to reset them (this is
        # valid because the environment is procgen).
        action = torch.as_tensor([-1, 0, -1, 0])
        new_time_step = self._env.step(action)
        self.assertEqual(
            torch.tensor(
                [StepType.LAST, StepType.MID, StepType.LAST, StepType.MID]),
            new_time_step.step_type)

        # Assert that the special handling (retain last frame for end of
        # episode) is done properly for the 1st and 3rd observation.

        def __observation_diff(a: torch.Tensor, b: torch.Tensor):
            return torch.sum(torch.abs(a - b))

        self.assertEqual(
            0,
            __observation_diff(time_step.observation['rgb'][0],
                               new_time_step.observation['rgb'][0]))
        self.assertNotEqual(
            0,
            __observation_diff(time_step.observation['rgb'][1],
                               new_time_step.observation['rgb'][1]))
        self.assertEqual(
            0,
            __observation_diff(time_step.observation['rgb'][2],
                               new_time_step.observation['rgb'][2]))
        self.assertNotEqual(
            0,
            __observation_diff(time_step.observation['rgb'][3],
                               new_time_step.observation['rgb'][3]))


if __name__ == "__main__":
    alf.test.main()
