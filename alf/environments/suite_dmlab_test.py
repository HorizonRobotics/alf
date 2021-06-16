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
import functools

import alf
from alf.environments import gym_wrappers, suite_dmlab, alf_environment
from alf.environments import parallel_environment, thread_environment


class SuiteDMLabTest(parameterized.TestCase, alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_dmlab.is_available():
            self.skipTest('suite_dmlab is not available.')

    def tearDown(self):
        super().tearDown()
        self._env.close()

    @parameterized.parameters(
        dict(
            scene='nav_maze_random_goal_03',
            action_config={'action_discretize.jump': ()},
            action_length=9),
        dict(
            scene='lt_chasm',
            action_config={
                'action_discretize.look_down_up_pixels_per_frame':
                    range(-90, 90, 5),
                'action_discretize.crouch': ()
            },
            action_length=42),
    )
    def test_action_discretize(self, scene, action_config, action_length):
        alf.reset_configs()
        alf.pre_config(action_config)
        self._env = suite_dmlab.DeepmindLabEnv(scene=scene)
        self.assertEqual(self._env.action_space.n, action_length)

    def test_process_env(self):
        scene = 'lt_chasm'
        self._env = suite_dmlab.load(
            scene=scene,
            gym_env_wrappers=[
                gym_wrappers.FrameGrayScale, gym_wrappers.FrameResize,
                gym_wrappers.FrameStack
            ],
            wrap_with_process=True)
        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        self.assertEqual((4, 84, 84), self._env.observation_spec().shape)

        for _ in range(10):
            actions = self._env.action_spec().sample()
            self._env.step(actions)

    def test_thread_env(self):
        scene = 'lt_chasm'
        self._env = thread_environment.ThreadEnvironment(
            lambda: suite_dmlab.load(
                scene=scene,
                gym_env_wrappers=[
                    gym_wrappers.FrameGrayScale, gym_wrappers.FrameResize,
                    gym_wrappers.FrameStack
                ],
                wrap_with_process=False))
        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        self.assertEqual((4, 84, 84), self._env.observation_spec().shape)

        for _ in range(10):
            actions = self._env.action_spec().sample()
            self._env.step(actions)

    def test_parallel_env(self):
        scene = 'lt_chasm'
        env_num = 8

        def ctor(scene, env_id=None):
            return suite_dmlab.load(
                scene=scene,
                gym_env_wrappers=[
                    gym_wrappers.FrameGrayScale, gym_wrappers.FrameResize,
                    gym_wrappers.FrameStack
                ],
                wrap_with_process=False)

        constructor = functools.partial(ctor, scene)

        self._env = parallel_environment.ParallelAlfEnvironment(
            [constructor] * env_num)
        self.assertTrue(self._env.batched)
        self.assertEqual(self._env.batch_size, env_num)
        self.assertEqual((4, 84, 84), self._env.observation_spec().shape)

    @parameterized.parameters([
        'nav_maze_random_goal_03', 'contributed/dmlab30/rooms_watermaze',
        'contributed/psychlab/arbitrary_visuomotor_mapping'
    ])
    def test_dmlab_env_run(self, scene):
        def ctor(scene, env_id=None):
            return suite_dmlab.load(
                scene=scene,
                gym_env_wrappers=[gym_wrappers.FrameResize],
                wrap_with_process=False)

        constructor = functools.partial(ctor, scene)

        self._env = parallel_environment.ParallelAlfEnvironment(
            [constructor] * 5)
        self.assertEqual((3, 84, 84), self._env.observation_spec().shape)

        for _ in range(10):
            actions = self._env.action_spec().sample(outer_dims=(5, ))
            self._env.step(actions)


if __name__ == '__main__':
    alf.test.main()
