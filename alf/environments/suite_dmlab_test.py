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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
import tensorflow as tf
from absl.testing import parameterized
from tf_agents.policies import random_tf_policy
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from alf.environments import wrappers
from alf.environments import suite_dmlab


class SuiteDMLabTest(parameterized.TestCase, tf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_dmlab.is_available():
            self.skipTest('suite_dmlab is not available.')
        else:
            gin.clear_config()

    def tearDown(self):
        super().tearDown()
        self._env.close()

    @parameterized.parameters(
        dict(
            scene='nav_maze_random_goal_03',
            action_config=['action_discretize.jump=()'],
            action_length=9),
        dict(
            scene='lt_chasm',
            action_config=[
                'action_discretize.look_down_up_pixels_per_frame="range(-90,90,5)"',
                'action_discretize.crouch=()'
            ],
            action_length=42),
    )
    def test_action_discretize(self, scene, action_config, action_length):
        with gin.unlock_config():
            gin.clear_config()
            gin.parse_config(action_config)
        self._env = suite_dmlab.DeepmindLabEnv(scene=scene)
        self.assertEqual(self._env.action_space.n, action_length)

    def test_dmlab_env(self):
        ctor = lambda: suite_dmlab.load(
            scene='lt_chasm',
            gym_env_wrappers=[
                wrappers.FrameGrayScale, wrappers.FrameResize, wrappers.
                FrameStack
            ],
            wrap_with_process=False)
        self._env = parallel_py_environment.ParallelPyEnvironment([ctor] * 2)
        env = tf_py_environment.TFPyEnvironment(self._env)
        self.assertEqual((84, 84, 4), env.observation_spec().shape)

    @parameterized.parameters([
        'nav_maze_random_goal_03', 'contributed/dmlab30/rooms_watermaze',
        'contributed/psychlab/arbitrary_visuomotor_mapping'
    ])
    def test_dmlab_env_run(self, scene):
        ctor = lambda: suite_dmlab.load(
            scene=scene,
            gym_env_wrappers=[wrappers.FrameResize],
            wrap_with_process=False)

        self._env = parallel_py_environment.ParallelPyEnvironment([ctor] * 4)
        env = tf_py_environment.TFPyEnvironment(self._env)
        self.assertEqual((84, 84, 3), env.observation_spec().shape)

        random_policy = random_tf_policy.RandomTFPolicy(
            env.time_step_spec(), env.action_spec())

        driver = dynamic_step_driver.DynamicStepDriver(
            env=env, policy=random_policy, observers=None, num_steps=10)

        driver.run(maximum_iterations=10)


if __name__ == '__main__':
    tf.test.main()
