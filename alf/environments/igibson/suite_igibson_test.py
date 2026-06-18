# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import functools
import os.path
import os
import gin

import alf
from alf.environments import alf_environment, thread_environment, parallel_environment
from alf.environments.igibson import suite_igibson
from alf.data_structures import StepType

import igibson
from igibson.render.profiler import Profiler
import logging


class SuiteGibsonTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        gin.clear_config()

    def tearDown(self):
        super().tearDown()
        self._env.close()

    def test_unwrapped_env(self):
        self._env = suite_igibson.load(
            config_file=os.path.join('configs',
                                     'turtlebot_point_nav_stadium.yaml'),
            env_mode='headless',
            task='visual_point_nav_random',
            physics_timestep=1.0 / 120.,
        )

        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        self._env.reset()
        for i in range(100):
            with Profiler('Environment action step'):
                actions = self._env.action_spec().sample()
                # unwrapped env (not thread_env or parallel_env) needs to convert
                # from tensor to array
                timestep = self._env.step(actions.cpu().numpy())
                if timestep.step_type == StepType.LAST:
                    logging.info(
                        "Episode finished after {} timesteps".format(i + 1))
                    break

    def test_thread_env(self):
        self._env = thread_environment.ThreadEnvironment(
            lambda: suite_igibson.load(
                config_file=os.path.join('configs',
                                         'turtlebot_point_nav_stadium.yaml'),
                env_mode='headless',
                task='visual_point_nav_random',
                physics_timestep=1.0 / 120.,
            ))
        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        self._env.reset()
        for i in range(100):
            with Profiler('Environment action step'):
                actions = self._env.action_spec().sample()
                timestep = self._env.step(actions)
                if timestep.step_type == StepType.LAST:
                    logging.info(
                        "Episode finished after {} timesteps".format(i + 1))
                    break

    def test_parallel_env(self):
        env_num = 4

        def ctor(env_id=None):
            return suite_igibson.load(
                config_file=os.path.join('configs',
                                         'turtlebot_point_nav_stadium.yaml'),
                env_mode='headless',
                task='visual_point_nav_random',
                physics_timestep=1.0 / 120.,
            )

        constructor = functools.partial(ctor)

        self._env = parallel_environment.ParallelAlfEnvironment(
            [constructor] * env_num)
        self.assertTrue(self._env.batched)
        self.assertEqual(self._env.batch_size, env_num)

        actions = self._env.action_spec().sample(outer_dims=(env_num, ))
        for _ in range(10):
            time_step = self._env.step(actions)

    def test_env_info(self):
        # test creating multiple envs in the same process
        l0_env = suite_igibson.load(
            config_file=os.path.join('configs',
                                     'turtlebot_point_nav_stadium.yaml'),
            env_mode='headless',
            task='visual_point_nav_random',
            physics_timestep=1.0 / 120.,
        )
        l1_env = suite_igibson.load(
            config_file=os.path.join('configs',
                                     'turtlebot_point_nav_stadium.yaml'),
            env_mode='headless',
            task='visual_point_nav_random',
            physics_timestep=1.0 / 120.,
        )

        # level 0 envs don't have costs
        actions = l0_env.action_spec().sample()
        time_step = l0_env.step(actions.cpu().numpy())
        # ['collision_step', 'done', 'episode_length', 'path_length', 'spl', 'success']
        self.assertEqual(len(time_step.env_info.keys()), 6)

        actions = l1_env.action_spec().sample()
        time_step = l1_env.step(actions.cpu().numpy())
        self.assertEqual(len(time_step.env_info.keys()), 6)
        l0_env.close()
        self._env = l1_env


if __name__ == '__main__':
    alf.test.main()
