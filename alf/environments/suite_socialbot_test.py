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
"""Test for alf.environments.suite_socialbot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import numpy as np

from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import parallel_py_environment, py_environment, tf_py_environment
from tf_agents.utils import common
from alf.environments import suite_socialbot

import gin.tf


class SuiteSocialbotTest(absltest.TestCase):
    def setUp(self):
        super(SuiteSocialbotTest, self).setUp()
        if not suite_socialbot.is_available():
            self.skipTest('suite_socialbot is not available.')
        else:
            gin.clear_config()

    def test_socialbot_env_registered(self):
        env = suite_socialbot.load('SocialBot-CartPole-v0')
        self.assertIsInstance(env, py_environment.PyEnvironment)

    def test_observation_spec(self):
        env = suite_socialbot.load('SocialBot-CartPole-v0')
        self.assertEqual(np.float32, env.observation_spec().dtype)
        self.assertEqual((4, ), env.observation_spec().shape)

    def test_action_spec(self):
        env = suite_socialbot.load('SocialBot-CartPole-v0')
        self.assertEqual(np.float32, env.action_spec().dtype)
        self.assertEqual((1, ), env.action_spec().shape)

    def test_parallel_envs(self):
        env_num = 5

        ctors = [lambda: suite_socialbot.load('SocialBot-CartPole-v0',
                                              wrap_with_process=False)] * env_num

        parallel_envs = parallel_py_environment.ParallelPyEnvironment(
            env_constructors=ctors, start_serially=False)
        tf_env = tf_py_environment.TFPyEnvironment(parallel_envs)

        self.assertTrue(tf_env.batched)
        self.assertEqual(tf_env.batch_size, env_num)

        random_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(), tf_env.action_spec())

        replay_buffer_capacity = 100
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            random_policy.trajectory_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity)

        steps = 100
        step_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            random_policy,
            observers=[replay_buffer.add_batch],
            num_steps=steps)
        step_driver.run = common.function(step_driver.run)
        step_driver.run()

        self.assertIsNotNone(replay_buffer.get_next())

        parallel_envs.close()


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    absltest.main()
