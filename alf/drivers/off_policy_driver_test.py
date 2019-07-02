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

import unittest
from absl.testing import parameterized

from absl import logging
import tensorflow as tf
import gin.tf

from tf_agents.environments.tf_py_environment import TFPyEnvironment

from alf.algorithms.ddpg_algorithm import create_ddpg_algorithm
from alf.algorithms.sac_algorithm import create_sac_algorithm
from alf.drivers.off_policy_driver import OffPolicyDriver
from alf.environments.suite_unittest import PolicyUnittestEnv
from alf.environments.suite_unittest import ActionType


def _create_sac_algorithm(env):
    return create_sac_algorithm(
        env=env,
        actor_fc_layers=(16, 16),
        critic_fc_layers=(16, 16),
        alpha_learning_rate=5e-3,
        actor_learning_rate=5e-3,
        critic_learning_rate=5e-3)


def _create_ddpg_algorithm(env):
    return create_ddpg_algorithm(
        env=env,
        actor_fc_layers=(16, 16),
        critic_fc_layers=(16, 16),
        actor_learning_rate=1e-2,
        critic_learning_rate=1e-1)


class OffPolicyDriverTest(parameterized.TestCase, unittest.TestCase):
    @parameterized.parameters(
        _create_sac_algorithm,
        _create_ddpg_algorithm,
    )
    def test_off_policy_algorithm(self, algorithm_ctor):
        batch_size = 100
        steps_per_episode = 13
        env = PolicyUnittestEnv(
            batch_size, steps_per_episode, action_type=ActionType.Continuous)
        env = TFPyEnvironment(env)

        eval_env = PolicyUnittestEnv(
            batch_size, steps_per_episode, action_type=ActionType.Continuous)
        eval_env = TFPyEnvironment(eval_env)

        algorithm = algorithm_ctor(env)
        driver = OffPolicyDriver(
            env,
            algorithm,
            debug_summaries=True,
            summarize_grads_and_vars=True)
        replay_buffer = driver.add_replay_buffer()
        eval_driver = OffPolicyDriver(eval_env, algorithm, greedy_predict=True)
        driver.run = tf.function(driver.run)
        eval_driver.run = tf.function(eval_driver.run)

        env.reset()
        eval_env.reset()
        time_step = driver.get_initial_time_step()
        policy_state = driver.get_initial_state()
        for i in range(5):
            time_step, policy_state = driver.run(
                max_num_steps=batch_size * steps_per_episode,
                time_step=time_step,
                policy_state=policy_state)

        for i in range(300):
            time_step, policy_state = driver.run(
                max_num_steps=batch_size * 4,
                time_step=time_step,
                policy_state=policy_state)
            experience, _ = replay_buffer.get_next(
                sample_batch_size=128, num_steps=2)
            driver.train(experience)
            eval_env.reset()
            eval_time_step, _ = eval_driver.run(
                max_num_steps=(steps_per_episode - 1) * batch_size)
            logging.info("%d reward=%f", i,
                         float(tf.reduce_mean(eval_time_step.reward)))

        eval_env.reset()
        eval_time_step, _ = eval_driver.run(
            max_num_steps=(steps_per_episode - 1) * batch_size)
        logging.info("reward=%f", float(tf.reduce_mean(eval_time_step.reward)))
        self.assertAlmostEqual(
            1.0, float(tf.reduce_mean(eval_time_step.reward)), delta=2e-1)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    unittest.main()
