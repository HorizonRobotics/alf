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

from absl import logging
import os
import psutil
import time

import unittest
import tensorflow as tf

from tf_agents.environments.tf_py_environment import TFPyEnvironment

from alf.algorithms.merlin_algorithm import create_merlin_algorithm
from alf.drivers.on_policy_driver import OnPolicyDriver
from alf.environments.suite_unittest import RNNPolicyUnittestEnv
from alf.utils.common import run_under_record_context


class MerlinAlgorithmTest(unittest.TestCase):
    def test_merlin_algorithm(self):
        batch_size = 100
        steps_per_episode = 15
        gap = 10
        env = RNNPolicyUnittestEnv(
            batch_size, steps_per_episode, gap, obs_dim=3)
        env = TFPyEnvironment(env)

        algorithm = create_merlin_algorithm(env, learning_rate=1e-3)
        driver = OnPolicyDriver(
            env,
            algorithm,
            train_interval=6,
            debug_summaries=True,
            summarize_grads_and_vars=False)

        eval_driver = OnPolicyDriver(env, algorithm, training=False)

        proc = psutil.Process(os.getpid())

        policy_state = driver.get_initial_policy_state()
        time_step = driver.get_initial_time_step()
        for i in range(100):
            t0 = time.time()
            time_step, policy_state = driver.run(
                max_num_steps=150 * batch_size,
                time_step=time_step,
                policy_state=policy_state)
            mem = proc.memory_info().rss // 1e6
            logging.info('%s time=%.3f mem=%s' % (i, time.time() - t0, mem))

        env.reset()
        time_step, _ = eval_driver.run(max_num_steps=14 * batch_size)
        logging.info("eval reward=%.3f" % tf.reduce_mean(time_step.reward))
        self.assertAlmostEqual(
            1.0, float(tf.reduce_mean(time_step.reward)), delta=1e-2)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()

    run_under_record_context(
        unittest.main,
        summary_dir="~/tmp/debug",
        summary_interval=1,
        flush_millis=1000)
