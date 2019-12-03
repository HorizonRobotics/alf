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

import tensorflow as tf

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.encoding_network import EncodingNetwork
from tf_agents.specs.tensor_spec import TensorSpec
from alf.algorithms.merlin_algorithm import MerlinAlgorithm
from alf.algorithms.decoding_algorithm import DecodingAlgorithm
from alf.drivers.on_policy_driver import OnPolicyDriver
from alf.environments.suite_unittest import RNNPolicyUnittestEnv
from alf.utils import common


def _create_merlin_algorithm(
        encoder_fc_layers=(3, ),
        latent_dim=3,
        lstm_size=(4, ),
        memory_size=20,
        learning_rate=1e-3,
        debug_summaries=True):
    observation_spec = common.get_observation_spec()
    action_spec = common.get_action_spec()
    algorithm = MerlinAlgorithm(
        action_spec=action_spec,
        encoders=EncodingNetwork(
            input_tensor_spec=observation_spec,
            fc_layer_params=encoder_fc_layers,
            activation_fn=None,
            name="ObsEncoder"),
        decoders=DecodingAlgorithm(
            decoder=EncodingNetwork(
                input_tensor_spec=TensorSpec((latent_dim, ), dtype=tf.float32),
                fc_layer_params=encoder_fc_layers,
                activation_fn=None,
                name="ObsDecoder"),
            loss_weight=100.),
        latent_dim=latent_dim,
        lstm_size=lstm_size,
        memory_size=memory_size,
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        debug_summaries=debug_summaries)

    return algorithm


class MerlinAlgorithmTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        if os.environ.get('SKIP_LONG_TIME_COST_TESTS', False):
            self.skipTest("It takes very long to run this test.")

    def test_merlin_algorithm(self):
        batch_size = 100
        steps_per_episode = 15
        gap = 10
        env = RNNPolicyUnittestEnv(
            batch_size, steps_per_episode, gap, obs_dim=3)
        env = TFPyEnvironment(env)

        common.set_global_env(env)

        algorithm = _create_merlin_algorithm(
            learning_rate=1e-3, debug_summaries=False)
        driver = OnPolicyDriver(env, algorithm, train_interval=6)

        eval_driver = OnPolicyDriver(env, algorithm, training=False)

        proc = psutil.Process(os.getpid())

        policy_state = driver.get_initial_policy_state()
        time_step = driver.get_initial_time_step()
        for i in range(100):
            t0 = time.time()
            time_step, policy_state, _ = driver.run(
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
    tf.test.main()
