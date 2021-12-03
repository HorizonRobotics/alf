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
import torch

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.merlin_algorithm import MerlinAlgorithm
from alf.algorithms.decoding_algorithm import DecodingAlgorithm
from alf.algorithms.ppo_algorithm_test import unroll
from alf.environments.suite_unittest import RNNPolicyUnittestEnv
from alf.utils import common, math_ops


def _create_merlin_algorithm(env,
                             encoder_fc_layers=(3, ),
                             latent_dim=4,
                             lstm_size=(4, ),
                             memory_size=20,
                             learning_rate=1e-3,
                             debug_summaries=True):
    config = TrainerConfig(root_dir="dummy", unroll_length=6)
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    algorithm = MerlinAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        env=env,
        config=config,
        encoders=alf.networks.EncodingNetwork(
            input_tensor_spec=observation_spec,
            fc_layer_params=encoder_fc_layers,
            activation=math_ops.identity,
            name="ObsEncoder"),
        decoders=DecodingAlgorithm(
            decoder=alf.networks.EncodingNetwork(
                input_tensor_spec=alf.TensorSpec((latent_dim, )),
                fc_layer_params=encoder_fc_layers,
                activation=math_ops.identity,
                name="ObsDecoder"),
            loss_weight=100.),
        latent_dim=latent_dim,
        lstm_size=lstm_size,
        memory_size=memory_size,
        optimizer=alf.optimizers.AdamTF(lr=learning_rate),
        debug_summaries=debug_summaries)

    return algorithm


class MerlinAlgorithmTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if os.environ.get('SKIP_LONG_TIME_COST_TESTS', False):
            self.skipTest("It takes very long to run this test.")

    def test_merlin_algorithm(self):
        batch_size = 200
        steps_per_episode = 15
        gap = 10
        env = RNNPolicyUnittestEnv(
            batch_size, steps_per_episode, gap, obs_dim=3)
        eval_env = RNNPolicyUnittestEnv(100, steps_per_episode, gap, obs_dim=3)

        algorithm = _create_merlin_algorithm(
            env, learning_rate=3e-3, debug_summaries=False)

        for i in range(300):
            algorithm.train_iter()
            if (i + 1) % 100 == 0:
                eval_env.reset()
                eval_time_step = unroll(eval_env, algorithm,
                                        steps_per_episode - 1)
                logging.info(
                    "%d reward=%f" % (i, float(eval_time_step.reward.mean())))

        self.assertAlmostEqual(
            1.0, float(eval_time_step.reward.mean()), delta=1e-2)


if __name__ == '__main__':
    alf.test.main()
