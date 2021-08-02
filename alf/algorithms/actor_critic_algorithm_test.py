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
from functools import partial
import torch
import torch.distributions as td
import unittest

import alf
from alf.utils import common, dist_utils, tensor_utils
from alf.data_structures import StepType, TimeStep
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.rl_algorithm_test import MyEnv


def create_algorithm(env):
    config = TrainerConfig(root_dir="dummy", unroll_length=5)
    obs_spec = alf.TensorSpec((2, ), dtype='float32')
    action_spec = alf.BoundedTensorSpec(
        shape=(), dtype='int32', minimum=0, maximum=2)

    fc_layer_params = (10, 8, 6)

    actor_network = partial(
        ActorDistributionNetwork,
        fc_layer_params=fc_layer_params,
        discrete_projection_net_ctor=alf.networks.CategoricalProjectionNetwork)

    value_network = partial(ValueNetwork, fc_layer_params=(10, 8, 1))

    alg = ActorCriticAlgorithm(
        observation_spec=obs_spec,
        action_spec=action_spec,
        reward_spec=env.reward_spec(),
        actor_network_ctor=actor_network,
        value_network_ctor=value_network,
        env=env,
        config=config,
        optimizer=alf.optimizers.Adam(lr=1e-2),
        debug_summaries=True,
        name="MyActorCritic")
    return alg


class ActorCriticAlgorithmTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((1, ), (3, ))
    def test_ac_algorithm(self, reward_dim):
        env = MyEnv(batch_size=3, reward_dim=reward_dim)
        alg1 = create_algorithm(env)

        iter_num = 50
        for _ in range(iter_num):
            alg1.train_iter()

        time_step = common.get_initial_time_step(env)
        state = alg1.get_initial_predict_state(env.batch_size)
        policy_step = alg1.rollout_step(time_step, state)
        logits = policy_step.info.action_distribution.log_prob(
            torch.arange(3).reshape(3, 1))
        print("logits: ", logits)
        self.assertTrue(torch.all(logits[1, :] > logits[0, :]))
        self.assertTrue(torch.all(logits[1, :] > logits[2, :]))

        # global counter is iter_num due to alg1
        self.assertTrue(alf.summary.get_global_counter() == iter_num)

    @parameterized.parameters((1, ), (3, ))
    def test_ac_algorithm_with_global_counter(self, reward_dim):
        env = MyEnv(batch_size=3, reward_dim=reward_dim)
        alg2 = create_algorithm(env)
        new_iter_num = 3
        for _ in range(new_iter_num):
            alg2.train_iter()
        # new_iter_num of iterations done in alg2
        self.assertTrue(alf.summary.get_global_counter() == new_iter_num)


if __name__ == '__main__':
    alf.test.main()
