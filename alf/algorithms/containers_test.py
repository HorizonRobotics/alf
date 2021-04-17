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

import torch
import alf

from alf.algorithms.algorithm import Algorithm
from alf.algorithms.actor_critic_algorithm import ActorCriticLoss, ActorCriticInfo
from alf.algorithms.config import TrainerConfig
from alf.algorithms.containers import _build_nested_fields, SequentialAlg, RLAlgWrapper
from alf.algorithms.rl_algorithm_test import MyEnv
from alf.data_structures import AlgStep, Experience
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.utils import common, dist_utils


class ActorCriticLossAlg(ActorCriticLoss, Algorithm):
    def rollout_step(self, inputs, state):
        return AlgStep()

    def calc_loss(self, inputs, train_info):
        return self(
            experience=Experience(
                reward=inputs['reward'],
                step_type=inputs['step_type'],
                discount=inputs['discount'],
                action=inputs['action']),
            train_info=ActorCriticInfo(
                value=inputs['value'],
                action_distribution=inputs['action_distribution']))


def create_algorithm(env):
    config = TrainerConfig(root_dir="dummy", unroll_length=5)

    value_net = ValueNetwork(
        input_tensor_spec=env.observation_spec(), fc_layer_params=(10, 8))

    actor_net = ActorDistributionNetwork(
        input_tensor_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        fc_layer_params=(10, 8),
        discrete_projection_net_ctor=alf.networks.CategoricalProjectionNetwork)

    alg = SequentialAlg(
        value=('input.observation', value_net),
        action_dist=('input.observation', actor_net),
        action=dist_utils.sample_action_distribution,
        loss=(dict(
            reward='input.reward',
            step_type='input.step_type',
            discount='input.discount',
            action_distribution='action_dist',
            action='action',
            value='value'), ActorCriticLossAlg()),
        output='action')

    return RLAlgWrapper(
        observation_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        algorithm=alg,
        env=env,
        config=config,
        optimizer=alf.optimizers.Adam(lr=1e-2),
        debug_summaries=True,
        name="MyActorCritic")


class ContainersTest(alf.test.TestCase):
    def test_build_nested_fields(self):
        nest = _build_nested_fields(['a.b', 'a', 'c.d'])
        self.assertEqual(nest, {'a': 'a', 'c': {'d': 'c.d'}})

        nest = _build_nested_fields(['a.b', 'a.c', 'a.b.c'])
        self.assertEqual(nest, {'a': {'b': 'a.b', 'c': 'a.c'}})

    def test_sequential_alg(self):
        env = MyEnv(batch_size=3)
        alg1 = create_algorithm(env)

        iter_num = 50
        for _ in range(iter_num):
            alg1.train_iter()

        time_step = common.get_initial_time_step(env)
        state = alg1.get_initial_predict_state(env.batch_size)
        policy_step = alg1.rollout_step(time_step, state)
        action_dist = alf.nest.find_field(policy_step.info, 'action_dist')[0]
        logits = action_dist.log_prob(torch.arange(3).reshape(3, 1))
        print("logits: ", logits)
        self.assertTrue(torch.all(logits[1, :] > logits[0, :]))
        self.assertTrue(torch.all(logits[1, :] > logits[2, :]))


if __name__ == '__main__':
    alf.test.main()
