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
from alf.algorithms.containers import SequentialAlg, RLAlgWrapper, EchoAlg
from alf.algorithms.rl_algorithm_test import MyEnv
from alf.data_structures import AlgStep, Experience
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.utils import common, dist_utils


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
        is_on_policy=True,
        value=('input.observation', value_net),
        action_dist=('input.observation', actor_net),
        action=dist_utils.sample_action_distribution,
        loss=(ActorCriticInfo(
            reward='input.reward',
            step_type='input.step_type',
            discount='input.discount',
            action_distribution='action_dist',
            action='action',
            value='value'), ActorCriticLoss()),
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
    def test_sequential_alg(self):
        env = MyEnv(batch_size=3)
        alg1 = create_algorithm(env)

        iter_num = 50
        for _ in range(iter_num):
            alg1.train_iter()

        time_step = common.get_initial_time_step(env)
        state = alg1.get_initial_predict_state(env.batch_size)
        policy_step = alg1.rollout_step(time_step, state)
        action_dist = alf.nest.find_field(policy_step.info,
                                          'action_distribution')[0]
        logits = action_dist.log_prob(torch.arange(3).reshape(3, 1))
        print("logits: ", logits)
        self.assertTrue(torch.all(logits[1, :] > logits[0, :]))
        self.assertTrue(torch.all(logits[1, :] > logits[2, :]))

    def test_echo_alg(self):
        echo_spec = alf.TensorSpec(())
        alg = SequentialAlg(
            a=lambda x: x['input'] + x['echo'],
            b=('input.echo', lambda x: 2 * x),
            output=dict(output='b', echo='a'))
        echo_alg = EchoAlg(alg, echo_spec=echo_spec)

        self.assertEqual(echo_alg.train_state_spec,
                         (alg.train_state_spec, echo_spec))
        self.assertEqual(echo_alg.predict_state_spec,
                         (alg.predict_state_spec, echo_spec))
        self.assertEqual(echo_alg.rollout_state_spec,
                         (alg.rollout_state_spec, echo_spec))

        state = echo_alg.get_initial_rollout_state(2)
        alg_step = echo_alg.rollout_step(torch.tensor([1.0, 1.0]), state)
        self.assertEqual(alg_step.output, torch.tensor([0., 0.]))
        self.assertEqual(alg_step.state[1], torch.tensor([1., 1.]))
        alg_step = echo_alg.rollout_step(
            torch.tensor([3.0, 2.0]), alg_step.state)
        self.assertEqual(alg_step.output, torch.tensor([2., 2.]))
        self.assertEqual(alg_step.state[1], torch.tensor([4., 3.]))
        alg_step = echo_alg.rollout_step(
            torch.tensor([2.0, 1.0]), alg_step.state)
        self.assertEqual(alg_step.output, torch.tensor([8., 6.]))
        self.assertEqual(alg_step.state[1], torch.tensor([6., 4.]))

        state = echo_alg.get_initial_predict_state(2)
        alg_step = echo_alg.predict_step(torch.tensor([1.0, 1.0]), state)
        self.assertEqual(alg_step.output, torch.tensor([0., 0.]))
        self.assertEqual(alg_step.state[1], torch.tensor([1., 1.]))
        alg_step = echo_alg.predict_step(
            torch.tensor([3.0, 2.0]), alg_step.state)
        self.assertEqual(alg_step.output, torch.tensor([2., 2.]))
        self.assertEqual(alg_step.state[1], torch.tensor([4., 3.]))
        alg_step = echo_alg.predict_step(
            torch.tensor([2.0, 1.0]), alg_step.state)
        self.assertEqual(alg_step.output, torch.tensor([8., 6.]))
        self.assertEqual(alg_step.state[1], torch.tensor([6., 4.]))

        state = echo_alg.get_initial_train_state(2)
        alg_step = echo_alg.train_step(torch.tensor([1.0, 1.0]), state, ())
        self.assertEqual(alg_step.output, torch.tensor([0., 0.]))
        self.assertEqual(alg_step.state[1], torch.tensor([1., 1.]))
        alg_step = echo_alg.train_step(
            torch.tensor([3.0, 2.0]), alg_step.state, ())
        self.assertEqual(alg_step.output, torch.tensor([2., 2.]))
        self.assertEqual(alg_step.state[1], torch.tensor([4., 3.]))
        alg_step = echo_alg.train_step(
            torch.tensor([2.0, 1.0]), alg_step.state, ())
        self.assertEqual(alg_step.output, torch.tensor([8., 6.]))
        self.assertEqual(alg_step.state[1], torch.tensor([6., 4.]))


if __name__ == '__main__':
    alf.test.main()
