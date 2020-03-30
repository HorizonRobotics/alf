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
import gin
import torch

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.environments.suite_unittest import PolicyUnittestEnv
from alf.environments.suite_unittest import ActionType
from alf.networks import (
    ActorDistributionNetwork, ActorDistributionRNNNetwork,
    StableNormalProjectionNetwork, ValueNetwork, ValueRNNNetwork)
from alf.utils import common

DEBUGGING = True


def create_algorithm(env, use_rnn=False, learning_rate=1e-1):
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    if use_rnn:
        actor_net = ActorDistributionRNNNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=(),
            lstm_hidden_size=(4, ),
            actor_fc_layer_params=())
        value_net = ValueRNNNetwork(
            observation_spec,
            fc_layer_params=(),
            lstm_hidden_size=(4, ),
            value_fc_layer_params=())
    else:
        actor_net = ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=(),
            continuous_projection_net_ctor=StableNormalProjectionNetwork)
        value_net = ValueNetwork(observation_spec, fc_layer_params=())

    optimizer = torch.optim.Adam(lr=learning_rate)

    config = TrainerConfig(
        root_dir="dummy",
        unroll_length=13,
        num_updates_per_train_step=4,
        mini_batch_size=25,
        summarize_grads_and_vars=DEBUGGING)

    return PPOAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        env=env,
        config=config,
        actor_network=actor_net,
        value_network=value_net,
        loss=PPOLoss(gamma=1.0, debug_summaries=DEBUGGING),
        optimizer=optimizer,
        debug_summaries=DEBUGGING)


class PpoTest(alf.test.TestCase):
    def test_ppo(self):
        env_class = PolicyUnittestEnv
        learning_rate = 1e-1
        iterations = 20
        batch_size = 100
        steps_per_episode = 13
        env = env_class(batch_size, steps_per_episode)

        eval_env = env_class(batch_size, steps_per_episode)

        algorithm = create_algorithm(env, learning_rate=learning_rate)

        env.reset()
        eval_env.reset()
        for i in range(iterations):
            algorithm.train_iter()

            eval_env.reset()
            eval_time_step = unroll(eval_env, algorithm, steps_per_episode - 1)
            logging.info("%d reward=%f", i,
                         float(eval_time_step.reward.mean()))

        self.assertAlmostEqual(
            1.0, float(eval_time_step.reward.mean()), delta=1e-1)


def unroll(env, algorithm, steps, epsilon_greedy=0.1):
    """Run `steps` environment steps using algoirthm.predict_step()."""
    time_step = common.get_initial_time_step(env)
    policy_state = algorithm.get_initial_predict_state(env.batch_size)
    for _ in range(steps):
        policy_state = common.reset_state_if_necessary(
            policy_state, algorithm.get_initial_predict_state(env.batch_size),
            time_step.is_first())
        transformed_time_step = algorithm.transform_timestep(time_step)
        policy_step = algorithm.predict_step(
            transformed_time_step, policy_state, epsilon_greedy=epsilon_greedy)
        time_step = env.step(policy_step.output)
        policy_state = policy_step.state
    return time_step


if __name__ == '__main__':
    alf.test.main()
