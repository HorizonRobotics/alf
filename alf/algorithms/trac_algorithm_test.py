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
from functools import partial
import torch

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.trac_algorithm import TracAlgorithm
from alf.algorithms.rl_algorithm_test import MyEnv
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.utils import common


def create_ac_algorithm(observation_spec, action_spec, debug_summaries):
    fc_layer_params = (10, 8, 6)

    actor_network = partial(
        ActorDistributionNetwork,
        fc_layer_params=fc_layer_params,
        discrete_projection_net_ctor=alf.networks.CategoricalProjectionNetwork)

    value_network = partial(ValueNetwork, fc_layer_params=(10, 8, 1))

    return ActorCriticAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        actor_network_ctor=actor_network,
        value_network_ctor=value_network,
        optimizer=alf.optimizers.Adam(lr=0.1),
        debug_summaries=debug_summaries,
        name="MyActorCritic")


class TracAlgorithmTest(alf.test.TestCase):
    def test_trac_algorithm(self):
        config = TrainerConfig(root_dir="dummy", unroll_length=5)
        env = MyEnv(batch_size=3)
        alg = TracAlgorithm(
            observation_spec=env.observation_spec(),
            action_spec=env.action_spec(),
            ac_algorithm_cls=create_ac_algorithm,
            env=env,
            config=config)

        for _ in range(50):
            alg.train_iter()

        time_step = common.get_initial_time_step(env)
        state = alg.get_initial_predict_state(env.batch_size)
        policy_step = alg.rollout_step(time_step, state)
        logits = policy_step.info.action_distribution.log_prob(
            torch.arange(3).reshape(3, 1))
        print("logits: ", logits)
        # action 1 gets the most reward. So its probability should be higher
        # than other actions after training.
        self.assertTrue(torch.all(logits[1, :] > logits[0, :]))
        self.assertTrue(torch.all(logits[1, :] > logits[2, :]))


if __name__ == '__main__':
    alf.test.main()
