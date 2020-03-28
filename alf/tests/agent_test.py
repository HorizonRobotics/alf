# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import functools

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.agent import Agent
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.data_structures import TimeStep, make_experience
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


class AgentTest(alf.test.TestCase):
    def test_agent_steps(self):
        batch_size = 1
        observation_spec = TensorSpec((10, ))
        action_spec = BoundedTensorSpec((), dtype='int64')
        time_step = TimeStep(
            observation=observation_spec.zeros(outer_dims=(batch_size, )),
            prev_action=action_spec.zeros(outer_dims=(batch_size, )))

        actor_net = ActorDistributionNetwork(
            input_tensor_spec=observation_spec,
            action_spec=action_spec,
            fc_layer_params=(100, ))
        value_net = ValueNetwork(
            input_tensor_spec=observation_spec, fc_layer_params=(100, ))

        # TODO: add a goal generator and an entropy target algorithm once they
        # are implemented.
        agent = Agent(
            observation_spec=observation_spec,
            action_spec=action_spec,
            rl_algorithm_cls=functools.partial(
                ActorCriticAlgorithm,
                actor_network=actor_net,
                value_network=value_net),
            intrinsic_reward_module=ICMAlgorithm(
                action_spec=action_spec, observation_spec=observation_spec))

        predict_state = agent.get_initial_predict_state(batch_size)
        rollout_state = agent.get_initial_rollout_state(batch_size)
        train_state = agent.get_initial_train_state(batch_size)

        pred_step = agent.predict_step(
            time_step, predict_state, epsilon_greedy=0.1)
        self.assertEqual(pred_step.state.irm, ())

        rollout_step = agent.rollout_step(time_step, rollout_state)
        self.assertNotEqual(rollout_step.state.irm, ())

        exp = make_experience(time_step, rollout_step, rollout_state)

        train_step = agent.train_step(exp, train_state)
        self.assertNotEqual(train_step.state.irm, ())

        self.assertTensorEqual(rollout_step.state.irm, train_step.state.irm)


if __name__ == "__main__":
    alf.test.main()
