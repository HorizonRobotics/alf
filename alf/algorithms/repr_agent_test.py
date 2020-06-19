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
from alf.algorithms.repr_agent import RepresentationAgent
from alf.algorithms.recurrent_state_space_model import RecurrentStateSpaceModel
from alf.data_structures import TimeStep, make_experience
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.networks import EncodingNetwork, LSTMEncodingNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.nest.utils import NestConcat


class RepresentationAgentTest(alf.test.TestCase):
    def test_agent_steps(self):
        batch_size = 1
        observation_dim = 10
        observation_spec = TensorSpec((observation_dim, ))
        action_spec = BoundedTensorSpec((), dtype='int64')
        reward_spec = TensorSpec(())
        time_step = TimeStep(
            observation=observation_spec.zeros(outer_dims=(batch_size, )),
            prev_action=action_spec.zeros(outer_dims=(batch_size, )),
            reward=reward_spec.zeros(outer_dims=(batch_size, )))

        # rl algorithm
        actor_net = functools.partial(
            ActorDistributionNetwork, fc_layer_params=(100, ))
        value_net = functools.partial(ValueNetwork, fc_layer_params=(100, ))

        # state representation model
        state_dim = 5
        recurrent_state_network_ctor = functools.partial(
            LSTMEncodingNetwork,
            input_preprocessors=(None,
                                 EmbeddingPreprocessor(
                                     input_tensor_spec=action_spec,
                                     embedding_dim=observation_dim)),
            preprocessing_combiner=NestConcat(),
            pre_fc_layer_params=(100, ),
            hidden_size=(state_dim, ))
        observation_network_ctor = functools.partial(
            EncodingNetwork,
            fc_layer_params=(100, ),
            last_layer_size=observation_dim,
            last_activation=lambda x: x)
        reward_network_ctor = functools.partial(
            EncodingNetwork, fc_layer_params=(100, ))
        state_posterior_network_ctor = functools.partial(
            EncodingNetwork, preprocessing_combiner=NestConcat())

        agent = RepresentationAgent(
            observation_spec=observation_spec,
            action_spec=action_spec,
            rl_algorithm_cls=functools.partial(
                ActorCriticAlgorithm,
                actor_network_ctor=actor_net,
                value_network_ctor=value_net),
            state_representation_model_cls=functools.partial(
                RecurrentStateSpaceModel,
                state_dim=state_dim,
                recurrent_state_network_ctor=recurrent_state_network_ctor,
                observation_network_ctor=observation_network_ctor,
                reward_network_ctor=reward_network_ctor,
                state_posterior_network_ctor=state_posterior_network_ctor))

        predict_state = agent.get_initial_predict_state(batch_size)
        rollout_state = agent.get_initial_rollout_state(batch_size)
        train_state = agent.get_initial_train_state(batch_size)

        pred_step = agent.predict_step(
            time_step, predict_state, epsilon_greedy=0.1)
        self.assertEqual(pred_step.info, ())

        rollout_step = agent.rollout_step(time_step, rollout_state)
        self.assertEqual(rollout_step.info.repr, ())
        self.assertNotEqual(rollout_step.state.repr, ())

        exp = make_experience(time_step, rollout_step, rollout_state)

        train_step = agent.train_step(exp, train_state)
        self.assertNotEqual(train_step.info.repr, ())
        self.assertNotEqual(train_step.state.repr, ())

        alf.nest.map_structure(self.assertTensorEqual,
                               rollout_step.state.repr.h,
                               train_step.state.repr.h)


if __name__ == "__main__":
    alf.test.main()
