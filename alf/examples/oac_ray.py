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

from functools import partial
import ray
import torch

import alf
from alf.algorithms.oac_algorithm import OacAlgorithm
from alf.algorithms.data_transformer import create_data_transformer
from alf.environments.utils import create_environment
from alf.networks import NormalProjectionNetwork, ActorDistributionNetwork, CriticNetwork
from alf.optimizers import Adam, AdamTF
from alf.utils import common
from alf.utils.math_ops import clipped_exp

fc_layer_params = (256, 256)

actor_network_cls = partial(
    ActorDistributionNetwork,
    fc_layer_params=fc_layer_params,
    continuous_projection_net_ctor=partial(
        NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=clipped_exp))

# actor_network_cls = partial(
#     ActorDistributionNetwork,
#     fc_layer_params=fc_layer_params,
#     continuous_projection_net_ctor=partial(
#         NormalProjectionNetwork,
#         state_dependent_std=True,
#         scale_distribution=True,
#         std_transform=partial(
#             clipped_exp, clip_value_min=-10, clip_value_max=2)))

critic_network_cls = partial(
    CriticNetwork, joint_fc_layer_params=fc_layer_params)


def _environment_creator(env_name, random_seed=None):
    env = create_environment(
        env_name=env_name, nonparallel=True, seed=random_seed)
    return env


def _algorithm_creator(env, observation_spec, config):
    algorithm = OacAlgorithm(
        observation_spec,
        action_spec=env.action_spec(),
        reward_spec=env.reward_spec(),
        env=env,
        actor_network_cls=actor_network_cls,
        critic_network_cls=critic_network_cls,
        explore=True,
        explore_delta=6.,
        target_update_tau=0.005,
        actor_optimizer=AdamTF(lr=3e-4),
        critic_optimizer=AdamTF(lr=3e-4),
        alpha_optimizer=AdamTF(lr=3e-4),
        config=config)
    return algorithm


@ray.remote(num_cpus=1)
class RemoteAlgorithmEvaluator(object):
    def __init__(self, env_seed, config):

        torch.set_num_threads(1)
        self._env = _environment_creator(config.env_name, random_seed=env_seed)

        data_transformer = create_data_transformer(
            config.data_transformer_ctor, self._env.observation_spec())
        observation_spec = data_transformer.transformed_observation_spec

        self._algorithm = _algorithm_creator(self._env, observation_spec,
                                             config)
        self._algorithm.eval()

        self._num_eval_episodes = config.num_eval_episodes
        self._epsilon_greedy = config.epsilon_greedy

        self._eval_metrics = [
            alf.metrics.AverageReturnMetric(
                buffer_size=self._num_eval_episodes,
                reward_shape=self._env.reward_spec().shape),
            alf.metrics.AverageEpisodeLengthMetric(
                buffer_size=self._num_eval_episodes),
            alf.metrics.AverageEnvInfoMetric(
                example_env_info=self._env.reset().env_info,
                batch_size=self._env.batch_size,
                buffer_size=self._num_eval_episodes),
            alf.metrics.AverageDiscountedReturnMetric(
                buffer_size=self._num_eval_episodes,
                reward_shape=self._env.reward_spec().shape),
        ]

        self._env.reset()

    @torch.no_grad()
    def eval(self, state_dict, train_step=None, step_metrics=()):
        self._train_step = train_step
        self._step_metrics = step_metrics
        actor_state_dict = state_dict
        self._algorithm._actor_network.load_state_dict(actor_state_dict)

        time_step = common.get_initial_time_step(self._env)
        policy_state = self._algorithm.get_initial_predict_state(
            self._env.batch_size)
        trans_state = self._algorithm.get_initial_transform_state(
            self._env.batch_size)
        episodes = 0
        steps = 0
        while episodes < self._num_eval_episodes:
            policy_state = common.reset_state_if_necessary(
                policy_state,
                self._algorithm.get_initial_predict_state(
                    self._env.batch_size), time_step.is_first())
            transformed_time_step, trans_state = self._algorithm.transform_timestep(
                time_step, trans_state)
            policy_step = self._algorithm.predict_step(
                transformed_time_step, policy_state, self._epsilon_greedy)
            steps += 1
            next_time_step = self._env.step(policy_step.output)
            for metric in self._eval_metrics:
                metric(time_step.cpu())
            time_step = next_time_step
            policy_state = policy_step.state

            if time_step.is_last():
                episodes += 1

    def get_eval_results(self):
        results = [metric.result() for metric in self._eval_metrics]
        return results, self._train_step, self._step_metrics
