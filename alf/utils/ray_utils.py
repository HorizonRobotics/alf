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

import ray
import torch

import alf
from alf.algorithms.data_transformer import create_data_transformer
from alf.environments.utils import create_environment
from alf.utils import common


@ray.remote(num_cpus=1)
class RemoteAlgorithmEvaluator(object):
    """Asynchronized algorithm evaluator managed by ray. """

    def __init__(self, config, conf_file):
        """
        Args:
            config (TrainerConfig): configuration used to construct the main trainer
            conf_file (str): full path to the configuration file
        """

        torch.set_num_threads(1)
        common.parse_conf_file(conf_file)
        # current parse_conf_file automatically creates a global env
        # if the conf_file is .gin file
        # for eval purpose, we need another nonparallel env
        alf.close_env()
        self._env = create_environment(
            nonparallel=True, seed=config.random_seed)

        data_transformer = create_data_transformer(
            config.data_transformer_ctor, self._env.observation_spec())
        observation_spec = data_transformer.transformed_observation_spec
        self._algorithm = config.algorithm_ctor(
            observation_spec=observation_spec,
            action_spec=self._env.action_spec(),
            env=self._env,
            config=config)

        self._algorithm.eval()

        self._num_eval_episodes = config.num_eval_episodes

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
        self._algorithm.load_predict_module_state(state_dict)

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
            policy_step = self._algorithm.predict_step(transformed_time_step,
                                                       policy_state)
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

    def close_env(self):
        if self._env is not None:
            self._env.close()
