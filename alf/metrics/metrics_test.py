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

import torch

import alf
from alf.metrics import (EnvironmentSteps, NumberOfEpisodes,
                         AverageReturnMetric, AverageDiscountedReturnMetric,
                         AverageEpisodeLengthMetric, AverageEnvInfoMetric)
from alf.utils.tensor_utils import to_tensor
from alf.data_structures import TimeStep, StepType

import unittest
from absl.testing import parameterized


def _create_timestep(reward, env_id, step_type, env_info):
    return TimeStep(
        step_type=to_tensor(step_type),
        reward=to_tensor(reward),
        env_info=env_info,
        env_id=to_tensor(env_id))


def timestep_first(reward, env_id, env_info):
    return _create_timestep(reward, env_id, [StepType.FIRST] * 2, env_info)


def timestep_mid(reward, env_id, env_info):
    return _create_timestep(reward, env_id, [StepType.MID] * 2, env_info)


def timestep_last(reward, env_id, env_info):
    return _create_timestep(reward, env_id, [StepType.LAST] * 2, env_info)


class THMetricsTest(parameterized.TestCase, unittest.TestCase):
    def _create_trajectories(self, vector_reward=False):
        # Order of args for timestep_* methods:
        # reward, env_id, env_info
        def _vectorize_reward(reward):
            # make vector reward by simply repeating
            if vector_reward:
                return [[ri] * 2 for ri in reward]
            else:
                return reward

        ts0 = timestep_first(
            _vectorize_reward([0, 0]), [1, 2],
            dict(x=to_tensor([1, 0]), y=to_tensor([1, 1])))
        ts1 = timestep_mid(
            _vectorize_reward([1, 2]), [1, 2],
            dict(x=to_tensor([1, 2]), y=to_tensor([0, 3])))
        ts2 = timestep_last(
            _vectorize_reward([3, 4]), [1, 2],
            dict(x=to_tensor([-1, -2]), y=to_tensor([1, -1])))
        ts3 = timestep_first(
            _vectorize_reward([0, 0]), [1, 2],
            dict(x=to_tensor([1, 1]), y=to_tensor([1, 1])))
        ts4 = timestep_mid(
            _vectorize_reward([5, 6]), [1, 2],
            dict(x=to_tensor([2, -2]), y=to_tensor([-1, -6])))
        ts5 = timestep_last(
            _vectorize_reward([7, 8]), [1, 2],
            dict(x=to_tensor([10, 10]), y=to_tensor([5, 5])))

        return [ts0, ts1, ts2, ts3, ts4, ts5]

    @parameterized.named_parameters(
        [('testEnvironmentSteps', EnvironmentSteps, 5, 6, False),
         ('testNumberOfEpisodes', NumberOfEpisodes, 4, 2, False),
         ('testAverageReturn', AverageReturnMetric, 6, 9.0, False),
         ('testAverageEpisodeLength', AverageEpisodeLengthMetric, 6, 2.0,
          False),
         ('testAverageEnvInfoMetric', AverageEnvInfoMetric, 6,
          dict(x=torch.as_tensor(5.), y=torch.as_tensor(1.5)), False),
         ('testAverageDiscountedReturnMetric', AverageDiscountedReturnMetric,
          6, 7.2225, False),
         ('testAverageReturnVectorReward', AverageReturnMetric, 6, [9.0, 9.0],
          True),
         ('testAverageDiscountedReturnMetricVectorReward',
          AverageDiscountedReturnMetric, 6, [7.2225, 7.2225], True)])
    def testMetric(self, metric_class, num_trajectories, expected_result,
                   vector_reward):
        trajectories = self._create_trajectories(vector_reward)
        if metric_class in [AverageEpisodeLengthMetric]:
            metric = metric_class(batch_size=2)
        elif metric_class in [
                AverageReturnMetric, AverageDiscountedReturnMetric
        ]:

            metric = metric_class(
                batch_size=2, reward_shape=(2, ) if vector_reward else ())
        elif metric_class == AverageEnvInfoMetric:
            metric = metric_class(
                batch_size=2, example_env_info=dict(x=0, y=0))
        else:
            metric = metric_class()

        for i in range(num_trajectories):
            metric(trajectories[i])

        self.assertEqual(expected_result, metric.result())

        metric.reset()

        if metric_class == AverageEnvInfoMetric:
            self.assertEqual(
                dict(x=torch.as_tensor(0.), y=torch.as_tensor(0.)),
                metric.result())
        else:
            self.assertEqual([0.0] * 2 if vector_reward else 0.0,
                             metric.result())


if __name__ == "__main__":
    unittest.main()
