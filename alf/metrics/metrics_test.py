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
                         AverageEpisodeLengthMetric, AverageEnvInfoMetric,
                         AverageEpisodicAggregationMetric,
                         EpisodicStartAverageDiscountedReturnMetric)
from alf.utils.tensor_utils import to_tensor
from alf.data_structures import TimeStep, StepType

import unittest
from absl.testing import parameterized


def _create_timestep(reward, env_id, step_type, env_info):
    return TimeStep(
        step_type=to_tensor(step_type),
        discount=torch.where(
            to_tensor(step_type) == StepType.LAST, torch.tensor(0.0),
            torch.tensor(1.0)),
        reward=to_tensor(reward),
        env_info=env_info,
        env_id=to_tensor(env_id))


def timestep_first(reward, env_id, env_info):
    return _create_timestep(reward, env_id, [StepType.FIRST] * 2, env_info)


def timestep_mid(reward, env_id, env_info):
    return _create_timestep(reward, env_id, [StepType.MID] * 2, env_info)


def timestep_last(reward, env_id, env_info):
    return _create_timestep(reward, env_id, [StepType.LAST] * 2, env_info)


class AverageDrivingMetric(AverageEpisodicAggregationMetric):
    """Metrics for computing the average velocity and accelration.

    This is purely for the purpose of unit testing the "@step" feature. It
    assumes the time step has velocity, acceleration and "success or not"logged
    in its ``env_info`` field.

    """

    def __init__(self,
                 example_time_step: TimeStep,
                 name='AverageDrivingMetric',
                 prefix='Metrics',
                 dtype=torch.float32,
                 buffer_size=10):
        super().__init__(
            name=name,
            dtype=dtype,
            prefix=prefix,
            buffer_size=buffer_size,
            example_time_step=example_time_step)

    def _extract_metric_values(self, time_step):
        return {
            'velocity@step':
                time_step.env_info['kinetics']['velocity'],
            'acceleration@step':
                time_step.env_info['kinetics']['acceleration'],
            'success':
                time_step.env_info['success']
        }


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
         ('testEpisodicStartAverageDiscountedReturnMetric',
          EpisodicStartAverageDiscountedReturnMetric, 6, 8.945, False),
         ('testAverageReturnVectorReward', AverageReturnMetric, 6, [9.0, 9.0],
          True),
         ('testAverageDiscountedReturnMetricVectorReward',
          AverageDiscountedReturnMetric, 6, [7.2225, 7.2225], True)])
    def testMetric(self, metric_class, num_trajectories, expected_result,
                   vector_reward):
        trajectories = self._create_trajectories(vector_reward)
        if metric_class in [
                AverageEpisodeLengthMetric, AverageReturnMetric,
                AverageDiscountedReturnMetric, AverageEnvInfoMetric,
                EpisodicStartAverageDiscountedReturnMetric
        ]:
            metric = metric_class(example_time_step=trajectories[0])
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

    def test_average_per_step(self):
        trajectories = []
        trajectories.append(
            timestep_first(
                0.0,
                env_id=[1, 2],
                env_info={
                    'kinetics': {
                        'velocity': to_tensor([4.0, 0.0]),
                        'acceleration': to_tensor([1.0, 0.0]),
                    },
                    'success': to_tensor([0.0, 0.0])
                }))
        trajectories.append(
            timestep_mid(
                0.0,
                env_id=[1, 2],
                env_info={
                    'kinetics': {
                        'velocity': to_tensor([4.0, 0.0]),
                        'acceleration': to_tensor([1.0, 0.0]),
                    },
                    'success': to_tensor([0.0, 0.0])
                }))
        trajectories.append(
            timestep_mid(
                0.0,
                env_id=[1, 2],
                env_info={
                    'kinetics': {
                        'velocity': to_tensor([5.0, 0.0]),
                        'acceleration': to_tensor([1.0, 0.0]),
                    },
                    'success': to_tensor([0.0, 0.0])
                }))
        trajectories.append(
            timestep_last(
                0.0,
                env_id=[1, 2],
                env_info={
                    'kinetics': {
                        'velocity': to_tensor([6.0, 0.0]),
                        'acceleration': to_tensor([1.0, 0.0]),
                    },
                    'success': to_tensor([1.0, 0.0])
                }))

        metric = AverageDrivingMetric(example_time_step=trajectories[0])

        for traj in trajectories:
            metric(traj)

        self.assertEqual(
            {
                # Sum is 15.0, divided by 3 (episode length) and 2 (batch size)
                # = 2.5
                'velocity@step': torch.as_tensor(2.5000),
                # Sum is 3.0, divided by 3 (episode length) and 2
                # (batch size) = 0.5
                'acceleration@step': torch.as_tensor(0.5000),
                # Sum is 1.0, divided by 2 (batch size) = 0.5
                'success': torch.as_tensor(0.5000)
            },
            metric.result())

    def test_accumulator_mask(self):
        traj = []
        neg_inf = -float('inf')
        traj.append(
            # First step values will be ignored
            timestep_first(
                0.0,
                env_id=[1, 2],
                env_info={
                    'velocity@max': to_tensor([-10, neg_inf]),
                    'success': to_tensor([0.0, 0.0]),
                    'value@step': to_tensor([0.0, 0.0]),
                }))
        traj.append(
            timestep_mid(
                0.0,
                env_id=[1, 2],
                env_info={
                    'velocity@max': to_tensor([neg_inf, -1.]),
                    'success': to_tensor([1.0, -neg_inf]),
                    'value@step': to_tensor([1.0, 2.0]),
                }))
        traj.append(
            timestep_last(
                0.0,
                env_id=[1, 2],
                env_info={
                    'velocity@max': to_tensor([neg_inf, -2.]),
                    'success': to_tensor([0.0, 0.0]),
                    'value@step': to_tensor([3.0, neg_inf]),
                }))
        ####
        traj.append(
            # First step values will be ignored
            timestep_first(
                0.0,
                env_id=[1, 2],
                env_info={
                    'velocity@max': to_tensor([-1., neg_inf]),
                    'success': to_tensor([neg_inf, 1.0]),
                    'value@step': to_tensor([0.0, 0.0]),
                }))
        traj.append(
            timestep_last(
                0.0,
                env_id=[1, 2],
                env_info={
                    'velocity@max': to_tensor([0., neg_inf]),
                    'success': to_tensor([neg_inf, 1.0]),
                    'value@step': to_tensor([neg_inf, 5.0]),
                }))

        metric = AverageEnvInfoMetric(example_time_step=traj[0])

        for step in traj:
            metric(step)

        self.assertEqual(
            {  # Only two episodes are valid for velocity: -1 and 0
                'velocity@max': torch.as_tensor(-0.5),
                # One episode is 1, one is 0, and the third one is 1
                'success': torch.as_tensor(2. / 3),
                # Two episodes are 2 and the third one is 5
                'value@step': torch.as_tensor(3.)
            },
            metric.result())


if __name__ == "__main__":
    unittest.main()
