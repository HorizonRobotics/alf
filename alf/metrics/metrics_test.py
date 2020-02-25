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

import alf
from alf.data_structures import timestep_first, timestep_mid, timestep_last
from alf.metrics import EnvironmentSteps, NumberOfEpisodes, AverageReturnMetric, AverageEpisodeLengthMetric
import torch as tc

import unittest
from absl.testing import parameterized


class THMetricsTest(parameterized.TestCase, unittest.TestCase):
    def _create_trajectories(self):
        def _concat_nested_tensors(nest1, nest2):
            return alf.nest.map_structure(
                lambda t1, t2: tc.cat([t1, t2], dim=0), nest1, nest2)

        # Order of args for timestep_* methods:
        # observation, prev_action, reward, discount, env_id
        ts0 = _concat_nested_tensors(
            timestep_first((), tc.tensor([1]), tc.tensor([0.],
                                                         dtype=tc.float32),
                           [1.], [1]),
            timestep_first((), tc.tensor([2]), tc.tensor([0.],
                                                         dtype=tc.float32),
                           [1.], [2]))
        ts1 = _concat_nested_tensors(
            timestep_mid((), tc.tensor([2]), tc.tensor([1.], dtype=tc.float32),
                         [1.], [1]),
            timestep_mid((), tc.tensor([1]), tc.tensor([2.], dtype=tc.float32),
                         [1.], [2]))
        ts2 = _concat_nested_tensors(
            timestep_last((), tc.tensor([1]),
                          tc.tensor([3.], dtype=tc.float32), [1.], [1]),
            timestep_last((), tc.tensor([1]),
                          tc.tensor([4.], dtype=tc.float32), [1.], [2]))
        ts3 = _concat_nested_tensors(
            timestep_first((), tc.tensor([2]), tc.tensor([0.],
                                                         dtype=tc.float32),
                           [1.], [1]),
            timestep_first((), tc.tensor([0]), tc.tensor([0.],
                                                         dtype=tc.float32),
                           [1.], [2]))
        ts4 = _concat_nested_tensors(
            timestep_mid((), tc.tensor([1]), tc.tensor([5.], dtype=tc.float32),
                         [1.], [1]),
            timestep_mid((), tc.tensor([1]), tc.tensor([6.], dtype=tc.float32),
                         [1.], [2]))
        ts5 = _concat_nested_tensors(
            timestep_last((), tc.tensor([1]),
                          tc.tensor([7.], dtype=tc.float32), [1.], [1]),
            timestep_last((), tc.tensor([1]),
                          tc.tensor([8.], dtype=tc.float32), [1.], [2]))

        return [ts0, ts1, ts2, ts3, ts4, ts5]

    @parameterized.named_parameters([
        ('testEnvironmentStepsGraph', EnvironmentSteps, 5, 6, 1),
        ('testNumberOfEpisodesGraph', NumberOfEpisodes, 4, 2, 1),
        ('testAverageReturnGraph', AverageReturnMetric, 6, 9.0, 4),
        ('testAverageEpisodeLengthGraph', AverageEpisodeLengthMetric, 6, 2.0,
         4),
    ])
    def testMetric(self, metric_class, num_trajectories, expected_result,
                   state_dict_length):
        trajectories = self._create_trajectories()
        if metric_class in [AverageReturnMetric, AverageEpisodeLengthMetric]:
            metric = metric_class(batch_size=2)
        else:
            metric = metric_class()

        for i in range(num_trajectories):
            metric(trajectories[i])

        self.assertEqual(expected_result, metric.result())

        self.assertEqual(state_dict_length, len(metric.state_dict()))

        metric.reset()
        self.assertEqual(0.0, metric.result())


if __name__ == "__main__":
    unittest.main()
