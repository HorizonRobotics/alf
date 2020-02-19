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

import unittest
from absl.testing import parameterized
from alf.metrics.metrics import EnvironmentSteps, NumberOfEpisodes, AverageReturnMetric, AverageEpisodeLengthMetric


class TFMetricsTest(parameterized.TestCase, unittest.TestCase):
    def _create_trajectories(self):
        def _concat_nested_tensors(nest1, nest2):
            return tf.nest.map_structure(
                lambda t1, t2: tf.concat([t1, t2], axis=0), nest1, nest2)

        # Order of args for trajectory methods:
        # observation, action, policy_info, reward, discount
        ts0 = _concat_nested_tensors(
            trajectory.boundary((), torch.tensor([1]), (),
                                torch.tensor([0.], dtype=torch.float32), [1.]),
            trajectory.boundary((), tf.constant([2]), (),
                                tf.constant([0.], dtype=tf.float32), [1.]))
        ts1 = _concat_nested_tensors(
            trajectory.first((), tf.constant([2]), (),
                             tf.constant([1.], dtype=tf.float32), [1.]),
            trajectory.first((), tf.constant([1]), (),
                             tf.constant([2.], dtype=tf.float32), [1.]))
        ts2 = _concat_nested_tensors(
            trajectory.last((), tf.constant([1]), (),
                            tf.constant([3.], dtype=tf.float32), [1.]),
            trajectory.last((), tf.constant([1]), (),
                            tf.constant([4.], dtype=tf.float32), [1.]))
        ts3 = _concat_nested_tensors(
            trajectory.boundary((), tf.constant([2]), (),
                                tf.constant([0.], dtype=tf.float32), [1.]),
            trajectory.boundary((), tf.constant([0]), (),
                                tf.constant([0.], dtype=tf.float32), [1.]))
        ts4 = _concat_nested_tensors(
            trajectory.first((), tf.constant([1]), (),
                             tf.constant([5.], dtype=tf.float32), [1.]),
            trajectory.first((), tf.constant([1]), (),
                             tf.constant([6.], dtype=tf.float32), [1.]))
        ts5 = _concat_nested_tensors(
            trajectory.last((), tf.constant([1]), (),
                            tf.constant([7.], dtype=tf.float32), [1.]),
            trajectory.last((), tf.constant([1]), (),
                            tf.constant([8.], dtype=tf.float32), [1.]))

        return [ts0, ts1, ts2, ts3, ts4, ts5]

    @parameterized.named_parameters([
        ('testEnvironmentStepsGraph', EnvironmentSteps, 5, 6),
        ('testNumberOfEpisodesGraph', NumberOfEpisodes, 4, 2),
        ('testAverageReturnGraph', AverageReturnMetric, 6, 9.0),
        ('testAverageEpisodeLengthGraph', AverageEpisodeLengthMetric, 6, 2.0),
    ])
    def testMetric(self, metric_class, num_trajectories, expected_result):
        trajectories = self._create_trajectories()
        if metric_class in [AverageReturnMetric, AverageEpisodeLengthMetric]:
            metric = metric_class(batch_size=2)
        else:
            metric = metric_class()

        for i in range(num_trajectories):
            metric(trajectories[i])

        self.assertEqual(expected_result, metric.result())
        metric.reset()
        self.assertEqual(0.0, metric.result())


if __name__ == "__main__":
    unittest.main()
