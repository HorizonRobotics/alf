# Copyright (c) 2022 Horizon Robotics. All Rights Reserved.
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

from absl.testing import parameterized
import unittest
import torch
import alf
from alf.trainers.policy_trainer import Trainer
from alf.utils.schedulers import CyclicalScheduler
import numpy as np


class CyclicalSchedulerTest(parameterized.TestCase, unittest.TestCase):
    """Tests for CyclicalScheduler.
    """

    @parameterized.parameters((1, "step"), (2, "step"), (3, "step"),
                              (1, "linear"), (2, "linear"), (3, "linear"))
    def test_step_switch_iterations(self, half_cycle_size, switch_mode):

        scheduler = CyclicalScheduler(
            progress_type="iterations",
            base_lr=0,
            bound_lr=1,
            half_cycle_size=half_cycle_size,
            switch_mode=switch_mode)

        trainer_progress = Trainer._trainer_progress
        # set the trainer_progress mode
        iter_num = 10
        trainer_progress.set_termination_criterion(num_iterations=iter_num)

        if switch_mode == "step":
            if half_cycle_size == 1:
                expected_values = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            elif half_cycle_size == 2:
                expected_values = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
            elif half_cycle_size == 3:
                expected_values = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
        elif switch_mode == "linear":
            if half_cycle_size == 1:
                expected_values = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            elif half_cycle_size == 2:
                expected_values = [
                    0.0, 0.5, 1, 0.5, 0.0, 0.5, 1, 0.5, 0.0, 0.5
                ]
            elif half_cycle_size == 3:
                expected_values = [
                    0.0, 0.3333333333333333, 0.6666666666666666, 1,
                    0.6666666666666667, 0.33333333333333337, 0.0,
                    0.3333333333333333, 0.6666666666666666, 1
                ]

        values = []
        for i in range(iter_num):
            trainer_progress.update(iter_num=i)
            value = scheduler()
            values.append(value)

        self.assertTrue(np.allclose(expected_values, values))

    @parameterized.parameters((1, "step"), (2, "step"), (3, "step"),
                              (1, "linear"), (2, "linear"), (3, "linear"))
    def test_step_switch_iterations_upper_bound_as_base(
            self, half_cycle_size, switch_mode):

        scheduler = CyclicalScheduler(
            progress_type="iterations",
            base_lr=1,
            bound_lr=0,
            half_cycle_size=half_cycle_size,
            switch_mode=switch_mode)

        trainer_progress = Trainer._trainer_progress
        # set the trainer_progress mode
        iter_num = 10
        trainer_progress.set_termination_criterion(num_iterations=iter_num)

        if switch_mode == "step":
            if half_cycle_size == 1:
                expected_values = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
            elif half_cycle_size == 2:
                expected_values = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            elif half_cycle_size == 3:
                expected_values = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0]
        elif switch_mode == "linear":
            if half_cycle_size == 1:
                expected_values = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
            elif half_cycle_size == 2:
                expected_values = [
                    1.0, 0.5, 0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5
                ]
            elif half_cycle_size == 3:
                expected_values = [
                    1.0, 0.6666666666666667, 0.33333333333333337, 0,
                    0.3333333333333333, 0.6666666666666666, 1.0,
                    0.6666666666666667, 0.33333333333333337, 0
                ]

        values = []
        for i in range(iter_num):
            trainer_progress.update(iter_num=i)
            value = scheduler()
            values.append(value)
        self.assertTrue(np.allclose(expected_values, values))


if __name__ == '__main__':
    unittest.main()
