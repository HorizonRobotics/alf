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

    @parameterized.parameters(
        ("iterations", 1, "step"),
        ("iterations", 2, "step"),
        ("iterations", 3, "step"),
        ("iterations", 1, "linear"),
        ("iterations", 2, "linear"),
        ("iterations", 3, "linear"),
        ("env_steps", 1, "step"),
        ("env_steps", 2, "step"),
        ("env_steps", 3, "step"),
        ("env_steps", 1, "linear"),
        ("env_steps", 2, "linear"),
        ("env_steps", 3, "linear"),
        ("percent", 0.1, "step"),
        ("percent", 0.2, "step"),
        ("percent", 0.3, "step"),
        ("percent", 0.1, "linear"),
        ("percent", 0.2, "linear"),
        ("percent", 0.3, "linear"),
    )
    def test_step_switch_iterations(self, progress_type, half_cycle_size,
                                    switch_mode):

        trainer_progress = Trainer.get_trainer_progress()
        # set the trainer_progress mode
        if progress_type in {"iterations", "percent"}:
            num_iterations = 10
            num_env_steps = 0
        elif progress_type == "env_steps":
            num_iterations = 0
            num_env_steps = 10

        trainer_progress.set_termination_criterion(
            num_iterations=num_iterations, num_env_steps=num_env_steps)

        scheduler = CyclicalScheduler(
            progress_type=progress_type,
            base_lr=0,
            bound_lr=1,
            half_cycle_size=half_cycle_size,
            switch_mode=switch_mode)

        effective_half_cycle_size = (
            half_cycle_size if half_cycle_size >= 1 else
            half_cycle_size * max(num_iterations, num_env_steps))

        if switch_mode == "step":
            if effective_half_cycle_size == 1:
                expected_values = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            elif effective_half_cycle_size == 2:
                expected_values = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
            elif effective_half_cycle_size == 3:
                expected_values = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
        elif switch_mode == "linear":
            if effective_half_cycle_size == 1:
                expected_values = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            elif effective_half_cycle_size == 2:
                expected_values = [
                    0.0, 0.5, 1, 0.5, 0.0, 0.5, 1, 0.5, 0.0, 0.5
                ]
            elif effective_half_cycle_size == 3:
                expected_values = [
                    0.0, 0.3333333333333333, 0.6666666666666666, 1,
                    0.6666666666666667, 0.33333333333333337, 0.0,
                    0.3333333333333333, 0.6666666666666666, 1
                ]

        values = []
        for i in range(max(num_iterations, num_env_steps)):
            if progress_type in {"iterations", "percent"}:
                trainer_progress.update(iter_num=i)
            elif progress_type == "env_steps":
                trainer_progress.update(env_steps=i)
            value = scheduler()
            values.append(value)

        self.assertTrue(np.allclose(expected_values, values, atol=5e-2))

    @parameterized.parameters(
        ("iterations", 1, "step"),
        ("iterations", 2, "step"),
        ("iterations", 3, "step"),
        ("iterations", 1, "linear"),
        ("iterations", 2, "linear"),
        ("iterations", 3, "linear"),
        ("env_steps", 1, "step"),
        ("env_steps", 2, "step"),
        ("env_steps", 3, "step"),
        ("env_steps", 1, "linear"),
        ("env_steps", 2, "linear"),
        ("env_steps", 3, "linear"),
        ("percent", 0.1, "step"),
        ("percent", 0.2, "step"),
        ("percent", 0.3, "step"),
        ("percent", 0.1, "linear"),
        ("percent", 0.2, "linear"),
        ("percent", 0.3, "linear"),
    )
    def test_step_switch_iterations_upper_bound_as_base(
            self, progress_type, half_cycle_size, switch_mode):

        trainer_progress = Trainer.get_trainer_progress()
        # set the trainer_progress mode
        if progress_type in {"iterations", "percent"}:
            num_iterations = 10
            num_env_steps = 0
        elif progress_type == "env_steps":
            num_iterations = 0
            num_env_steps = 10

        trainer_progress.set_termination_criterion(
            num_iterations=num_iterations, num_env_steps=num_env_steps)

        scheduler = CyclicalScheduler(
            progress_type=progress_type,
            base_lr=1,
            bound_lr=0,
            half_cycle_size=half_cycle_size,
            switch_mode=switch_mode)

        effective_half_cycle_size = (
            half_cycle_size if half_cycle_size >= 1 else
            half_cycle_size * max(num_iterations, num_env_steps))

        if switch_mode == "step":
            if effective_half_cycle_size == 1:
                expected_values = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
            elif effective_half_cycle_size == 2:
                expected_values = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            elif effective_half_cycle_size == 3:
                expected_values = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0]
        elif switch_mode == "linear":
            if effective_half_cycle_size == 1:
                expected_values = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
            elif effective_half_cycle_size == 2:
                expected_values = [
                    1.0, 0.5, 0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5
                ]
            elif effective_half_cycle_size == 3:
                expected_values = [
                    1.0, 0.6666666666666667, 0.33333333333333337, 0,
                    0.3333333333333333, 0.6666666666666666, 1.0,
                    0.6666666666666667, 0.33333333333333337, 0
                ]

        values = []
        for i in range(max(num_iterations, num_env_steps)):
            if progress_type in {"iterations", "percent"}:
                trainer_progress.update(iter_num=i)
            elif progress_type == "env_steps":
                trainer_progress.update(env_steps=i)
            value = scheduler()
            values.append(value)

        self.assertTrue(np.allclose(expected_values, values, atol=5e-2))


if __name__ == '__main__':
    unittest.main()
