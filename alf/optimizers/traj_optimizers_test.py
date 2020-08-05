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

from absl import logging
from collections import namedtuple
import torch
import alf
from alf.optimizers.traj_optimizers import CEMOptimizer

DataItem = namedtuple("DataItem", ["observation"])


class CEMOptimizerTest(alf.test.TestCase):
    def test_cem_optimizer(self):
        opt = CEMOptimizer(
            planning_horizon=10,
            action_dim=2,
            population_size=1000,
            top_percent=0.05,
            iterations=100,
            init_mean=0.,
            init_var=20.,
            bounds=(-10, 10))

        def _costs(time_step, state, samples):
            costs = torch.sum(samples, dim=1)  # sum of all action values
            return -costs

        opt.set_cost(_costs)
        solution = opt.obtain_solution(
            DataItem(observation=torch.zeros(2, )), ())
        self.assertTensorClose(solution, torch.Tensor([10.0]), epsilon=1e-5)
