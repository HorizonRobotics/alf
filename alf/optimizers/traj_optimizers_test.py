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

DataItem = namedtuple("DataItem",
                      ["observation", "achieved_goal", "desired_goal"])


class CEMOptimizerTest(alf.test.TestCase):
    def test_cem_optimizer(self):
        opt = CEMOptimizer(
            planning_horizon=10,
            action_dim=2,
            population_size=1000,
            top_percent=0.05,
            iterations=500,
            init_mean=0.,
            init_var=20.,
            bounds=(-10, 10))

        def _costs(time_step, state, samples):
            costs = torch.sum(samples, dim=1)  # sum of all action values
            return -costs

        def _costs_agg_dist(time_step, state, samples, squared=True):
            n_samples = samples.shape[0]
            start = time_step.achieved_goal.expand(n_samples, 1, 2)
            end = time_step.desired_goal.expand(n_samples, 1, 2)
            samples_e = torch.cat(
                [start, samples.reshape(n_samples, -1, 2), end], dim=1)
            # Minimizing sum of squared distance would result in evenly spaced midpoints
            norm = torch.norm(
                samples_e[:, 1:, :] - samples_e[:, :-1, :], dim=2)
            if squared:
                norm = norm**2
            agg_dist = torch.sum(norm, dim=1)
            return agg_dist

        fake_item = DataItem(
            observation=torch.zeros(2, ), achieved_goal=0, desired_goal=0)
        opt.set_cost(_costs)
        solution = opt.obtain_solution(fake_item, ())
        self.assertTensorClose(solution, torch.Tensor([10.0]), epsilon=1e-1)

        start = torch.Tensor([[-1, -1]])
        end = torch.Tensor([[10, 10]])
        item = DataItem(
            observation=torch.zeros(1, ),
            achieved_goal=start,
            desired_goal=end)
        opt.set_cost(_costs_agg_dist)
        solution = opt.obtain_solution(item, ())
        self.assertTensorClose(
            solution.reshape(-1, 2),
            start +
            (1. + torch.arange(10)).reshape(-1, 1) * (end - start) / 11,
            epsilon=1e-1)
