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
        planning_horizon = 10
        act_dim = 2
        pop_size = 1000
        opt = CEMOptimizer(
            planning_horizon=planning_horizon,
            action_dim=act_dim,
            population_size=pop_size,
            top_percent=0.05,
            iterations=500,
            bounds=(-10, 10),
            init_mean=0.,
            init_var=20.)

        def _costs(time_step, state, samples):
            batch_size = time_step.observation.shape[0]
            costs = torch.sum(
                samples.reshape(batch_size, pop_size, -1),
                dim=2)  # sum of all action values
            return -costs

        def _costs_agg_dist(time_step, state, samples, squared=True):
            batch_size = time_step.observation.shape[0]
            samples = samples.reshape(batch_size, pop_size, -1, act_dim)
            start = time_step.achieved_goal.reshape(
                batch_size, 1, 1, act_dim).expand(batch_size, pop_size, 1,
                                                  act_dim)
            end = time_step.desired_goal.reshape(batch_size, 1, 1,
                                                 act_dim).expand(
                                                     batch_size, pop_size, 1,
                                                     act_dim)
            samples_e = torch.cat([start, samples, end], dim=2)
            # Minimizing sum of squared distance would result in evenly spaced midpoints
            norm = torch.norm(
                samples_e[:, :, 1:, :] - samples_e[:, :, :-1, :], dim=3)
            if squared:
                norm = norm**2
            agg_dist = torch.sum(norm.reshape(batch_size, pop_size, -1), dim=2)
            return agg_dist

        batch_size = 2
        fake_item = DataItem(
            observation=torch.zeros(batch_size, ),
            achieved_goal=0,
            desired_goal=0)
        opt.set_cost(_costs)
        solution = opt.obtain_solution(fake_item, ())
        self.assertEqual(solution.shape,
                         (batch_size, planning_horizon, act_dim))
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
            solution.reshape(-1, act_dim),
            start +
            (1. + torch.arange(10)).reshape(-1, 1) * (end - start) / 11,
            epsilon=0.1)

        # action dimensions with different scales
        opt2 = CEMOptimizer(
            planning_horizon=4,
            action_dim=act_dim,
            population_size=pop_size,
            top_percent=0.05,
            iterations=100,
            bounds=([-10, -1], [10, 1]))
        opt2.set_cost(_costs_agg_dist)
        start2 = torch.Tensor([[-10, -1]])
        end2 = torch.Tensor([[10, 1]])
        item2 = DataItem(
            observation=torch.zeros(1, ),
            achieved_goal=start2,
            desired_goal=end2)
        solution2 = opt2.obtain_solution(item2, ())
        self.assertTensorClose(
            solution2,
            torch.Tensor([[[-6., -0.6], [-2., -0.2], [2, 0.2], [6, 0.6]]]),
            epsilon=0.1)
