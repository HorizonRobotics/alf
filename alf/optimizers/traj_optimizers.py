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
import numpy as np
import torch
from torch.distributions import Normal

from alf.data_structures import TimeStep
from alf.utils import tensor_utils


class TrajOptimizer(object):
    """Trajectory Optimizer Module

    This module generates optimized trajectories by minimizing a trajectory
        cost function set through ``set_cost``.
    """

    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def set_cost(self, cost_function):
        """Set cost function for miminizing.
        cost_function (Callable): the cost function to be minimized. It
            takes as input:
            (1) time_step (TimeStep) for next step prediction
            (2) state: input state for next step prediction
            (3) action_sequence (Tensor of shape [batch_size,
                population_size, solution_dim])
            and returns a cost Tensor of shape [batch_size, population_size]
        """
        self.cost_function = cost_function

    def obtain_solution(self, *args, **kwargs):
        pass


class RandomOptimizer(TrajOptimizer):
    def __init__(self,
                 solution_dim,
                 population_size,
                 upper_bound=None,
                 lower_bound=None):
        """Random Trajectory Optimizer

        This module conducts trajectory optimization via random-shooting-based
            optimization, i.e., generating a random population for each sample
            in the batch and select those having the lowest cost as the solution.

        Args:
            solution_dim (int): The dimensionality of the problem space
            population_size (int): The number of candidate solutions to be
                sampled at every iteration
            upper_bound (int|Tensor): upper bounds for elements in solution
            lower_bound (int|Tensor): lower bounds for elements in solution
        """
        super().__init__()
        self._solution_dim = solution_dim
        self._population_size = population_size
        self._upper_bound = upper_bound
        self._lower_bound = lower_bound

    def obtain_solution(self, time_step: TimeStep, state):
        """Minimize the cost function provided

        Args:
            time_step (TimeStep): the initial time_step to start rollout
            state: input state to start rollout
        """
        init_obs = time_step.observation
        batch_size = init_obs.shape[0]
        solutions = torch.rand(
            batch_size, self._population_size, self._solution_dim
        ) * (self._upper_bound - self._lower_bound) + self._lower_bound * 1.0
        costs = self.cost_function(time_step, state, solutions)
        min_ind = torch.argmin(costs, dim=-1).long()
        # solutions [B, pop_size, sol_dim] -> [B, sol_dim]
        solution = solutions[(torch.arange(batch_size), min_ind)]
        return solution


class CEMOptimizer(TrajOptimizer):
    def __init__(self,
                 planning_horizon,
                 action_dim,
                 population_size,
                 top_percent,
                 iterations,
                 init_mean=None,
                 init_var=None,
                 bounds=None):
        """Cross Entropy Method Trajectory Optimizer

        This module conducts trajectory optimization via sampling from a number
            of multivariate normal distributions (one per planning horizon), 
            calculating the cost, taking top m% of the trajectories and
            recompute mean and covariance of the distributions based on the top
            trajectories.

        Args:
            planning_horizon (int): The number of steps to plan
            action_dim (int): The dimensionality of each action executed at
                each step
            population_size (int): population to compute cost and optimize
            top_percent (float): percentage of top samples to use for optimization
            iterations (int): number of iterations to run CEM optimization
            init_mean (float|Tensor): mean to initialize the normal distributions
            init_var (float|Tensor): var to initialize the normal distributions
            bounds (pair of float): min and max bounds of the samples
        """
        super().__init__()
        self._planning_horizon = planning_horizon
        self._action_dim = action_dim
        self._population_size = population_size
        self._top_percent = top_percent
        self._iterations = iterations
        self._init_mean = init_mean
        self._init_var = init_var
        self._bounds = bounds
        assert self._top_percent * self._population_size > 1, "too few samples"

    def _init_distr(self, batch_size):
        means = self._init_mean * torch.zeros(
            batch_size * self._planning_horizon * self._action_dim)
        std = self._init_var * torch.ones(
            batch_size * self._planning_horizon * self._action_dim)
        return means, torch.sqrt(std)

    def obtain_solution(self, time_step: TimeStep, state):
        """Minimize the cost function provided

        Args:
            time_step (TimeStep): the initial ``time_step`` to start planning
            state: input state to start planning

        Returns:
            solution Tensor of length ``batch_size`` * ``planning_horizon`` *
                ``action_dim``
        """
        init_obs = time_step.observation
        batch_size = init_obs.shape[0]
        distr = Normal(*self._init_distr(batch_size))
        solution = None
        for i in range(self._iterations):
            samples = distr.sample(
                (self._population_size, )).clamp(*self._bounds)
            costs = self.cost_function(time_step, state, samples)
            assert costs.shape[0] == samples.shape[0], "bad cost function"
            min_inds = torch.topk(
                costs,
                int(self._population_size * self._top_percent),
                largest=False,
                dim=0)[1].long()
            # samples [pop_size, B * horizon * act_dim]
            tops = samples[min_inds]

            if i == self._iterations - 1:
                min_ind = torch.argmin(costs, dim=0).long()
                solution = samples[min_ind]
                break
            else:
                means = torch.mean(tops, dim=0)
                # minimum cov of 1e-4 tends to work well with planning horizon
                # of 10 with simpler as well as harder cost functions.
                std = torch.sum(
                    (tops - means)**2, dim=0) / (
                        self._top_percent * self._population_size - 1) + 1.e-4
                distr = Normal(means, torch.sqrt(std))
        return solution
