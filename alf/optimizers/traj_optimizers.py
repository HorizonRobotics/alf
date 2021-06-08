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
import gin
import numpy as np
import torch
from torch.distributions import Normal

import alf
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

    def obtain_solution(self, time_step: TimeStep, state, info=None):
        """Minimize the cost function provided

        Args:
            time_step (TimeStep): the initial time_step to start rollout
            state: input state to start rollout

        Returns:
            - solution (Tensor): shape (``batch_size``, ``planning_horizon``, ``action_dim``)
            - cost (Tensor): costs corresponding to the best solution
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
        return solution, costs[(torch.arange(batch_size),
                                min_ind)].unsqueeze(-1)


@gin.configurable
class CEMOptimizer(TrajOptimizer):
    def __init__(self,
                 planning_horizon,
                 action_dim,
                 bounds,
                 population_size=100,
                 top_percent=0.1,
                 iterations=2,
                 init_mean=None,
                 init_var=None,
                 use_target_networks=False,
                 use_replica_min=True):
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
            bounds (pair of float Tensors): min and max bounds of the samples
            population_size (int): population to compute cost and optimize
            top_percent (float): percentage of top samples to use for optimization
            iterations (int): number of iterations to run CEM optimization
            init_mean (float|Tensor): mean to initialize the normal distributions
            init_var (float|Tensor): var to initialize the normal distributions
            use_target_networks (bool): whether to use target networks to compute value
            use_replica_min (bool): whether use min of replicated critics or first critic
        """
        super().__init__()
        self._planning_horizon = planning_horizon
        self._action_dim = action_dim
        self._population_size = population_size
        self._top_percent = top_percent
        self._iterations = iterations
        assert len(bounds) == 2, "must provide (min, max) bounds"
        bounds = (torch.as_tensor(bounds[0], dtype=torch.float32),
                  torch.as_tensor(bounds[1], dtype=torch.float32))
        if bounds[0].shape == ():
            bounds = (bounds[0].expand(action_dim),
                      bounds[1].expand(action_dim))
        assert bounds[0].shape == (action_dim, ), "bounds not of action_dim"
        assert bounds[0].shape == bounds[1].shape, "shape mismatch"
        if init_mean is None:
            init_mean = (bounds[0] + bounds[1]) / 2.
        else:
            init_mean = torch.as_tensor(init_mean)
            if init_mean.shape == ():
                init_mean = init_mean.expand(action_dim)
        self._init_mean = torch.as_tensor(init_mean).reshape(1, 1, action_dim)
        if init_var is None:
            init_var = bounds[1] - bounds[0]
        else:
            init_var = torch.as_tensor(init_var)
            if init_var.shape == ():
                init_var = init_var.expand(action_dim)
        self._init_var = torch.as_tensor(init_var).reshape(1, 1, action_dim)
        self._use_target_networks = use_target_networks
        self._use_replica_min = use_replica_min
        self._bounds = (bounds[0].expand(1, 1, planning_horizon, action_dim),
                        bounds[1].expand(1, 1, planning_horizon, action_dim))
        assert self._top_percent * self._population_size > 1, "too few samples"

    def set_initial_distributions(self, mean, m2, set_bounds=False):
        """Use the stats in normalizer to re-initialize the distributions before obtain_solution.

        Args:
            normalizer (Normalizer): to compute a running average of mean and std.
        """
        self._init_mean = mean
        self._init_var = m2 - alf.utils.math_ops.square(mean) + 1e-8
        if set_bounds:
            self._bounds = ((mean - 5 * torch.sqrt(self._init_var)).expand(
                1, 1, self._planning_horizon, self._action_dim), (
                    mean + 5 * torch.sqrt(self._init_var).expand(
                        1, 1, self._planning_horizon, self._action_dim)))

    def _init_distr(self, batch_size):
        means = self._init_mean.expand(batch_size, self._planning_horizon,
                                       self._action_dim) + torch.zeros(
                                           batch_size, self._planning_horizon,
                                           self._action_dim)
        std = self._init_var.expand(
            batch_size, self._planning_horizon, self._action_dim) * torch.ones(
                batch_size, self._planning_horizon, self._action_dim)
        return means, torch.sqrt(std)

    def obtain_solution(self, time_step: TimeStep, state, info=None):
        """Minimize the cost function provided

        Args:
            time_step (TimeStep): the initial ``time_step`` to start planning
            state: input state to start planning
            info (dict): if not None, populate with costs per segment.

        Returns:
            - solution (Tensor): shape (``batch_size``, ``planning_horizon``, ``action_dim``)
            - cost (Tensor): costs corresponding to the best solution
        """
        init_obs = time_step.observation
        batch_size = alf.nest.get_nest_batch_size(init_obs)
        distr = Normal(*self._init_distr(batch_size))
        solution = None

        def _clamp(v, min_v, max_v):
            # v: (pop_size, B, plan_horizon, act_dim)
            # max_v: (1, 1, plan_horizon, act_dim)
            return torch.max(torch.min(v, max_v), min_v)

        for i in range(self._iterations):
            # samples shape (B, pop_size, horizon, act_dim)
            samples = _clamp(
                distr.sample((self._population_size, )),
                *self._bounds).transpose(0, 1)
            costs = self.cost_function(
                time_step,
                state,
                samples,
                info=info,
                use_target_networks=self._use_target_networks,
                use_replica_min=self._use_replica_min)
            assert costs.shape == samples.shape[:2], "bad cost function"
            min_inds = torch.topk(
                costs,
                int(self._population_size * self._top_percent),
                largest=False,
                dim=1)[1].long()
            # min_inds shape: (B, pop_size * top_percent)
            tops = samples[(torch.arange(batch_size).unsqueeze(1), min_inds)]

            # if (i + 1) % (self._iterations / 5) == 0:
            #     logging.warning(
            #         "%d: distr means: %s, var: %s", i,
            #         str(
            #             distr.loc.reshape(batch_size, self._planning_horizon,
            #                               self._action_dim)),
            #         str(
            #             distr.scale.reshape(batch_size, self._planning_horizon,
            #                                 self._action_dim)))
            if i == self._iterations - 1:
                min_ind = torch.argmin(costs, dim=1).long()
                solution = samples[(torch.arange(batch_size), min_ind)]
                if info is not None:
                    info["segment_costs"] = info["all_population_costs"][(
                        torch.arange(batch_size), min_ind)]
                break
            else:
                means = torch.mean(tops, dim=1)
                # minimum cov of 1e-4 tends to work well with planning horizon
                # of 10 with simpler as well as harder cost functions.
                std = torch.sum(
                    (tops - means.unsqueeze(1))**2, dim=1) / (
                        self._top_percent * self._population_size - 1) + 1.e-4
                distr = Normal(means, torch.sqrt(std))
        return solution, costs[(torch.arange(batch_size),
                                min_ind)].unsqueeze(-1)
