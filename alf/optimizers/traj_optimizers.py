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

import numpy as np
import torch

import alf
from alf.data_structures import TimeStep
from alf.utils import tensor_utils


class TrajOptimizer(object):
    """Trajectory Optimizer Module

    This module generates optimized solution by minimizing a given
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
                 solution_dim,
                 population_size,
                 upper_bound,
                 lower_bound,
                 elite_size=50,
                 max_iter_num=5,
                 epsilon=0.01,
                 tau=0.9,
                 min_var=1e-5):
        """Creates a CEM Optimizer

        This module optimizes a given cost function via the `Cross-Enrtopy
        Method <https://en.wikipedia.org/wiki/Cross-entropy_method>`_,
        which iterates between evaluating a population of samples generated
        from a probability distribution and updating the distribution based
        on the evaluation for generating better samples in the next
        iteration. In practice, a multi-dimensional Gaussian distribution
        with a diagonal covariance matrix is used.

        Args:
            solution_dim (int): the dimensionality of the problem space
            population_size (int): the number of candidate solutions to be
                sampled at every iteration
            upper_bound (float|Tensor): upper bounds for elements in solution
            lower_bound (float|Tensor): lower bounds for elements in solution
            elite_size (int): the number of elites selected in each round.
                Elites represent the group of the top-elite_size members from
                the population based on their cost values. They are used to
                update the mean and variance of the Gaussian population
                generation distribution.
            max_iter_num (int|Tensor): the maximum number of CEM iterations
            epsilon (float): a minimum variance threshold. If the maximum
                variance of the population falls below it, the CEM iteration
                will stop.
            tau (float): a value in (0, 1) for softly updating the population
                mean and variance:
                    mean = (1 - tau) * mean + tau * new_mean
                    var = (1 - tau) * var + tau * new_var
            min_var (float): minimum value of the variance for the Gaussian
                distribution to sample from
        """
        super().__init__()
        self._solution_dim = solution_dim
        self._population_size = population_size
        self._upper_bound = upper_bound
        self._lower_bound = lower_bound
        self._elite_size = elite_size
        self._max_iter_num = max_iter_num
        self._epsilon = epsilon
        self._tau = tau
        self._min_var = min_var

    def obtain_solution(self,
                        time_step: TimeStep,
                        state,
                        init_mean=None,
                        init_var=None):
        """Minimize the cost function provided by using the CEM method.

        Args:
            time_step (TimeStep): the initial time_step to start rollout
            state: the initial state to start rollout
            init_mean (None|Tensor): initial mean of the population. If None,
                the mean is initialized as zeros.
            init_var (None|Tensor): initial variance of the population. If None,
                the variance is initialized to have value as
                0.5 * (upper_bound - lower_bound).
        """
        init_obs = time_step.observation
        batch_size = init_obs.shape[0]

        if init_mean is None:
            # [B, 1, solution_dim]
            init_mean = torch.zeros(batch_size, 1, self._solution_dim)
        else:
            assert init_mean.shape == (batch_size, 1, self._solution_dim)

        if init_var is None:
            init_var = torch.ones(batch_size, 1, self._solution_dim) * \
                    (self._upper_bound - self._lower_bound) / 2.
        else:
            assert init_var.shape == (batch_size, 1, self._solution_dim)

        i = 0
        pop_mean = init_mean
        pop_var = init_var

        # [B, population, solution_dim]
        trunc_normal_samples = torch.zeros(batch_size, self._population_size,
                                           self._solution_dim)

        while i < self._max_iter_num and pop_var.max() > self._epsilon:
            # compute the distance to lower and upper bound
            distance_to_lb = pop_mean - self._lower_bound
            distance_to_ub = self._upper_bound - pop_mean

            # compute the constrained var based on the computed distance
            constrained_var = torch.min(
                torch.min((distance_to_lb / 2.0)**2, (distance_to_ub / 2.0)
                          **2), pop_var)
            constrained_var = constrained_var.clamp(min=self._min_var)

            trunc_normal_samples = alf.initializers.trunc_normal_(
                trunc_normal_samples)
            samples = trunc_normal_samples * torch.sqrt(
                constrained_var) + pop_mean

            costs = self.cost_function(time_step, state, samples.clone())

            # select elite set from the population
            ind = torch.topk(-costs, self._elite_size)[1]
            # samples: [batch, population, solution_dim]
            elites = samples[torch.arange(batch_size).unsqueeze(-1), ind]

            # update mean and var based on the elite set
            new_mean = torch.mean(elites, dim=1, keepdim=True)
            new_var = torch.var(elites, dim=1, keepdim=True)

            pop_mean = (1 - self._tau) * pop_mean + self._tau * new_mean
            pop_var = (1 - self._tau) * pop_var + self._tau * new_var
            i = i + 1

        # [B, 1, solution_dim] -> [B, solution_dim]
        return pop_mean.squeeze(1)
