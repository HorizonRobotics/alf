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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .optimizer import Optimizer


class RandomOptimizer(Optimizer):
    def __init__(self, sol_dim, popsize, upper_bound=None, lower_bound=None):
        """Creates an instance of this class.

        Args:
            sol_dim (int): The dimensionality of the problem space
            popsize (int): The number of candidate solutions to be sampled at every iteration
            cost_function (func): the cost function to be minimized
            upper_bound (int|tf.Array): upper bounds for elements in solution
            lower_bound (int|tf.Array): lower bounds for elements in solution
        """
        super().__init__()
        self.sol_dim = sol_dim
        self.popsize = popsize
        self.ub = upper_bound
        self.lb = lower_bound

    def reset(self):
        pass

    def obtain_solution(self, init_obs, *args, **kwargs):
        """Minimize the cost function provided

        Arguments:
            init_obs (tf.Tensor): the initial observation to start the rollout
        """
        solutions = tf.random.uniform([self.popsize, self.sol_dim], self.ub,
                                      self.lb)
        costs = self.cost_function(init_obs, solutions)
        solution = tf.gather(
            solutions, tf.cast(tf.argmin(costs, axis=-1), tf.int32), axis=0)
        return solution
