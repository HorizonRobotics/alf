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


class Optimizer(object):
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def set_cost(self, cost_function):
        """Set cost function for miminizing.
        cost_function (Callable): the cost function to be minimized. It
            takes as input:
            (1) time_step (ActionTimeStep) for next step prediction
            (2) state: input state for next step prediction
            (3) action_sequence (tf.Tensor of shape [batch_size,
                population_size, solution_dim])
            and returns a cost Tensor of shape [batch_size, population_size]
        """
        self.cost_function = cost_function

    def obtain_solution(self, *args, **kwargs):
        pass
