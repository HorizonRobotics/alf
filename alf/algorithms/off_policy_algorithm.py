# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

import abc
from alf.algorithms import policy_algorithm


class OffPolicyAlgorithm(policy_algorithm.PolicyAlgorithm):

    def __init__(self,
                 action_spec,
                 train_state_spec,
                 action_distribution_spec,
                 predict_state_spec=None,
                 optimizer=None,
                 gradient_clipping=None,
                 train_step_counter=None,
                 debug_summaries=False,
                 name="OffPolicyAlgorithm"):
        super().__init__(action_spec,
                         train_state_spec,
                         action_distribution_spec,
                         predict_state_spec,
                         optimizer,
                         gradient_clipping,
                         train_step_counter,
                         debug_summaries,
                         name)

    @abc.abstractmethod
    def train_step(self, time_step=None, state=None):
        pass

    @abc.abstractmethod
    def train_complete(self, experience=None):
        pass
