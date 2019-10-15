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

import gin.tf
from alf.drivers.on_policy_driver import OnPolicyDriver
from alf.trainers.policy_trainer import Trainer


@gin.configurable("on_policy_trainer")
class OnPolicyTrainer(Trainer):
    def __init__(self, config):
        """Perform on-policy training using OnPolicyDriver
        Args:
            config (TrainerConfig): configuration used to construct this trainer
        """
        super().__init__(config)
        self._num_steps_per_iter = config.num_steps_per_iter

    def init_driver(self):
        return OnPolicyDriver(
            env=self._envs[0],
            algorithm=self._algorithm,
            train_interval=self._unroll_length)

    def train_iter(self, iter_num, policy_state, time_step):
        time_step, policy_state, num_steps = self._driver.run(
            max_num_steps=self._num_steps_per_iter,
            time_step=time_step,
            policy_state=policy_state)
        return time_step, policy_state, num_steps
