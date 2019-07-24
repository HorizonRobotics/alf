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


@gin.configurable
class OnPolicyTrainer(Trainer):
    def __init__(self,
                 root_dir,
                 train_interval=20,
                 num_steps_per_iter=10000,
                 **kwargs):
        """Perform on-policy training using OnPolicyDriver
        Args:
            root_dir (str): directory for saving summary and checkpoints
            train_interval (int): update parameter every so many env.step().
            num_steps_per_iter (int): number of steps for one iteration. It is the
                total steps from all individual environment in the batch
                environment.
        """
        super().__init__(root_dir, **kwargs)
        self._train_interval = train_interval
        self._num_steps_per_iter = num_steps_per_iter

    def init_driver(self):
        return OnPolicyDriver(
            env=self._env,
            algorithm=self._algorithm,
            train_interval=self._train_interval,
            debug_summaries=self._debug_summaries,
            summarize_grads_and_vars=self._summarize_grads_and_vars)

    def train_iter(self, iter_num, policy_state, time_step):
        return self._driver.run(
            max_num_steps=self._num_steps_per_iter,
            time_step=time_step,
            policy_state=policy_state)
