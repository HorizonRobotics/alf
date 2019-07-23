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
from alf.drivers.async_off_policy_driver import AsyncOffPolicyDriver
from alf.drivers.sync_off_policy_driver import SyncOffPolicyDriver
from alf.environments.utils import create_environment
from alf.trainers.policy_trainer import Trainer


@gin.configurable
class OffPolicyTrainer(Trainer):
    def __init__(self,
                 root_dir,
                 initial_collect_steps=0,
                 num_updates_per_train_step=4,
                 unroll_length=8,
                 mini_batch_length=20,
                 mini_batch_size=256,
                 clear_replay_buffer=True):
        """Abstract base class for off policy trainer

        Args:
            initial_collect_steps (int): if positive, number of steps each single
                environment steps before perform first update
            num_updates_per_train_step (int): number of optimization steps for
                one iteration
            unroll_length (int): number of time steps each environment proceeds per
                iteration. The total number of time steps from all environments per
                iteration can be computed as: `num_envs` * `env_batch_size`
                * `unroll_length`.
            mini_batch_size (int): number of sequences for each minibatch
            mini_batch_length (int): the length of the sequence for each
                sample in the minibatch
            clear_replay_buffer (bool): whether use all data in replay buffer to
                perform one update and then wiped clean

        """
        super().__init__(root_dir)
        self._initial_collect_steps = initial_collect_steps
        self._num_updates_per_train_step = num_updates_per_train_step
        self._unroll_length = unroll_length
        self._mini_batch_length = mini_batch_length
        self._mini_batch_size = mini_batch_size
        self._clear_replay_buffer = clear_replay_buffer

    def get_exp(self):
        """Get experience from replay buffer for training"""
        replay_buffer = self._driver.exp_replayer
        if self._clear_replay_buffer:
            experience = replay_buffer.replay_all()
            replay_buffer.clear()
        else:
            experience, _ = replay_buffer.replay(
                sample_batch_size=self._mini_batch_size,
                mini_batch_length=self._mini_batch_length)
        return experience


@gin.configurable
class SyncOffPolicyTrainer(OffPolicyTrainer):
    """Perform off-policy training using SyncOffPolicyDriver"""

    def init_driver(self):
        return SyncOffPolicyDriver(
            env=self._env,
            algorithm=self._algorithm,
            debug_summaries=self._debug_summaries,
            summarize_grads_and_vars=self._summarize_grads_and_vars)

    def train_iter(self, iter_num, policy_state, time_step):
        max_num_steps = self._unroll_length * self._env.batch_size
        if iter_num == 0 and self._initial_collect_steps != 0:
            max_num_steps = self._initial_collect_steps
        time_step, policy_state = self._driver.run(
            max_num_steps=max_num_steps,
            time_step=time_step,
            policy_state=policy_state)

        self._driver.train(
            self.get_exp(),
            num_updates=self._num_updates_per_train_step,
            mini_batch_length=self._mini_batch_length,
            mini_batch_size=self._mini_batch_size)

        return time_step, policy_state


@gin.configurable
class AsyncOffPolicyTrainer(OffPolicyTrainer):
    """Perform off-policy training using AsyncOffPolicyDriver"""

    def init_driver(self):
        driver = AsyncOffPolicyDriver(
            env_f=create_environment,
            algorithm=self._algorithm,
            unroll_length=self._unroll_length,
            debug_summaries=self._debug_summaries,
            summarize_grads_and_vars=self._summarize_grads_and_vars)
        driver.start()
        return driver

    def train_iter(self, iter_num, policy_state, time_step):
        if iter_num == 0 and self._initial_collect_steps != 0:
            steps = 0
            while steps < self._initial_collect_steps:
                steps += self._driver.run_async()
        else:
            self._driver.run_async()
        self._driver.train(
            self.get_exp(),
            num_updates=self._num_updates_per_train_step,
            mini_batch_length=self._mini_batch_length,
            mini_batch_size=self._mini_batch_size)
        return time_step, policy_state
