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

import time

import gin.tf
import tensorflow as tf

from alf.drivers.async_off_policy_driver import AsyncOffPolicyDriver
from alf.drivers.sync_off_policy_driver import SyncOffPolicyDriver
from alf.environments.utils import create_environment
from alf.trainers.policy_trainer import Trainer


class OffPolicyTrainer(Trainer):
    def __init__(self, config):
        """Abstract base class for off policy trainer

        Args:
            config (TrainerConfig): configuration used to construct this trainer
        """
        super().__init__(config)
        self._initial_collect_steps = config.initial_collect_steps
        self._num_updates_per_train_step = config.num_updates_per_train_step
        self._mini_batch_length = config.mini_batch_length
        if self._mini_batch_length is None:
            self._mini_batch_length = self._unroll_length
        self._mini_batch_size = config.mini_batch_size
        self._clear_replay_buffer = config.clear_replay_buffer

    def get_exp(self):
        """Get experience from replay buffer for training

        Returns:
            exp (Experience): each item has the shape [B, T ...] where B = batch size, T = steps
        """
        replay_buffer = self._driver.exp_replayer
        if self._mini_batch_size is None:
            self._mini_batch_size = replay_buffer.batch_size
        if self._clear_replay_buffer:
            experience = replay_buffer.replay_all()
            replay_buffer.clear()
        else:
            experience, _ = replay_buffer.replay(
                sample_batch_size=self._mini_batch_size,
                mini_batch_length=self._mini_batch_length)
        return experience


@gin.configurable("sync_off_policy_trainer")
class SyncOffPolicyTrainer(OffPolicyTrainer):
    """Perform off-policy training using SyncOffPolicyDriver"""

    def init_driver(self):
        return SyncOffPolicyDriver(
            env=self._env,
            use_rollout_state=self._config.use_rollout_state,
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
        t0 = time.time()
        self._driver.train(
            self.get_exp(),
            num_updates=self._num_updates_per_train_step,
            mini_batch_length=self._mini_batch_length,
            mini_batch_size=self._mini_batch_size)
        tf.summary.scalar("time/train", time.time() - t0)
        return time_step, policy_state, max_num_steps


@gin.configurable("async_off_policy_trainer")
class AsyncOffPolicyTrainer(OffPolicyTrainer):
    """Perform off-policy training using AsyncOffPolicyDriver"""

    def init_driver(self):
        driver = AsyncOffPolicyDriver(
            env_f=create_environment,
            algorithm=self._algorithm,
            use_rollout_state=self._config.use_rollout_state,
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
            steps = self._driver.run_async()
        t0 = time.time()
        self._driver.train(
            self.get_exp(),
            num_updates=self._num_updates_per_train_step,
            mini_batch_length=self._mini_batch_length,
            mini_batch_size=self._mini_batch_size)
        tf.summary.scalar("time/train", time.time() - t0)
        return time_step, policy_state, steps
