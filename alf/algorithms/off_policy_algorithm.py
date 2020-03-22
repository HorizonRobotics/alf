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
"""Base class for off policy algorithms."""

from absl import logging
from collections import namedtuple
import math
from typing import Callable
import torch

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, StepType, TrainingInfo
from alf.utils import common, dist_utils
from alf.utils.summary_utils import record_time


class OffPolicyAlgorithm(RLAlgorithm):
    """`OffPolicyAlgorithm` implements basic off-policy training pipeline.

       User needs to implement `rollout_step()` and `train_step()`.

       `rollout_step()` is called to generate actions at every environment step.

       `train_step()` is called to generate necessary information for training.

       The following is the pseudo code to illustrate how `OffPolicyAlgorithm`
       is used:

       ```python
        # (1) collect stage
        for _ in range(steps_per_collection):
            # collect experience and store to replay buffer
            policy_step = rollout_step(time_step, policy_step.state)
            experience = make_experience(time_step, policy_step)
            store experience to replay buffer
            action = sample action from policy_step.action
            time_step = env.step(action)

        # (2) train stage
        for _ in range(training_per_collection):
            # sample experiences and perform training
            experiences = sample batch from replay_buffer
            batched_training_info = []
            for experience in experiences:
                policy_step = train_step(experience, state)
                train_info = make_training_info(info, ...)
                write train_info to batched_training_info
            loss = calc_loss(batched_training_info)
            update_with_gradient(loss)
    ```
    """

    @property
    def train_info_spec(self):
        """The spec for the `AlgStep.info` returned from train_step()."""
        assert self._train_info_spec is not None, (
            "train_step() has not been used. train_info_spec is not available")
        return self._train_info_spec

    @property
    def processed_experience_spec(self):
        """Spec for processed experience.

        Returns:
            Spec for the experience returned by preprocess_experience().
        """
        assert self._processed_experience_spec is not None, (
            "preprocess_experience() has not been used. processed_experience_spec"
            "is not available")
        return self._processed_experience_spec

    def preprocess_experience(self, experience: Experience):
        """Preprocess experience.

        `preprocess_experience()` is called on the experiences got from a replay
        buffer. An example usage of this function is to calculate advantages and
        returns in `PPOAlgorithm`.

        The shapes of tensors in experience are assumed to be (B, T, ...).

        Args:
            experience (Experience): original experience
        Returns:
            processed experience
        """
        return experience

    def is_on_policy(self):
        return False

    def _train_iter_off_policy(self):
        """User may override this for their own training procedure."""
        config: TrainerConfig = self._config

        if (alf.summary.get_global_counter() == 0
                and config.initial_collect_steps != 0):
            unroll_steps = config.unroll_length * self._env.batch_size
            num_unrolls = math.ceil(
                config.initial_collect_steps / unroll_steps)
        else:
            num_unrolls = 1

        if not config.update_counter_every_mini_batch:
            alf.summary.get_global_counter().add_(1)

        with record_time("time/unroll"):
            with torch.no_grad():
                for _ in range(num_unrolls):
                    training_info = self.unroll(config.unroll_length)
                    self.summarize_rollout(training_info)
                    self.summarize_metrics()

        with record_time("time/replay"):
            mini_batch_size = config.mini_batch_size
            if mini_batch_size is None:
                mini_batch_size = self._exp_replayer.batch_size
            if config.whole_replay_buffer_training:
                experience = self._exp_replayer.replay_all()
                if config.clear_replay_buffer:
                    self._exp_replayer.clear()
            else:
                experience = self._exp_replayer.replay(
                    sample_batch_size=mini_batch_size,
                    mini_batch_length=config.mini_batch_length)

        with record_time("time/train"):
            return self._train_experience(
                experience, config.num_updates_per_train_step, mini_batch_size,
                config.mini_batch_length,
                config.update_counter_every_mini_batch)

    def _train_experience(self, experience, num_updates, mini_batch_size,
                          mini_batch_length, update_counter_every_mini_batch):
        """Train using experience."""
        experience = dist_utils.params_to_distributions(
            experience, self.experience_spec)
        experience = self.transform_timestep(experience)
        experience = self.preprocess_experience(experience)
        if self._processed_experience_spec is None:
            self._processed_experience_spec = dist_utils.extract_spec(
                experience, from_dim=2)
        experience = dist_utils.distributions_to_params(experience)

        length = experience.step_type.shape[1]
        mini_batch_length = (mini_batch_length or length)
        assert length % mini_batch_length == 0, (
            "length=%s not a multiple of mini_batch_length=%s" %
            (length, mini_batch_length))

        if len(alf.nest.flatten(
                self.train_state_spec)) > 0 and not self._use_rollout_state:
            if mini_batch_length == 1:
                logging.fatal(
                    "Should use TrainerConfig.use_rollout_state=True "
                    "for off-policy training of RNN when minibatch_length==1.")
            else:
                common.warning_once(
                    "Consider using TrainerConfig.use_rollout_state=True "
                    "for off-policy training of RNN.")

        experience = alf.nest.map_structure(
            lambda x: x.reshape(-1, mini_batch_length, *x.shape[2:]),
            experience)

        batch_size = experience.step_type.shape[0]
        mini_batch_size = (mini_batch_size or batch_size)

        def _make_time_major(nest):
            """Put the time dim to axis=0."""
            return alf.nest.map_structure(lambda x: x.transpose(0, 1), nest)

        for u in range(num_updates):
            if mini_batch_size < batch_size:
                indices = torch.randperm(batch_size)
                experience = alf.nest.map_structure(lambda x: x[indices],
                                                    experience)
            for b in range(0, batch_size, mini_batch_size):
                if update_counter_every_mini_batch:
                    alf.summary.get_global_counter().add_(1)
                is_last_mini_batch = (u == num_updates - 1
                                      and b + mini_batch_size >= batch_size)
                do_summary = (is_last_mini_batch
                              or update_counter_every_mini_batch)
                alf.summary.enable_summary(do_summary)
                batch = alf.nest.map_structure(
                    lambda x: x[b:min(batch_size, b + mini_batch_size)],
                    experience)
                batch = _make_time_major(batch)
                training_info, loss_info, params = self._update(
                    batch, weight=batch.step_type.shape[1] / mini_batch_size)
                if do_summary:
                    self.summarize_train(training_info, loss_info, params)

        train_steps = batch_size * mini_batch_length * num_updates
        return train_steps

    def _update(self, experience, weight):
        batch_size = experience.step_type.shape[1]
        initial_train_state = self.get_initial_train_state(batch_size)
        if self._use_rollout_state:
            policy_state = alf.nest.map_structure(lambda state: state[0, ...],
                                                  experience.state)
        else:
            policy_state = initial_train_state

        num_steps = experience.step_type.shape[0]
        info_list = []
        for counter in range(num_steps):
            exp = alf.nest.map_structure(lambda ta: ta[counter], experience)
            exp = dist_utils.params_to_distributions(
                exp, self.processed_experience_spec)
            policy_state = common.reset_state_if_necessary(
                policy_state, initial_train_state,
                exp.step_type == StepType.FIRST)
            policy_step = self.train_step(exp, policy_state)
            if self._train_info_spec is None:
                self._train_info_spec = dist_utils.extract_spec(
                    policy_step.info)
            info_list.append(
                dist_utils.distributions_to_params(policy_step.info))
            policy_state = policy_step.state

        info = alf.nest.utils.stack_nests(info_list)
        info = dist_utils.params_to_distributions(info, self.train_info_spec)
        experience = dist_utils.params_to_distributions(
            experience, self.processed_experience_spec)
        training_info = TrainingInfo(
            action=experience.action,
            reward=experience.reward,
            discount=experience.discount,
            step_type=experience.step_type,
            rollout_info=experience.rollout_info,
            info=info,
            env_id=experience.env_id)

        loss_info = self.calc_loss(training_info)
        valid_masks = (training_info.step_type != StepType.LAST).to(
            torch.float32)
        loss_info, params = self.update_with_gradient(loss_info, valid_masks)
        self.after_update(training_info)

        return training_info, loss_info, params
