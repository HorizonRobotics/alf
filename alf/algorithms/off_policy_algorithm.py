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

import abc
from collections import namedtuple
from typing import Callable

from absl import logging
import tensorflow as tf
import tensorflow_probability as tfp

from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import ActionTimeStep, Experience, StepType, TrainingInfo
from alf.utils import common, nest_utils


class OffPolicyAlgorithm(RLAlgorithm):
    """
       OffPolicyAlgorithm works with alf.drivers.off_policy_driver to do training

       User needs to implement rollout() and train_step().

       rollout() is called to generate actions for every environment step.

       train_step() is called to generate necessary information for training.

       The following is the pseudo code to illustrate how OffPolicyAlgorithm is used
       with OffPolicyDriver:

       ```python
        # (1) collect stage
        for _ in range(steps_per_collection):
            # collect experience and store to replay buffer
            policy_step = rollout(time_step, policy_step.state)
            experience = make_experience(time_step, policy_step)
            store experience to replay buffer
            action = sample action from policy_step.action
            time_step = env.step(action)

        # (2) train stage
        for _ in range(training_per_collection):
            # sample experiences and perform training
            experiences = sample batch from replay_buffer
            with tf.GradientTape() as tape:
                batched_training_info = []
                for experience in experiences:
                    policy_step = train_step(experience, state)
                    train_info = make_training_info(info, ...)
                    write train_info to batched_training_info
                train_complete(tape, batched_training_info,...)
    ```
    """

    def need_full_rollout_state(self):
        return self._is_rnn and self._use_rollout_state

    @property
    def exp_replayer(self):
        """Return experience replayer."""
        return self._exp_replayer

    @abc.abstractmethod
    def train_step(self, experience: Experience, state):
        """Perform one step of training computation.

        Args:
            experience (Experience):
            state (nested Tensor): should be consistent with train_state_spec

        Returns (PolicyStep):
            action (nested tf.distribution): should be consistent with
                `action_distribution_spec`
            state (nested Tensor): should be consistent with `train_state_spec`
            info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OffPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        pass

    def preprocess_experience(self, experience: Experience):
        """Preprocess experience.

        preprocess_experience is called for the experiences got from replay
        buffer. An example is to calculate advantages and returns in PPOAlgorithm.

        The shapes of tensors in experience are assumed to be (B, T, ...)

        Args:
            experience (Experience): original experience
        Returns:
            processed experience
        """
        return experience

    def train(self,
              num_updates=1,
              mini_batch_size=None,
              mini_batch_length=None,
              clear_replay_buffer=True):
        """Train algorithm.

        Args:
            num_updates (int): number of optimization steps
            mini_batch_size (int): number of sequences for each minibatch
            mini_batch_length (int): the length of the sequence for each
                sample in the minibatch
            clear_replay_buffer (bool): whether use all data in replay buffer to
                perform one update and then wiped clean

        Returns:
            train_steps (int): the actual number of time steps that have been
                trained (a step might be trained multiple times)
        """

        if mini_batch_size is None:
            mini_batch_size = self._exp_replayer.batch_size
        if clear_replay_buffer:
            experience = self._exp_replayer.replay_all()
            self._exp_replayer.clear()
        else:
            experience = self._exp_replayer.replay(
                sample_batch_size=mini_batch_size,
                mini_batch_length=mini_batch_length)
        return self._train(experience, num_updates, mini_batch_size,
                           mini_batch_length)

    @tf.function
    def _train(self, experience, num_updates, mini_batch_size,
               mini_batch_length):
        """Train using experience."""
        experience = nest_utils.params_to_distributions(
            experience, self.experience_spec)
        experience = self.transform_timestep(experience)
        experience = self.preprocess_experience(experience)
        experience = nest_utils.distributions_to_params(experience)

        length = experience.step_type.shape[1]
        mini_batch_length = (mini_batch_length or length)
        assert length % mini_batch_length == 0, (
            "length=%s not a multiple of mini_batch_length=%s" %
            (length, mini_batch_length))

        if len(tf.nest.flatten(
                self.train_state_spec)) > 0 and not self._use_rollout_state:
            if mini_batch_length == 1:
                logging.fatal(
                    "Should use TrainerConfig.use_rollout_state=True "
                    "for off-policy training of RNN when minibatch_length==1.")
            else:
                common.warning_once(
                    "Consider using TrainerConfig.use_rollout_state=True "
                    "for off-policy training of RNN.")

        experience = tf.nest.map_structure(
            lambda x: tf.reshape(
                x, common.concat_shape([-1, mini_batch_length],
                                       tf.shape(x)[2:])), experience)

        batch_size = tf.shape(experience.step_type)[0]
        mini_batch_size = (mini_batch_size or batch_size)

        def _make_time_major(nest):
            """Put the time dim to axis=0."""
            return tf.nest.map_structure(lambda x: common.transpose2(x, 0, 1),
                                         nest)

        for u in tf.range(num_updates):
            if mini_batch_size < batch_size:
                indices = tf.random.shuffle(
                    tf.range(tf.shape(experience.step_type)[0]))
                experience = tf.nest.map_structure(
                    lambda x: tf.gather(x, indices), experience)
            for b in tf.range(0, batch_size, mini_batch_size):
                batch = tf.nest.map_structure(
                    lambda x: x[b:tf.minimum(batch_size, b + mini_batch_size)],
                    experience)
                batch = _make_time_major(batch)
                training_info, loss_info, grads_and_vars = self._update(
                    batch,
                    weight=tf.cast(tf.shape(batch.step_type)[1], tf.float32) /
                    float(mini_batch_size))
                common.get_global_counter().assign_add(1)
                self.training_summary(training_info, loss_info, grads_and_vars)

        self.metric_summary()
        train_steps = batch_size * mini_batch_length * num_updates
        return train_steps

    def _update(self, experience, weight):
        batch_size = tf.shape(experience.step_type)[1]
        counter = tf.zeros((), tf.int32)
        initial_train_state = common.get_initial_policy_state(
            batch_size, self.train_state_spec)
        if self._use_rollout_state:
            first_train_state = tf.nest.map_structure(
                lambda state: state[0, ...], experience.state)
        else:
            first_train_state = initial_train_state
        num_steps = tf.shape(experience.step_type)[0]

        def create_ta(s):
            # TensorArray cannot use Tensor (batch_size) as element_shape
            ta_batch_size = experience.step_type.shape[1]
            return tf.TensorArray(
                dtype=s.dtype,
                size=num_steps,
                element_shape=tf.TensorShape([ta_batch_size]).concatenate(
                    s.shape))

        experience_ta = tf.nest.map_structure(
            create_ta,
            nest_utils.to_distribution_param_spec(
                self.processed_experience_spec))
        experience_ta = tf.nest.map_structure(
            lambda elem, ta: ta.unstack(elem), experience, experience_ta)
        info_ta = tf.nest.map_structure(
            create_ta,
            nest_utils.to_distribution_param_spec(self.train_step_info_spec))

        def _train_loop_body(counter, policy_state, info_ta):
            exp = tf.nest.map_structure(lambda ta: ta.read(counter),
                                        experience_ta)
            exp = nest_utils.params_to_distributions(
                exp, self.processed_experience_spec)
            policy_state = common.reset_state_if_necessary(
                policy_state, initial_train_state,
                tf.equal(exp.step_type, StepType.FIRST))

            policy_step = self.train_step(exp, policy_state)

            info_ta = tf.nest.map_structure(
                lambda ta, x: ta.write(counter, x), info_ta,
                nest_utils.distributions_to_params(policy_step.info))

            counter += 1

            return [counter, policy_step.state, info_ta]

        with tf.GradientTape(
                persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.trainable_variables)
            [_, _, info_ta] = tf.while_loop(
                cond=lambda counter, *_: tf.less(counter, num_steps),
                body=_train_loop_body,
                loop_vars=[counter, first_train_state, info_ta],
                back_prop=True,
                name="train_loop")
            info = tf.nest.map_structure(lambda ta: ta.stack(), info_ta)
            info = nest_utils.params_to_distributions(
                info, self.train_step_info_spec)
            experience = nest_utils.params_to_distributions(
                experience, self.processed_experience_spec)
            training_info = TrainingInfo(
                action=experience.action,
                reward=experience.reward,
                discount=experience.discount,
                step_type=experience.step_type,
                rollout_info=experience.rollout_info,
                info=info,
                env_id=experience.env_id)

        loss_info, grads_and_vars = self.train_complete(
            tape=tape, training_info=training_info, weight=weight)

        del tape

        return training_info, loss_info, grads_and_vars
