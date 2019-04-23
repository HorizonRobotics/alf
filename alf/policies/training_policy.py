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

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.specs.distribution_spec import nested_distributions_from_specs
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.policies import tf_policy
from tf_agents.utils import eager_utils

from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.policies.policy_training_info import TrainingInfo
from alf.utils.common import add_loss_summaries, get_distribution_params
from alf.utils.common import reset_state_if_necessary


@gin.configurable
class TrainingPolicy(tf_policy.Base):
    """A policy which also does training.

    To use it, you need to implement an OnPolicyAlgorithm and pass as argument
    to the constructor.

    See example/actor_critic.py for example.
    """

    def __init__(self,
                 algorithm: OnPolicyAlgorithm,
                 time_step_spec: TimeStep,
                 training=True,
                 train_interval=4,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None):
        """Create a TrainingPolicy

        Args:
          algorithm (OnPolicyAlgorithm): the algorithm this policy will use.
          time_step_spec: A `TimeStep` spec of the expected time_steps.
          training (bool): If True, will perform training by calling
            algorithm.train_step() at every step, otherwise it will call
            algorithm.predict() at every step.
          train_interval (int): number of steps for each iteration. It will call
            algorithm.train_complete() every so many steps.
          debug_summaries: A bool to gather debug summaries.
          summarize_grads_and_vars: If True, gradient and network variable
            summaries will be written during training.
          train_step_counter: An optional counter to increment every time the
            a new iteration is started.
        """
        self._algorithm = algorithm
        self._training = training
        self._steps = 0
        self._train_interval = train_interval
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars

        if train_step_counter is None:
            train_step_counter = tf.Variable(0, trainable=False)
        self._train_step_counter = train_step_counter

        self._action_distribution_spec = algorithm.action_distribution_spec

        self._batch_size = None
        self._initial_state = None

        if self._training:
            policy_state_spec = algorithm.train_state_spec
            self._tape = tf.GradientTape()
            self._new_iter()
        else:
            policy_state_spec = algorithm.predict_state_spec

        super(TrainingPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=algorithm.action_spec,
            policy_state_spec=policy_state_spec)

    def _get_initial_state(self, batch_size):
        if batch_size == self._batch_size and self._initial_state is not None:
            return self._initial_state
        self._initial_state = super(TrainingPolicy,
                                    self)._get_initial_state(batch_size)
        self._batch_size = batch_size
        return self._initial_state

    def _action(self, time_step: TimeStep, policy_state, seed):
        if self._training:
            return self._train(time_step, policy_state, seed)
        else:
            policy_step = self._algorithm.predict(
                time_step, state=policy_state)
            action = self._sample_action_distribution(policy_step.action, seed)
            return policy_step._replace(action=action)

    def _sample_action_distribution(self, action_distribution, seed):
        seed_stream = tfp.distributions.SeedStream(seed=seed, salt='ac_policy')
        return tf.nest.map_structure(lambda d: d.sample(seed=seed_stream()),
                                     action_distribution)

    def _new_iter(self):
        """Start a new training iteration"""
        self._train_step_counter.assign_add(1)
        self._training_info = []

    def _train(self, time_step: TimeStep, policy_state, seed):
        self._steps += 1
        finish_train = len(self._training_info) == self._train_interval
        batch_size = time_step.reward.shape[0]

        policy_state = reset_state_if_necessary(
            policy_state, self.get_initial_state(batch_size),
            time_step.is_first())

        if finish_train:
            tape = self._tape
            self._tape = tf.GradientTape()

        with self._tape:
            policy_step = self._algorithm.train_step(
                time_step, state=policy_state)

        if finish_train:
            self._train_complete(tape, time_step, policy_step)
            self._new_iter()

        action_distribution = policy_step.action
        action = self._sample_action_distribution(action_distribution, seed)

        action_distribution_param = get_distribution_params(
            action_distribution)

        self._fill_reward_and_discount(time_step)

        self._training_info.append(
            TrainingInfo(
                action_distribution=action_distribution_param,
                action=action,
                next_reward=None,
                next_discount=None,
                step_type=time_step.step_type,
                info=policy_step.info))

        return policy_step._replace(action=action, info=())

    def _fill_reward_and_discount(self, time_step):
        """Fill the reward and discount of the last training_info
        """
        if self._training_info:
            info = self._training_info[-1]
            discount = time_step.discount
            discount *= 1 - tf.cast(time_step.is_last(), tf.float32)
            self._training_info[-1] = info._replace(
                next_discount=discount, next_reward=time_step.reward)

    def _train_complete(self, tape, final_time_step, final_policy_step):
        self._fill_reward_and_discount(final_time_step)

        with tape:
            training_info = tf.nest.map_structure(lambda *args: tf.stack(args),
                                                  *self._training_info)

            action_distribution = nested_distributions_from_specs(
                self._action_distribution_spec,
                training_info.action_distribution)

            training_info = training_info._replace(
                action_distribution=action_distribution)

        loss_info, grads_and_vars = self._algorithm.train_complete(
            tape, training_info, final_time_step, final_policy_step)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self._train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self._train_step_counter)
        if self._debug_summaries:
            add_loss_summaries(loss_info, self._train_step_counter)
