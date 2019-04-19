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

from collections import namedtuple

import gin.tf

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.specs.distribution_spec import nested_distributions_from_specs
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.networks.network import Network
from tf_agents.policies import tf_policy
from tf_agents.specs.tensor_spec import TensorSpec
from tf_agents.utils import common as tfa_common
from tf_agents.utils import eager_utils
from tf_agents.utils import value_ops

from alf.utils.common import add_loss_summaries, get_distribution_params
from alf.utils.losses import element_wise_squared_loss

ActorCriticState = namedtuple("ActorCriticPolicyState",
                              ["actor_state", "value_state"])

ActorCriticLossInfo = namedtuple("ActorCriticLossInfo",
                                 ["pg_loss", "td_loss", "entropy_loss"])

TrainingInfo = namedtuple("TrainingInfo", [
    "action_distribution_param", "action", "value", "reward", "discount",
    "is_last"
])


@gin.configurable
class ActorCriticPolicy(tf_policy.Base):
    def __init__(self,
                 actor_network: Network,
                 value_network: Network,
                 time_step_spec: TimeStep,
                 action_spec,
                 training=True,
                 train_interval=4,
                 optimizer=None,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 use_gae=False,
                 td_lambda=0.95,
                 entropy_regularization=None,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None):
        """Construct an instance of ActorCriticPolicy

        Args:
          actor_network: A function that returns nested action distribution for
            each observation.
          value_network: A function that returns value tensor from neural net
            predictions for each observation. Takes nested observation and
            returns batch of value_preds.
          time_step_spec: A `TimeStep` spec of the expected time_steps.
          action_spec: A nest of BoundedTensorSpec representing the actions.
          training (bool): Whether do training
          train_interval (int): train policy every so many steps
          optimizer (tf.optimizers.Optimizer): The optimizer to use for training.
          gamma: A discount factor for future rewards.
          td_errors_loss_fn: A function for computing the TD errors loss. This
            function takes as input the target and the estimated values and
            returns the loss for each element of the batch.
          use_gae: If True (default False), uses generalized advantage
            estimation for computing per-timestep advantage. Else, just
            subtracts value predictions from empirical return.
          td_lambda: Lambda parameter for generalized advantage estimation
            computation.
          entropy_regularization: Coefficient for entropy regularization loss
            term.
          debug_summaries: A bool to gather debug summaries.
          summarize_grads_and_vars: If True, gradient and network variable
            summaries will be written during training.
          train_step_counter (tf.Variable): An optional counter to increment
            every time a new training iteration is started.
        """
        self._actor_network = actor_network
        self._value_network = value_network
        self._training = training
        if training:
            assert optimizer is not None, \
                   "optimizer need to be provided for training"
        self._optimizer = optimizer
        self._steps = 0
        self._td_loss_weight = 1.0
        self._gamma = gamma
        self._train_interval = train_interval
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = False
        self._td_error_loss_fn = td_error_loss_fn
        self._use_gae = use_gae
        self._lambda = td_lambda
        self._entropy_regularization = entropy_regularization

        if train_step_counter is None:
            train_step_counter = tf.Variable(0, trainable=False)
        self._train_step_counter = train_step_counter

        state_spec = ActorCriticState(
            actor_state=actor_network.state_spec,
            value_state=value_network.state_spec)

        if self._training:
            self._new_iter()

        super(ActorCriticPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=state_spec)

    def _action(self, time_step: TimeStep, policy_state, seed):
        if self._training:
            return self._train(time_step, policy_state, seed)

        action_distribution, actor_state = self._actor_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=policy_state)

        action = self._sample_action_distribution(action_distribution, seed)

        return PolicyStep(action=action, state=actor_state, info=())

    def _sample_action_distribution(self, action_distribution, seed):
        seed_stream = tfp.distributions.SeedStream(seed=seed, salt='ac_policy')
        return tf.nest.map_structure(lambda d: d.sample(seed=seed_stream()),
                                     action_distribution)

    def _new_iter(self):
        """Start a new training iteration"""
        self._tape = tf.GradientTape()
        self._train_step_counter.assign_add(1)
        self._training_info = []

    def _train(self, time_step: TimeStep, policy_state, seed):
        step_type = time_step.step_type
        is_last = tf.cast(time_step.is_last(), tf.float32)

        self._steps += 1

        if len(self._training_info) == self._train_interval:
            with self._tape:
                loss_info = self._calc_loss(time_step, policy_state)
            self._update(loss_info)
            self._new_iter()

        with self._tape:
            value, value_state = self._value_network(
                time_step.observation,
                step_type=step_type,
                network_state=policy_state.value_state)
            action_distribution, actor_state = self._actor_network(
                time_step.observation,
                step_type=step_type,
                network_state=policy_state.actor_state)

        action = self._sample_action_distribution(action_distribution, seed)
        action_distribution_param = get_distribution_params(
            action_distribution)

        self._fill_reward_and_discount(time_step)
        self._training_info.append(
            TrainingInfo(
                action_distribution_param=action_distribution_param,
                action=action,
                value=value,
                reward=None,
                discount=None,
                is_last=is_last))

        policy_state = ActorCriticState(
            actor_state=actor_state, value_state=value_state)

        return PolicyStep(action=action, state=policy_state, info=())

    def _fill_reward_and_discount(self, time_step):
        if self._training_info:
            info = self._training_info[-1]
            self._training_info[-1] = info._replace(
                discount=time_step.discount, reward=time_step.reward)

    def _calc_loss(self, final_time_step, policy_state):
        # TODO: refactor to allow reuse of this function

        final_value, _ = self._value_network(
            final_time_step.observation,
            step_type=final_time_step.step_type,
            network_state=policy_state.value_state)

        self._fill_reward_and_discount(final_time_step)

        training_info = tf.nest.map_structure(lambda *args: tf.stack(args),
                                              *self._training_info)

        action_distribution = nested_distributions_from_specs(
            self._actor_network.output_spec,
            training_info.action_distribution_param)

        returns = value_ops.discounted_return(
            training_info.reward, training_info.discount, final_value)

        valid_masks = 1 - training_info.is_last

        action_log_prob = tfa_common.log_probability(
            action_distribution, training_info.action, self.action_spec)

        if not self._use_gae:
            advantages = returns - training_info.value
        else:
            advantages = value_ops.generalized_advantage_estimation(
                values=training_info.value,
                final_value=final_value,
                rewards=training_info.reward,
                discounts=training_info.discount,
                td_lambda=self._lambda)

        pg_loss = -tf.stop_gradient(advantages) * action_log_prob
        pg_loss = tf.reduce_mean(pg_loss * valid_masks)

        td_loss = self._td_error_loss_fn(
            tf.stop_gradient(returns), training_info.value)
        td_loss = tf.reduce_mean(td_loss * valid_masks)

        loss = pg_loss + self._td_loss_weight * td_loss

        entropy_loss = 0
        if self._entropy_regularization is not None:
            entropies = tfa_common.entropy(action_distribution,
                                           self.action_spec)
            entropy_loss = -tf.reduce_mean(input_tensor=entropies)
            loss += self._entropy_regularization * entropy_loss

        return LossInfo(
            loss,
            ActorCriticLossInfo(
                td_loss=td_loss, pg_loss=pg_loss, entropy_loss=entropy_loss))

    def _update(self, loss_info):
        vars = self._actor_network.variables + self._value_network.variables
        grads = self._tape.gradient(loss_info.loss, vars)
        grads_and_vars = tuple(zip(grads, vars))
        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self._train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self._train_step_counter)
        self._optimizer.apply_gradients(grads_and_vars)

        if self._debug_summaries:
            add_loss_summaries(loss_info, self._train_step_counter)
