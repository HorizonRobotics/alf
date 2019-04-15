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

import gin
import tensorflow as tf
from tf_agents.environments import trajectory
from tf_agents.agents import tf_agent
from tf_agents.policies import actor_policy
from tf_agents.utils import eager_utils, common
from tf_agents.utils import value_ops


@gin.configurable
class A2CAgent(tf_agent.TFAgent):
    def __init__(self,
                 time_step_spec,
                 action_spec,
                 value_net,
                 actor_net,
                 optimizer,
                 gamma=1.0,
                 gradient_clipping=None,
                 td_errors_loss_fn=None,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 entropy_regularization=None,
                 train_step_counter=None,
                 name=None):
        """Creates a A2C Agent.

            Args:
                time_step_spec: A `TimeStep` spec of the expected time_steps.
                action_spec: A nest of BoundedTensorSpec representing the actions.
                value_net: A tf_agents.network.Network to be used by the agent. The
                    network will be called with call(observation, step_type).
                actor_net: A tf_agents.network.Network to be used by the agent. The
                    network will be called with call(observation, step_type).
                optimizer: Optimizer for networks.
                gamma: A discount factor for future rewards.
                gradient_clipping: Norm length to clip gradients.
                debug_summaries: A bool to gather debug summaries.
                td_errors_loss_fn: A function for computing the TD errors loss. If None,
                    a default value of elementwise huber_loss is used.
                debug_summaries:  A bool to gather debug summaries.
                summarize_grads_and_vars: If True, gradient and network variable summaries
                    will be written during training.
                entropy_regularization: Coefficient for entropy regularization loss term.
                train_step_counter: An optional counter to increment every time the train
                    op is run.  Defaults to the global_step.
                name: The name of this agent. All variables in this module will fall
                    under that name. Defaults to the class name.
            """
        tf.Module.__init__(self, name=name)

        self._actor_net = actor_net
        self._value_net = value_net
        self._optimizer = optimizer
        self._gradient_clipping = gradient_clipping
        self._td_errors_loss_fn = td_errors_loss_fn or common.element_wise_huber_loss
        self._gamma = gamma
        self._debug_summaries = debug_summaries
        self._entropy_regularization = entropy_regularization

        policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_net,
            clip=True)

        super(A2CAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy=policy,
            collect_policy=policy,
            train_sequence_length=None,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)

    def _initialize(self):
        pass

    def _train(self, experience, weights=None):
        # [BxTxD] -> [BxTxD]
        transitions = trajectory.to_transition(experience)
        time_steps, policy_steps, next_time_steps = transitions
        actions = policy_steps.action

        with tf.GradientTape() as tape:
            loss_info = self._loss(
                time_steps, actions, next_time_steps, weights)
        grad_variables = self._value_net.variables + self._actor_net.variables
        grads = tape.gradient(loss_info.loss, grad_variables)
        self._apply_gradients(grads, grad_variables, self._optimizer)
        self.train_step_counter.assign_add(1)
        return tf.nest.map_structure(tf.identity, loss_info)

    def _calculate_values_and_returns(self, time_steps, next_time_steps):
        values = self._value_net(time_steps.observation)[0]
        final_values = self._value_net(next_time_steps.observation[:, -1])[0]
        # Make discount 0.0 at end of each episode to restart cumulative sum
        #   end of each episode.
        episode_mask = common.get_episode_mask(next_time_steps)
        discounts = next_time_steps.discount * self._gamma
        discounts *= episode_mask
        returns = value_ops.discounted_return(
            next_time_steps.reward,
            discounts=discounts,
            final_value=final_values,
            time_major=False)
        if self._debug_summaries:
            with tf.name_scope('Metrics/'):
                tf.compat.v2.summary.scalar(
                    name='values',
                    data=tf.reduce_mean(values),
                    step=self.train_step_counter)
                tf.compat.v2.summary.scalar(
                    name='returns',
                    data=tf.reduce_mean(returns),
                    step=self.train_step_counter)
        return values, returns

    def _loss(self, time_steps, actions, next_time_steps, weights):
        values, returns = self._calculate_values_and_returns(
            time_steps, next_time_steps)
        batch_size = (tf.compat.dimension_at_index(time_steps.discount.shape, 0) or
                      tf.shape(time_steps.discount)[0])
        policy_state = self._collect_policy.get_initial_state(batch_size=batch_size)
        actions_distribution = self.collect_policy.distribution(
            time_steps, policy_state).action
        action_log_prob = common.log_probability(
            actions_distribution, actions,
            self.action_spec)
        valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
        action_log_prob *= valid_mask
        advantage = tf.stop_gradient(returns - values)
        policy_loss = -tf.reduce_sum(action_log_prob * advantage)

        td_error = self._td_errors_loss_fn(values, returns)
        value_loss = tf.reduce_mean(td_error)

        entropy_loss = tf.constant(0.0, dtype=tf.float32)
        if self._entropy_regularization:
            entropy = common.entropy(actions_distribution, self.action_spec)
            entropy = tf.reduce_mean(-tf.cast(entropy, tf.float32))
            entropy_loss = self._entropy_regularization * entropy

        loss = policy_loss + value_loss + entropy_loss

        if self._debug_summaries:
            with tf.name_scope('Metrics/'):
                tf.compat.v2.summary.scalar(
                    name='advantage',
                    data=tf.reduce_mean(advantage),
                    step=self.train_step_counter)

        with tf.name_scope('Losses/'):
            tf.compat.v2.summary.scalar(
                name='policy_loss',
                data=policy_loss,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='value_loss',
                data=value_loss,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='entropy_loss',
                data=entropy_loss,
                step=self.train_step_counter)
        return tf_agent.LossInfo(loss, ())

    def _apply_gradients(self, gradients, variables, optimizer):
        grads_and_vars = tuple(zip(gradients, variables))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(
                grads_and_vars, self._gradient_clipping)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(
                grads_and_vars,
                self.train_step_counter)
            eager_utils.add_gradients_summaries(
                grads_and_vars,
                self.train_step_counter)

        optimizer.apply_gradients(grads_and_vars)
