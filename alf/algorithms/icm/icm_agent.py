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
from tf_agents.utils import eager_utils, common, value_ops
from tf_agents.specs import tensor_spec


@gin.configurable
class ICMAgent(tf_agent.TFAgent):
    def __init__(self,
                 time_step_spec,
                 action_spec,
                 value_net,
                 actor_net,
                 encoding_net,
                 forward_net,
                 inverse_net,
                 optimizer,
                 gamma=1.0,
                 gradient_clipping=None,
                 td_errors_loss_fn=None,
                 alpha=1.0,
                 eta=1.0,
                 beta=0.5,
                 entropy_regularization=None,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None,
                 name=None):
        """Creates a ICMAgent Agent.

            Args:
              time_step_spec: A `TimeStep` spec of the expected time_steps.
              action_spec: A nest of BoundedTensorSpec representing the actions.
              value_net: A tf_agents.network.Network to be used by the agent to approximate values
                    The network will be called with call(observation, step_type).
              actor_net: A tf_agents.network.Network to be used by the agent
                    The network will be called with call(observation, step_type).
              encoding_net: A tf_agents.network.Network to be used by the agent to encode observation
                    The network will be called with call(observation, step_type).
              forward_net:  A tf_agents.network.Network to be used by the agent to predict next_features
                    The network will be called with call((feature, actions), step_type).
              inverse_net:  A tf_agents.network.Network to be used by the agent to predict actions taken
                    The network will be called with call((feature, next_feature), step_type).
              optimizer: Optimizer for networks.
              gamma: A discount factor for future rewards.
              gradient_clipping: Norm length to clip gradients.
              td_errors_loss_fn: A function for computing the TD errors loss. If None,
                    a default value of elementwise huber_loss is used.
              alpha: Coefficient for reinforce loss (a2c)
              eta:  Coefficient for intrinsic rewards.
              beta: Coefficient for forward and inverse loss term
              entropy_regularization: Coefficient for entropy  loss term
              debug_summaries: A bool to gather debug summaries.
              summarize_grads_and_vars: If True, gradient and network variable summaries
                    will be written during training.
              train_step_counter: An optional counter to increment every time the train
                    op is run.  Defaults to the global_step.
              name: The name of this agent. All variables in this module will fall under
                that name. Defaults to the class name.
            """
        tf.Module.__init__(self, name=name)

        self._actor_net = actor_net
        self._value_net = value_net
        self._encoding_net = encoding_net
        self._forward_net = forward_net
        self._inverse_net = inverse_net
        self._optimizer = optimizer
        self._gradient_clipping = gradient_clipping
        self._td_errors_loss_fn = td_errors_loss_fn or common.element_wise_huber_loss
        self._gamma = gamma
        self._alpha = alpha
        self._eta = eta
        self._beta = beta
        self._debug_summaries = debug_summaries
        self._entropy_regularization = entropy_regularization

        policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_net,
            clip=True)

        super(ICMAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy=policy,
            collect_policy=policy,
            train_sequence_length=None,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)

    def _train(self, experience, weights=None):
        # [BxTxD] -> [BxTxD]
        transitions = trajectory.to_transition(experience)
        time_steps, policy_steps, next_time_steps = transitions
        actions = policy_steps.action
        with tf.GradientTape() as tape:
            loss_info = self._loss(
                time_steps, actions, next_time_steps, weights)
        grad_variables = self._value_net.variables + \
                         self._actor_net.variables + \
                         self._encoding_net.variables + \
                         self._forward_net.variables + \
                         self._inverse_net.variables
        grads = tape.gradient(loss_info.loss, grad_variables)
        self._apply_gradients(grads, grad_variables, self._optimizer)
        self.train_step_counter.assign_add(1)
        return tf.nest.map_structure(tf.identity, loss_info)

    def _loss(self, time_steps, actions, next_time_steps, weights):

        """
        alpha * (policy_loss + value_loss + entropy_coef * entropy_loss ) +
        beta * forward_loss  + (1-beta) * inverse_loss

        """

        a2c_loss = self._a2c_loss(
            time_steps, actions, next_time_steps, weights)
        forward_inverse_loss = self._forward_inverse_loss(
            time_steps, actions, next_time_steps, weights)
        total_loss = self._alpha * a2c_loss + forward_inverse_loss
        self._summary('Losses', ('total_loss', total_loss))
        return tf_agent.LossInfo(total_loss, {})

    def _calculate_values_and_returns(self, time_steps, actions, next_time_steps, weights):
        features, _ = self._encoding_net(time_steps.observation)
        next_features, _ = self._encoding_net(next_time_steps.observation)
        next_features_pred, _ = self._forward_net((features, actions))
        intrinsic_rewards = tf.reduce_sum(
            tf.square(next_features - next_features_pred), axis=-1)
        intrinsic_rewards = tf.stop_gradient(intrinsic_rewards)
        intrinsic_rewards = 0.5 * self._eta * intrinsic_rewards

        values = self._value_net(time_steps.observation)[0]
        final_values = self._value_net(next_time_steps.observation[:, -1])[0]
        episode_mask = common.get_episode_mask(next_time_steps)
        discounts = next_time_steps.discount * self._gamma
        discounts *= episode_mask

        rewards = next_time_steps.reward + intrinsic_rewards
        returns = value_ops.discounted_return(
            rewards,
            discounts=discounts,
            final_value=final_values,
            time_major=False)
        if self._debug_summaries:
            self._summary('Infos', [
                ('extrinsic_rewards', next_time_steps.reward),
                ('intrinsic_rewards', intrinsic_rewards),
                ('values', values),
                ('returns', returns)])
        return values, returns

    def _forward_inverse_loss(self, time_steps, actions, next_time_steps, weights):
        features, _ = self._encoding_net(time_steps.observation)
        next_features, _ = self._encoding_net(next_time_steps.observation)

        next_features_pred, _ = self._forward_net(
            (tf.stop_gradient(features), actions))
        action_pred, _ = self._inverse_net((features, next_features))
        valid_mask = tf.cast(~time_steps.is_last(), tf.float32)

        # forward  loss
        forward_loss = tf.square(next_features_pred - tf.stop_gradient(next_features))
        forward_loss = 0.5 * tf.reduce_sum(forward_loss, axis=-1)
        forward_loss = tf.reduce_mean(valid_mask * forward_loss)
        forward_loss = self._beta * forward_loss

        action_spec = tf.nest.flatten(self._action_spec)[0]
        if (tensor_spec.is_discrete(action_spec)):
            actions = tf.expand_dims(actions, -1)
            inverse_loss = tf.keras.losses.categorical_crossentropy(
                actions, action_pred)
            # tf.reduce_sum(inverse_loss, axis=-1) # multi dims
        else:
            inverse_loss = tf.square(actions - action_pred)
            inverse_loss = tf.reduce_sum(inverse_loss, axis=-1)
        inverse_loss = tf.reduce_mean(valid_mask * inverse_loss)

        inverse_loss = (1 - self._beta) * inverse_loss
        self._summary('Losses', [
            ('forward_loss', forward_loss),
            ('inverse_loss', inverse_loss)])

        return forward_loss + inverse_loss

    def _a2c_loss(self, time_steps, actions, next_time_steps, weights):
        values, returns = self._calculate_values_and_returns(
            time_steps, actions, next_time_steps, weights)
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
            self._summary('Infos', ('advantage', advantage))

        self._summary('Losses', [
            ('policy_loss', policy_loss),
            ('value_loss', value_loss),
            ('entropy_loss', entropy_loss)])

        return loss

    def _summary(self, name_scope, key_values):
        if not isinstance(key_values, list):
            key_values = [key_values]
        with tf.name_scope(name_scope + '/'):
            for (name, data) in key_values:
                tf.compat.v2.summary.scalar(
                    name=name, data=tf.reduce_mean(data),
                    step=self.train_step_counter)

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
