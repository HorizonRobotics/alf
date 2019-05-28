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


import collections
import gin.tf
import tensorflow as tf

import numpy as np
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.utils import common as tfa_common
from tf_agents.agents.tf_agent import LossInfo
from alf.algorithms import off_policy_algorithm
from alf.utils import losses, common
from tf_agents.utils import nest_utils

SacLossInfo = collections.namedtuple(
    "SacLossInfo", ("ActorLoss", "CriticLoss", "AlphaLoss"))


@gin.configurable
class SacAlgorithm(off_policy_algorithm.OffPolicyAlgorithm):
    def __init__(self, action_spec,
                 actor_network,
                 critic_network,
                 initial_log_alpha=0.0,
                 target_update_tau=0.005,
                 target_update_period=1,
                 gamma=0.99,
                 td_errors_loss_fn=losses.element_wise_squared_loss,
                 optimizer=None,
                 gradient_clipping=None,
                 train_step_counter=None,
                 debug_summaries=False,
                 name="SacAlgorithm"):
        super().__init__(
            action_spec,
            train_state_spec=actor_network.state_spec,
            action_distribution_spec=action_spec,
            predict_state_spec=actor_network.state_spec,
            optimizer=optimizer,
            gradient_clipping=gradient_clipping,
            train_step_counter=train_step_counter,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network
        self._critic_network1 = critic_network
        self._critic_network2 = critic_network.copy(name='CriticNetwork2')

        self._td_errors_loss_fn = td_errors_loss_fn
        self._target_critic_network1 = critic_network.copy(name='TargetCriticNetwork1')
        self._target_critic_network2 = critic_network.copy(name='TargetCriticNetwork2')

        self._log_alpha = tfa_common.create_variable(
            'initial_log_alpha',
            initial_value=initial_log_alpha,
            dtype=tf.float32,
            trainable=True)

        self._gamma = gamma

        flat_action_spec = tf.nest.flatten(self._action_spec)
        self._target_entropy = -np.sum([
            np.product(single_spec.shape.as_list())
            for single_spec in flat_action_spec])

        self._update_target = self._get_target_updater(
            target_update_tau, target_update_period)

        tfa_common.soft_variables_update(
            self._critic_network1.variables,
            self._target_critic_network1.variables,
            tau=1.0)

        tfa_common.soft_variables_update(
            self._critic_network2.variables,
            self._target_critic_network2.variables,
            tau=1.0)

    def train_step(self, time_step=None, state=None):
        action_distribution, state = self._actor_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=state)
        return PolicyStep(action=action_distribution, state=state, info=())

    def train_complete(self, experience=None):
        time_steps, actions, next_time_steps = common.to_transitions(
            experience, self._actor_network.state_spec)

        # calc critic loss
        critic_variables = self._critic_network1.variables + \
                           self._critic_network2.variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(critic_variables)
            critic_loss = self._calc_critic_loss(
                time_steps, actions, next_time_steps)
        critic_grads = tape.gradient(critic_loss, critic_variables)
        # calc actor loss
        actor_variables = self._actor_network.variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(actor_variables)
            actor_loss = self._calc_actor_loss(time_steps)
        actor_grads = tape.gradient(actor_loss, actor_variables)
        # calc alpha loss
        alpha_variables = [self._log_alpha]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(alpha_variables)
            alpha_loss = self._cal_alpha_loss(time_steps)
        alpha_grads = tape.gradient(alpha_loss, alpha_variables)
        # apply gradients
        grads_and_vars = tuple(zip(
            critic_grads + actor_grads + alpha_grads,
            critic_variables + actor_variables + alpha_variables))
        self._optimizer.apply_gradients(grads_and_vars)
        self._update_target()

        loss = critic_loss + actor_loss + alpha_loss
        loss_info = LossInfo(loss, SacLossInfo(critic_loss, actor_loss, alpha_loss))

        return loss_info, grads_and_vars

    def _calc_critic_loss(self, time_steps, actions, next_time_steps):
        next_actions, next_log_pis = self._actions_and_log_probs(next_time_steps)
        target_q_value_input = (next_time_steps.observation, next_actions)
        target_q_values1, _ = self._target_critic_network1(
            target_q_value_input, next_time_steps.step_type)
        target_q_values2, _ = self._target_critic_network2(
            target_q_value_input, next_time_steps.step_type)
        target_q_values = (tf.minimum(target_q_values1, target_q_values2) -
                           tf.exp(self._log_alpha) * next_log_pis)
        td_targets = tf.stop_gradient(
            next_time_steps.reward +
            self._gamma * next_time_steps.discount * target_q_values)
        q_value_input = (time_steps.observation, actions)
        q_values1, _ = self._critic_network1(q_value_input, time_steps.step_type)
        q_values2, _ = self._critic_network2(q_value_input, time_steps.step_type)
        critic_loss1 = self._td_errors_loss_fn(q_values1, td_targets)
        critic_loss2 = self._td_errors_loss_fn(q_values2, td_targets)
        valid_masks = tf.cast(~time_steps.is_last(), tf.float32)
        critic_loss = tf.reduce_mean(valid_masks * (critic_loss1 + critic_loss2))
        if self._debug_summaries:
            tf.summary.scalar('target_q_value', tf.reduce_mean(target_q_values))
        return critic_loss

    def _calc_actor_loss(self, time_steps):
        actions, log_pis = self._actions_and_log_probs(time_steps)
        q_values_input = (time_steps.observation, actions)
        q_values1, _ = self._critic_network1(q_values_input, time_steps.step_type)
        q_values2, _ = self._critic_network2(q_values_input, time_steps.step_type)
        q_values = tf.minimum(q_values1, q_values2)
        actor_loss = tf.exp(self._log_alpha) * log_pis - q_values
        valid_masks = tf.cast(~time_steps.is_last(), tf.float32)
        actor_loss = tf.reduce_mean(valid_masks * actor_loss)
        return actor_loss

    def _cal_alpha_loss(self, time_steps):
        _, log_pis = self._actions_and_log_probs(time_steps)
        valid_masks = tf.cast(~time_steps.is_last(), tf.float32)
        alpha_loss = (self._log_alpha * tf.stop_gradient(
            -log_pis - self._target_entropy))
        alpha_loss = tf.reduce_mean(alpha_loss * valid_masks)
        return alpha_loss

    def _actions_and_log_probs(self, time_steps):
        action_distribution, _ = self._actor_network(
            time_steps.observation, time_steps.step_type, None)
        actions = tf.nest.map_structure(lambda d: d.sample(), action_distribution)
        log_pi = tfa_common.log_probability(
            action_distribution, actions,
            self.action_spec)
        return actions, log_pi

    def _get_target_updater(self, tau=1.0, period=1):
        def update():
            critic_network1_update = tfa_common.soft_variables_update(
                self._critic_network1.variables,
                self._target_critic_network1.variables, tau)
            critic_network2_update = tfa_common.soft_variables_update(
                self._critic_network2.variables,
                self._target_critic_network2.variables, tau)
            return tf.group([critic_network1_update, critic_network2_update])

        return tfa_common.Periodically(update, period)
