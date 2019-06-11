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

from tensorflow_probability import distributions as tfd

from tf_agents.networks.network import Network
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.utils import common as tfa_common
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm, Experience
from alf.drivers.policy_driver import ActionTimeStep
from alf.utils import losses, common

DdpgLossInfo = namedtuple('DdpgLossInfo', ('actor_loss', 'critic_loss'))

DDPGCriticInfo = namedtuple("DDPGCriticInfo", ["q_value", "target_q_value"])
DDPGCriticState = namedtuple("DDPGCriticState",
                             ['critic', 'target_actor', 'target_critic'])
DDPGInfo = namedtuple("DDPGInfo", ["actor", "critic"])
DDPGState = namedtuple("DDPGState", ['actor', 'critic'])
DDPGActorState = namedtuple("DDPGActorState", ['actor', 'critic'])


@gin.configurable
class DdpgAlgorithm(off_policy_algorithm.OffPolicyAlgorithm):
    def __init__(self,
                 action_spec,
                 actor_network: Network,
                 critic_network: Network,
                 gamma=0.99,
                 ou_stddev=1.0,
                 ou_damping=1.0,
                 td_errors_loss_fn=losses.element_wise_squared_loss,
                 actor_loss_weight=0.1,
                 target_update_tau=0.05,
                 target_update_period=1,
                 optimizer=None,
                 gradient_clipping=None,
                 train_step_counter=None,
                 debug_summaries=False,
                 name="DdpgAlgorithm"):
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
        self._critic_network = critic_network

        self._target_actor_network = actor_network.copy(
            name='target_actor_network')
        self._target_critic_network = critic_network.copy(
            name='target_critic_network')

        self._actor_variables = self._actor_network.variables
        self._critic_variables = self._critic_network.variables

        self._ou_stddev = ou_stddev
        self._ou_damping = ou_damping

        self._gamma = gamma
        self._td_errors_loss_fn = td_errors_loss_fn

        self._ou_process = self._create_ou_process(ou_stddev, ou_damping)
        self._update_target = self._get_target_updater(target_update_tau,
                                                       target_update_period)

        self._actor_loss_weight = actor_loss_weight
        tfa_common.soft_variables_update(
            self._critic_network.variables,
            self._target_critic_network.variables,
            tau=1.0)
        tfa_common.soft_variables_update(
            self._actor_network.variables,
            self._target_actor_network.variables,
            tau=1.0)

    def predict(self, time_step: ActionTimeStep, state=None):
        action, state = self._actor_network(time_step.observation,
                                            time_step.step_type, state)
        action = tf.nest.map_structure(lambda a, ou: a + ou(), action,
                                       self._ou_process)
        return PolicyStep(action=action, state=state)

    # def train_step(self, time_step=None, state=None):
    #     action, state = self._actor_network(
    #         time_step.observation, time_step.step_type, state)

    #     action = tf.nest.map_structure(
    #         lambda a, ou: a + ou(), action, self._ou_process)
    #     return PolicyStep(action=action, state=state)

    def critic_train_step(self, exp: Experience, state: DDPGCriticState):
        target_action, target_actor_state = self._target_actor_network(
            exp.observation, step_type=exp.step_type, state=state.target_actor)
        target_q_value, target_critic_state = self._target_critic_network(
            (exp.observation, target_action), state=state.target_critic)

        q_value, critic_state = self._critic_network(
            (exp.observation, exp.action), state=state.critic)

        state = DDPGCriticState(
            critic=critic_state,
            target_actor=target_actor_state,
            target_critic=target_critic_state)
        info = DDPGCriticInfo(q_value=q_value, target_q_value=target_q_value)

        return state, info

    def actor_train_step(self, exp: Experience, state: DDPGActorState):
        action, actor_state = self._actor_network(
            exp.observation, exp.step_type, state=state.actor)

        with tf.GradientTape(watch_accessed_variables=False) as atape:
            tape.watch(action)
            q_value, critic_state = self._critic_network(
                (exp.observation, action), state=state.critic)

        dqda = atape.gradient(actor_q_value, action)

        def actor_loss_fn(dqdq, action):
            if self._dqda_clipping is not None:
                dqda = tf.clip_by_value(dqda, -self._dqda_clipping,
                                        self._dqda_clipping)
            loss = common.element_wise_squared_loss(
                tf.stop_gradient(dqda + action), action)

        actor_loss = tf.nest.map_structure(actor_loss_fn, dqdq, action)
        state = DDPGActorState(actor=actor_state, critic=critic_state)
        info = LossInfo(
            loss=tf.add_n(tf.nest.flatten(actor_loss)), extra=actor_loss)
        return state, info

    def train_step(self, exp: Experience, state: DDPGState):
        critic_state, critic_info = self.critic_train_step(
            exp=exp, state=state.critic)
        actor_state, actor_loss = self.actor_train_step(
            exp=exp, state=state.critic)
        return (DDPGState(actor=actor_state, critic=critic_state),
                DDPGInfo(critic=critic_info, actor_loss=actor_loss))

    def calc_loss(self, training_info, final_time_step, final_info):
        critic_loss = self._critic_loss(
            training_info, traininfo_info.info.critic.q_value,
            training_info.info.critic.target_q_value, final_time_step,
            final_info.critic.target_q_value)

        actor_loss = training_info.info.actor_loss

        return LossInfo(
            loss=critic_loss + self._actor_loss_weight * actor_loss.loss,
            extra=DDPGLossInfo(critic=critic_loss, actor=actor_loss.extra))

    def train_complete(self, tape: tf.GradientTape,
                       training_info: TrainingInfo,
                       final_time_step: ActionTimeStep, final_info):
        retu = super(DdpgAlgorithm, self).__train_complete(
            tape=tape,
            training_info=training_info,
            final_time_step=final_time_step,
            final_policy_step=final_policy_step)
        self._update_target()
        return ret

    def _calc_critic_loss(self, time_steps, actions, next_time_steps):
        target_actions, _ = self._target_actor_network(
            next_time_steps.observation, next_time_steps.step_type)
        target_q_values, _ = self._target_critic_network(
            (next_time_steps.observation, target_actions))
        td_targets = tf.stop_gradient(
            next_time_steps.reward +
            self._gamma * next_time_steps.discount * target_q_values)
        q_values, _ = self._critic_network((time_steps.observation, actions))

        if self._debug_summaries:
            tf.summary.scalar('target_q_value',
                              tf.reduce_mean(target_q_values))
        td_loss = self._td_errors_loss_fn(td_targets, q_values)
        critic_loss = tf.reduce_mean(td_loss)
        return critic_loss

    def _calc_actor_loss(self, time_steps):
        actions, _ = self._actor_network(time_steps.observation,
                                         time_steps.step_type)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(actions)
            q_values, _ = self._critic_network(
                (time_steps.observation, actions), time_steps.step_type)
            actions = tf.nest.flatten(actions)
        dqdas = tape.gradient([q_values], actions)
        actor_losses = []
        for dqda, action in zip(dqdas, actions):
            actor_losses.append(tf.reduce_mean(-dqda * action))
        actor_loss = self._actor_loss_weight * tf.add_n(actor_losses)
        return actor_loss

    def _create_ou_process(self, ou_stddev, ou_damping):
        # todo with seed None
        seed_stream = tfd.SeedStream(seed=None, salt='ou_noise')

        def _create_ou_process(action_spec):
            return tfa_common.OUProcess(
                lambda: tf.zeros(action_spec.shape, dtype=action_spec.dtype),
                ou_damping,
                ou_stddev,
                seed=seed_stream())

        ou_process = tf.nest.map_structure(_create_ou_process,
                                           self._action_spec)
        return ou_process

    def _get_target_updater(self, tau=1.0, period=1):
        def update():
            critic_update = tfa_common.soft_variables_update(
                self._critic_network.variables,
                self._target_critic_network.variables, tau)
            actor_update = tfa_common.soft_variables_update(
                self._actor_network.variables,
                self._target_actor_network.variables, tau)
            return tf.group(critic_update, actor_update)

        return tfa_common.Periodically(update, period,
                                       'periodic_update_targets')


@gin.configurable
class QLoss(object):
    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 debug_summaries=False):
        """
        Args:
            gamma (float): A discount factor for future rewards.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated 
                Q values and returns the loss for each element of the batch.
        """
        self._gamma = gamma
        self._td_error_loss_fn = td_error_loss_fn
        self._debug_summaries = debug_summaries

    def __call__(self, training_info: TrainingInfo, value, target_value,
                 final_time_step, final_target_value):
        returns = value_ops.one_step_discounted_return(
            rewards=training_info.reward,
            values=target_value,
            step_types=training_info.step_type,
            discounts=training_info.discount * self._gamma,
            final_value=final_target_value,
            final_time_step=final_time_step)
        return self._td_error_loss_fn(tf.stop_gradient(returns), value)
