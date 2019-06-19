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
import numpy as np
import tensorflow as tf
import gin.tf
from tf_agents.networks.network import Network
from tf_agents.utils import common as tfa_common
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.trajectories.policy_step import PolicyStep
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm, Experience
from alf.algorithms.rl_algorithm import TrainingInfo, ActionTimeStep
from alf.utils import losses, common
from alf.algorithms.one_step_loss import OneStepTDLoss

SacActorState = namedtuple(
    "SacActorState", ["actor", "critic1", "critic2"])

SacCriticState = namedtuple(
    "SacCriticState", ["actor", "critic1", "critic2",
                       "target_critic1", "target_critic2"])
SacAlphaState = namedtuple(
    "SacAlphaState", ["actor"])

SacState = namedtuple(
    "SacState", ["actor", "critic", "alpha"])

SacActorInfo = namedtuple(
    "SacActorInfo", ["log_pi", "critic1", "critic2", "loss"])

SacCriticInfo = namedtuple(
    "SacActorInfo", ["log_pi", "critic1", "critic2",
                     "target_critic1", "target_critic2"])

SacAlphaInfo = namedtuple(
    "SacAlphaInfo", ["log_pi"])

SacInfo = namedtuple(
    "SacInfo", ["actor", "critic", "alpha"])

SacLossInfo = namedtuple('SacLossInfo', ('actor', 'critic', 'alpha'))


@gin.configurable
class SacAlgorithm(OffPolicyAlgorithm):
    """Soft Actor Critic

    It's described in:
    Haarnoja et al "Soft Actor-Critic Algorithms and Applications" arXiv:1812.05905v2
    """

    def __init__(self, action_spec,
                 actor_network: Network,
                 critic_network: Network,
                 critic_loss=None,
                 initial_log_alpha=0.0,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 gradient_clipping=None,
                 train_step_counter=None,
                 debug_summaries=False,
                 name="SacAlgorithm"):
        """Create a SacAlgorithm

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            actor_network (Network): The network will be called with
                call(observation, step_type).
            critic_network (Network): The network will be called with
                call(observation, action, step_type).
            critic_loss (None|OneStepTDLoss): an object for calculating critic loss.
                If None, a default OneStepTDLoss will be used.
            initial_log_alpha (float): initial value for variable log_alpha
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between [-dqda_clipping, dqda_clipping].
                Does not perform clipping if dqda_clipping == 0.
            actor_optimizer (tf.optimizers.Optimizer): The optimizer for actor.
            critic_optimizer (tf.optimizers.Optimizer): The optimizer for critic.
            alpha_optimizer (tf.optimizers.Optimizer): The optimizer for alpha.
            gradient_clipping (float): Norm length to clip gradients.
            train_step_counter (tf.Variable): An optional counter to increment
                every time the a new iteration is started. If None, it will use
                tf.summary.experimental.get_step(). If this is still None, a
                counter will be created.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        critic_network1 = critic_network
        critic_network2 = critic_network.copy(name='CriticNetwork2')
        log_alpha = tfa_common.create_variable(
            'initial_log_alpha',
            initial_value=initial_log_alpha,
            dtype=tf.float32,
            trainable=True)
        super().__init__(
            action_spec,
            train_state_spec=SacState(
                actor=SacActorState(
                    actor=actor_network.state_spec,
                    critic1=critic_network.state_spec,
                    critic2=critic_network.state_spec),
                critic=SacCriticState(
                    actor=actor_network.state_spec,
                    critic1=critic_network.state_spec,
                    critic2=critic_network.state_spec,
                    target_critic1=critic_network.state_spec,
                    target_critic2=critic_network.state_spec),
                alpha=SacAlphaState(actor=actor_network.state_spec)),
            action_distribution_spec=actor_network.output_spec,
            predict_state_spec=actor_network.state_spec,
            optimizer=[
                actor_optimizer,
                critic_optimizer,
                alpha_optimizer],
            get_trainable_variables_func=[
                lambda: actor_network.trainable_variables,
                lambda: (critic_network1.trainable_variables +
                         critic_network2.trainable_variables),
                lambda: [log_alpha]],
            gradient_clipping=gradient_clipping,
            train_step_counter=train_step_counter,
            debug_summaries=debug_summaries,
            name=name)

        self._log_alpha = log_alpha
        self._actor_network = actor_network
        self._critic_network1 = critic_network1
        self._critic_network2 = critic_network2
        self._target_critic_network1 = self._critic_network1.copy(
            name='TargetCriticNetwork1')
        self._target_critic_network2 = self._critic_network2.copy(
            name='TargetCriticNetwork2')
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._alpha_optimizer = alpha_optimizer

        if critic_loss is None:
            critic_loss = OneStepTDLoss(debug_summaries=debug_summaries)
        self._critic_loss = critic_loss

        flat_action_spec = tf.nest.flatten(self._action_spec)
        self._target_entropy = -np.sum([
            np.product(single_spec.shape.as_list())
            for single_spec in flat_action_spec])

        self._dqda_clipping = dqda_clipping

        self._update_target = common.get_target_updater(
            models=[self._critic_network1, self._critic_network2],
            target_models=[
                self._target_critic_network1, self._target_critic_network2
            ],
            tau=target_update_tau,
            period=target_update_period)

        tfa_common.soft_variables_update(
            self._critic_network1.variables,
            self._target_critic_network1.variables,
            tau=1.0)

        tfa_common.soft_variables_update(
            self._critic_network2.variables,
            self._target_critic_network2.variables,
            tau=1.0)

    def greedy_predict(self, time_step: ActionTimeStep, state=None):
        action, state = self._actor_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=state)
        return PolicyStep(action=action, state=state, info=())

    def predict(self, time_step: ActionTimeStep, state=None):
        return self.greedy_predict(time_step=time_step, state=state)

    def _actions_and_log_probs(self, exp: Experience, state):
        action_distribution, actor_state = self._actor_network(
            exp.observation,
            step_type=exp.step_type,
            network_state=state.actor)
        action = tf.nest.map_structure(
            lambda d: d.sample(), action_distribution)
        log_pi = tfa_common.log_probability(
            action_distribution,
            tf.nest.map_structure(lambda a: tf.stop_gradient(a), action),
            self._action_spec)
        return action, log_pi, actor_state

    def _actor_train_step(self, exp: Experience, state: SacActorState):
        action, log_pi, actor_state = self._actions_and_log_probs(exp, state)
        critic_input = (exp.observation, action)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(action)
            critic1, critic1_state = self._critic_network1(
                critic_input,
                step_type=exp.step_type,
                network_state=state.critic1)

            critic2, critic2_state = self._critic_network2(
                critic_input,
                step_type=exp.step_type,
                network_state=state.critic2)

            target_q_value = tf.minimum(critic1, critic2)

        dqda = tape.gradient(target_q_value, action)

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = tf.clip_by_value(
                    dqda, -self._dqda_clipping,
                    self._dqda_clipping)
            loss = losses.element_wise_squared_loss(
                tf.stop_gradient(dqda + action), action)
            loss = tf.reduce_sum(loss, axis=loss.shape[1:])
            return loss

        actor_loss = tf.nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss += tf.stop_gradient(tf.exp(self._log_alpha)) * log_pi

        state = SacActorState(
            actor=actor_state,
            critic1=critic1_state,
            critic2=critic2_state)
        info = SacActorInfo(
            log_pi=log_pi, critic1=critic1,
            critic2=critic2, loss=LossInfo(loss=actor_loss, extra=actor_loss))
        return state, info

    def _critic_train_step(self, exp: Experience, state: SacCriticState):
        action, log_pi, actor_state = self._actions_and_log_probs(exp, state)

        critic_input = (exp.observation, exp.action)
        critic1, critic1_state = self._critic_network1(
            critic_input,
            step_type=exp.step_type,
            network_state=state.critic1)

        critic2, critic2_state = self._critic_network2(
            critic_input,
            step_type=exp.step_type,
            network_state=state.critic2)

        target_critic_input = (exp.observation, action)

        target_critic1, target_critic1_state = self._target_critic_network1(
            target_critic_input,
            step_type=exp.step_type,
            network_state=state.target_critic1)

        target_critic2, target_critic2_state = self._target_critic_network2(
            target_critic_input,
            step_type=exp.step_type,
            network_state=state.target_critic2)

        state = SacCriticState(
            actor=actor_state,
            critic1=critic1_state,
            critic2=critic2_state,
            target_critic1=target_critic1_state,
            target_critic2=target_critic2_state)

        info = SacCriticInfo(
            log_pi=log_pi,
            critic1=critic1, critic2=critic2,
            target_critic1=target_critic1,
            target_critic2=target_critic2)

        return state, info

    def _alpha_train_step(self, exp: Experience, state: SacActorState):
        action, log_pi, actor_state = self._actions_and_log_probs(exp, state)
        state = SacAlphaState(actor=actor_state)
        info = SacAlphaInfo(log_pi=log_pi)
        return state, info

    def train_step(self, exp: Experience, state: SacState):
        actor_state, actor_info = self._actor_train_step(exp, state.actor)
        critic_state, critic_info = self._critic_train_step(exp, state.critic)
        alpha_state, alpha_info = self._alpha_train_step(exp, state.alpha)
        state = SacState(actor=actor_state, critic=critic_state, alpha=alpha_state)
        info = SacInfo(actor=actor_info, critic=critic_info, alpha=alpha_info)
        return state, info

    def train_complete(self, tape: tf.GradientTape,
                       training_info: TrainingInfo,
                       final_time_step: ActionTimeStep,
                       final_info):
        ret = super().train_complete(
            tape, training_info,
            final_time_step, final_info)
        self._update_target()
        return ret

    def calc_loss(self, training_info: TrainingInfo,
                  final_time_step: ActionTimeStep, final_info: SacInfo):
        critic_loss = self._calc_critic_loss(
            training_info, final_time_step, final_info)
        alpha_loss = self._calc_alpha_loss(training_info)
        actor_loss = training_info.info.actor.loss
        return LossInfo(
            loss=actor_loss.loss + critic_loss.loss + alpha_loss.loss,
            extra=SacLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss.extra))

    def _calc_critic_loss(self, training_info, final_time_step, final_info):
        critic_info = training_info.info.critic
        final_critic_info = final_info.critic
        target_q_fn = lambda target_q1, target_q2, log_pi: (
                tf.minimum(target_q1, target_q2) -
                tf.stop_gradient(tf.exp(self._log_alpha) * log_pi))

        target_q = target_q_fn(
            critic_info.target_critic1,
            critic_info.target_critic2,
            critic_info.log_pi)

        final_target_q = target_q_fn(
            final_critic_info.target_critic1,
            final_critic_info.target_critic2,
            final_critic_info.log_pi)

        critic_loss1 = self._critic_loss(
            training_info=training_info,
            value=critic_info.critic1,
            target_value=target_q,
            final_time_step=final_time_step,
            final_target_value=final_target_q)

        critic_loss2 = self._critic_loss(
            training_info=training_info,
            value=critic_info.critic2,
            target_value=target_q,
            final_time_step=final_time_step,
            final_target_value=final_target_q)

        critic_loss = critic_loss1.loss + critic_loss2.loss
        return LossInfo(loss=critic_loss, extra=critic_loss)

    def _calc_alpha_loss(self, training_info):
        alpha_info = training_info.info.alpha
        log_pis = alpha_info.log_pi
        alpha_loss = self._log_alpha * tf.stop_gradient(-log_pis - self._target_entropy)
        return LossInfo(loss=alpha_loss, extra=alpha_loss)
