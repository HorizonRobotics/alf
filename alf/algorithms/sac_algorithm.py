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
"""Soft Actor Critic Algorithm."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gin.tf

from tf_agents.specs import tensor_spec
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.critic_rnn_network import CriticRnnNetwork
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks.q_rnn_network import QRnnNetwork
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.network import Network, DistributionNetwork
from tf_agents.utils import common as tfa_common

from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import ActionTimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import PolicyStep, TrainingInfo
from alf.utils import losses, common, dist_utils

SacShareState = namedtuple("SacShareState", ["actor"])

SacActorState = namedtuple("SacActorState", ["critic1", "critic2"])

SacCriticState = namedtuple(
    "SacCriticState",
    ["critic1", "critic2", "target_critic1", "target_critic2"])

SacState = namedtuple("SacState", ["share", "actor", "critic"])

SacActorInfo = namedtuple("SacActorInfo", ["loss"])

SacCriticInfo = namedtuple("SacCriticInfo",
                           ["critic1", "critic2", "target_critic"])

SacAlphaInfo = namedtuple("SacAlphaInfo", ["loss"])

SacInfo = namedtuple(
    "SacInfo", ["action_distribution", "actor", "critic", "alpha"],
    default_value=())

SacLossInfo = namedtuple('SacLossInfo', ('actor', 'critic', 'alpha'))


@gin.configurable
class SacAlgorithm(OffPolicyAlgorithm):
    """Soft Actor Critic

    It's described in:
    Haarnoja et al "Soft Actor-Critic Algorithms and Applications" arXiv:1812.05905v2

    There are 3 points different with `tf_agents.agents.sac.sac_agent`:

    1. To reduce computation, here we sample actions only once for calculating
    actor, critic, and alpha loss while `tf_agents.agents.sac.sac_agent` samples
    actions for each loss. This difference has little influence on the training
    performance.

    2. We calculate losses for every sampled steps.
    (s_t, a_t), (s_{t+1}, a_{t+1}) in sampled transition are used to calculate
    actor, critic and alpha loss while `tf_agents.agents.sac.sac_agent` only
    uses (s_t, a_t) and critic loss for s_{t+1} is 0. You should handle this
    carefully, it is equivalent to applying a coefficient of 0.5 on the critic
    loss.

    3. We mask out `StepType.LAST` steps when calculating losses but
    `tf_agents.agents.sac.sac_agent` does not. We believe the correct
    implementation should mask out LAST steps. And this may make different
    performance on same tasks.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 actor_network: DistributionNetwork,
                 critic_network: Network,
                 critic_loss=None,
                 target_entropy=None,
                 initial_log_alpha=0.0,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 gradient_clipping=None,
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
            target_entropy (float|None): The target average policy entropy, for updating alpha.
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
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        critic_network1 = critic_network
        critic_network2 = critic_network.copy(name='CriticNetwork2')
        log_alpha = tfa_common.create_variable(
            name='log_alpha',
            initial_value=initial_log_alpha,
            dtype=tf.float32,
            trainable=True)
        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=SacState(
                share=SacShareState(actor=actor_network.state_spec),
                actor=SacActorState(
                    critic1=critic_network.state_spec,
                    critic2=critic_network.state_spec),
                critic=SacCriticState(
                    critic1=critic_network.state_spec,
                    critic2=critic_network.state_spec,
                    target_critic1=critic_network.state_spec,
                    target_critic2=critic_network.state_spec)),
            optimizer=[actor_optimizer, critic_optimizer, alpha_optimizer],
            trainable_module_sets=[[actor_network],
                                   [critic_network1, critic_network2],
                                   [log_alpha]],
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            name=name)

        self._log_alpha = log_alpha
        self._actor_network = actor_network
        self._critic_network1 = critic_network1
        self._critic_network2 = critic_network2
        self._target_critic_network1 = self._critic_network1.copy(
            name='target_critic_network1')
        self._target_critic_network2 = self._critic_network2.copy(
            name='target_critic_network2')
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._alpha_optimizer = alpha_optimizer

        if critic_loss is None:
            critic_loss = OneStepTDLoss(debug_summaries=debug_summaries)
        self._critic_loss = critic_loss

        flat_action_spec = tf.nest.flatten(self._action_spec)
        self._is_continuous = tensor_spec.is_continuous(flat_action_spec[0])
        if target_entropy is None:
            target_entropy = np.sum(
                list(
                    map(dist_utils.calc_default_target_entropy,
                        flat_action_spec)))
        self._target_entropy = target_entropy

        self._dqda_clipping = dqda_clipping

        self._update_target = common.get_target_updater(
            models=[self._critic_network1, self._critic_network2],
            target_models=[
                self._target_critic_network1, self._target_critic_network2
            ],
            tau=target_update_tau,
            period=target_update_period)

    def _predict(self,
                 time_step: ActionTimeStep,
                 state=None,
                 epsilon_greedy=1.):
        action_dist, state = self._actor_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=state.share.actor)
        empty_state = tf.nest.map_structure(lambda x: (),
                                            self.train_state_spec)
        state = empty_state._replace(share=SacShareState(actor=state))
        action = common.epsilon_greedy_sample(action_dist, epsilon_greedy)
        return PolicyStep(
            action=action,
            state=state,
            info=SacInfo(action_distribution=action_dist))

    def predict(self, time_step: ActionTimeStep, state, epsilon_greedy):
        return self._predict(time_step, state, epsilon_greedy)

    def rollout(self, time_step: ActionTimeStep, state, mode):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by SacAlgorithm")
        return self._predict(time_step, state, epsilon_greedy=1.0)

    def _actor_train_step(self, exp: Experience, state: SacActorState,
                          action_distribution, action, log_pi):

        if self._is_continuous:
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
                    dqda = tf.clip_by_value(dqda, -self._dqda_clipping,
                                            self._dqda_clipping)
                loss = 0.5 * losses.element_wise_squared_loss(
                    tf.stop_gradient(dqda + action), action)
                loss = tf.reduce_sum(
                    loss, axis=list(range(1, len(loss.shape))))
                return loss

            actor_loss = tf.nest.map_structure(actor_loss_fn, dqda, action)
            alpha = tf.stop_gradient(tf.exp(self._log_alpha))
            actor_loss += alpha * log_pi
        else:
            critic1, critic1_state = self._critic_network1(
                exp.observation,
                step_type=exp.step_type,
                network_state=state.critic1)

            critic2, critic2_state = self._critic_network2(
                exp.observation,
                step_type=exp.step_type,
                network_state=state.critic2)

            assert isinstance(
                action_distribution, tfp.distributions.Categorical), \
                "Only `tfp.distributions.Categorical` was supported, received:" + str(type(action_distribution))

            action_probs = action_distribution.probs
            log_action_probs = tf.math.log(action_probs + 1e-8)

            target_q_value = tf.stop_gradient(tf.minimum(critic1, critic2))
            alpha = tf.stop_gradient(tf.exp(self._log_alpha))
            actor_loss = tf.reduce_mean(
                action_probs * (alpha * log_action_probs - target_q_value),
                axis=-1)

        state = SacActorState(critic1=critic1_state, critic2=critic2_state)
        info = SacActorInfo(loss=LossInfo(loss=actor_loss, extra=actor_loss))
        return state, info

    def _critic_train_step(self, exp: Experience, state: SacCriticState,
                           action, log_pi):
        if self._is_continuous:
            critic_input = (exp.observation, exp.action)
            target_critic_input = (exp.observation, action)
        else:
            critic_input = exp.observation
            target_critic_input = exp.observation

        critic1, critic1_state = self._critic_network1(
            critic_input, step_type=exp.step_type, network_state=state.critic1)

        critic2, critic2_state = self._critic_network2(
            critic_input, step_type=exp.step_type, network_state=state.critic2)

        target_critic1, target_critic1_state = self._target_critic_network1(
            target_critic_input,
            step_type=exp.step_type,
            network_state=state.target_critic1)

        target_critic2, target_critic2_state = self._target_critic_network2(
            target_critic_input,
            step_type=exp.step_type,
            network_state=state.target_critic2)

        if not self._is_continuous:
            exp_action = tf.cast(exp.action, tf.int32)
            critic1 = tfa_common.index_with_actions(critic1, exp_action)
            critic2 = tfa_common.index_with_actions(critic2, exp_action)
            sampled_action = tf.cast(action, tf.int32)
            target_critic1 = tfa_common.index_with_actions(
                target_critic1, sampled_action)
            target_critic2 = tfa_common.index_with_actions(
                target_critic2, sampled_action)

        target_critic = (tf.minimum(target_critic1, target_critic2) -
                         tf.stop_gradient(tf.exp(self._log_alpha) * log_pi))

        state = SacCriticState(
            critic1=critic1_state,
            critic2=critic2_state,
            target_critic1=target_critic1_state,
            target_critic2=target_critic2_state)

        info = SacCriticInfo(
            critic1=critic1, critic2=critic2, target_critic=target_critic)

        return state, info

    def _alpha_train_step(self, log_pi):
        alpha_loss = self._log_alpha * tf.stop_gradient(-log_pi -
                                                        self._target_entropy)
        info = SacAlphaInfo(loss=LossInfo(loss=alpha_loss, extra=alpha_loss))
        return info

    def train_step(self, exp: Experience, state: SacState):
        action_distribution, share_actor_state = self._actor_network(
            exp.observation,
            step_type=exp.step_type,
            network_state=state.share.actor)
        action = tf.nest.map_structure(lambda d: d.sample(),
                                       action_distribution)
        log_pi = tfa_common.log_probability(action_distribution, action,
                                            self._action_spec)

        actor_state, actor_info = self._actor_train_step(
            exp, state.actor, action_distribution, action, log_pi)
        critic_state, critic_info = self._critic_train_step(
            exp, state.critic, action, log_pi)
        alpha_info = self._alpha_train_step(log_pi)
        state = SacState(
            share=SacShareState(actor=share_actor_state),
            actor=actor_state,
            critic=critic_state)
        info = SacInfo(
            action_distribution=action_distribution,
            actor=actor_info,
            critic=critic_info,
            alpha=alpha_info)
        return PolicyStep(action, state, info)

    def after_train(self, training_info):
        self._update_target()

    def calc_loss(self, training_info: TrainingInfo):
        critic_loss = self._calc_critic_loss(training_info)
        alpha_loss = training_info.info.alpha.loss
        actor_loss = training_info.info.actor.loss
        return LossInfo(
            loss=actor_loss.loss + critic_loss.loss + alpha_loss.loss,
            extra=SacLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss.extra))

    def _calc_critic_loss(self, training_info):
        critic_info = training_info.info.critic

        target_critic = critic_info.target_critic

        critic_loss1 = self._critic_loss(
            training_info=training_info,
            value=critic_info.critic1,
            target_value=target_critic)

        critic_loss2 = self._critic_loss(
            training_info=training_info,
            value=critic_info.critic2,
            target_value=target_critic)

        critic_loss = critic_loss1.loss + critic_loss2.loss
        return LossInfo(loss=critic_loss, extra=critic_loss)

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_network1', '_target_critic_network2']
