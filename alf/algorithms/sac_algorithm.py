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

from collections import namedtuple
import numpy as np
import tensorflow as tf
import gin.tf
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.critic_rnn_network import CriticRnnNetwork
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.network import Network
from tf_agents.utils import common as tfa_common
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.trajectories.policy_step import PolicyStep
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm, Experience
from alf.algorithms.rl_algorithm import TrainingInfo, ActionTimeStep
from alf.utils import losses, common
from alf.algorithms.one_step_loss import OneStepTDLoss

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

SacInfo = namedtuple("SacInfo", ["actor", "critic", "alpha"])

SacLossInfo = namedtuple('SacLossInfo', ('actor', 'critic', 'alpha'))


@gin.configurable
class SacAlgorithm(OffPolicyAlgorithm):
    """Soft Actor Critic

    It's described in:
    Haarnoja et al "Soft Actor-Critic Algorithms and Applications" arXiv:1812.05905v2
    """

    def __init__(self,
                 action_spec,
                 actor_network: Network,
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
            name='log_alpha',
            initial_value=initial_log_alpha,
            dtype=tf.float32,
            trainable=True)
        super().__init__(
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
            action_distribution_spec=actor_network.output_spec,
            predict_state_spec=actor_network.state_spec,
            optimizer=[actor_optimizer, critic_optimizer, alpha_optimizer],
            get_trainable_variables_func=[
                lambda: actor_network.trainable_variables, lambda:
                (critic_network1.trainable_variables + critic_network2.
                 trainable_variables), lambda: [log_alpha]
            ],
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

        if target_entropy is None:
            flat_action_spec = tf.nest.flatten(self._action_spec)
            target_entropy = -np.sum([
                np.product(single_spec.shape.as_list())
                for single_spec in flat_action_spec
            ])
        self._target_entropy = target_entropy

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

    def predict(self, time_step: ActionTimeStep, state=None):
        action, state = self._actor_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=state)
        return PolicyStep(action=action, state=state, info=())

    def _actor_train_step(self, exp: Experience, state: SacActorState, action,
                          log_pi):
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
            loss = tf.reduce_sum(loss, axis=list(range(1, len(loss.shape))))
            return loss

        actor_loss = tf.nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss += tf.stop_gradient(tf.exp(self._log_alpha)) * log_pi

        state = SacActorState(critic1=critic1_state, critic2=critic2_state)
        info = SacActorInfo(loss=LossInfo(loss=actor_loss, extra=actor_loss))
        return state, info

    def _critic_train_step(self, exp: Experience, state: SacCriticState,
                           action, log_pi):
        critic_input = (exp.observation, exp.action)
        critic1, critic1_state = self._critic_network1(
            critic_input, step_type=exp.step_type, network_state=state.critic1)

        critic2, critic2_state = self._critic_network2(
            critic_input, step_type=exp.step_type, network_state=state.critic2)

        target_critic_input = (exp.observation, action)

        target_critic1, target_critic1_state = self._target_critic_network1(
            target_critic_input,
            step_type=exp.step_type,
            network_state=state.target_critic1)

        target_critic2, target_critic2_state = self._target_critic_network2(
            target_critic_input,
            step_type=exp.step_type,
            network_state=state.target_critic2)

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
            exp, state.actor, action, log_pi)
        critic_state, critic_info = self._critic_train_step(
            exp, state.critic, action, log_pi)
        alpha_info = self._alpha_train_step(log_pi)
        state = SacState(
            share=SacShareState(actor=share_actor_state),
            actor=actor_state,
            critic=critic_state)
        info = SacInfo(actor=actor_info, critic=critic_info, alpha=alpha_info)
        return PolicyStep(action_distribution, state, info)

    def train_complete(self,
                       tape: tf.GradientTape,
                       training_info: TrainingInfo,
                       weight=1.0):
        ret = super().train_complete(
            tape=tape, training_info=training_info, weight=weight)
        self._update_target()
        return ret

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


@gin.configurable
def create_sac_algorithm(env,
                         actor_fc_layers=(100, 100),
                         critic_fc_layers=(100, 100),
                         use_rnns=False,
                         alpha_learning_rate=5e-3,
                         actor_learning_rate=5e-3,
                         critic_learning_rate=5e-3,
                         debug_summaries=False):
    """Create a simple SacAlgorithm.

    Args:
        env (TFEnvironment): A TFEnvironment
        actor_fc_layers (list[int]): list of fc layers parameters for actor network
        critic_fc_layers (list[int]): list of fc layers parameters for critic network
        use_rnns (bool): True if rnn should be used
        alpha_learning_rate (float): learning rate for alpha
        actor_learning_rate (float) : learning rate for actor network
        critic_learning_rate (float) : learning rate for critic network
        debug_summaries (bool): True if debug summaries should be created
    """

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    if use_rnns:
        actor_net = ActorDistributionRnnNetwork(
            observation_spec,
            action_spec,
            input_fc_layer_params=actor_fc_layers,
            output_fc_layer_params=())
        critic_net = CriticRnnNetwork((observation_spec, action_spec),
                                      observation_fc_layer_params=(),
                                      action_fc_layer_params=(),
                                      output_fc_layer_params=(),
                                      joint_fc_layer_params=critic_fc_layers)
    else:
        actor_net = ActorDistributionNetwork(
            observation_spec, action_spec, fc_layer_params=actor_fc_layers)
        critic_net = CriticNetwork((observation_spec, action_spec),
                                   joint_fc_layer_params=critic_fc_layers)

    actor_optimizer = tf.optimizers.Adam(learning_rate=actor_learning_rate)
    critic_optimizer = tf.optimizers.Adam(learning_rate=critic_learning_rate)
    alpha_optimizer = tf.optimizers.Adam(learning_rate=alpha_learning_rate)
    return SacAlgorithm(
        action_spec=action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        debug_summaries=debug_summaries)
