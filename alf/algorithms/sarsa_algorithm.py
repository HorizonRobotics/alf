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
"""SARSA Algorithm."""

import gin
import tensorflow as tf
from tf_agents.networks.network import Network, DistributionNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.critic_rnn_network import CriticRnnNetwork
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.trajectories.time_step import StepType
from tf_agents.utils import common as tfa_common

from alf.algorithms.ddpg_algorithm import create_ou_process
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.data_structures import ActionTimeStep, LossInfo, PolicyStep, TrainingInfo
from alf.data_structures import namedtuple
from alf.utils import common, dist_utils, losses
from alf.utils.summary_utils import safe_mean_hist_summary

SarsaState = namedtuple(
    'SarsaState', [
        'prev_observation', 'prev_step_type', 'actor', 'target_actor',
        'critic', 'target_critic'
    ],
    default_value=())
SarsaInfo = namedtuple(
    'SarsaInfo', ['action_distribution', 'actor_loss', 'critic', 'returns'])
SarsaLossInfo = namedtuple(
    'SarsaLossInfo', ['actor', 'critic'], default_value=())


@gin.configurable
class SarsaAlgorithm(OnPolicyAlgorithm):
    """SARSA Algorithm.

    SARSA update Q function in an online manner using the following loss:
        ||Q(s_t,a_t) - stop_gradient(r_t, \gamma * Q(s_{t+1}, a_{t+1})||^2
    See https://en.wikipedia.org/wiki/State-action-reward-state-action

    Currently, this is only implemented for continuous action problems.
    The policy is dervied by a DDPG/SAC manner by maximizing Q(a(s_t), s_t),
    where a(s_t) is the action.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 actor_network: DistributionNetwork,
                 critic_network: Network,
                 gamma=0.99,
                 ou_stddev=0.2,
                 ou_damping=0.15,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 target_update_tau=0.05,
                 target_update_period=10,
                 dqda_clipping=None,
                 gradient_clipping=None,
                 debug_summaries=False,
                 name="SarsaAlgorithm"):
        """Create an SarsaAlgorithm.

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            observation_spec (nested TensorSpec): spec for observation.
            actor_network (Network|DistributionNetwork):  The network will be
                called with call(observation, step_type). If it is DistributionNetwork
                an action will be sampled.
            critic_network (Network): The network will be called with
                call(observation, action, step_type).
            gamma (float): discount rate for reward
            ou_stddev (float): Only used for DDPG. Standard deviation for the
                Ornstein-Uhlenbeck (OU) noise added in the default collect policy.
            ou_damping (float): Only used for DDPG. Damping factor for the OU
                noise added in the default collect policy.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between [-dqda_clipping, dqda_clipping].
                Does not perform clipping if dqda_clipping == 0.
            actor_optimizer (tf.optimizers.Optimizer): The optimizer for actor.
            critic_optimizer (tf.optimizers.Optimizer): The optimizer for actor.
            gradient_clipping (float): Norm length to clip gradients.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        if isinstance(actor_network, DistributionNetwork):
            self._action_distribution_spec = actor_network.output_spec
        elif isinstance(actor_network, Network):
            self._action_distribution_spec = action_spec
        else:
            raise ValueError("Expect DistributionNetwork or Network for"
                             " `actor_network`, got %s" % type(actor_network))

        super().__init__(
            observation_spec,
            action_spec,
            predict_state_spec=SarsaState(
                prev_observation=observation_spec,
                prev_step_type=tf.TensorSpec((), tf.int32),
                actor=actor_network.state_spec),
            train_state_spec=SarsaState(
                prev_observation=observation_spec,
                prev_step_type=tf.TensorSpec((), tf.int32),
                actor=actor_network.state_spec,
                target_actor=actor_network.state_spec,
                critic=critic_network.state_spec,
                target_critic=critic_network.state_spec,
            ),
            optimizer=[actor_optimizer, critic_optimizer],
            trainable_module_sets=[[actor_network], [critic_network]],
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            name=name)
        self._actor_network = actor_network
        self._critic_network = critic_network
        self._target_actor_network = actor_network.copy(
            name='target_actor_network')
        self._target_critic_network = critic_network.copy(
            name='target_critic_network')
        self._update_target = common.get_target_updater(
            models=[self._actor_network, self._critic_network],
            target_models=[
                self._target_actor_network, self._target_critic_network
            ],
            tau=target_update_tau,
            period=target_update_period)
        self._dqda_clipping = dqda_clipping
        self._gamma = gamma
        self._ou_process = create_ou_process(action_spec, ou_stddev,
                                             ou_damping)

    def _trainable_attributes_to_ignore(self):
        return ["_target_actor_network", "_target_critic_network"]

    def _get_action(self, actor_network, time_step, state, epsilon_greedy=1.0):
        action_distribution, state = actor_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=state)
        if isinstance(actor_network, DistributionNetwork):
            action = common.epsilon_greedy_sample(action_distribution,
                                                  epsilon_greedy)
        else:

            def _sample(a, ou):
                return tf.cond(
                    tf.less(tf.random.uniform((), 0, 1),
                            epsilon_greedy), lambda: a + ou(), lambda: a)

            action = tf.nest.map_structure(_sample, action_distribution,
                                           self._ou_process)
            action_distribution = ()
        return action_distribution, action, state

    def predict(self, time_step: ActionTimeStep, state, epsilon_greedy):
        _, action, actor_state = self._get_action(
            self._actor_network, time_step, state.actor, epsilon_greedy)
        return PolicyStep(
            action=action,
            state=SarsaState(
                actor=actor_state,
                prev_observation=time_step.observation,
                prev_step_type=time_step.step_type),
            info=())

    def rollout(self, time_step: ActionTimeStep, state: SarsaState, mode):
        not_first_step = tf.not_equal(time_step.step_type, StepType.FIRST)
        prev_critic, critic_state = self._critic_network(
            inputs=(state.prev_observation, time_step.prev_action),
            step_type=state.prev_step_type,
            network_state=state.critic)
        critic_state = tf.nest.map_structure(
            lambda new_s, s: tf.where(not_first_step, new_s, s), critic_state,
            state.critic)

        action_distribution, action, actor_state = self._get_action(
            self._actor_network, time_step, state.actor)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(action)
            critic, _ = self._critic_network((time_step.observation, action),
                                             step_type=time_step.step_type,
                                             network_state=critic_state)
        dqda = tape.gradient(critic, action)

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = tf.clip_by_value(dqda, -self._dqda_clipping,
                                        self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                tf.stop_gradient(dqda + action), action)
            loss = tf.reduce_sum(loss, axis=list(range(1, len(loss.shape))))
            return loss

        actor_loss = tf.nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss = tf.add_n(tf.nest.flatten(actor_loss))

        _, target_action, target_actor_state = self._get_action(
            self._target_actor_network, time_step, state.target_actor)

        target_critic, target_critic_state = self._target_critic_network(
            (time_step.observation, target_action),
            step_type=time_step.step_type,
            network_state=state.target_critic)

        prev_return = tf.stop_gradient(time_step.reward +
                                       time_step.discount * target_critic)
        info = SarsaInfo(
            action_distribution=action_distribution,
            actor_loss=actor_loss,
            critic=prev_critic,
            returns=prev_return)

        rl_state = SarsaState(
            prev_observation=time_step.observation,
            prev_step_type=time_step.step_type,
            actor=actor_state,
            target_actor=target_actor_state,
            critic=critic_state,
            target_critic=target_critic_state)

        return PolicyStep(action, rl_state, info)

    def calc_loss(self, training_info: TrainingInfo):
        info = training_info.info  # SarsaInfo
        critic_loss = losses.element_wise_squared_loss(info.returns,
                                                       info.critic)
        not_first_step = tf.not_equal(training_info.step_type, StepType.FIRST)
        critic_loss *= tf.cast(not_first_step, tf.float32)

        def _summary():
            with self.name_scope:
                tf.summary.scalar("values", tf.reduce_mean(info.critic))
                tf.summary.scalar("returns", tf.reduce_mean(info.returns))
                safe_mean_hist_summary("td_error", info.returns - info.critic)
                tf.summary.scalar(
                    "explained_variance_of_return_by_value",
                    common.explained_variance(info.critic, info.returns))

        if self._debug_summaries:
            common.run_if(common.should_record_summaries(), _summary)

        return LossInfo(
            loss=info.actor_loss,
            # put critic_loss to scalar_loss because loss will be masked by
            # ~is_last at train_complete(). The critic_loss here should be
            # masked by ~is_first instead, which is done above.
            scalar_loss=critic_loss,
            extra=SarsaLossInfo(actor=info.actor_loss, critic=critic_loss))

    def after_train(self, training_info):
        self._update_target()
