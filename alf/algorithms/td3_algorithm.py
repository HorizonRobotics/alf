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

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.networks.network import Network
from tf_agents.utils import common as tfa_common
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.actor_rnn_network import ActorRnnNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.critic_rnn_network import CriticRnnNetwork
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import ActionTimeStep, Experience, namedtuple, TrainingInfo
from alf.algorithms.one_step_loss import OneStepTDLoss, LossInfo
from alf.utils import losses, common
import gin.tf

tfpd = tfp.distributions

TD3State = namedtuple('TD3State', ["actor", "critic"])
TD3ActorState = namedtuple('TD3ActorState', ["actor", "critic"])
TD3CriticState = namedtuple(
    'TD3CriticState',
    ["actor", "critic1", "critic2", "target_critic1", "target_critic2"])
TD3CriticInfo = namedtuple("TD3CriticInfo",
                           ["critic1", "critic2", "target_critic"])
TD3Info = namedtuple("TD3Info", ["critic", "actor_loss"])
TD3LossInfo = namedtuple('TD3LossInfo', ['actor', 'critic'])


@gin.configurable
class TD3Algorithm(OffPolicyAlgorithm):
    """Twin Delayed Deep Deterministic Policy Gradients (TD3).

    Reference:
    Fujimoto et al "Addressing Function Approximation Error in Actor-Critic Methods"
    https://arxiv.org/abs/arXiv:1802.09477

    TODO: optimize the actor network every actor_update_period training steps.
        actor and critic network are optimized at the same frequency in this
        implementation
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 actor_network: Network,
                 critic_network: Network,
                 critic_loss=None,
                 exploration_policy_noise=0.1,
                 target_policy_noise=0.2,
                 target_policy_noise_clip=0.5,
                 target_update_tau=0.05,
                 target_update_period=5,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 gradient_clipping=None,
                 debug_summaries=False,
                 name="TD3Algorithm"):
        """Create a TD3Algorithm

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            actor_network (Network): The network will be called with
                call(observation, step_type).
            critic_network (Network): The network will be called with
                call(observation, action, step_type).
            critic_loss (None|OneStepTDLoss): an object for calculating critic loss.
                If None, a default OneStepTDLoss will be used.
            exploration_policy_noise (float): Scale factor on exploration policy noise.
            target_policy_noise (float): Scale factor on target action noise
            target_policy_noise_clip (float): Value to clip noise.
            target_update_tau (float): Factor for soft update of the target networks.
            target_update_period (int): Period for soft update of the target networks.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between [-dqda_clipping, dqda_clipping].
                Does not perform clipping if dqda_clipping == 0.
            actor_optimizer (tf.optimizers.Optimizer): The optimizer for actor.
            critic_optimizer (tf.optimizers.Optimizer): The optimizer for critic.
            gradient_clipping (float): Norm length to clip gradients.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        critic_network1 = critic_network
        critic_network2 = critic_network.copy(name='critic_network2')

        train_state_spec = TD3State(
            actor=TD3ActorState(
                actor=actor_network.state_spec,
                critic=critic_network.state_spec),
            critic=TD3CriticState(
                actor=actor_network.state_spec,
                critic1=critic_network.state_spec,
                critic2=critic_network.state_spec,
                target_critic1=critic_network.state_spec,
                target_critic2=critic_network.state_spec))
        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=train_state_spec,
            optimizer=[actor_optimizer, critic_optimizer],
            trainable_module_sets=[[actor_network],
                                   [critic_network1, critic_network2]],
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network
        self._critic_network1 = critic_network1
        self._critic_network2 = critic_network2

        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer

        self._target_actor_network = actor_network.copy(
            name='target_actor_network')
        self._target_critic_network1 = critic_network1.copy(
            name='target_critic_network1')
        self._target_critic_network2 = critic_network2.copy(
            name='target_critic_network2')

        if critic_loss is None:
            critic_loss = OneStepTDLoss(debug_summaries=debug_summaries)
        self._critic_loss = critic_loss

        self._exploration_noise = create_gaussian_noise(
            action_spec, scale=exploration_policy_noise)
        self._target_noise = create_gaussian_noise(
            action_spec,
            scale=target_policy_noise,
            clip_value=target_policy_noise_clip)

        self._update_target = common.get_target_updater(
            models=[
                self._actor_network, self._critic_network1,
                self._critic_network2
            ],
            target_models=[
                self._target_actor_network, self._target_critic_network1,
                self._target_critic_network2
            ],
            tau=target_update_tau,
            period=target_update_period)

        self._dqda_clipping = dqda_clipping

        tfa_common.soft_variables_update(
            self._critic_network1.variables,
            self._target_critic_network1.variables,
            tau=1.0)
        tfa_common.soft_variables_update(
            self._critic_network2.variables,
            self._target_critic_network2.variables,
            tau=1.0)
        tfa_common.soft_variables_update(
            self._actor_network.variables,
            self._target_actor_network.variables,
            tau=1.0)

    def predict(self, time_step: ActionTimeStep, state, epsilon_greedy):
        action, state = self._actor_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=state.actor.actor)
        empty_state = tf.nest.map_structure(lambda x: (),
                                            self.train_state_spec)

        def _add_noise(a, noise):
            return tf.cond(
                tf.less(tf.random.uniform((), 0, 1),
                        epsilon_greedy), lambda: a + noise(), lambda: a)

        action = tf.nest.map_structure(_add_noise, action,
                                       self._exploration_noise)
        state = empty_state._replace(
            actor=TD3ActorState(actor=state, critic=()))
        return PolicyStep(action=action, state=state, info=())

    def rollout(self,
                time_step: ActionTimeStep,
                state=None,
                mode=RLAlgorithm.ROLLOUT):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by TD3Algorithm")
        return self.predict(time_step, state, epsilon_greedy=1.0)

    def train_step(self, exp: Experience, state):
        critic_state, critic_info = self._critic_train_step(
            exp=exp, state=state.critic)
        policy_step = self._actor_train_step(exp=exp, state=state.actor)
        return policy_step._replace(
            state=TD3State(actor=policy_step.state, critic=critic_state),
            info=TD3Info(critic=critic_info, actor_loss=policy_step.info))

    def _critic_train_step(self, exp: Experience, state: TD3CriticState):
        critic_input = (exp.observation, exp.action)
        critic1, critic1_state = self._critic_network1(
            critic_input, step_type=exp.step_type, network_state=state.critic1)
        critic2, critic2_state = self._critic_network2(
            critic_input, step_type=exp.step_type, network_state=state.critic2)

        action, actor_state = self._target_actor_network(
            exp.observation, exp.step_type, network_state=state.actor)
        action = tf.nest.map_structure(lambda a, noise: a + noise(), action,
                                       self._target_noise)
        target_critic_input = (exp.observation, action)
        target_critic1, target_critic1_state = self._target_critic_network1(
            target_critic_input,
            step_type=exp.step_type,
            network_state=state.target_critic1)
        target_critic2, target_critic2_state = self._target_critic_network2(
            target_critic_input,
            step_type=exp.step_type,
            network_state=state.target_critic2)

        target_critic = tf.minimum(target_critic1, target_critic2)

        state = TD3CriticState(
            actor=actor_state,
            critic1=critic1_state,
            critic2=critic2_state,
            target_critic1=target_critic1_state,
            target_critic2=target_critic2_state)

        info = TD3CriticInfo(
            critic1=critic1, critic2=critic2, target_critic=target_critic)

        return state, info

    def _actor_train_step(self, exp: Experience, state: TD3ActorState):
        action, actor_state = self._actor_network(
            exp.observation, exp.step_type, network_state=state.actor)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(action)
            q_value, critic_state = self._critic_network1(
                (exp.observation, action), network_state=state.critic)

        dqda = tape.gradient(q_value, action)

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = tf.clip_by_value(dqda, -self._dqda_clipping,
                                        self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                tf.stop_gradient(dqda + action), action)
            loss = tf.reduce_sum(loss, axis=list(range(1, len(loss.shape))))
            return loss

        actor_loss = tf.nest.map_structure(actor_loss_fn, dqda, action)
        state = TD3ActorState(actor=actor_state, critic=critic_state)
        info = LossInfo(
            loss=tf.add_n(tf.nest.flatten(actor_loss)), extra=actor_loss)
        return PolicyStep(action=action, state=state, info=info)

    def calc_loss(self, training_info: TrainingInfo):
        critic_loss = self._calc_critic_loss(training_info)
        actor_loss = training_info.info.actor_loss

        return LossInfo(
            loss=actor_loss.loss + critic_loss.loss,
            extra=TD3LossInfo(
                actor=actor_loss.extra, critic=critic_loss.extra))

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

    def after_train(self, training_info):
        self._update_target()

    def _trainable_attributes_to_ignore(self):
        return [
            '_target_actor_network', '_target_critic_network1',
            '_target_critic_network2'
        ]


def create_gaussian_noise(action_spec, scale, clip_value=None):
    """Create gaussian noise

    Args:
        action_spec (nested BoundedTensorSpec): representing the actions.
        scale (float):  Scale factor on gaussian noise.
        clip_value (float|None): Value to clip noise. If not None, clips
            noise between [-clip_value, clip_value]
    """

    def _create_gaussian_noise(action_spec):
        distribution = tfpd.Normal(
            loc=tf.zeros(action_spec.shape, dtype=action_spec.dtype),
            scale=tf.ones(action_spec.shape, dtype=action_spec.dtype) * scale)

        def _gaussian_noise():
            noise = distribution.sample()
            if clip_value is not None:
                noise = tf.clip_by_value(noise, -clip_value, clip_value)
            return noise

        return _gaussian_noise

    return tf.nest.map_structure(_create_gaussian_noise, action_spec)
