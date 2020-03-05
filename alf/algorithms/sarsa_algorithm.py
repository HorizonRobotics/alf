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
import copy

from absl import logging
import numpy as np
import gin
import tensorflow as tf

from tf_agents.networks.network import Network, DistributionNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.critic_rnn_network import CriticRnnNetwork
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.specs import tensor_spec
from tf_agents.utils import common as tfa_common

from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.ddpg_algorithm import create_ou_process
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.data_structures import ActionTimeStep, LossInfo, PolicyStep, StepType, TrainingInfo
from alf.data_structures import Experience, experience_to_time_step, namedtuple
from alf.utils import common, dist_utils, losses, math_ops
from alf.utils.summary_utils import safe_mean_hist_summary

SarsaState = namedtuple(
    'SarsaState', [
        'prev_observation', 'prev_step_type', 'actor', 'critics',
        'target_critics'
    ],
    default_value=())
SarsaInfo = namedtuple(
    'SarsaInfo', [
        'action_distribution', 'actor_loss', 'alpha_loss', 'critics',
        'target_critics'
    ],
    default_value=())
SarsaLossInfo = namedtuple('SarsaLossInfo', ['actor', 'critic', 'alpha'])

from alf.algorithms.algorithm import Algorithm, AlgorithmStep
from alf.utils.averager import AdaptiveAverager


class FastCriticBias(tf.Module):
    """
    FastCriticBias estimates expected return `b` by minimizing the following
    expectation:
        min_b E(b - r - discount * b)^2
    where r is reward. `b` can be analytically solved as:
        b = E(r) / (1 - E(discount))

    We explicitly enforce that E(critic - b) = 0 by estimating E(critic) and
    use b - E(critic) as bias to update critic.

    Pseudocode of using it:
        critic = critic_network(...)
        fast_bias.update(critic, reward, discount, mask)
        new_critic = critic + fast_bias.get()
        # use new_critic for training
    """

    def __init__(self, num_critics=1, speed=10, name="CriticBias"):
        """Create an instance of FastCriticBias.

        Args:
            num_critics (int): number of critic functions to process
            speeed (float): how fast to update the averages
            name (str): name of this object
        """
        super().__init__(name=name)
        with self.name_scope:
            spec = tf.TensorSpec((), tf.float32)
            self._averager = AdaptiveAverager(
                tensor_spec=[[spec] * num_critics, spec, spec], speed=speed)
            self._biases = [
                tf.Variable(initial_value=0., trainable=False)
                for _ in range(num_critics)
            ]

    def update(self, critics, reward, discount, mask=None):
        """Update internal statistics.

        Args:
            critics (Tensor|list[Tensor]): critics to process.
            reward (Tensor): reward received for the current step
            discount (Tensor): discount for the future steps
            mask (bool Tensor): If provided, only use the entries whose
                corresponding element in `mask` is True.
        """

        def _update(critics, reward, discount, mask):
            if isinstance(critics, tf.Tensor):
                critics = [critics]
            critics, reward, discount = math_ops.weighted_reduce_mean(
                [critics, reward, discount], mask)
            self._averager.average([critics, reward, discount])

            critics, reward, discount = self._averager.get()
            b = reward / (1 - discount)
            tf.nest.map_structure(lambda v, c: v.assign(b - c), self._biases,
                                  critics)

        common.run_if(
            tf.reduce_any(mask), lambda: _update(critics, reward, discount,
                                                 mask))

    def get(self, i=None):
        """Get the critic bias.

        Args:
            i (None|int): If None, return the list of biases for all the
                critics. If not None, return the bias for the i-th critic.
        Returns:
            Tensor if `i` is not None or `num_critics` is 1.
            list[Tensor] otherwise.
        """
        if i is not None:
            return self._biases[i].value()
        elif len(self._biases) == 1:
            return self._biases[0].value()
        else:
            return [v.value() for v in self._biases]

    def summarize(self):
        critics, reward, discount = self._averager.get()
        tf.summary.scalar("critic_bias", self.get(0))
        tf.summary.scalar("critic_avg", critics[0])
        tf.summary.scalar("reward_avg", reward)
        tf.summary.scalar("discount_avg", discount)


@gin.configurable
class SarsaAlgorithm(OnPolicyAlgorithm):
    """SARSA Algorithm.

    SARSA update Q function using the following loss:
        ||Q(s_t,a_t) - stop_gradient(r_t + \gamma * Q(s_{t+1}, a_{t+1})||^2
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
                 critic_loss_cls=OneStepTDLoss,
                 target_entropy=None,
                 initial_alpha=1.0,
                 num_replicas=2,
                 fast_critic_bias_speed=0.,
                 ou_stddev=0.2,
                 ou_damping=0.15,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 target_update_tau=0.05,
                 target_update_period=10,
                 use_smoothed_actor=False,
                 dqda_clipping=None,
                 gradient_clipping=None,
                 on_policy=False,
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
            fast_critic_bias_speed (float): If >=1, use FastCriticBias to learn
                critic bias.
            initial_alpha (float|None): If provided, will add -alpha*entropy to
                the loss to encourage diverse action.
            target_entropy (float|None): The target average policy entropy, for
                updating alpha. Only used if `initial_alpha` is not None
            ou_stddev (float): Only used for DDPG. Standard deviation for the
                Ornstein-Uhlenbeck (OU) noise added in the default collect policy.
            ou_damping (float): Only used for DDPG. Damping factor for the OU
                noise added in the default collect policy.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            use_smoothed_actor (bool): use a smoothed version of actor for
                predict and rollout. This option can be used if `on_policy` is
                False.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between [-dqda_clipping, dqda_clipping].
                Does not perform clipping if dqda_clipping == 0.
            actor_optimizer (tf.optimizers.Optimizer): The optimizer for actor.
            critic_optimizer (tf.optimizers.Optimizer): The optimizer for critic
                networks. If None, will use actor_optimizer.
            alpha_optimizer (tf.optimizers.Optimizer): The optimizer for alpha.
                Only used if `initial_alpha` is not None. If None, will use
                actor_optimizer.
            gradient_clipping (float): Norm length to clip gradients.
            on_policy (bool): whether it is used as an on-policy algorithm.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        tf.Module.__init__(self, name=name)

        flat_action_spec = tf.nest.flatten(action_spec)
        is_continuous = min(map(tensor_spec.is_continuous, flat_action_spec))
        assert is_continuous, (
            "SarsaAlgorithm only supports continuous action."
            " action_spec: %s" % action_spec)

        if isinstance(actor_network, DistributionNetwork):
            self._action_distribution_spec = actor_network.output_spec
        elif isinstance(actor_network, Network):
            self._action_distribution_spec = action_spec
        else:
            raise ValueError("Expect DistributionNetwork or Network for"
                             " `actor_network`, got %s" % type(actor_network))

        # TODO: implement ParallelCriticNetwork to speed up computation
        critic_networks = [
            critic_network.copy(name='critic_network%s' % i)
            for i in range(num_replicas)
        ]

        self._on_policy = on_policy

        optimizers = [actor_optimizer, critic_optimizer]
        trainable_module_sets = [[actor_network], critic_networks]

        if initial_alpha is not None:
            if target_entropy is None:
                target_entropy = np.sum(
                    list(
                        map(dist_utils.calc_default_target_entropy,
                            flat_action_spec)))
            self._target_entropy = target_entropy
            logging.info("Sarsa target_entropy=%s" % target_entropy)
            with self.name_scope:
                self._log_alpha = tf.Variable(
                    name='log_alpha',
                    initial_value=np.log(initial_alpha),
                    dtype=tf.float32,
                    trainable=True)
            optimizers.append(alpha_optimizer)
            trainable_module_sets.append([self._log_alpha])

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
                critics=[critic_network.state_spec] * num_replicas,
                target_critics=[critic_network.state_spec] * num_replicas,
            ),
            optimizer=optimizers,
            trainable_module_sets=trainable_module_sets,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            name=name)
        self._actor_network = actor_network
        self._num_replicas = num_replicas
        self._critic_networks = critic_networks
        self._target_critic_networks = [
            critic_network.copy(name='target_critic_network%s' % i)
            for i in range(num_replicas)
        ]

        models = copy.copy(self._critic_networks)
        target_models = copy.copy(self._target_critic_networks)

        self._rollout_actor_network = self._actor_network
        if use_smoothed_actor:
            assert not on_policy, ("use_smoothed_actor can only be used in "
                                   "off-policy training")
            self._rollout_actor_network = actor_network.copy(
                name='rollout_actor_network')
            models.append(self._actor_network)
            target_models.append(self._rollout_actor_network)

        with self.name_scope:
            self._update_target = common.get_target_updater(
                models=models,
                target_models=target_models,
                tau=target_update_tau,
                period=target_update_period)

        self._dqda_clipping = dqda_clipping

        # TODO: make a batched OUProcess. The current one is not batched.
        # Different environments use the same OU process, which is not good.
        self._ou_process = create_ou_process(action_spec, ou_stddev,
                                             ou_damping)
        self._critic_losses = []
        for i in range(num_replicas):
            self._critic_losses.append(
                critic_loss_cls(debug_summaries=debug_summaries and i == 0))

        with self.name_scope:
            self._critic_bias = None
            if fast_critic_bias_speed >= 1:
                self._critic_bias = FastCriticBias(
                    num_critics=num_replicas, speed=fast_critic_bias_speed)

    def _trainable_attributes_to_ignore(self):
        return ["_target_critic_networks", "_rollout_actor_network"]

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
                if epsilon_greedy >= 1.0:
                    return a + ou()
                else:
                    return tf.where(
                        tf.random.uniform(tf.shape(a)[:1]) < epsilon_greedy,
                        a + ou(), a)

            action = tf.nest.map_structure(_sample, action_distribution,
                                           self._ou_process)
        return action_distribution, action, state

    def predict(self, time_step: ActionTimeStep, state, epsilon_greedy):
        action_distribution, action, actor_state = self._get_action(
            self._rollout_actor_network, time_step, state.actor,
            epsilon_greedy)
        return PolicyStep(
            action=action,
            state=SarsaState(
                actor=actor_state,
                prev_observation=time_step.observation,
                prev_step_type=time_step.step_type),
            info=SarsaInfo(action_distribution=action_distribution))

    def convert_train_state_to_predict_state(self, state: SarsaState):
        return state._replace(critics=(), target_critics=())

    def _calc_critics(self, critic_networks, inputs, step_type,
                      network_states):
        critics = []
        states = []
        if self._critic_bias is not None:
            biases = self._critic_bias.get()
        for i in range(self._num_replicas):
            critic, state = critic_networks[i](
                inputs=inputs,
                step_type=step_type,
                network_state=network_states[i])
            if self._critic_bias is not None:
                critic += biases[i]
            critics.append(critic)
            states.append(state)
        return critics, states

    def rollout(self, time_step: ActionTimeStep, state: SarsaState, mode):
        if self._on_policy:
            return self._train_step(time_step, state, mode)

        if len(tf.nest.flatten(state.target_critics)) == 0:
            critic_states = state.critics
        else:
            _, critic_states = self._calc_critics(
                self._critic_networks,
                inputs=(state.prev_observation, time_step.prev_action),
                step_type=state.prev_step_type,
                network_states=state.critics)

            not_first_step = tf.not_equal(time_step.step_type, StepType.FIRST)

            critic_states = tf.nest.map_structure(
                lambda new_s, s: tf.where(not_first_step, new_s, s),
                critic_states, state.critics)

        action_distribution, action, actor_state = self._get_action(
            self._rollout_actor_network, time_step, state.actor)

        if len(tf.nest.flatten(state.target_critics)) == 0:
            target_critic_states = state.target_critics
        else:
            _, target_critic_states = self._calc_critics(
                self._target_critic_networks,
                inputs=(time_step.observation, action),
                step_type=time_step.step_type,
                network_states=state.target_critics)

        info = SarsaInfo(action_distribution=action_distribution)

        rl_state = SarsaState(
            prev_observation=time_step.observation,
            prev_step_type=time_step.step_type,
            actor=actor_state,
            critics=critic_states,
            target_critics=target_critic_states)

        return PolicyStep(action, rl_state, info)

    def train_step(self, time_step: Experience, state: SarsaState, mode):
        return self._train_step(
            experience_to_time_step(time_step), state, mode)

    def _train_step(self, time_step: ActionTimeStep, state: SarsaState, mode):
        not_first_step = tf.not_equal(time_step.step_type, StepType.FIRST)
        prev_critics, critic_states = self._calc_critics(
            self._critic_networks,
            inputs=(state.prev_observation, time_step.prev_action),
            step_type=state.prev_step_type,
            network_states=state.critics)

        if self._critic_bias is not None and mode != RLAlgorithm.PREPARE_SPEC:
            self._critic_bias.update(
                critics=[
                    c - b
                    for c, b in zip(prev_critics, self._critic_bias.get())
                ],
                reward=time_step.reward,
                discount=time_step.discount * self._critic_losses[0].discount,
                mask=not_first_step)

        critic_states = tf.nest.map_structure(
            lambda new_s, s: tf.where(not_first_step, new_s, s), critic_states,
            state.critics)

        action_distribution, action, actor_state = self._get_action(
            self._actor_network, time_step, state.actor)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(action)
            critics, _ = self._calc_critics(
                self._critic_networks,
                inputs=(time_step.observation, action),
                step_type=time_step.step_type,
                network_states=critic_states)
            critic = math_ops.min_n(critics)
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

        if self._log_alpha is not None:
            log_pi = tfa_common.log_probability(action_distribution, action,
                                                self._action_spec)
            alpha = tf.stop_gradient(tf.exp(self._log_alpha))
            actor_loss += alpha * log_pi
            alpha_loss = self._log_alpha * tf.stop_gradient(
                -log_pi - self._target_entropy)
        else:
            alpha_loss = ()

        target_critics, target_critic_states = self._calc_critics(
            self._target_critic_networks,
            inputs=(time_step.observation, action),
            step_type=time_step.step_type,
            network_states=state.target_critics)

        info = SarsaInfo(
            action_distribution=action_distribution,
            actor_loss=actor_loss,
            alpha_loss=alpha_loss,
            critics=prev_critics,
            target_critics=target_critics)

        rl_state = SarsaState(
            prev_observation=time_step.observation,
            prev_step_type=time_step.step_type,
            actor=actor_state,
            critics=critic_states,
            target_critics=target_critic_states)

        return PolicyStep(action, rl_state, info)

    def calc_loss(self, training_info: TrainingInfo):
        info: SarsaInfo = training_info.info

        step_type0 = training_info.step_type[0]
        step_type0 = tf.where(step_type0 == StepType.LAST, StepType.MID,
                              step_type0)
        step_type0 = tf.where(step_type0 == StepType.FIRST, StepType.LAST,
                              step_type0)
        shifted_training_info = training_info._replace(
            discount=common.tensor_prepend_zero(training_info.discount),
            reward=common.tensor_prepend_zero(training_info.reward),
            step_type=common.tensor_prepend(training_info.step_type,
                                            step_type0))
        critic_losses = []
        for i in range(self._num_replicas):
            critic = common.tensor_extend_zero(
                shifted_training_info.info.critics[i])
            target_critic = common.tensor_prepend_zero(
                shifted_training_info.info.target_critics[i])
            loss_info = self._critic_losses[i](shifted_training_info, critic,
                                               target_critic)
            critic_losses.append(
                tf.nest.map_structure(lambda l: l[:-1], loss_info.loss))

        critic_loss = tf.add_n(critic_losses)

        # returns = [
        #     tf.stop_gradient(training_info.reward +
        #                      0.99 * training_info.discount * tc)
        #     for tc in info.target_critics
        # ]
        # critic_loss = tf.add_n([losses.element_wise_huber_loss(c, r)
        #     for c, r in zip(info.critics, returns)])

        not_first_step = tf.not_equal(training_info.step_type, StepType.FIRST)
        # put critic_loss to scalar_loss because loss will be masked by
        # ~is_last at train_complete(). The critic_loss here should be
        # masked by ~is_first instead, which is done above
        critic_loss = tf.reduce_mean(
            critic_loss * tf.cast(not_first_step, tf.float32))
        scalar_loss = critic_loss

        def _summary():
            with self.name_scope:
                if self._critic_bias is not None:
                    self._critic_bias.summarize()

        if self._debug_summaries:
            common.run_if(common.should_record_summaries(), _summary)

        return LossInfo(
            loss=math_ops.add_ignore_empty(info.actor_loss, info.alpha_loss),
            scalar_loss=scalar_loss,
            extra=SarsaLossInfo(
                actor=info.actor_loss,
                critic=critic_loss,
                alpha=info.alpha_loss))

    def after_train(self, training_info):
        self._update_target()
