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

from absl import logging
import copy
import gin
import numpy as np
import torch
import torch.nn as nn

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.data_structures import (AlgStep, Experience, experience_to_time_step,
                                 LossInfo, namedtuple, StepType, TimeStep,
                                 TrainingInfo)
from alf.networks import Network, DistributionNetwork
from alf.utils import common, dist_utils, losses, math_ops, tensor_utils
from alf.utils.averager import AdaptiveAverager
from alf.utils.summary_utils import safe_mean_hist_summary

SarsaState = namedtuple(
    'SarsaState', [
        'prev_observation', 'prev_step_type', 'actor', 'critics',
        'target_critics', 'noise'
    ],
    default_value=())
SarsaInfo = namedtuple(
    'SarsaInfo', [
        'action_distribution', 'actor_loss', 'critics', 'target_critics',
        'neg_entropy'
    ],
    default_value=())
SarsaLossInfo = namedtuple('SarsaLossInfo',
                           ['actor', 'critic', 'alpha', 'neg_entropy'])

nest_map = alf.nest.map_structure


class FastCriticBias(nn.Module):
    """
    The motivation of this is to prevent the critic value becomes greater and
    greater as training goes on because of the max operation.

    FastCriticBias estimates expected return `b` by minimizing the following
    expectation:
        min_b E(b - r - discount * b)^2
    where r is reward. `b` can be analytically solved as:
        b = E(r) / (1 - E(discount))

    We explicitly enforce that E(critic - b) = 0 by estimating E(critic) and
    use b - E(critic) as bias to offset critic.

    Pseudocode of using it:
        critic = critic_network(...)
        fast_bias.update(critic, reward, discount, mask)
        new_critic = critic + fast_bias.get()
        # use new_critic for training
    """

    def __init__(self, num_critics=1, speed=10):
        """Create an instance of FastCriticBias.

        Args:
            num_critics (int): number of critic functions to process
            speeed (float): how fast to update the averages
            name (str): name of this object
        """
        super().__init__()
        spec = alf.TensorSpec((), torch.float32)
        self._averager = AdaptiveAverager(
            tensor_spec=[[spec] * num_critics, spec, spec], speed=speed)

        self._biases = []
        for i in range(num_critics):
            buf = torch.tensor(0.)
            self._biases.append(buf)
            self.register_buffer("_bias%d" % i, buf)

    def update(self, critics, reward, discount, mask=None):
        """Update internal statistics.

        Args:
            critics (Tensor|list[Tensor]): critics to process.
            reward (Tensor): reward received for the current step
            discount (Tensor): discount for the future steps
            mask (bool Tensor): If provided, only use the entries whose
                corresponding element in `mask` is True.
        """
        if torch.any(mask):
            if isinstance(critics, torch.Tensor):
                critics = [critics]
            critics, reward, discount = math_ops.weighted_reduce_mean(
                [critics, reward, discount], mask)
            critics, reward, discount = self._averager.average(
                [critics, reward, discount])
            b = reward / (1 - discount)
            nest_map(lambda v, c: v.copy_((b - c).detach()), self._biases,
                     critics)

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
            return self._biases[i]
        elif len(self._biases) == 1:
            return self._biases[0]
        else:
            return copy.copy(self._biases)

    def summarize(self):
        critics, reward, discount = self._averager.get()
        alf.summary.scalar("critic_bias", self.get(0))
        alf.summary.scalar("critic_avg", critics[0])
        alf.summary.scalar("reward_avg", reward)
        alf.summary.scalar("discount_avg", discount)


@gin.configurable
class SarsaAlgorithm(OnPolicyAlgorithm):
    r"""SARSA Algorithm.

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
                 env=None,
                 config=None,
                 critic_loss_cls=OneStepTDLoss,
                 target_entropy=None,
                 use_entropy_reward=False,
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
                 dqda_clipping=0.,
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
            env (Environment): The environment to interact with. `env` is a
                batched environment, which means that it runs multiple
                simulations simultaneously. Running multiple environments in
                parallel is crucial to on-policy algorithms as it increases the
                diversity of data and decreases temporal correlation. `env` only
                needs to be provided to the root `Algorithm`.
            config (TrainerConfig): config for training. `config` only needs to
                be provided to the algorithm which performs `train_iter()` by
                itself.
            fast_critic_bias_speed (float): If >=1, use FastCriticBias to learn
                critic bias.
            initial_alpha (float|None): If provided, will add -alpha*entropy to
                the loss to encourage diverse action.
            target_entropy (float|None): The target average policy entropy, for
                updating alpha. Only used if `initial_alpha` is not None
            use_entropy_reward (bool): If True, will use alpha*entropy as
                additional reward.
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
            actor_optimizer (torch.optim.Optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.Optimizer): The optimizer for critic
                networks.
            alpha_optimizer (torch.optim.Optimizer): The optimizer for alpha.
                Only used if `initial_alpha` is not None.
            gradient_clipping (float): Norm length to clip gradients.
            on_policy (bool): whether it is used as an on-policy algorithm.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        flat_action_spec = alf.nest.flatten(action_spec)
        is_continuous = min(
            map(lambda spec: spec.is_continuous, flat_action_spec))
        assert is_continuous, (
            "SarsaAlgorithm only supports continuous action."
            " action_spec: %s" % action_spec)

        # TODO: implement ParallelCriticNetwork to speed up computation
        critic_networks = [
            critic_network.copy(name='critic_network%s' % i)
            for i in range(num_replicas)
        ]

        self._on_policy = on_policy

        if not isinstance(actor_network, DistributionNetwork):
            noise_process = alf.networks.OUProcess(
                state_spec=action_spec, damping=ou_damping, stddev=ou_stddev)
            noise_state = noise_process.state_spec
        else:
            noise_process = None
            noise_state = ()

        super().__init__(
            observation_spec,
            action_spec,
            env=env,
            config=config,
            predict_state_spec=SarsaState(
                noise=noise_state,
                prev_observation=observation_spec,
                prev_step_type=alf.TensorSpec((), torch.int32),
                actor=actor_network.state_spec),
            train_state_spec=SarsaState(
                noise=noise_state,
                prev_observation=observation_spec,
                prev_step_type=alf.TensorSpec((), torch.int32),
                actor=actor_network.state_spec,
                critics=[critic_network.state_spec] * num_replicas,
                target_critics=[critic_network.state_spec] * num_replicas,
            ),
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            name=name)
        self._actor_network = actor_network
        self._num_replicas = num_replicas
        self._critic_networks = nn.ModuleList(critic_networks)
        self._target_critic_networks = [
            critic_network.copy(name='target_critic_network%s' % i)
            for i in range(num_replicas)
        ]
        self.add_optimizer(actor_optimizer, [actor_network])
        self.add_optimizer(critic_optimizer, critic_networks)

        self._log_alpha = None
        self._use_entropy_reward = False
        if initial_alpha is not None:
            if isinstance(actor_network, DistributionNetwork):
                if target_entropy is None:
                    target_entropy = np.sum(
                        list(
                            map(dist_utils.calc_default_target_entropy,
                                flat_action_spec)))
                self._target_entropy = target_entropy
                logging.info("Sarsa target_entropy=%s" % target_entropy)
                log_alpha = torch.tensor(
                    np.log(initial_alpha), dtype=torch.float32)
                if alpha_optimizer is None:
                    self._log_alpha = log_alpha
                else:
                    self._log_alpha = nn.Parameter(log_alpha)
                    self.add_optimizer(alpha_optimizer, [self._log_alpha])
                self._use_entropy_reward = use_entropy_reward
            else:
                logging.info(
                    "initial_alpha and alpha_optimizer is ignored. "
                    "The `actor_network` needs to a DistributionNetwork in "
                    "order to use entropy as regularization or reward")

        models = copy.copy(critic_networks)
        target_models = copy.copy(self._target_critic_networks)

        self._rollout_actor_network = self._actor_network
        if use_smoothed_actor:
            assert not on_policy, ("use_smoothed_actor can only be used in "
                                   "off-policy training")
            self._rollout_actor_network = actor_network.copy(
                name='rollout_actor_network')
            models.append(self._actor_network)
            target_models.append(self._rollout_actor_network)

        self._update_target = common.get_target_updater(
            models=models,
            target_models=target_models,
            tau=target_update_tau,
            period=target_update_period)

        self._dqda_clipping = dqda_clipping

        self._noise_process = noise_process
        self._critic_losses = []
        for i in range(num_replicas):
            self._critic_losses.append(
                critic_loss_cls(debug_summaries=debug_summaries and i == 0))

        self._critic_bias = None
        if fast_critic_bias_speed >= 1:
            self._critic_bias = FastCriticBias(
                num_critics=num_replicas, speed=fast_critic_bias_speed)

        self._is_rnn = len(alf.nest.flatten(critic_network.state_spec)) > 0

    def is_on_policy(self):
        return self._on_policy

    def _trainable_attributes_to_ignore(self):
        return ["_target_critic_networks", "_rollout_actor_network"]

    def _get_action(self,
                    actor_network,
                    time_step: TimeStep,
                    state: SarsaState,
                    epsilon_greedy=1.0):
        action_distribution, actor_state = actor_network(
            time_step.observation, state=state.actor)
        if isinstance(actor_network, DistributionNetwork):
            if epsilon_greedy == 1.0:
                action = dist_utils.rsample_action_distribution(
                    action_distribution)
            else:
                action = dist_utils.epsilon_greedy_sample(
                    action_distribution, epsilon_greedy)
            noise_state = ()
        else:

            def _sample(a, noise):
                if epsilon_greedy >= 1.0:
                    return a + noise
                else:
                    choose_random_action = (torch.rand(a.shape[:1]) <
                                            epsilon_greedy)
                    return torch.where(
                        common.expand_dims_as(choose_random_action, a),
                        a + noise, a)

            noise, noise_state = self._noise_process(state.noise)
            action = nest_map(_sample, action_distribution, noise)
        return action_distribution, action, actor_state, noise_state

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        action_distribution, action, actor_state, noise_state = self._get_action(
            self._rollout_actor_network, time_step, state, epsilon_greedy)
        return AlgStep(
            output=action,
            state=SarsaState(
                noise=noise_state,
                actor=actor_state,
                prev_observation=time_step.observation,
                prev_step_type=time_step.step_type),
            info=SarsaInfo(action_distribution=action_distribution))

    def convert_train_state_to_predict_state(self, state: SarsaState):
        return state._replace(critics=(), target_critics=())

    def _calc_critics(self, critic_networks, inputs, states):
        critics = []
        new_states = []
        if self._critic_bias is not None:
            biases = self._critic_bias.get()
        for i in range(self._num_replicas):
            critic, state = critic_networks[i](inputs=inputs, state=states[i])
            if self._critic_bias is not None:
                critic += biases[i]
            critics.append(critic)
            new_states.append(state)
        return critics, new_states

    def rollout_step(self, time_step: TimeStep, state: SarsaState):
        if self._on_policy:
            return self._train_step(time_step, state)

        if not self._is_rnn:
            critic_states = state.critics
        else:
            _, critic_states = self._calc_critics(
                self._critic_networks,
                inputs=(state.prev_observation, time_step.prev_action),
                states=state.critics)

            not_first_step = time_step.step_type != StepType.FIRST

            critic_states = common.reset_state_if_necessary(
                state.critics, critic_states, not_first_step)

        action_distribution, action, actor_state, noise_state = self._get_action(
            self._rollout_actor_network, time_step, state)

        if not self._is_rnn:
            target_critic_states = state.target_critics
        else:
            _, target_critic_states = self._calc_critics(
                self._target_critic_networks,
                inputs=(time_step.observation, action),
                states=state.target_critics)

        info = SarsaInfo(action_distribution=action_distribution)

        rl_state = SarsaState(
            noise=noise_state,
            prev_observation=time_step.observation,
            prev_step_type=time_step.step_type,
            actor=actor_state,
            critics=critic_states,
            target_critics=target_critic_states)

        return AlgStep(action, rl_state, info)

    def train_step(self, time_step: Experience, state: SarsaState):
        return self._train_step(experience_to_time_step(time_step), state)

    def _train_step(self, time_step: TimeStep, state: SarsaState):
        not_first_step = time_step.step_type != StepType.FIRST
        prev_critics, critic_states = self._calc_critics(
            self._critic_networks,
            inputs=(state.prev_observation, time_step.prev_action),
            states=state.critics)

        if self._critic_bias is not None:
            self._critic_bias.update(
                critics=[
                    c - b
                    for c, b in zip(prev_critics, self._critic_bias.get())
                ],
                reward=time_step.reward,
                discount=time_step.discount * self._critic_losses[0].discount,
                mask=not_first_step)

        critic_states = common.reset_state_if_necessary(
            state.critics, critic_states, not_first_step)

        action_distribution, action, actor_state, noise_state = self._get_action(
            self._actor_network, time_step, state)

        critics, _ = self._calc_critics(
            self._critic_networks,
            inputs=(time_step.observation, action),
            states=critic_states)
        critic = math_ops.min_n(critics)
        dqda = alf.nest.pack_sequence_as(
            action, torch.autograd.grad(critic.sum(),
                                        alf.nest.flatten(action)))

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = dqda.clamp(-self._dqda_clipping, self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            loss = loss.sum(list(range(1, loss.ndim)))
            return loss

        actor_loss = nest_map(actor_loss_fn, dqda, action)
        actor_loss = math_ops.add_n(alf.nest.flatten(actor_loss))

        neg_entropy = ()
        if self._log_alpha is not None:
            neg_entropy = dist_utils.compute_log_probability(
                action_distribution, action)

        target_critics, target_critic_states = self._calc_critics(
            self._target_critic_networks,
            inputs=(time_step.observation, action),
            states=state.target_critics)

        info = SarsaInfo(
            action_distribution=action_distribution,
            actor_loss=actor_loss,
            critics=prev_critics,
            neg_entropy=neg_entropy,
            target_critics=target_critics)

        rl_state = SarsaState(
            noise=noise_state,
            prev_observation=time_step.observation,
            prev_step_type=time_step.step_type,
            actor=actor_state,
            critics=critic_states,
            target_critics=target_critic_states)

        return AlgStep(action, rl_state, info)

    def calc_loss(self, training_info: TrainingInfo):
        info: SarsaInfo = training_info.info

        loss = info.actor_loss
        if self._log_alpha is not None:
            alpha = self._log_alpha.exp().detach()
            alpha_loss = self._log_alpha * (
                -info.neg_entropy - self._target_entropy).detach()
            loss = loss + alpha * info.neg_entropy + alpha_loss
        else:
            alpha_loss = ()

        # For sarsa, info.critics is actually the critics for the previous step.
        # And info.target_critics is the critics for the current step. So we
        # need to rearrange training_info to match the requirement for `OneStepTDLoss`.
        step_type0 = training_info.step_type[0]
        step_type0 = torch.where(step_type0 == StepType.LAST,
                                 torch.tensor(StepType.MID), step_type0)
        step_type0 = torch.where(step_type0 == StepType.FIRST,
                                 torch.tensor(StepType.LAST), step_type0)

        reward = training_info.reward
        if self._use_entropy_reward:
            reward -= (self._log_alpha.exp() * info.neg_entropy).detach()
        shifted_training_info = training_info._replace(
            discount=tensor_utils.tensor_prepend_zero(training_info.discount),
            reward=tensor_utils.tensor_prepend_zero(reward),
            step_type=tensor_utils.tensor_prepend(training_info.step_type,
                                                  step_type0))
        critic_losses = []
        for i in range(self._num_replicas):
            critic = tensor_utils.tensor_extend_zero(
                training_info.info.critics[i])
            target_critic = tensor_utils.tensor_prepend_zero(
                training_info.info.target_critics[i])
            loss_info = self._critic_losses[i](shifted_training_info, critic,
                                               target_critic)
            critic_losses.append(nest_map(lambda l: l[:-1], loss_info.loss))

        critic_loss = math_ops.add_n(critic_losses)

        not_first_step = training_info.step_type != StepType.FIRST
        # put critic_loss to scalar_loss because loss will be masked by
        # ~is_last at train_complete(). The critic_loss here should be
        # masked by ~is_first instead, which is done above
        critic_loss = (critic_loss * not_first_step.to(torch.float32)).mean()
        scalar_loss = critic_loss

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                if self._critic_bias is not None:
                    self._critic_bias.summarize()
                if self._log_alpha is not None:
                    alf.summary.scalar("alpha", alpha)

        return LossInfo(
            loss=loss,
            scalar_loss=scalar_loss,
            extra=SarsaLossInfo(
                actor=info.actor_loss,
                critic=critic_loss,
                alpha=alpha_loss,
                neg_entropy=info.neg_entropy))

    def after_update(self, training_info):
        self._update_target()
