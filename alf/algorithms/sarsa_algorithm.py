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
import numpy as np
import torch
import torch.nn as nn

import alf
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import AlgStep, LossInfo, namedtuple, StepType, TimeStep
from alf.utils import common, dist_utils, losses, math_ops, tensor_utils
import alf.nest.utils as nest_utils
from alf.tensor_specs import TensorSpec

SarsaState = namedtuple(
    'SarsaState', [
        'prev_observation', 'prev_step_type', 'actor', 'critics',
        'target_critics', 'noise'
    ],
    default_value=())
SarsaInfo = namedtuple(
    'SarsaInfo', [
        'reward', 'step_type', 'discount', 'action_distribution', 'actor_loss',
        'critics', 'target_critics', 'neg_entropy'
    ],
    default_value=())
SarsaLossInfo = namedtuple('SarsaLossInfo',
                           ['actor', 'critic', 'alpha', 'neg_entropy'])

nest_map = alf.nest.map_structure


@alf.configurable
class SarsaAlgorithm(RLAlgorithm):
    r"""SARSA Algorithm.

    SARSA update Q function using the following loss:

    .. math::

        ||Q(s_t,a_t) - \text{nograd}(r_t + \gamma * Q(s_{t+1}, a_{t+1}))||^2

    See https://en.wikipedia.org/wiki/State-action-reward-state-action

    Currently, this is only implemented for continuous action problems.
    The policy is derived by a DDPG/SAC manner by maximizing :math:`Q(a(s_t), s_t)`,
    where :math:`a(s_t)` is the action.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 actor_network_ctor,
                 critic_network_ctor,
                 reward_spec=TensorSpec(()),
                 num_critic_replicas=2,
                 env=None,
                 config=None,
                 critic_loss_cls=OneStepTDLoss,
                 target_entropy=None,
                 epsilon_greedy=None,
                 use_entropy_reward=False,
                 calculate_priority=False,
                 initial_alpha=1.0,
                 ou_stddev=0.2,
                 ou_damping=0.15,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 target_update_tau=0.05,
                 target_update_period=10,
                 use_smoothed_actor=False,
                 dqda_clipping=0.,
                 on_policy=False,
                 checkpoint=None,
                 debug_summaries=False,
                 name="SarsaAlgorithm"):
        """
        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            observation_spec (nested TensorSpec): spec for observation.
            actor_network_ctor (Callable): Function to construct the actor network.
                ``actor_network_ctor`` needs to accept ``input_tensor_spec`` and
                ``action_spec`` as its arguments and return an actor network.
                The constructed network will be called with ``forward(observation, state)``.
            critic_network_ctor (Callable): Function to construct the critic
                network. ``critic_netwrok_ctor`` needs to accept ``input_tensor_spec``
                which is a tuple of ``(observation_spec, action_spec)``. The
                constructed network will be called with
                ``forward((observation, action), state)``.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            num_critic_replicas (int): number of critics to be used. Default is 2.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple
                simulations simultaneously. Running multiple environments in
                parallel is crucial to on-policy algorithms as it increases the
                diversity of data and decreases temporal correlation. ``env`` only
                needs to be provided to the root ``Algorithm``.
            config (TrainerConfig): config for training. ``config`` only needs to
                be provided to the algorithm which performs ``train_iter()`` by
                itself.
            initial_alpha (float|None): If provided, will add ``-alpha*entropy``
                to the loss to encourage diverse action.
            target_entropy (float|Callable|None): If a floating value, it's the
                target average policy entropy, for updating ``alpha``. If a
                callable function, then it will be called on the action spec to
                calculate a target entropy. If ``None``, a default entropy will
                be calculated.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            use_entropy_reward (bool): If ``True``, will use alpha*entropy as
                additional reward.
            calculate_priority (bool): whether to calculate priority. This is
                only useful if priority replay is enabled.
            ou_stddev (float): Only used for DDPG. Standard deviation for the
                Ornstein-Uhlenbeck (OU) noise added in the default collect policy.
            ou_damping (float): Only used for DDPG. Damping factor for the OU
                noise added in the default collect policy.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            use_smoothed_actor (bool): use a smoothed version of actor for
                predict and rollout. This option can be used if ``on_policy`` is
                ``False``.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient ``dqda`` element-wise between
                ``[-dqda_clipping, dqda_clipping]``. Does not perform clipping
                if ``dqda_clipping == 0``.
            actor_optimizer (torch.optim.Optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.Optimizer): The optimizer for critic
                networks.
            alpha_optimizer (torch.optim.Optimizer): The optimizer for alpha.
                Only used if ``initial_alpha`` is not ``None``.
            on_policy (bool): whether it is used as an on-policy algorithm.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries (bool): ``True`` if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        self._calculate_priority = calculate_priority
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy
        critic_network = critic_network_ctor(
            input_tensor_spec=(observation_spec, action_spec))
        actor_network = actor_network_ctor(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        flat_action_spec = alf.nest.flatten(action_spec)
        is_continuous = min(
            map(lambda spec: spec.is_continuous, flat_action_spec))
        assert is_continuous, (
            "SarsaAlgorithm only supports continuous action."
            " action_spec: %s" % action_spec)

        critic_networks = critic_network.make_parallel(num_critic_replicas)

        if not actor_network.is_distribution_output:
            noise_process = alf.networks.OUProcess(
                state_spec=action_spec, damping=ou_damping, stddev=ou_stddev)
            noise_state = noise_process.state_spec
        else:
            noise_process = None
            noise_state = ()

        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            env=env,
            is_on_policy=on_policy,
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
                critics=critic_networks.state_spec,
                target_critics=critic_networks.state_spec,
            ),
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)
        self._actor_network = actor_network
        self._num_critic_replicas = num_critic_replicas
        self._critic_networks = critic_networks
        self._target_critic_networks = critic_networks.copy(
            name='target_critic_networks')
        self.add_optimizer(actor_optimizer, [actor_network])
        self.add_optimizer(critic_optimizer, [critic_networks])

        self._log_alpha = None
        self._use_entropy_reward = False
        if initial_alpha is not None:
            if actor_network.is_distribution_output:
                self._target_entropy = _set_target_entropy(
                    self.name, target_entropy, flat_action_spec)
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
                    "The `actor_network` needs to output Distribution in "
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

        self._update_target = common.TargetUpdater(
            models=models,
            target_models=target_models,
            tau=target_update_tau,
            period=target_update_period)

        self._dqda_clipping = dqda_clipping

        self._noise_process = noise_process
        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(
                critic_loss_cls(debug_summaries=debug_summaries and i == 0))

        self._is_rnn = len(alf.nest.flatten(critic_network.state_spec)) > 0

    def _trainable_attributes_to_ignore(self):
        return ["_target_critic_networks", "_rollout_actor_network"]

    def _get_action(self,
                    actor_network,
                    time_step: TimeStep,
                    state: SarsaState,
                    epsilon_greedy=1.0):
        action_distribution, actor_state = actor_network(
            time_step.observation, state=state.actor)
        if actor_network.is_distribution_output:
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

    def predict_step(self, inputs: TimeStep, state: SarsaState):
        action_distribution, action, actor_state, noise_state = self._get_action(
            self._rollout_actor_network, inputs, state, self._epsilon_greedy)
        return AlgStep(
            output=action,
            state=SarsaState(
                noise=noise_state,
                actor=actor_state,
                prev_observation=inputs.observation,
                prev_step_type=inputs.step_type),
            info=SarsaInfo(action_distribution=action_distribution))

    def convert_train_state_to_predict_state(self, state: SarsaState):
        return state._replace(critics=(), target_critics=())

    def rollout_step(self, inputs: TimeStep, state: SarsaState):
        if self.on_policy:
            return self._train_step(inputs, state)

        if not self._is_rnn:
            critic_states = state.critics
        else:
            _, critic_states = self._critic_networks(
                (state.prev_observation, inputs.prev_action), state.critics)

            not_first_step = inputs.step_type != StepType.FIRST

            critic_states = common.reset_state_if_necessary(
                state.critics, critic_states, not_first_step)

        action_distribution, action, actor_state, noise_state = self._get_action(
            self._rollout_actor_network, inputs, state)

        if not self._is_rnn:
            target_critic_states = state.target_critics
        else:
            _, target_critic_states = self._target_critic_networks(
                (inputs.observation, action), state.target_critics)

        info = SarsaInfo(action_distribution=action_distribution)

        rl_state = SarsaState(
            noise=noise_state,
            prev_observation=inputs.observation,
            prev_step_type=inputs.step_type,
            actor=actor_state,
            critics=critic_states,
            target_critics=target_critic_states)

        return AlgStep(action, rl_state, info)

    def train_step(self, inputs: TimeStep, state: SarsaState, rollout_info):
        return self._train_step(inputs, state)

    def _train_step(
            self,
            time_step: TimeStep,
            state: SarsaState,
    ):
        not_first_step = time_step.step_type != StepType.FIRST
        prev_critics, critic_states = self._critic_networks(
            (state.prev_observation, time_step.prev_action), state.critics)

        critic_states = common.reset_state_if_necessary(
            state.critics, critic_states, not_first_step)

        action_distribution, action, actor_state, noise_state = self._get_action(
            self._actor_network, time_step, state)

        critics, _ = self._critic_networks((time_step.observation, action),
                                           critic_states)
        critic = critics.min(dim=1)[0]
        dqda = nest_utils.grad(action, critic.sum())

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

        target_critics, target_critic_states = self._target_critic_networks(
            (time_step.observation, action), state.target_critics)

        info = SarsaInfo(
            reward=time_step.reward,
            step_type=time_step.step_type,
            discount=time_step.discount,
            action_distribution=action_distribution,
            actor_loss=actor_loss,
            critics=prev_critics,
            neg_entropy=neg_entropy,
            target_critics=target_critics.min(dim=1)[0])

        rl_state = SarsaState(
            noise=noise_state,
            prev_observation=time_step.observation,
            prev_step_type=time_step.step_type,
            actor=actor_state,
            critics=critic_states,
            target_critics=target_critic_states)

        return AlgStep(action, rl_state, info)

    def calc_loss(self, info: SarsaInfo):
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
        # need to rearrange ``experience``` to match the requirement for
        # `OneStepTDLoss`.
        step_type0 = info.step_type[0]
        step_type0 = torch.where(step_type0 == StepType.LAST,
                                 torch.tensor(StepType.MID), step_type0)
        step_type0 = torch.where(step_type0 == StepType.FIRST,
                                 torch.tensor(StepType.LAST), step_type0)

        gamma = self._critic_losses[0].gamma
        reward = info.reward
        if self._use_entropy_reward:
            reward -= gamma * (
                self._log_alpha.exp() * info.neg_entropy).detach()
        shifted_experience = info._replace(
            discount=tensor_utils.tensor_prepend_zero(info.discount),
            reward=tensor_utils.tensor_prepend_zero(reward),
            step_type=tensor_utils.tensor_prepend(info.step_type, step_type0))
        critic_losses = []
        for i in range(self._num_critic_replicas):
            critic = tensor_utils.tensor_extend_zero(info.critics[..., i])
            target_critic = tensor_utils.tensor_prepend_zero(
                info.target_critics)
            loss_info = self._critic_losses[i](shifted_experience, critic,
                                               target_critic)
            critic_losses.append(nest_map(lambda l: l[:-1], loss_info.loss))

        critic_loss = math_ops.add_n(critic_losses)

        not_first_step = (info.step_type != StepType.FIRST).to(torch.float32)
        critic_loss = critic_loss * not_first_step
        if self._calculate_priority:
            valid_n = torch.clamp(not_first_step.sum(dim=0), min=1.0)
            priority = (critic_loss.sum(dim=0) / valid_n).sqrt()
        else:
            priority = ()

        # put critic_loss to scalar_loss because loss will be masked by
        # ~is_last at train_complete(). The critic_loss here should be
        # masked by ~is_first instead, which is done above
        scalar_loss = critic_loss.mean()

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                if self._log_alpha is not None:
                    alf.summary.scalar("alpha", alpha)

        return LossInfo(
            loss=loss,
            scalar_loss=scalar_loss,
            priority=priority,
            extra=SarsaLossInfo(
                actor=info.actor_loss,
                critic=critic_loss,
                alpha=alpha_loss,
                neg_entropy=info.neg_entropy))

    def after_update(self, root_inputs, info: SarsaInfo):
        self._update_target()
