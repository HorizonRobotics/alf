# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import math
from typing import NamedTuple, Union, Callable

import torch
from torch import Tensor
import torch.distributions as td

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import TimeStep, LossInfo
from alf.data_structures import AlgStep, StepType
from alf.networks import QNetwork
from alf.nest.utils import convert_device
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, losses, dist_utils, tensor_utils, value_ops, schedulers
from alf.utils.summary_utils import safe_mean_hist_summary
from alf.experience_replayers.replay_buffer import BatchInfo
from alf.utils import losses, tensor_utils, value_ops


class DQNXState(NamedTuple):
    q: Tensor
    target_q: Tensor = ()


class DQNXInfo(NamedTuple):
    action: Tensor = ()
    reward: Tensor = ()
    step_type: Tensor = ()
    discount: Tensor = ()
    action_distribution: td.Distribution = ()
    q_values: Tensor = ()  # [B, num_replicas, q_dim]
    target_q_values: Tensor = ()  # [B, q_dim]
    entropy: Tensor = ()  # [B]
    log_pi: Tensor = ()  # [B]


@alf.configurable
class MDQNAlgorithm(OffPolicyAlgorithm):
    def __init__(
            self,
            observation_spec,
            action_spec: BoundedTensorSpec,
            reward_spec=TensorSpec(()),
            q_network_ctor=QNetwork,
            num_replicas=1,
            target_update_tau=0.05,
            target_update_period=8000,
            entropy_regularization=0.03,
            target_entropy: Union[None, float, Callable] = None,
            use_entropy_reward=True,
            alpha=0.9,
            log_pi_clip=-1.0,
            separate_q_m=False,
            use_sac_style_target=False,
            gamma=0.99,
            td_lambda=0.95,
            td_error_loss_fn=losses.element_wise_huber_loss,
            reward_weights=None,
            epsilon_greedy=None,
            rollout_epsilon_greedy: Union[float, schedulers.Scheduler] = 1.0,
            epsilon_greedy_uniform=True,
            top_k_sample=0,
            optimizer=None,
            config: TrainerConfig = None,
            debug_summaries=False,
            name="DQNX"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                be a mixture of discrete and continuous actions. The number of
                continuous actions can be arbitrary while only one discrete
                action is allowed currently. If it's a mixture, then it must be
                a tuple/list ``(discrete_action_spec, continuous_action_spec)``.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            q_network_ctor (Callable): is used to construct QNetwork for estimating ``Q(s,a)``
                given that the action is discrete. Its output spec must be consistent with
                the discrete action in ``action_spec``.
            num_replicas=1,
            target_update_tau=0.05,
            target_update_period=8000,
            entropy_regularization:
            target_entropy: If None, will use fixed entropy_regularization.
                If a float, will use this value as target entropy.
                If a callable function, will call it with action_spec to get target entropy.
                The entropy_regularization will be adjusted to match the entropy
                of the action distribution to the target entropy.
            use_entropy_reward: If True, add entropy to the reward.
            alpha: alpha parameter in Munchausen RL.
            log_pi_clip=-1.0,
            use_sac_style_target: If True, when calculating entropy and the target
                Q value, use the action distribution of the current policy instead.
                Otherwise, use the action distribution of the target policy.
            gamma=0.99,
            td_lambda=0.95,
            td_error_loss_fn=losses.element_wise_squared_loss,
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor if reward_weights is not None.
                Otherwise, the sum of the q values is used.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            rollout_epsilon_greedy: epsilon greedy parameter for rollout.
            epsilon_greedy_uniform: If True, exploration with uniform distribution.
                If False, exploration with learned distribution.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        assert isinstance(action_spec, BoundedTensorSpec)
        assert action_spec.is_discrete
        assert action_spec.shape == ()
        assert entropy_regularization > 0, "Not supported"
        assert 0 <= alpha <= 1, f"Invalid alpha: {alpha}"

        self._use_sac_style_target = use_sac_style_target
        self._num_replicas = num_replicas
        self._q_dim = reward_spec.numel
        if separate_q_m:
            self._q_dim += 1
        self._separate_q_m = separate_q_m
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy
        self._rollout_epsilon_greedy = schedulers.as_scheduler(
            rollout_epsilon_greedy)
        self._epsilon_greedy_uniform = epsilon_greedy_uniform
        self._top_k_sample = top_k_sample
        if callable(target_entropy):
            target_entropy = target_entropy(action_spec)
        self._target_entropy = target_entropy

        original_observation_spec = observation_spec
        q_networks = q_network_ctor(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        if num_replicas * self._q_dim > 1:
            q_networks = q_networks.make_parallel(num_replicas * self._q_dim)

        train_state_spec = DQNXState(
            q=q_networks.state_spec,
            target_q=q_networks.state_spec if target_update_period > 0 else ())
        super().__init__(
            observation_spec=original_observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            reward_weights=reward_weights,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._critic_networks = q_networks
        self._alpha = alpha
        self._log_pi_clip = log_pi_clip
        self._gamma = gamma
        self._td_lambda = td_lambda
        self._td_error_loss_fn = td_error_loss_fn

        reward_weights = torch.ones(self._q_dim)
        if self._reward_weights is not None:
            # User provided reward weights is for reward part only
            reward_weights[:self._reward_spec.numel] = self._reward_weights
        self._reward_weights = reward_weights
        self._use_entropy_reward = use_entropy_reward

        self._target_q_networks = q_networks.copy(name='target_q_networks')
        self._target_q_networks.requires_grad_(False)
        self._update_target = common.TargetUpdater(
            models=[q_networks],
            target_models=[self._target_q_networks],
            tau=target_update_tau,
            period=target_update_period)
        self._log_entropy_reg = torch.tensor(math.log(entropy_regularization))
        if target_entropy is not None:
            self._log_entropy_reg = torch.nn.Parameter(self._log_entropy_reg)

    def _trainable_attributes_to_ignore(self):
        return ['_target_q_networks']

    def after_update(self, root_inputs, info: DQNXInfo):
        if self._target_q_networks is not None:
            self._update_target()

    @property
    def entropy_regularization(self):
        return torch.exp(self._log_entropy_reg).item()

    def _compute_q_values(self, q_networks, observation, state):
        """

        Returns:
        - q_values: [B, num_replicas, num_actions, q_dim]
        - min_q_values: min q_values across replicas, [B, num_actions, q_dim]
        - action_dist:
        - state: the updated state
        """

        def _calc_q(net, state):
            # [B, num_replicas * q_dim, num_actions]
            q_values, state = net(observation, state)
            q_values = q_values.reshape(
                q_values.size(0), self._num_replicas, self._q_dim, -1)
            # [B, num_replicas, num_actions, q_dim]
            q_values = q_values.transpose(2, 3)
            return q_values, state

        def _min_q(q):
            if self._num_replicas == 1:
                return q.squeeze(1)
            elif self.has_multidim_reward():
                sign = self.reward_weights.sign()
                return (q * sign).min(dim=1)[0] * sign
            else:
                return q.min(dim=1)[0]

        q_values, new_state = _calc_q(q_networks, state)
        min_q_values = _min_q(q_values)

        summed_q_values = min_q_values @ self._reward_weights
        entropy_regularization = self.entropy_regularization
        # Need this so that the gradient imitation loss will not overwhelm the
        # TD loss
        summed_q_values = tensor_utils.scale_gradient(summed_q_values,
                                                      entropy_regularization)

        action_logits = summed_q_values / entropy_regularization
        action_dist = td.Categorical(logits=action_logits)

        return q_values, min_q_values, action_dist, new_state

    def _step(self, inputs: TimeStep, state: DQNXState, epsilon_greedy: float):
        _, _, action_dist, new_state = self._compute_q_values(
            self._critic_networks, inputs.observation, state)
        if self._epsilon_greedy_uniform:
            logits = action_dist.logits
            greedy_action = logits.argmax(dim=1)
            random_action = torch.randint_like(greedy_action, 0,
                                               logits.size(1))
            r = torch.rand_like(greedy_action, dtype=torch.float32)
            action = torch.where(r < epsilon_greedy, random_action,
                                 greedy_action)
        elif self._top_k_sample > 0:
            action = dist_utils.top_k_sample(action_dist, self._top_k_sample)
        else:
            action = dist_utils.epsilon_greedy_sample(action_dist,
                                                      epsilon_greedy)

        return AlgStep(
            output=action,
            state=new_state,
            info=DQNXInfo(action_distribution=action_dist, action=action))

    def predict_setp(self, inputs: TimeStep, state: DQNXState):
        return self._step(inputs, state, self._epsilon_greedy)

    def rollout_step(self, inputs: TimeStep, state: DQNXState):
        return self._step(inputs, state, self._rollout_epsilon_greedy())

    def train_step(self, inputs: TimeStep, state: DQNXState,
                   rollout_info: DQNXInfo):
        q_values, _, action_dist, new_state = self._compute_q_values(
            self._critic_networks, inputs.observation, state)
        with torch.no_grad():
            _, target_q_values, target_action_dist, new_target_state = self._compute_q_values(
                self._target_q_networks, inputs.observation, state)
            # Note: SAC uses action_dist instead of target_action_dist for target_q_values
            if self._use_sac_style_target:
                target_q_values = torch.einsum('bar,ba->br', target_q_values,
                                               action_dist.probs)
                entropy = action_dist.entropy()
            else:
                target_q_values = torch.einsum('bar,ba->br', target_q_values,
                                               target_action_dist.probs)
                entropy = target_action_dist.entropy()

        action = rollout_info.action
        B = torch.arange(action.shape[0])
        action_q_values = q_values[B, :, action]
        log_pi = target_action_dist.logits[B, action]
        return AlgStep(
            output=action,
            state=DQNXState(q=new_state, target_q=new_target_state),
            info=DQNXInfo(
                action=action,
                action_distribution=action_dist,
                step_type=inputs.step_type,
                reward=inputs.reward,
                discount=inputs.discount,
                q_values=action_q_values,
                entropy=entropy.detach(),
                log_pi=log_pi.detach(),
                target_q_values=target_q_values))

    def calc_loss(self, info: DQNXInfo):
        entropy_regularization = self.entropy_regularization
        log_pi = torch.clamp(
            info.log_pi, min=self._log_pi_clip / entropy_regularization)
        discount = self._gamma * info.discount
        entropy_reward = discount * entropy_regularization * info.entropy
        if not self._separate_q_m:
            if self._use_entropy_reward:
                info = info._replace(
                    reward=(info.reward + common.expand_dims_as(
                        entropy_reward / self._q_dim, info.reward)))

            # [T-1, B, q_dim]
            target_q_values = value_ops.compute_td_target(
                info, self._gamma, self._td_lambda, info.target_q_values)
            log_pi = log_pi[:-1, :, None]
            target_q_values += self._alpha * entropy_regularization / self._q_dim * log_pi
        else:
            target_q_values = value_ops.compute_td_target(
                info, self._gamma, self._td_lambda,
                info.target_q_values[..., :-1])
            target_q_m_values = self._alpha * entropy_regularization * log_pi[:
                                                                              -1, :]
            if self._use_entropy_reward:
                target_q_m_values += value_ops.one_step_discounted_return(
                    rewards=entropy_reward,
                    values=info.target_q_values[:, :, -1],
                    step_types=info.step_type,
                    discounts=discount,
                    time_major=True)
            target_q_values = torch.cat(
                [target_q_values,
                 target_q_m_values.unsqueeze(-1)], dim=-1)

        q_values = info.q_values[:-1]  # [T-1, B, num_replicas, q_dim]
        # [T-1, B, num_replicas, q_dim]
        target_q_values = target_q_values[:, :, None, :].expand_as(q_values)
        td_error = target_q_values - q_values
        loss = self._td_error_loss_fn(target_q_values, q_values)
        loss = loss.reshape(*loss.shape[:2], -1).mean(-1)
        # The shape of the loss expected by Algorithm.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type[:-1] != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))
                    safe_mean_hist_summary('critics' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("critic_td_error" + suffix, td,
                                           mask)

                num_replicas = q_values.size(2)
                for r in range(num_replicas):
                    for i in range(q_values.size(3)):
                        suffix = f'/replica_{r}/{i}'
                        _summarize(q_values[..., r, i],
                                   target_q_values[..., r, i],
                                   td_error[..., r, i], suffix)
                alf.summary.scalar("entropy_regularization",
                                   entropy_regularization)

        T = info.step_type.size(0)
        loss = T / (T - 1) * loss
        extra = {'critic': loss, 'neg_entropy': -info.entropy}

        if self._target_entropy is not None:
            entropy_reg_loss = self._log_entropy_reg * (
                info.entropy - self._target_entropy).detach()
            loss += entropy_reg_loss
            extra['entropy_reg_loss'] = entropy_reg_loss,
        return LossInfo(loss=loss, extra=extra)

    @torch.no_grad()
    def set_reward_weights(self, reward_weights):
        """Update reward weights; this function can be called at any step during
        training. Once called, the updated reward weights are expected to be used
        by the algorithm in the next.

        Args:
            reward_weights (Tensor): a tensor that is compatible with
                ``self._reward_spec``.
        """
        assert self.has_multidim_reward(), (
            "Can't update weights for a scalar reward!")
        self._reward_weights[:self._reward_spec.numel] = reward_weights
