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

from absl import logging
import math
import random
from typing import Callable, NamedTuple, Union

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
from alf.utils import common, losses, dist_utils, tensor_utils, value_ops
from alf.utils.summary_utils import safe_mean_hist_summary, safe_mean_summary


class DQNXState(NamedTuple):
    q: Tensor
    target_q: Tensor = ()
    imitation_policy: Tensor = ()


class DQNXInfo(NamedTuple):
    action: Tensor = ()
    reward: Tensor = ()
    step_type: Tensor = ()
    discount: Tensor = ()
    action_distribution: td.Distribution = ()
    imitation_distribution: td.Distribution = ()
    q_values_for_action_dist: Tensor = ()  # [B, num_actions]
    v_values: Tensor = ()  # [B, q_dim]
    q_values: Tensor = ()  # [B, num_replicas, q_dim]
    target_q_values: Tensor = ()  # [B, q_dim]
    entropy: Tensor = ()  # [B]
    log_pi: Tensor = ()  # [B]
    rollout_log_pi: Tensor = ()  # [B]
    env_id: Tensor = ()


@alf.configurable
class DQNXAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 q_network_ctor=QNetwork,
                 num_replicas=1,
                 replica_training_sample_ratio=1.0,
                 num_sampled_critic_targets=0,
                 entropy_regularization=0.003,
                 imitation_prior_regularization=0.0,
                 alpha=0.99,
                 target_entropy: Union[None, float, Callable] = None,
                 fixed_prior_and_entropy_reg_ratio: bool = False,
                 temperature_uncertainty_ratio=0.0,
                 target_update_period=0,
                 use_entropy_reward=True,
                 log_pi_clip=-1.0,
                 delta_log_pi_clip=0.2,
                 gamma=0.99,
                 td_lambda=0.95,
                 calc_value_using_max_q=False,
                 normalize_q_value_using_advantages: bool = False,
                 normalize_q_value_using_pseudo_advantages: bool = False,
                 advantage_norm_momentum: float = 0.1,
                 td_error_loss_fn=losses.element_wise_squared_loss,
                 reward_weights=None,
                 epsilon_greedy=None,
                 epsilon_greedy_uniform=False,
                 top_k_sample=0,
                 rollout_action_sample_same_as_predict=False,
                 optimizer=None,
                 entropy_regularization_optimizer=None,
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
            replica_training_sample_ratio: the ratio of samples used for training
                each replica.
            entropy_regularization:
            imitation_prior_regularization:
            alpha=0.9: alpha in M-RL paper
                `(entropy_regularization + imitation_prior_regularization) / (1 - alpha)`
                is tau in M-RL paper
            log_pi_clip=-1.0,
            delta_log_pi_clip:
            target_entropy: If None, will use fixed entropy_regularization.
                If a float, will use this value as target entropy.
                If a callable function, will call it with action_spec to get target
                entropy. The `entropy_regularization` and `imitation_prior_regularization`
                will be adjusted to match the entropy of the action distribution
                to the target entropy.
            fixed_prior_and_entropy_reg_ratio: If True, when adjusting regularization for
                target entropy, will keep the ratio between entropy regularization
                and imitation_prior_regularization fixed and only adjust the sum.
                If False, will adjust only entropy regularization and do not
                change imitation_prior_regularization.
            gamma=0.99,
            td_lambda=0.95,
            calc_value_using_max_q: If True, use the max q value to calculate the
                state value. Otherwise, use the expected q value under the action
                distribution.
            normalize_q_values_using_advantages: If True, use the standard deviation
                of advantages to normalize the q values for calculating the action
                distribution. It seems that using the actual advantage to do normalization
                can lead to very concentrated action distribution if entropy_regularization+imitation_prior_regularization
                is not large enough. See https://colab.research.google.com/drive/12Pf-mo0wt-n6U07pGh2LUey-iPw5Igr2?usp=sharing
            normalize_q_value_using_pseudo_advantages: If True, use the standard
                deviation of psudo-advantages to normalize the q values for
                calculating the action distribution.
            advantage_norm_momentum (float): Momentum for moving average of
                the variance of advantages (same as the momentum for nn.BatchNorm1d).
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
        assert entropy_regularization + imitation_prior_regularization > 0, "Not supported"
        assert 0 < alpha < 1, f"Invalid alpha: {alpha}"

        self._num_replicas = num_replicas
        assert 0 <= num_sampled_critic_targets <= num_replicas
        self._num_sampled_critic_targets = num_sampled_critic_targets or num_replicas
        self._replica_training_sample_ratio = replica_training_sample_ratio

        self._q_dim = reward_spec.numel
        if target_update_period == 0:
            self._q_dim += 1  # one for entropy part
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy
        self._epsilon_greedy_uniform = epsilon_greedy_uniform
        self._top_k_sample = top_k_sample
        self._rollout_action_sample_same_as_predict = rollout_action_sample_same_as_predict

        original_observation_spec = observation_spec
        q_networks = q_network_ctor(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        if num_replicas * self._q_dim > 1:
            q_networks = q_networks.make_parallel(num_replicas * self._q_dim)

        imitation_actor_network = None
        if imitation_prior_regularization > 0:
            imitation_actor_network = alf.nn.Sequential(
                q_network_ctor(
                    input_tensor_spec=observation_spec,
                    action_spec=action_spec), lambda logits: td.Categorical(
                        logits=logits))
        train_state_spec = DQNXState(
            q=q_networks.state_spec,
            target_q=q_networks.state_spec if target_update_period > 0 else (),
            imitation_policy=imitation_actor_network.state_spec
            if imitation_actor_network else ())
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

        self._q_networks = q_networks
        self._imitation_actor_network = imitation_actor_network
        self._entropy_regularization = entropy_regularization
        self._imitation_prior_regularization = imitation_prior_regularization
        self._fixed_prior_and_entropy_reg_ratio = fixed_prior_and_entropy_reg_ratio
        self._alpha = alpha
        self._log_pi_clip = log_pi_clip
        self._gamma = torch.tensor(gamma)
        self._td_lambda = td_lambda
        self._calc_value_using_max_q = calc_value_using_max_q
        self._td_error_loss_fn = td_error_loss_fn

        self._temperature_uncertainty_ratio = temperature_uncertainty_ratio
        if temperature_uncertainty_ratio > 0:
            assert target_entropy is None, (
                "target_entropy should be None when temperature_uncertainty_ratio > 0"
            )

        if callable(target_entropy):
            target_entropy = target_entropy(action_spec)
        self._target_entropy = target_entropy
        if entropy_regularization == 0:
            entropy_regularization = 1e-30
        if fixed_prior_and_entropy_reg_ratio:
            log_entropy_reg = 0.0
        else:
            log_entropy_reg = math.log(entropy_regularization)
        self._log_entropy_reg = torch.tensor(log_entropy_reg)
        if target_entropy is not None:
            self._log_entropy_reg = torch.nn.Parameter(self._log_entropy_reg)
        if target_entropy is not None and entropy_regularization_optimizer is not None:
            self._log_entropy_reg_optimizer = self.add_optimizer(
                entropy_regularization_optimizer, [self._log_entropy_reg])

        reward_weights = torch.ones(self._q_dim)
        if self._reward_weights is not None:
            # User provided reward weights is for reward part only
            reward_weights[:self._reward_spec.numel] = self._reward_weights
        self._reward_weights = reward_weights
        self._use_entropy_reward = use_entropy_reward
        self._delta_log_pi_clip = delta_log_pi_clip

        self._target_q_networks = None
        if target_update_period > 0:
            self._target_q_networks = self._q_networks.copy(
                name='target_q_networks')
            self._target_q_networks.requires_grad_(False)
            self._update_target = common.TargetUpdater(
                models=[self._q_networks],
                target_models=[self._target_q_networks],
                tau=1 - alpha,
                period=target_update_period)
        # TODO: properly implementation for target_update_period==0 for imitation_prior_regularization > 0

        self._adv_norm = None
        assert not (
            normalize_q_value_using_advantages
            and normalize_q_value_using_pseudo_advantages), (
                "Can't use both normalize_q_value_using_advantages and "
                "normalize_q_value_using_pseudo_advantages at the same time!")
        normalize_q_values = normalize_q_value_using_advantages or normalize_q_value_using_pseudo_advantages
        if normalize_q_values:
            self._adv_norm = torch.nn.BatchNorm1d(
                num_features=1,
                eps=1e-8,
                momentum=advantage_norm_momentum,
                affine=False,
                track_running_stats=True)
            self._adv_norm.running_mean.fill_(1.0)
        self._normalize_q_values = normalize_q_values
        self._normalize_q_value_using_advantages = normalize_q_value_using_advantages

    def _trainable_attributes_to_ignore(self):
        return ['_target_q_networks']

    def after_update(self, root_inputs, info: DQNXInfo):
        if self._target_q_networks is not None:
            self._update_target()

    def _compute_q_values(self, observation, state):
        """

        Returns:
        - q_values: [B, num_replicas, num_actions, q_dim]
        - min_q_values: min q_values across replicas, [B, num_actions, q_dim]
        - q_values_for_action_dist: q values used for computing action_dist, [B, num_actions]
        - action_dist:
        - imitation_dist:
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
                return q[:, 0, :, :]

            if self._num_sampled_critic_targets < self._num_replicas:
                indices = random.sample(
                    range(self._num_replicas),
                    self._num_sampled_critic_targets)
                q = q[:, indices, :, :]

            if self.has_multidim_reward():
                sign = self.reward_weights.sign()
                return (q * sign).min(dim=1)[0] * sign
            else:
                return q.min(dim=1)[0]

        def _mean_q(q):
            if self._num_replicas == 1:
                return q[:, 0, :, :]
            else:
                return q.mean(dim=1)

        q_values, q_state = _calc_q(self._q_networks, state.q)
        min_q_values = _min_q(q_values)

        target_q_state = ()
        imitation_policy_state = ()
        imitation_dist = None
        if self._target_q_networks is None:
            summed_q_values = min_q_values @ self._reward_weights
            # Need this so that the gradient of imitation loss will not overwhelm
            # the TD loss
            summed_q_values = tensor_utils.scale_gradient(
                summed_q_values, self.total_regularization)
            q_values_for_action_dist = summed_q_values.to(torch.float32)
            action_logits = q_values_for_action_dist / (
                self.adv_std * self.total_regularization)
        elif self._temperature_uncertainty_ratio > 0:
            with torch.no_grad():
                target_q_values, target_q_state = _calc_q(
                    self._target_q_networks, state.target_q)
            # (1-alpha)*q_values + alpha*target_q_values
            combined_q_values = torch.lerp(q_values, target_q_values,
                                           self._alpha)
            # [B, num_replicas, num_actions]
            summed_q_values = combined_q_values @ self._reward_weights
            q_std, q_mean = torch.std_mean(summed_q_values, dim=1)
            q_values_for_action_dist = q_mean
            q_std += 1e-6
            temperature = q_std * self._temperature_uncertainty_ratio
            if self._imitation_actor_network is not None:
                imitation_dist, imitation_policy_state = self._imitation_actor_network(
                    observation, state.imitation_policy)
                imitation_reg = self.imitation_prior_regularization
                entropy_reg = self.entropy_regularization
                reg = imitation_reg + entropy_reg
                q_mean = q_mean + imitation_dist.logits * (
                    imitation_reg / reg) * temperature
            action_logits = calc_soft_policy(
                q_mean.to(torch.float32), temperature.to(torch.float32))
        else:
            with torch.no_grad():
                target_q_values, target_q_state = _calc_q(
                    self._target_q_networks, state.target_q)
            # (1-alpha)*q_values + alpha*target_q_values
            combined_q_values = torch.lerp(q_values, target_q_values,
                                           self._alpha)
            summed_q_values = _mean_q(combined_q_values) @ self._reward_weights
            # summed_q_values = _min_q(target_q_values) @ self._reward_weights
            q_values_for_action_dist = summed_q_values.to(torch.float32)
            imitation_reg = self.imitation_prior_regularization
            entropy_reg = self.entropy_regularization
            reg = imitation_reg + entropy_reg
            action_logits = q_values_for_action_dist / (self.adv_std * reg)
            if self._imitation_prior_regularization > 0:
                imitation_dist, imitation_policy_state = self._imitation_actor_network(
                    observation, state.imitation_policy)
                action_logits += imitation_dist.logits * (imitation_reg / reg)

        action_dist = td.Categorical(logits=action_logits)
        if imitation_dist is None:
            imitation_dist = action_dist

        return q_values, min_q_values, q_values_for_action_dist, action_dist, imitation_dist, DQNXState(
            q=q_state,
            target_q=target_q_state,
            imitation_policy=imitation_policy_state)

    @property
    def entropy_regularization(self):
        if self._fixed_prior_and_entropy_reg_ratio:
            return self._entropy_regularization * math.exp(
                self._log_entropy_reg.item())
        else:
            return math.exp(self._log_entropy_reg.item())

    @property
    def imitation_prior_regularization(self):
        if self._fixed_prior_and_entropy_reg_ratio:
            return self._imitation_prior_regularization * math.exp(
                self._log_entropy_reg.item())
        else:
            return self._imitation_prior_regularization

    @property
    def total_regularization(self):
        # total_regularization is tau in M-RL paper
        return (self.entropy_regularization +
                self.imitation_prior_regularization) / (1 - self._alpha)

    @property
    def adv_std(self):
        if self._normalize_q_values:
            return (
                self._adv_norm.running_mean.item() + self._adv_norm.eps)**0.5
        else:
            return 1.0

    def sample_action(self, action_dist):
        if self._epsilon_greedy_uniform:
            logits = action_dist.logits
            greedy_action = logits.argmax(dim=1)
            random_action = torch.randint_like(greedy_action, 0,
                                               logits.size(1))
            r = torch.rand_like(greedy_action, dtype=torch.float32)
            action = torch.where(r < self._epsilon_greedy, random_action,
                                 greedy_action)
        elif self._top_k_sample > 0:
            action = dist_utils.top_k_sample(action_dist, self._top_k_sample)
        else:
            action = dist_utils.epsilon_greedy_sample(action_dist,
                                                      self._epsilon_greedy)
        return action

    def predict_step(self, inputs: TimeStep, state: DQNXState):
        _, _, _, action_dist, _, new_state = self._compute_q_values(
            inputs.observation, state)
        action = self.sample_action(action_dist)
        return AlgStep(output=action, state=new_state)

    def rollout_step(self, inputs: TimeStep, state: DQNXState):
        _, q_values, q_values_for_action_dist, action_dist, imitation_dist, new_state = self._compute_q_values(
            inputs.observation, state)
        if self._rollout_action_sample_same_as_predict:
            action = self.sample_action(action_dist)
        else:
            action = dist_utils.sample_action_distribution(action_dist)
        if self._calc_value_using_max_q:
            # [B]
            action_indices = (q_values @ self._reward_weights).argmax(dim=1)
            # [B, num_rewards]
            v_values = q_values[torch.arange(action.shape[0]), action_indices]
        else:
            # [B, num_rewards]
            v_values = torch.einsum('bar,ba->br', q_values, action_dist.probs)

        entropy = action_dist.entropy()
        B = torch.arange(action.shape[0])
        log_pi = action_dist.logits[B, action]

        return AlgStep(
            output=action,
            state=new_state,
            info=DQNXInfo(
                action=action,
                action_distribution=action_dist,
                imitation_distribution=imitation_dist,  # debug
                q_values_for_action_dist=q_values_for_action_dist,
                q_values=q_values,
                v_values=v_values,
                entropy=entropy,
                log_pi=log_pi))

    def train_step(self, inputs: TimeStep, state: DQNXState,
                   rollout_info: DQNXInfo):
        q_values, _, q_values_for_action_dist, action_dist, imitation_dist, new_state = self._compute_q_values(
            inputs.observation, state)
        action = rollout_info.action
        B = torch.arange(action.shape[0])
        action_q_values = q_values[B, :, action]
        log_pi = action_dist.logits[B, action]
        return AlgStep(
            output=action,
            state=new_state,
            info=DQNXInfo(
                env_id=rollout_info.env_id,
                action_distribution=
                imitation_dist,  # This will be used for imitation loss
                step_type=inputs.step_type,
                entropy=rollout_info.entropy,
                q_values=action_q_values,
                log_pi=log_pi.detach(),
                rollout_log_pi=rollout_info.log_pi,
                target_q_values=rollout_info.target_q_values))

    def calc_loss(self, info: DQNXInfo):
        q_values = info.q_values  # [T, B, num_replicas, q_dim]
        # [T, B, num_replicas, q_dim]
        target_q_values = info.target_q_values[:, :, None, :].expand_as(
            q_values)
        td_error = target_q_values - q_values
        delta_log_pi = info.log_pi - info.rollout_log_pi
        delta_log_pi = delta_log_pi[:, :, None, None]
        clipped = ((delta_log_pi > self._delta_log_pi_clip) &
                   (td_error > 0)) | (
                       (delta_log_pi < -self._delta_log_pi_clip) &
                       (td_error < 0))
        critic_loss = self._td_error_loss_fn(target_q_values, q_values)

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, td_loss, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))
                    safe_mean_hist_summary('critics' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("critic_td_error" + suffix, td,
                                           mask)
                    safe_mean_summary("critic_loss" + suffix, td_loss, mask)
                    alf.summary.scalar("td_error_clip_ratio",
                                       clipped.sum() / clipped.numel())

                num_replicas = q_values.size(2)
                for r in range(min(2, num_replicas)):
                    for i in range(q_values.size(3)):
                        suffix = f'/replica_{r}/{i}'
                        _summarize(q_values[..., r, i],
                                   target_q_values[..., r, i],
                                   td_error[..., r, i], critic_loss[..., r, i],
                                   suffix)
                if self._normalize_q_values:
                    alf.summary.scalar("advantage_running_std",
                                       self._adv_norm.running_mean.item()**0.5)
                alf.summary.scalar("entropy_regularization",
                                   math.exp(self._log_entropy_reg.item()))

        critic_loss = critic_loss * ~clipped
        if self._replica_training_sample_ratio < 1:
            # [B, T, num_replicas]
            mask = pseudo_random_bit(
                info.env_id.unsqueeze(-1), torch.arange(self._num_replicas),
                self._replica_training_sample_ratio)
            critic_loss = critic_loss * mask.unsqueeze(-1)
        critic_loss = critic_loss.reshape(*critic_loss.shape[:2], -1).mean(-1)
        loss = critic_loss
        extra = {'critic': critic_loss, 'neg_entropy': -info.entropy}

        if self._target_entropy is not None:
            entropy_reg_loss = self._log_entropy_reg * (
                info.entropy - self._target_entropy).detach()
            loss = loss + entropy_reg_loss
            extra['entropy_reg_loss'] = entropy_reg_loss

        return LossInfo(loss=loss, extra=extra)

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        """Compute advantages and put it into exp.rollout_info."""

        # The device of rollout_info can be different from the default device
        # when ReplayBuffer.gather_all.convert_to_default_device is configured
        # to False to save gpu memory.
        step_type = convert_device(root_inputs.step_type)
        B, T = step_type.shape
        discount = convert_device(root_inputs.discount)
        reward = convert_device(root_inputs.reward).reshape(B, T, -1)
        value = convert_device(rollout_info.v_values)
        log_pi = convert_device(rollout_info.log_pi)
        discounts = discount.unsqueeze(-1) * self._gamma  # [B, T, reward_dim]

        reward_dim = reward.size(-1)
        advantages = value_ops.generalized_advantage_estimation(
            rewards=reward,
            values=value[:, :, :reward_dim],
            step_types=step_type,
            discounts=discounts,
            td_lambda=self._td_lambda,
            time_major=False)
        # [B, T, q_dim-1]
        advantages = tensor_utils.tensor_extend_zero(advantages, dim=1)
        target_q_values = value[:, :, :reward_dim] + advantages

        if self._target_q_networks is None:
            if self._log_pi_clip < 0:
                log_pi = log_pi.clamp(
                    min=self._log_pi_clip / self.total_regularization)

            if self._use_entropy_reward:
                entropy = convert_device(rollout_info.entropy)
                entropy = discount * entropy
                if self._alpha > 0:
                    entropy[:, 1:] += self._alpha * log_pi[:, :-1]
                # [B, T-1]
                target_q_m = value_ops.one_step_discounted_return(
                    rewards=self.total_regularization * entropy,
                    values=value[:, :, -1],
                    step_types=step_type,
                    discounts=discounts[:, :,
                                        0],  # use the same gamma as reward 0
                    time_major=False)
                # [B, T]
                target_q_m = torch.cat([target_q_m, value[:, -1:, -1]], dim=-1)
            elif self._alpha > 0:
                target_q_m = self._alpha * self.total_regularization * log_pi
            else:
                target_q_m = torch.zeros_like(log_pi)

            target_q_values = torch.cat(
                [target_q_values, target_q_m.unsqueeze(-1)], dim=-1)

        if self._normalize_q_values:
            q_values_for_action_dist = convert_device(
                rollout_info.q_values_for_action_dist)
            if self._normalize_q_value_using_advantages:
                probs = convert_device(rollout_info.action_distribution.probs)
                values = torch.einsum('bta,bta->bt', probs,
                                      q_values_for_action_dist).unsqueeze(-1)
                adv = q_values_for_action_dist - values
                adv_var = torch.einsum('bta,bta->bt', probs, adv**2)
            else:
                values = q_values_for_action_dist.mean(dim=-1, keepdim=True)
                adv = q_values_for_action_dist - values
                adv_var = (adv**2).mean(dim=-1)
            momentum = self._adv_norm.momentum
            if self._adv_norm.num_batches_tracked * momentum < 1.0:
                # For the first few batches, we do cumulative moving average
                self._adv_norm.momentum = None
            self._adv_norm(adv_var.reshape(-1, 1))
            self._adv_norm.momentum = momentum

        env_id = ()
        if self._replica_training_sample_ratio < 1:
            env_id = batch_info.env_ids[:, None].expand(B, T)

        return root_inputs, rollout_info._replace(
            env_id=env_id, target_q_values=target_q_values)

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


def calc_soft_policy(q, tau, tol=1e-6):
    r"""Solve weighted entropy regularized policy optimization problem.

    .. math::

        \max_p \sum_i p_i q_i - \sum_i \tau_i p_i \log(p_i)

    The solution has the following form:

    .. math::

        p_i = \exp\left(\frac{q_i-\tau_i-\lambda}{\tau_i}\right)

    where :math:`\lambda` is such that :math:`\sum_i p_i=1`

    We use Newton's method to solve for :math:`\lambda`.

    Args:
        q (Tensor): [B, num_actions]
        tau (Tensor): [B, num_actions]
        tol (float): tolerance for convergence
    """
    q = q - tau
    max_q = q.max(dim=-1)[0]
    lmbda = torch.zeros_like(max_q)
    q = q - max_q.unsqueeze(-1)

    for _ in range(30):
        logits = (q - lmbda.unsqueeze(-1)) / tau
        p = logits.exp()
        sum_p = p.sum(dim=-1)
        if sum_p.max().item() - 1 < tol:
            return logits
        deriv = (p / tau).sum(dim=-1)
        delta = (sum_p - 1) / deriv
        lmbda += delta

    raise ValueError(f"Failed to converge! sum_p.max()={sum_p.max().item()}")


def pseudo_random_bit(a, b, p):
    # Combine the two integers into a single hash value.
    # Here we multiply each by a large prime and use bitwise XOR to mix the bits.
    h = (a.to(torch.int64) * (0x9E3779B185EBCA87 // 2)) ^ (
        b.to(torch.int64) * (0xC2B2AE3D27D4EB4F // 2))
    # Ensure h is in the range [0, 2**32 - 1]
    h = h & 0xffffffff  # This is equivalent to taking h modulo 2^32.
    return h < (p * 2**32)
