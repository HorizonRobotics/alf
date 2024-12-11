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

from typing import NamedTuple

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
from alf.utils import losses, dist_utils, tensor_utils, value_ops
from alf.utils.summary_utils import safe_mean_hist_summary


class DQNXState(NamedTuple):
    q: Tensor


class DQNXInfo(NamedTuple):
    action: Tensor = ()
    reward: Tensor = ()
    step_type: Tensor = ()
    discount: Tensor = ()
    action_distribution: td.Distribution = ()
    v_values: Tensor = ()  # [B, q_dim]
    q_values: Tensor = ()  # [B, num_critic_replicas, q_dim]
    target_q_values: Tensor = ()  # [B, q_dim]
    entropy: Tensor = ()  # [B]
    log_pi: Tensor = ()  # [B]


@alf.configurable
class DQNXAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 q_network_ctor=QNetwork,
                 num_critic_replicas=1,
                 entropy_regularization=0.03,
                 alpha=0.9,
                 log_pi_clip=-1.0,
                 gamma=0.99,
                 td_lambda=0.95,
                 td_error_loss_fn=losses.element_wise_squared_loss,
                 reward_weights=None,
                 epsilon_greedy=None,
                 epsilon_greedy_uniform=False,
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
            num_critic_replicas=1,
            entropy_regularization=0.03,
            alpha=0.9,
            log_pi_clip=-1.0,
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

        self._num_critic_replicas = num_critic_replicas
        self._q_dim = reward_spec.numel + 1  # one for entropy part
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy
        self._epsilon_greedy_uniform = epsilon_greedy_uniform
        self._top_k_sample = top_k_sample

        original_observation_spec = observation_spec
        q_network = q_network_ctor(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        q_networks = q_network.make_parallel(num_critic_replicas * self._q_dim)

        train_state_spec = DQNXState(q=q_networks.state_spec)
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
        self._entropy_regularization = entropy_regularization
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

    def _compute_q_values(self, observation, state):
        """

        Returns:
        - q_values: [B, num_critic_replicas, num_actions, q_dim]
        - min_q_values: min q_values across replicas, [B, num_actions, q_dim]
        - action_dist:
        - state: the updated state
        """
        # [B, num_replicas * q_dim, num_actions]
        q_values, state = self._q_networks(observation, state)
        q_values = q_values.reshape(
            q_values.size(0), self._num_critic_replicas, self._q_dim, -1)
        # [B, num_critic_replicas, num_actions, q_dim]
        q_values = q_values.transpose(2, 3)

        if self._num_critic_replicas == 1:
            min_q_values = q_values[:, 0, :, :]
        elif self.has_multidim_reward():
            sign = self.reward_weights.sign()
            min_q_values = (q_values * sign).min(dim=1)[0] * sign
        else:
            min_q_values = q_values.min(dim=1)[0]

        summed_q_values = min_q_values @ self._reward_weights
        # Need this so that the gradient imitation loss will not overwhelm the
        # TD loss
        action_logits = tensor_utils.scale_gradient(
            summed_q_values,
            self._entropy_regularization) / self._entropy_regularization
        action_dist = td.Categorical(logits=action_logits)

        return q_values, min_q_values, action_dist, state

    def predict_step(self, inputs: TimeStep, state: DQNXState):
        _, q_values, action_dist, new_q_state = self._compute_q_values(
            inputs.observation, state.q)
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

        return AlgStep(output=action, state=DQNXState(q=new_q_state))

    def rollout_step(self, inputs: TimeStep, state: DQNXState):
        _, q_values, action_dist, new_q_state = self._compute_q_values(
            inputs.observation, state.q)
        action = dist_utils.sample_action_distribution(action_dist)
        # [B, num_rewards]
        v_values = torch.einsum('bar,ba->br', q_values, action_dist.probs)

        entropy = action_dist.entropy()

        log_pi = ()
        if self._alpha > 0:
            B = torch.arange(action.shape[0])
            log_pi = action_dist.logits[B, action]

        return AlgStep(
            output=action,
            state=DQNXState(q=new_q_state),
            info=DQNXInfo(
                action=action,
                action_distribution=action_dist,
                reward=inputs.reward,
                step_type=inputs.step_type,
                discount=inputs.discount,
                v_values=v_values,
                entropy=entropy,
                log_pi=log_pi))

    def train_step(self, inputs: TimeStep, state: DQNXState,
                   rollout_info: DQNXInfo):
        q_values, _, action_dist, new_q_state = self._compute_q_values(
            inputs.observation, state.q)
        action = rollout_info.action
        B = torch.arange(action.shape[0])
        action_q_values = q_values[B, :, action]
        return AlgStep(
            output=action,
            state=DQNXState(q=new_q_state),
            info=DQNXInfo(
                action_distribution=action_dist,
                step_type=rollout_info.step_type,
                entropy=rollout_info.entropy,
                q_values=action_q_values,
                target_q_values=rollout_info.target_q_values))

    def calc_loss(self, info: DQNXInfo):
        q_values = info.q_values  # [T, B, num_critic_replicas, q_dim]
        # [T, B, num_critic_replicas, q_dim]
        target_q_values = info.target_q_values[:, :, None, :].expand_as(
            q_values)
        td_error = target_q_values - q_values

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))
                    safe_mean_hist_summary('critics' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("critic_td_error" + suffix, td,
                                           mask)

                num_critic_replicas = q_values.size(2)
                for r in range(num_critic_replicas):
                    for i in range(q_values.size(3)):
                        suffix = f'/replica_{r}/{i}'
                        _summarize(q_values[..., r, i],
                                   target_q_values[..., r, i],
                                   td_error[..., r, i], suffix)

        loss = self._td_error_loss_fn(target_q_values, q_values)
        loss = loss.reshape(*loss.shape[:2], -1).mean(-1)

        return LossInfo(
            loss=loss, extra={
                'critic': loss,
                'neg_entropy': -info.entropy
            })

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        """Compute advantages and put it into exp.rollout_info."""

        # The device of rollout_info can be different from the default device
        # when ReplayBuffer.gather_all.convert_to_default_device is configured
        # to False to save gpu memory.
        step_type = convert_device(rollout_info.step_type)
        B, T = step_type.shape
        discount = convert_device(rollout_info.discount)
        reward = convert_device(rollout_info.reward).reshape(B, T, -1)
        value = convert_device(rollout_info.v_values)
        discounts = discount * self._gamma

        advantages = value_ops.generalized_advantage_estimation(
            rewards=reward,
            values=value[:, :, :-1],
            step_types=step_type,
            discounts=discounts,
            td_lambda=self._td_lambda,
            time_major=False)
        advantages = tensor_utils.tensor_extend_zero(advantages, dim=1)
        target_q_values = value[:, :, :-1] + advantages

        entropy = convert_device(rollout_info.entropy)
        entropy = discount * entropy
        if self._alpha > 0:
            log_pi = convert_device(rollout_info.log_pi)[:, :-1]
            if self._log_pi_clip < 0:
                log_pi = log_pi.clamp(
                    min=self._log_pi_clip / self._entropy_regularization)
            entropy[:, 1:] += self._alpha * log_pi
        target_q_m = value_ops.one_step_discounted_return(
            rewards=self._entropy_regularization * entropy,
            values=value[:, :, -1],
            step_types=step_type,
            discounts=discounts,
            time_major=False)
        target_q_m = torch.cat([target_q_m, value[:, -1:, -1]], dim=-1)

        target_q_values = torch.cat(
            [target_q_values, target_q_m.unsqueeze(-1)], dim=-1)
        return root_inputs, rollout_info._replace(
            target_q_values=target_q_values)

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
