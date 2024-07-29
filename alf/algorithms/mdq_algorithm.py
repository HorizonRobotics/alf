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
"""Multi-Dimensional Q-Learning Algorithm."""

import functools

import torch
import torch.nn as nn
from typing import Callable

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep
from alf.nest import nest
from alf.networks import MdqCriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import (losses, common, dist_utils, math_ops, spec_utils,
                       tensor_utils)

MdqCriticState = namedtuple("MdqCriticState", ['critic', 'target_critic'])
MdqCriticInfo = namedtuple("MdqCriticInfo", [
    "critic_free_form", "target_critic_free_form", "critic_adv_form",
    "distill_target", "kl_wrt_prior"
])

MdqState = namedtuple("MdqState", ['critic'])
MdqAlphaInfo = namedtuple("MdqAlphaInfo", ["alpha_loss", "neg_entropy"])
MdqInfo = namedtuple(
    "MdqInfo",
    ["reward", "step_type", "discount", "action", "critic", "alpha"],
    default_value=())

MdqLossInfo = namedtuple('MdqLossInfo', ['critic', 'distill', 'alpha'])


@alf.configurable
class MdqAlgorithm(OffPolicyAlgorithm):
    """Multi-Dimensional Q-Learning Algorithm.
    """

    def __init__(
            self,
            observation_spec,
            action_spec: BoundedTensorSpec,
            critic_network: MdqCriticNetwork,
            reward_spec=TensorSpec(()),
            epsilon_greedy=None,
            env=None,
            config: TrainerConfig = None,
            critic_loss_ctor=None,
            target_entropy=dist_utils.calc_default_target_entropy_quantized,
            initial_log_alpha=0.0,
            target_update_tau=0.05,
            target_update_period=1,
            distill_noise=0.01,
            critic_optimizer=None,
            alpha_optimizer=None,
            debug_summaries=False,
            name="MdqAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            critic_network (MdqCriticNetwork): an instance of MdqCriticNetwork
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            initial_log_alpha (float): initial value for variable ``log_alpha``.
            target_entropy (float|Callable): If a floating value, it's the
                target average policy entropy, for updating ``alpha``. If a
                callable function, then it will be called on the action spec to
                calculate a target entropy. Note that in MDQ algorithm, as the
                continuous action is represented by a discrete distribution for
                each action dimension, ``calc_default_target_entropy_quantized``
                is used to compute the target entropy by default.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            distill_noise (int): the std of random Gaussian noise added to the
                action used for distillation.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """

        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy

        critic_networks = critic_network
        target_critic_networks = critic_networks.copy(
            name='target_critic_networks')

        train_state_spec = MdqState(
            critic=MdqCriticState(
                critic=critic_networks.state_spec,
                target_critic=critic_networks.state_spec))

        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        self._critic_networks = critic_networks
        self._target_critic_networks = target_critic_networks

        self.add_optimizer(critic_optimizer, [critic_networks])

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)

        flat_action_spec = nest.flatten(self._action_spec)
        self._flat_action_spec = flat_action_spec
        self._action_dim = flat_action_spec[0].shape[0]
        self._log_pi_uniform_prior = self._critic_networks.get_uniform_prior_logpi(
        )

        self._num_critic_replicas = self._critic_networks._num_critic_replicas

        self._critic_losses = []

        for i in range(self._num_critic_replicas):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))

        self._is_continuous = flat_action_spec[0].is_continuous
        self._target_entropy = _set_target_entropy(self.name, target_entropy,
                                                   flat_action_spec)

        log_alpha = nn.Parameter(torch.tensor(float(initial_log_alpha)))
        self._log_alpha = log_alpha

        self._update_target = common.TargetUpdater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, [log_alpha])
        self._distill_noise = distill_noise

    def _predict(self, time_step: TimeStep, state=None, epsilon_greedy=1.):
        # Note that here get_action will do greedy sampling only if
        # epsilon_greedy is 0. This option is provided for evaluation purpose
        # if greedy sampling is desirable.
        action, _ = self._critic_networks.get_action(
            time_step.observation,
            alpha=torch.exp(self._log_alpha).detach(),
            greedy=(epsilon_greedy == 0))

        # slice over action when num_critic_replicas > 1
        # [B, n, d] -> [B, d]
        action = action[:, 0, :]

        empty_state = nest.map_structure(lambda x: (), self.train_state_spec)

        return AlgStep(
            output=action, state=empty_state, info=MdqInfo(action=action))

    def predict_step(self, time_step: TimeStep, state):
        return self._predict(time_step, state, self._epsilon_greedy)

    def rollout_step(self, time_step: TimeStep, state):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by SacAlgorithm")

        return self._predict(time_step, state, epsilon_greedy=1.0)

    def _critic_train_step(self, inputs: TimeStep, state: MdqCriticState,
                           rollout_action, action, log_pi_per_dim):
        alpha = self._log_alpha.exp().detach()

        critic_input = (inputs.observation, rollout_action.to(torch.float32))
        target_critic_input = (inputs.observation, action.detach())

        # [B, n]
        critic, critic_state = self._critic_networks(
            torch.cat(critic_input, -1),
            alpha=alpha,
            state=state.critic,
            free_form=True)

        noisy_distill_action = self._get_noisy_action(
            action,
            self._action_spec,
            self._distill_noise,
            noise_clip=0,
            spec_clip=True)

        critic_distill_input = (inputs.observation,
                                noisy_distill_action.detach())

        # [B, n, action_dim]
        critic_adv_form, critic_state = self._critic_networks(
            critic_distill_input,
            alpha=alpha,
            state=state.critic,
            free_form=False)

        target_critic_input_new = (tensor_utils.tensor_extend_new_dim(
            target_critic_input[0], dim=1, n=self._num_critic_replicas),
                                   target_critic_input[1])

        distill_critic_input_new = (tensor_utils.tensor_extend_new_dim(
            critic_distill_input[0], dim=1, n=self._num_critic_replicas),
                                    critic_distill_input[1])

        target_critic, target_critic_state = self._target_critic_networks(
            torch.cat(target_critic_input_new, -1),
            alpha=alpha,
            state=state.target_critic,
            free_form=True)

        # Note that in MDQ we distill from the target_critic_network.
        distill_target, _ = self._target_critic_networks(
            torch.cat(distill_critic_input_new, -1),
            alpha=alpha,
            state=state.target_critic,
            free_form=True)

        kl_wrt_prior_per_dim = log_pi_per_dim - self._log_pi_uniform_prior
        # keeping the KL of all actions dimensions in case it is useful
        # in some cases in the future, e.g., per-action target correction using
        # the corresponding KL
        kl_wrt_prior = tensor_utils.reverse_cumsum(
            kl_wrt_prior_per_dim, dim=-1)
        info = MdqCriticInfo(
            critic_free_form=critic,
            target_critic_free_form=target_critic,
            distill_target=distill_target,
            critic_adv_form=critic_adv_form,
            kl_wrt_prior=kl_wrt_prior)

        state = MdqCriticState(
            critic=critic_state, target_critic=target_critic_state)

        return state, info

    def _alpha_train_step(self, log_pi_per_dim):
        """ Adjusting alpha according to target entropy.
        Args:
            log_pi_per_dim (torch.Tensor): a tensor of the shape
                [B, n, action_dim] representing the log_pi for each dimension
                of the sampled multi-dimensional action
        """

        log_pi_full = log_pi_per_dim.sum(dim=-1)
        alpha_loss = self._log_alpha * (
            -log_pi_full - self._target_entropy).detach()

        # mean over critic
        alpha_loss = torch.mean(alpha_loss, -1).view(-1)

        neg_entropy = torch.mean(log_pi_full.squeeze(-1), -1).view(-1)

        info = LossInfo(
            loss=alpha_loss,
            extra=MdqAlphaInfo(alpha_loss=alpha_loss, neg_entropy=neg_entropy))
        return info

    def train_step(self, inputs: TimeStep, state: MdqState, rollout_info):

        alpha = torch.exp(self._log_alpha).detach()

        action, log_pi_per_dim = self._critic_networks.get_action(
            inputs.observation, alpha=alpha, greedy=False)
        action = action[:, 0:1, :].expand_as(action)
        log_pi_per_dim = log_pi_per_dim[:, 0:1, :].expand_as(log_pi_per_dim)

        critic_state, critic_info = self._critic_train_step(
            inputs, state.critic, rollout_info.action, action, log_pi_per_dim)

        alpha_info = self._alpha_train_step(log_pi_per_dim)

        state = MdqState(critic=critic_state)
        info = MdqInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            critic=critic_info,
            alpha=alpha_info)
        return AlgStep(action, state, info)

    def after_update(self, root_inputs, info: MdqInfo):
        # sync parallel/non-parallel network parameters
        # need to syn net first in the case of using target net as policy
        self._critic_networks.sync_net()
        self._update_target()

    def calc_loss(self, info: MdqInfo):
        alpha_loss = info.alpha
        critic_loss, distill_loss = self._calc_critic_loss(info)

        total_loss = critic_loss.loss + distill_loss + alpha_loss.loss.squeeze(
            -1)
        return LossInfo(
            loss=total_loss,
            extra=MdqLossInfo(
                critic=critic_loss.extra,
                alpha=alpha_loss.extra,
                distill=distill_loss))

    def _calc_critic_loss(self, train_info: MdqInfo):
        critic_info = train_info.critic

        # [t, B, n]
        critic_free_form = critic_info.critic_free_form
        # [t, B, n, action_dim]
        critic_adv_form = critic_info.critic_adv_form
        target_critic_free_form = critic_info.target_critic_free_form
        distill_target = critic_info.distill_target

        num_critic_replicas = critic_free_form.shape[2]

        alpha = torch.exp(self._log_alpha).detach()
        kl_wrt_prior = critic_info.kl_wrt_prior

        # [t, B, n, action_dim] -> [t, B]
        # note that currently the kl_wrt_prior is independent of ensembles,
        # we therefore slice over ensemble by taking the first element;
        # for the action dimension, the first element is the full KL
        kl_wrt_prior = kl_wrt_prior[..., 0, 0]

        # [t, B, n] -> [t, B]
        target_critic, min_target_ind = torch.min(
            target_critic_free_form, dim=2)

        # [t, B, n] -> [t, B]
        distill_target, _ = torch.min(distill_target, dim=2)

        target_critic_corrected = target_critic - alpha * kl_wrt_prior

        critic_losses = []
        for j in range(num_critic_replicas):
            critic_losses.append(self._critic_losses[j](
                info=train_info,
                value=critic_free_form[:, :, j],
                target_value=target_critic_corrected).loss)

        critic_loss = math_ops.add_n(critic_losses)

        distill_loss = (
            critic_adv_form[..., -1] - distill_target.unsqueeze(2).detach())**2
        # mean over replica
        distill_loss = distill_loss.mean(dim=2)

        return LossInfo(
            loss=critic_loss,
            extra=critic_loss / len(critic_losses)), distill_loss

    def _get_noisy_action(self,
                          actions,
                          action_specs,
                          noise_level,
                          noise_clip=0.0,
                          spec_clip=True):
        if noise_level > 0:
            max_action = torch.as_tensor(action_specs.maximum)
            noise = torch.randn_like(actions) * noise_level * max_action
            if noise_clip > 0:
                noise = noise.clamp(
                    min=-noise_clip * max_action, max=noise_clip * max_action)
            noisy_action = actions + noise
            if spec_clip:
                clipped_noisy_action = spec_utils.clip_to_spec(
                    noisy_action, action_specs)
            else:
                clipped_noisy_action = noisy_action

            return clipped_noisy_action
        else:
            return actions

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']
