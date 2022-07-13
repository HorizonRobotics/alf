# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from functools import partial
import torch
from functorch import make_functional_with_buffers, vmap, grad_and_value

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.sac_algorithm import SacAlgorithm, SacInfo, ActionType
from alf.algorithms.sac_algorithm import SacCriticState
from alf.data_structures import StepType, TimeStep, namedtuple, LossInfo
from alf.nest import nest
from alf.networks import ActorDistributionNetwork, EncodingNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, math_ops, tensor_utils
from alf.utils.summary_utils import safe_mean_hist_summary

StableTDCriticInfo = namedtuple("StableTDCriticInfo", [
    "critics",
    "target_critic",
    "per_sample_critic_grad",
],
                                default_value=())
StableTDCriticLossInfo = namedtuple("StableTDCriticLossInfo",
                                    ["critic_loss", "semi_grad_norm"],
                                    default_value=())


@alf.configurable
class StableTDAlgorithm(SacAlgorithm):
    def __init__(
        self,
        observation_spec,
        action_spec: BoundedTensorSpec,
        reward_spec=TensorSpec(()),
        actor_network_cls=ActorDistributionNetwork,
        critic_network_cls=EncodingNetwork,
        epsilon_greedy=None,
        use_entropy_reward=True,
        calculate_priority=False,
        num_critic_replicas=2,
        env=None,
        config: TrainerConfig = None,
        critic_loss_ctor=None,
        target_entropy=None,
        prior_actor_ctor=None,
        coef=1.0,
        target_kld_per_dim=3.0,
        initial_log_alpha=0.0,
        max_log_alpha=None,
        target_update_tau=0.05,
        target_update_period=1,
        dqda_clipping=None,
        actor_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        debug_summaries=False,
        name="StableTDAlgorithm",
    ):

        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            critic_network_cls=critic_network_cls,
            epsilon_greedy=epsilon_greedy,
            use_entropy_reward=use_entropy_reward,
            calculate_priority=calculate_priority,
            num_critic_replicas=num_critic_replicas,
            env=env,
            config=config,
            critic_loss_ctor=critic_loss_ctor,
            target_entropy=target_entropy,
            prior_actor_ctor=prior_actor_ctor,
            target_kld_per_dim=target_kld_per_dim,
            initial_log_alpha=initial_log_alpha,
            max_log_alpha=max_log_alpha,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            dqda_clipping=dqda_clipping,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            debug_summaries=debug_summaries,
            name=name,
        )
        del self._target_critic_networks
        self._coef = coef
        self._f_critics, self._critic_params, self._critic_buffers = None, None, None
        self._mini_batch_size = self._config.mini_batch_size

    def _get_per_sample_critic_grad(self, observation, action):
        def compute_critics_stateless(params, buffers, observation, action):
            assert self._act_type == ActionType.Continuous
            assert not self.has_multidim_reward()

            observation = observation.unsqueeze(0)
            action = action.unsqueeze(0)
            observation = (observation, action)

            critics, _ = self._f_critics(params, buffers, observation)
            return critics.sum(1).squeeze(), critics.squeeze(0)

        grads_loss_output = grad_and_value(compute_critics_stateless,
                                           has_aux=True)
        compute_sample_grad = vmap(grads_loss_output,
                                   in_dims=(None, None, 0, 0))
        ps_grads, (_, critics) = compute_sample_grad(self._critic_params,
                                                     self._critic_buffers,
                                                     observation, action)

        return [element for element in ps_grads], critics

    def _critic_train_step(
        self,
        inputs: TimeStep,
        state: SacCriticState,
        rollout_info: SacInfo,
        action: torch.Tensor,
        action_distribution,
    ):
        bsz = self._mini_batch_size
        self._f_critics, self._critic_params, self._critic_buffers = \
            make_functional_with_buffers(self._critic_networks)

        # Obtain per-sample gradient
        ps_critic_grad, critics = self._get_per_sample_critic_grad(
            inputs.observation[:-bsz], rollout_info.action[:-bsz])

        observation = (inputs.observation[bsz:], action[bsz:])
        target_critics, _ = self._f_critics(self._critic_params,
                                            self._critic_buffers, observation)
        target_critic = target_critics.min(dim=1)[0]

        # Prepend or Extend with zeros so that the tensors can be correctly
        # reshape to [T, B, ...]
        zeros = torch.zeros_like(critics[:bsz])
        critics = torch.cat((critics, zeros), dim=0)
        zeros = torch.zeros_like(target_critic[:bsz])
        target_critic = torch.cat((zeros, target_critic), dim=0)

        for i in range(len(ps_critic_grad)):
            zeros = torch.zeros_like(ps_critic_grad[i][:bsz])
            ps_critic_grad[i] = torch.cat((ps_critic_grad[i], zeros), dim=0)

        state = SacCriticState(critics=(), target_critics=())
        info = StableTDCriticInfo(
            critics=critics,
            target_critic=target_critic,
            per_sample_critic_grad=ps_critic_grad,
        )

        return state, info

    def _calc_critic_loss(self, info: SacInfo):
        gamma = self._critic_losses[0].gamma

        if self._use_entropy_reward:
            with torch.no_grad():
                log_pi = info.log_pi
                if self._entropy_normalizer is not None:
                    log_pi = self._entropy_normalizer.normalize(log_pi)
                entropy_reward = nest.map_structure(
                    lambda la, lp: -torch.exp(la) * lp, self._log_alpha,
                    log_pi)
                entropy_reward = sum(nest.flatten(entropy_reward))
                discount = gamma * info.discount
                info = info._replace(
                    reward=(info.reward +
                            common.expand_dims_as(entropy_reward *
                                                  discount, info.reward)))

        critic_info = info.critic
        ps_critic_grad = critic_info.per_sample_critic_grad

        critic_losses = []
        td_errors = []
        for i, l in enumerate(self._critic_losses):
            loss = l(
                info=info,
                value=critic_info.critics[:, :, i, ...],
                target_value=critic_info.target_critic,
            )
            critic_losses.append(loss.loss)
            td_errors.append(loss.extra)

        critic_loss = math_ops.add_n(critic_losses)
        td_error = torch.stack(td_errors, dim=-1)

        loss_gradients = []
        for grad in ps_critic_grad:
            # First calculate gradient for TD Loss
            loss_grad = -2 * grad * common.expand_dims_as(td_error, grad)
            loss_grad = loss_grad.mean(dim=(0, 1))
            loss_gradients.append(loss_grad)

        stable_loss = math_ops.add_n(
            [0.5 * (grad * grad).sum() for grad in loss_gradients])
        stable_gradients = torch.autograd.grad(stable_loss,
                                               self._critic_params)

        surrogate_losses = []
        for loss_grad, stable_grad, p in zip(
                loss_gradients, stable_gradients,
                self._critic_networks.parameters()):
            surrogate_losses.append(
                ((loss_grad + self._coef * stable_grad).detach() * p).sum())

        surrogate_loss = math_ops.add_n(surrogate_losses)

        if self._calculate_priority:
            valid_masks = (info.step_type != StepType.LAST).to(torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = ((critic_loss * valid_masks).sum(dim=0) /
                        valid_n).sqrt()
        else:
            priority = ()

        return LossInfo(
            loss=surrogate_loss,
            priority=priority,
            extra=StableTDCriticLossInfo(critic_loss=critic_loss /
                                         float(self._num_critic_replicas),
                                         semi_grad_norm=stable_loss),
        )

    def after_update(self, root_inputs, info: SacInfo):
        if self._max_log_alpha is not None:
            nest.map_structure(
                lambda la: la.data.copy_(torch.min(la, self._max_log_alpha)),
                self._log_alpha)

    def _trainable_attributes_to_ignore(self):
        return []
