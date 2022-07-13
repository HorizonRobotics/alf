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
from functorch import grad as func_grad

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

RegSacCriticInfo = namedtuple("RegSacCriticInfo", [
    "critics", "target_critic", "per_sample_critic_grad",
    "per_sample_target_critic_grad"
],
                              default_value=())
RegCriticLossInfo = namedtuple("RegCriticLossInfo", ["critic_loss"],
                               default_value=())


@alf.configurable
class RegSacAlgorithm(SacAlgorithm):
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
        reg=1.0,
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
        name="RegSacAlgorithm",
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
        self._reg = reg
        self._mini_batch_size = self._config.mini_batch_size

    def _get_per_sample_critic_grad(self, critic_net, observation, action,
                                    replica_min):
        f_critics, critic_params, critic_buffers = make_functional_with_buffers(
            critic_net)

        def compute_critics_stateless(params, buffers, observation, action):
            assert self._act_type == ActionType.Continuous
            assert not self.has_multidim_reward()

            observation = observation.unsqueeze(0)
            action = action.unsqueeze(0)
            observation = (observation, action)

            critics, _ = f_critics(params, buffers, observation)
            if replica_min:
                critics = critics.min(dim=1)[0]
                return critics.squeeze(), critics.squeeze(0)
            else:
                return critics.sum(1).squeeze(), critics.squeeze(0)

        grads_loss_output = grad_and_value(compute_critics_stateless,
                                           has_aux=True)
        compute_sample_grad = vmap(grads_loss_output,
                                   in_dims=(None, None, 0, 0))
        per_sample_grads, (_, critics) = compute_sample_grad(
            critic_params, critic_buffers, observation, action)

        return [element for element in per_sample_grads], critics

    def _critic_train_step(
        self,
        inputs: TimeStep,
        state: SacCriticState,
        rollout_info: SacInfo,
        action: torch.Tensor,
        action_distribution,
    ):
        bsz = self._mini_batch_size
        # Calculate the critics and value for both the current observation
        # and next observation
        # with torch.no_grad():
        # Obtain per-sample gradient
        ps_critic_grad, critics = self._get_per_sample_critic_grad(
            self._critic_networks,
            inputs.observation[:-bsz],
            rollout_info.action[:-bsz],
            replica_min=False)

        ps_target_critic_grad, target_critic = self._get_per_sample_critic_grad(
            self._critic_networks,
            inputs.observation[bsz:],
            action[bsz:],
            replica_min=True)

        # Prepend or Extend with zeros so that the tensors can be correctly
        # reshape to [T, B, ...]
        zeros = torch.zeros_like(critics[:bsz])
        critics = torch.cat((critics, zeros), dim=0)
        zeros = torch.zeros_like(target_critic[:bsz])
        target_critic = torch.cat((zeros, target_critic), dim=0)

        assert len(ps_critic_grad) == len(ps_target_critic_grad)
        for i in range(len(ps_critic_grad)):
            zeros = torch.zeros_like(ps_critic_grad[i][:bsz])
            ps_critic_grad[i] = torch.cat((ps_critic_grad[i], zeros), dim=0)
            ps_target_critic_grad[i] = torch.cat(
                (zeros, ps_target_critic_grad[i]), dim=0)

        state = SacCriticState(critics=(), target_critics=())
        info = RegSacCriticInfo(
            critics=critics,
            target_critic=target_critic.detach(),
            per_sample_critic_grad=ps_critic_grad,
            per_sample_target_critic_grad=ps_target_critic_grad,
        )

        return state, info

    def _calc_critic_loss(self, info: SacInfo):
        bsz = self._mini_batch_size
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
        ps_target_critic_grad = critic_info.per_sample_target_critic_grad

        critic_losses = []
        losses = []
        T = ps_critic_grad[0].shape[0]
        optimizer = self._module_to_optimizer.get(self._critic_networks)
        lr = optimizer.param_groups[0]['lr']
        inner_prodcuts = []
        for i, l in enumerate(self._critic_losses):
            loss = l(
                info=info,
                value=critic_info.critics[:, :, i, ...],
                target_value=critic_info.target_critic,
            )
            td_error = loss.extra[:, :bsz // 2]
            surrogate_loss = []
            inner_prod_list = []
            for grad, target_grad, p in zip(
                    ps_critic_grad, ps_target_critic_grad,
                    self._critic_networks.parameters()):
                with torch.no_grad():
                    # First calculate gradient for TD Loss
                    nabla_critic = grad[:, :bsz // 2, i]
                    loss_grad = -2 * nabla_critic * common.expand_dims_as(
                        td_error, nabla_critic)
                    loss_grad = loss_grad.mean(dim=(0, 1))

                    # # Calcuate the compensation term
                    # nabla_critic = grad[:-1, bsz // 2:, i]
                    # nabla_target_critic = target_grad[1:, bsz // 2:, i]
                    # inner_prodcut = (nabla_target_critic * loss_grad).reshape(
                    #     T - 1, bsz // 2, -1).sum(-1)
                    # compensation = nabla_critic * common.expand_dims_as(
                    #     inner_prodcut, nabla_critic)
                    # compensation = tensor_utils.tensor_extend_zero(
                    #     compensation)
                    # compensation = (0.5 * lr * gamma *
                    #                 compensation).mean(dim=(0, 1))

                    # Calcuate the compensation term
                    nabla_critic = grad[:-1, bsz // 2:, i]
                    nabla_target_critic = target_grad[1:, bsz // 2:, i]
                    inner_prodcut = (nabla_critic * loss_grad).reshape(
                        T - 1, bsz // 2, -1).sum(-1)
                    diff = nabla_critic - gamma * nabla_target_critic
                    stable_grad = diff * common.expand_dims_as(
                        inner_prodcut, diff)

                    stable_grad = tensor_utils.tensor_extend_zero(
                        stable_grad).mean(dim=(0, 1))

                inner_prod_list.append(inner_prodcut)
                # surrogate_loss.append(
                #     ((loss_grad - self._reg * compensation).detach() *
                #      p[i]).sum())
                surrogate_loss.append((stable_grad.detach() * p[i]).sum())

                # crtic_loss.append((loss_grad.detach() * p[i]).sum())

            losses.append(math_ops.add_n(surrogate_loss))
            critic_losses.append(loss.loss)
            inner_prodcuts.append(math_ops.add_n(inner_prod_list))

        critic_loss = math_ops.add_n(critic_losses)
        loss = math_ops.add_n(losses)
        inner_prod = math_ops.add_n(inner_prodcuts)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                safe_mean_hist_summary(
                    'nablda Q(s, a) norm',
                    torch.cat([
                        grad[:-1].reshape(T - 1, bsz, -1)
                        for grad in ps_critic_grad
                    ],
                              dim=-1).norm(dim=-1))

                safe_mean_hist_summary(
                    "nablda Q(s', a') norm",
                    torch.cat([
                        grad[1:].reshape(T - 1, bsz, -1)
                        for grad in ps_target_critic_grad
                    ],
                              dim=-1).norm(dim=-1))

                safe_mean_hist_summary("Inner Product", inner_prod / 2)

        if self._calculate_priority:
            valid_masks = (info.step_type != StepType.LAST).to(torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = ((critic_loss * valid_masks).sum(dim=0) /
                        valid_n).sqrt()
        else:
            priority = ()

        return LossInfo(
            loss=loss,
            priority=priority,
            extra=RegCriticLossInfo(critic_loss=critic_loss /
                                    float(self._num_critic_replicas), ),
        )

    def after_update(self, root_inputs, info: SacInfo):
        if self._max_log_alpha is not None:
            nest.map_structure(
                lambda la: la.data.copy_(torch.min(la, self._max_log_alpha)),
                self._log_alpha)

    def _trainable_attributes_to_ignore(self):
        return []
