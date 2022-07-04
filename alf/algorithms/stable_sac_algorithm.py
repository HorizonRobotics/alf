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
import torch
from functorch import make_functional_with_buffers, vmap
from functorch import grad as func_grad

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.sac_algorithm import SacAlgorithm, SacInfo, ActionType
from alf.algorithms.sac_algorithm import SacCriticState, SacLossInfo
from alf.data_structures import StepType, TimeStep, namedtuple, LossInfo
from alf.nest import nest
from alf.networks import ActorDistributionNetwork, CriticNetwork, EncodingNetwork
from alf.networks.containers import _Sequential
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, math_ops

StableSacCriticInfo = namedtuple("SacCriticInfo", [
    "critics", "target_critic", "features", "target_features",
    "per_sample_critic_grad"
])
StableCriticLossInfo = namedtuple("DOacCriticLossInfo",
                                  ["critic_loss", "interf_reg"],
                                  default_value=())


@alf.configurable
class StableSacAlgorithm(SacAlgorithm):
    def __init__(
        self,
        observation_spec,
        action_spec: BoundedTensorSpec,
        reward_spec=TensorSpec(()),
        actor_network_cls=ActorDistributionNetwork,
        feat_network_cls=CriticNetwork,
        critic_network_cls=EncodingNetwork,
        use_target_network=True,
        epsilon_greedy=None,
        use_entropy_reward=True,
        calculate_priority=False,
        num_critic_replicas=2,
        env=None,
        config: TrainerConfig = None,
        critic_loss_ctor=None,
        target_entropy=None,
        prior_actor_ctor=None,
        discourage_interf=False,
        inferf_reg_coef=1e-4,
        target_kld_per_dim=3.0,
        initial_log_alpha=0.0,
        max_log_alpha=None,
        target_update_tau=0.05,
        target_update_period=1,
        dqda_clipping=None,
        actor_optimizer=None,
        feat_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        debug_summaries=False,
        name="StableSacAlgorithm",
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

        assert (self._act_type == ActionType.Continuous
                ), "Only continuous action space is supported."

        feat_networks = feat_network_cls(
            input_tensor_spec=(observation_spec, action_spec),
            use_naive_parallel_network=True,
        )
        self._feat_networks = feat_networks.make_parallel(num_critic_replicas *
                                                          reward_spec.numel)
        self._target_feat_networks = self._feat_networks.copy(
            name="target_feat_networks")
        self.add_optimizer(feat_optimizer, [self._feat_networks])

        self._f_critics = None

        self._update_target_feat = common.TargetUpdater(
            models=[self._feat_networks],
            target_models=[self._target_feat_networks],
            tau=target_update_tau,
            period=target_update_period,
        )

        self._inferf_reg_coef = inferf_reg_coef
        self._discourage_interf = discourage_interf

        self._use_target_network = use_target_network
        self._mini_batch_size = self._config.mini_batch_size

    def _make_networks(
        self,
        observation_spec,
        action_spec,
        reward_spec,
        continuous_actor_network_cls,
        critic_network_cls,
        q_network_cls,
    ):
        def _make_parallel(net):
            return net.make_parallel(self._num_critic_replicas *
                                     reward_spec.numel)

        assert continuous_actor_network_cls is not None, (
            "If there are continuous actions, then a ActorDistributionNetwork "
            "must be provided for sampling continuous actions!")
        actor_network = continuous_actor_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        act_type = ActionType.Continuous
        assert critic_network_cls is not None, (
            "If only continuous actions exist, then a CriticNetwork must"
            " be provided!")
        assert (
            "input_tensor_spec" in critic_network_cls.keywords
        ), "The 'input_tensor_spec' should be specificied in the config!"
        critic_network = critic_network_cls()
        critic_networks = _make_parallel(critic_network)

        return critic_networks, actor_network, act_type

    def _compute_critics(
        self,
        critic_net,
        observation,
        action,
        critics_state,
        replica_min: bool = True,
        use_target_feat: bool = False,
        return_features: bool = False,
    ):
        assert not self.has_multidim_reward()

        observation = (observation, action)
        feat_net = (self._target_feat_networks
                    if use_target_feat else self._feat_networks)
        features, _ = feat_net(observation)

        critics, critics_state = critic_net(features, state=critics_state)
        assert not self.has_multidim_reward()
        if replica_min:
            critics = critics.min(dim=1)[0]

        if return_features:
            return features, critics, critics_state
        else:
            return critics, critics_state

    def _compute_critics_stateless(self, params, buffers, observation, action):
        assert not self.has_multidim_reward()

        observation = observation.unsqueeze(0)
        action = action.unsqueeze(0)

        critics, _ = self._f_critics(params, buffers, (observation, action))
        assert not self.has_multidim_reward()

        return critics.sum(1).squeeze()

    def _get_per_sample_critic_grad(self, observation, action):
        self._f_critics, critic_params, critic_buffers = make_functional_with_buffers(
            _Sequential(
                [self._feat_networks, self._critic_networks],
                input_tensor_spec=self._feat_networks._input_tensor_spec))

        ft_compute_grad = func_grad(self._compute_critics_stateless)
        ft_compute_sample_grad = vmap(ft_compute_grad,
                                      in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(critic_params,
                                                     critic_buffers,
                                                     observation, action)

        num_modules_total = (len(ft_per_sample_grads) - 2)
        num_feat_modules = num_modules_total // self._num_critic_replicas
        ft_per_sample_grads_new = []
        bsz = ft_per_sample_grads[0].shape[0]
        for i in range(num_feat_modules):
            indices = range(i, num_modules_total, num_feat_modules)
            grad = torch.cat([
                ft_per_sample_grads[idx].unsqueeze(1).reshape(bsz, 1, -1)
                for idx in indices
            ],
                             dim=1)
            ft_per_sample_grads_new.append(grad)
        ft_per_sample_grads_new.extend([
            grad.reshape(bsz, self._num_critic_replicas, -1)
            for grad in ft_per_sample_grads[-2:]
        ])
        ft_per_sample_grads_new = torch.cat(ft_per_sample_grads_new, dim=-1)

        return ft_per_sample_grads_new

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
        features, critics, critics_state = self._compute_critics(
            self._critic_networks,
            inputs.observation[:-bsz],
            rollout_info.action[:-bsz],
            state.critics,
            replica_min=False,
            return_features=True,
        )

        # Extend with zeros so that the tensors can be correctly reshape to
        # [T, B, ...]
        zeros = torch.zeros_like(critics[:bsz])
        critics = torch.cat((critics, zeros), dim=0)
        zeros = torch.zeros_like(features[:bsz])
        features = torch.cat((features, zeros), dim=0)

        # Obtain per-sample gradient
        per_sample_critic_grad = self._get_per_sample_critic_grad(
            inputs.observation[:-bsz], rollout_info.action[:-bsz])

        zeros = torch.zeros_like(per_sample_critic_grad[:bsz])
        per_sample_critic_grad = torch.cat((per_sample_critic_grad, zeros),
                                           dim=0)

        (
            target_features,
            target_critic,
            target_critics_state,
        ) = self._compute_critics(
            self._target_critic_networks,
            inputs.observation[bsz:],
            action[bsz:],
            state.target_critics,
            use_target_feat=True,
            return_features=True,
        )
        # Prepend with zeros so that the tensors can be correctly reshape to
        # [T, B, ...]
        zeros = torch.zeros_like(target_critic[:bsz])
        target_critic = torch.cat((zeros, target_critic), dim=0)
        zeros = torch.zeros_like(target_features[:bsz])
        target_features = torch.cat((zeros, target_features), dim=0)

        target_critic = target_critic.detach()

        state = SacCriticState(critics=critics_state,
                               target_critics=target_critics_state)
        info = StableSacCriticInfo(
            critics=critics,
            target_critic=target_critic,
            features=features,
            target_features=target_features,
            per_sample_critic_grad=per_sample_critic_grad,
        )

        return state, info

    def calc_loss(self, info: SacInfo):
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_loss = info.actor

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha", self._log_alpha.exp())

        assert not self._reproduce_locomotion
        loss = math_ops.add_ignore_empty(actor_loss.loss,
                                         critic_loss.loss + alpha_loss)

        return LossInfo(
            loss=loss,
            priority=critic_loss.priority,
            extra=SacLossInfo(actor=actor_loss.extra,
                              critic=critic_loss.extra,
                              alpha=alpha_loss),
        )

    def _calc_critic_loss(self, info: SacInfo):
        if self._use_entropy_reward:
            with torch.no_grad():
                log_pi = info.log_pi
                if self._entropy_normalizer is not None:
                    log_pi = self._entropy_normalizer.normalize(log_pi)
                entropy_reward = nest.map_structure(
                    lambda la, lp: -torch.exp(la) * lp, self._log_alpha,
                    log_pi)
                entropy_reward = sum(nest.flatten(entropy_reward))
                discount = self._critic_losses[0].gamma * info.discount
                info = info._replace(
                    reward=(info.reward +
                            common.expand_dims_as(entropy_reward *
                                                  discount, info.reward)))

        critic_info = info.critic
        per_sample_grad = critic_info.per_sample_critic_grad[:-1]
        critic_losses = []
        interf_regs = []
        for i, l in enumerate(self._critic_losses):
            loss = l(
                info=info,
                value=critic_info.critics[:, :, i, ...],
                target_value=critic_info.target_critic,
            )
            grad = per_sample_grad[:, :, i]
            ntk_matrix = torch.bmm(grad, grad.transpose(1, 2))
            td_error = loss.extra
            value_diff = ntk_matrix * td_error
            diag_diff = value_diff.diagonal(dim1=1, dim2=2)
            if self._discourage_interf:
                reg = (value_diff.sum(-1) - diag_diff) / (value_diff.shape[1] -
                                                          1)
            else:
                # Should we detach the diag_diff to make it like a regression task？
                reg = (value_diff.sum(-1) - diag_diff) / (value_diff.shape[1] -
                                                          1) - diag_diff

            critic_losses.append(loss.loss)
            interf_regs.append(reg.abs())

        critic_loss = math_ops.add_n(critic_losses)
        interf_reg = math_ops.add_n(interf_regs)
        loss = critic_loss + self._inferf_reg_coef * interf_reg

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
            extra=StableCriticLossInfo(
                critic_loss=critic_loss / float(self._num_critic_replicas),
                interf_reg=interf_reg / float(self._num_critic_replicas),
            ),
        )

    def after_update(self, root_inputs, info: SacInfo):
        super().after_update(root_inputs, info)
        self._update_target_feat()

    def _trainable_attributes_to_ignore(self):
        return [
            "_target_critic_networks",
            "_target_feat_networks",
        ]
