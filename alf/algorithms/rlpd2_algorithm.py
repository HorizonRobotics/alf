# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""RLPD Algorithm."""

import functools
import torch
import torch.nn as nn
from typing import Callable, Optional, Union

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rlpd_algorithm import RlpdAlgorithm
from alf.algorithms.sac_algorithm import SacActionState
from alf.algorithms.sac_algorithm import ActionType, SacInfo, SacState
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.data_structures import LossInfo, namedtuple, StepType
from alf.nest import nest
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, math_ops
from alf.utils.schedulers import Scheduler
from alf.utils.summary_utils import safe_mean_hist_summary

RlpdCriticState = namedtuple(
    "RlpdCriticState", [
        "critics", "target_critics", "aux_critics", "target_aux_critics"
    ],
    default_value=())

RlpdCriticInfo = namedtuple(
    "RlpdCriticInfo", [
        "critics", "target_critic", "aux_critics", "target_aux_critic"
    ],
    default_value=())


@alf.configurable
class Rlpd2Algorithm(RlpdAlgorithm):
    r"""A variant of the following RLPD algorithm:

    ::

        Ball et al "Efficient Online Reinforcement Learning with Offline Data", arXiv:2302.02948

    Currently, only continuous action spaces are supported.

    There are two difference with the above algorithm:

    1. Only online data buffer is used, so it is a pure off-policy algorithm.

    2. Optimization uncertainty/std of the critics is estimated by maintaining 
    auxiliary critics that are trained with perturbed TD learning. This optimization 
    uncertainty is further used to determined the importance weights of training
    sampled during critics training.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 repr_alg_ctor: Optional[Callable] = None,
                 reward_weights=None,
                 train_eps_greedy=1.0,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 normalize_entropy_reward=False,
                 calculate_priority=False,
                 num_critic_replicas=2,
                 num_critic_targets=2,
                 num_aux_critics=0,
                 use_bootstrap_critics=False,
                 bootstrap_mask_prob=0.8,
                 critic_actor_utd_ratio=1,
                 aux_critic_use_common_target=True,
                 critic_training_weight=1.0,
                 use_total_std_norm_ctw=False,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_entropy=None,
                 prior_actor_ctor=None,
                 target_kld_per_dim=3.,
                 initial_log_alpha=0.0,
                 max_log_alpha=None,
                 target_update_tau: Union[float, Scheduler] = 0.05,
                 target_update_period: Union[int, Scheduler] = 1,
                 parameter_reset_period: Union[int, Scheduler] = -1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 aux_critic_optimizer=None,
                 alpha_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 name="Rlpd2Algorithm"):
        """
        Refer to RlpdAlgorithm for details of arguments besides the following,

            num_aux_critics (int): Number of optimization-perturbed critics 
                for critics optimization uncertainty estimation.
            aux_critic_use_common_target (bool): whether to use the same TD target
                critic as default critic for aux critics training.
            critic_training_weight (float): each training sample will be weighted
                according the critic optimization std with exponent 
                ``critic_training_weignt``.
            use_total_std_norm_ctw (bool): whether to use the total std of critics 
                to normalize the critic_training_weignt
        """
        self._num_critic_replicas = num_critic_replicas
        self._num_critic_targets = num_critic_targets
        self._num_aux_critics = num_aux_critics
        self._use_bootstrap_critics = use_bootstrap_critics
        self._bootstrap_mask_prob = bootstrap_mask_prob
        self._bootstrap_mask = None
        self._aux_critic_use_common_target = aux_critic_use_common_target
        self._calculate_priority = calculate_priority
        self._train_eps_greedy = train_eps_greedy
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy
        self._critic_training_weight = critic_training_weight
        self._use_total_std_norm_ctw = use_total_std_norm_ctw
        self._critic_actor_utd_ratio = critic_actor_utd_ratio
        self._critic_train_counter = 0

        original_observation_spec = observation_spec
        if repr_alg_ctor is not None:
            repr_alg = repr_alg_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                reward_spec=reward_spec,
                debug_summaries=debug_summaries,
                config=config)
            target_repr_alg = repr_alg_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                reward_spec=reward_spec,
                debug_summaries=debug_summaries,
                config=config)
            assert hasattr(repr_alg,
                           'output_spec'), "repr_alg must have output_spec"
            observation_spec = repr_alg.output_spec
        else:
            repr_alg = None
            target_repr_alg = None

        critic_networks, actor_network, aux_critic_networks, self._act_type = \
            self._make_networks(
                observation_spec, action_spec, reward_spec, actor_network_cls,
                critic_network_cls)

        assert self._act_type == ActionType.Continuous, (
            "RLPD algorithm only supports continuous action spaces.")

        self._use_entropy_reward = use_entropy_reward

        if reward_spec.numel > 1:
            assert self._act_type != ActionType.Mixed, (
                "Only continuous/discrete action is supported for multidimensional reward"
            )

        def _init_log_alpha():
            return nn.Parameter(torch.tensor(float(initial_log_alpha)))

        log_alpha = _init_log_alpha()

        action_state_spec = SacActionState(
            actor_network=actor_network.state_spec,
            critic=())
        critic_state = RlpdCriticState(
            critics=critic_networks.state_spec,
            target_critics=critic_networks.state_spec)
        if num_aux_critics > 0:
            critic_state._replace(
                aux_critics=aux_critic_networks.state_spec,
                target_aux_critics=aux_critic_networks.state_spec)
        train_state_spec = SacState(
            action=action_state_spec,
            actor=critic_networks.state_spec,
            critic=critic_state,
            repr=repr_alg.train_state_spec if repr_alg else (),
            target_repr=target_repr_alg.predict_state_spec
            if target_repr_alg else ())

        OffPolicyAlgorithm.__init__(
            self,
            observation_spec=original_observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            rollout_state_spec=train_state_spec._replace(
                repr=repr_alg.rollout_state_spec if repr_alg else ()),
            predict_state_spec=SacState(
                action=action_state_spec,
                repr=repr_alg.predict_state_spec if repr_alg else ()),
            reward_weights=reward_weights,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        if not self._is_eval:
            assert critic_networks is not None, (
                "critic_networks must be provided for training RLPD")

        if actor_optimizer is not None and actor_network is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None and critic_networks is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if aux_critic_optimizer is not None and aux_critic_networks is not None:
            self.add_optimizer(aux_critic_optimizer, [aux_critic_networks])
        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, nest.flatten(log_alpha))
        self._log_alpha = log_alpha
        if self._act_type == ActionType.Mixed:
            self._log_alpha_paralist = nn.ParameterList(
                nest.flatten(log_alpha))

        if max_log_alpha is not None:
            self._max_log_alpha = torch.tensor(float(max_log_alpha))
        else:
            self._max_log_alpha = None

        self._actor_network = actor_network
        self._critic_networks = critic_networks
        self._aux_critic_networks = aux_critic_networks
        self._target_critic_networks = None
        self._target_aux_critic_networks = None
        # Note, q_network (discrete actions) is still needed for evaluating the algorithm.
        if critic_networks:
            self._target_critic_networks = self._critic_networks.copy(
                name='target_critic_networks')

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        # Have different names to separate their summary curves
        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))

        if aux_critic_networks:
            self._aux_weights = torch.empty((num_aux_critics, ))
            self._target_aux_critic_networks = self._aux_critic_networks.copy(
                name='target_aux_critic_networks')
            self._aux_critic_losses = []
            for i in range(num_aux_critics):
                self._aux_critic_losses.append(
                    critic_loss_ctor(name="aux_critic_loss%d" % (i + 1)))

        self._prior_actor = None
        if prior_actor_ctor is not None:
            self._prior_actor = prior_actor_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                debug_summaries=debug_summaries)
            total_action_dims = sum(
                [spec.numel for spec in alf.nest.flatten(action_spec)])
            self._target_entropy = -target_kld_per_dim * total_action_dims
        else:
            self._target_entropy = _set_target_entropy(
                self.name, target_entropy, nest.flatten(self._action_spec))

        self._dqda_clipping = dqda_clipping
        self._training_started = False
        self._reproduce_locomotion = False  # for compatibility with SAC

        self._entropy_normalizer = None
        if normalize_entropy_reward:
            self._entropy_normalizer = ScalarAdaptiveNormalizer(unit_std=True)

        self._repr_alg = repr_alg
        self._target_repr_alg = target_repr_alg

        def _filter(x):
            return list(filter(lambda x: x is not None, x))

        def _create_target_updater():
            self._update_target = common.TargetUpdater(
                models=_filter([
                    self._critic_networks, self._aux_critic_networks, repr_alg]),
                target_models=_filter([self._target_critic_networks, 
                                       self._target_aux_critic_networks, 
                                       target_repr_alg]),
                tau=target_update_tau,
                period=target_update_period)

        _create_target_updater()

        # no need to include ``target_critic_networks`` and ``target_repr_alg``
        # since their parameter values will be copied from ``self._critic_networks``
        # and ``repr_alg`` upon each reset via ``post_processings``
        self._periodic_reset = common.PeriodicReset(
            models=_filter([
                self._actor_network, self._critic_networks, repr_alg,
                self._log_alpha
            ]),
            post_processings=[_create_target_updater],
            period=parameter_reset_period)

        # The following checkpoint loading hook handles the case when critic
        # network is not constructed. In this case the critic network parameters
        # present in the checkpoint should be ignored.
        def _deployment_hook(state_dict, prefix: str, unused_loacl_metadata,
                             unused_strict, unused_missing_keys,
                             unused_unexpected_keys, unused_error_msgs):
            to_delete = []
            for key in state_dict:
                if not key.startswith(prefix):
                    continue
                if critic_networks is None:
                    if key[len(prefix):].startswith("_critic_networks") or key[
                            len(prefix):].startswith(
                                "_target_critic_networks"):
                        to_delete.append(key)
            for key in to_delete:
                state_dict.pop(key)

        self._register_load_state_dict_pre_hook(_deployment_hook)

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       continuous_actor_network_cls, critic_network_cls):
        def _make_parallel(net):
            return net.make_parallel(
                self._num_critic_replicas * reward_spec.numel)

        def _check_spec_equal(spec1, spec2):
            assert nest.flatten(spec1) == nest.flatten(spec2), (
                "Unmatched action specs: {} vs. {}".format(spec1, spec2))

        actor_network = None
        critic_networks = None
        aux_critic_networks = None
        assert continuous_actor_network_cls is not None, (
            "If there are continuous actions, then a ActorDistributionNetwork "
            "must be provided for sampling continuous actions!")
        actor_network = continuous_actor_network_cls(
            input_tensor_spec=observation_spec,
            action_spec=action_spec)
        act_type = ActionType.Continuous
        if critic_network_cls is not None:
            critic_network = critic_network_cls(
                input_tensor_spec=(observation_spec, action_spec))
            critic_networks = _make_parallel(critic_network)
            if self._num_aux_critics > 0:
                aux_critic_networks = alf.networks.NaiveParallelNetwork(
                    critic_network, self._num_aux_critics, deepcopy=True) 

        return critic_networks, actor_network, aux_critic_networks, act_type

    def _critic_train_step(self, observation, target_observation,
                           state: RlpdCriticState, rollout_info: SacInfo,
                           action, action_distribution):
        critics, critics_state = self._compute_critics(
            self._critic_networks,
            observation,
            rollout_info.action,
            state.critics,
            replica_consensus=None,
            apply_reward_weights=False)

        with torch.no_grad():
            target_critics, target_critics_state = self._compute_critics(
                self._target_critic_networks,
                target_observation,
                action,
                state.target_critics,
                replica_consensus='min',
                sample_subset=True,
                apply_reward_weights=False)

        target_critic = target_critics.reshape(target_critics.shape[0],
                                               *self._reward_spec.shape)

        target_critic = target_critic.detach()

        state = RlpdCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = RlpdCriticInfo(critics=critics, target_critic=target_critic)

        if self._num_aux_critics > 0:
            aux_critics, aux_critics_state = self._compute_critics(
                self._aux_critic_networks,
                observation,
                rollout_info.action,
                state.aux_critics,
                replica_consensus=None,
                apply_reward_weights=False)

            state = state._replace(
                aux_critics=aux_critics_state)
            info = info._replace(
                aux_critics=aux_critics)

            if not self._aux_critic_use_common_target:
                with torch.no_grad():
                    target_aux_critics, target_aux_critics_state = self._compute_critics(
                        self._target_aux_critic_networks,
                        target_observation,
                        action,
                        state.target_aux_critics,
                        replica_consensus='min',
                        apply_reward_weights=False)

                target_aux_critic = target_aux_critics.reshape(
                    target_aux_critics.shape[0], *self._reward_spec.shape)

                target_aux_critic = target_aux_critic.detach()

                state = state._replace(
                    target_aux_critics=target_aux_critics_state)
                info = info._replace(
                    target_aux_critic=target_aux_critic)

        return state, info

    def _calc_critic_loss(self, info: SacInfo):
        """
        We need to put entropy reward in ``experience.reward`` instead of ``target_critics``
        because in the case of multi-step TD learning, the entropy should also
        appear in intermediate steps! This doesn't affect one-step TD loss, however.

        Following the SAC official implementation,
        https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py#L32
        for StepType.LAST with discount=0, we mask out both the entropy reward
        and the target Q value. The reason is that there is no guarantee of what
        the last entropy will look like because the policy is never trained on
        that. If the entropy is very small, the the agent might hesitate to terminate
        the episode.
        (There is an issue in their implementation: their "terminals" can't
        differentiate between discount=0 (NormalEnd) and discount=1 (TimeOut).
        In the latter case, masking should not be performed.)

        When the reward is multi-dim, the entropy reward will be added to *all*
        dims.
        """
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
                    reward=(info.reward + common.expand_dims_as(
                        entropy_reward * discount, info.reward)))

        critic_info = info.critic
        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            critic_loss = l(info=info,
                            value=critic_info.critics[:, :, i, ...],
                            target_value=critic_info.target_critic).loss
            if self._use_bootstrap_critics:
                bootstrap_mask = info.bootstrap_mask[:, :, i] / self._bootstrap_mask_prob
                critic_loss = critic_loss * bootstrap_mask
            critic_losses.append(critic_loss)

        # for auxiliary critics
        if self._num_aux_critics > 0:
            if self._aux_critic_use_common_target:
                target_aux_critic = critic_info.target_critic
            else:
                target_aux_critic = critic_info.target_aux_critic
            self._aux_weights.exponential_(1.0)
            for i, l in enumerate(self._aux_critic_losses):
                weights = self._aux_weights[i]
                critic_losses.append(weights * l(
                    info=info,
                    value=critic_info.aux_critics[:, :, i, ...],
                    target_value=target_aux_critic).loss)

        critic_loss = math_ops.add_n(critic_losses)

        # compute the std of target_aux_critics as the estimation of 
        # optimization uncertainty
        if self._num_aux_critics > 0:
            base_q = critic_info.critics[:, :, :1, ...]
            q_bootstrap = critic_info.critics[:, :, 1:, ...]
            q_bootstrap_diff = q_bootstrap - base_q
            q_total_std = (q_bootstrap_diff**2).mean(dim=2).sqrt()
            q_aux_diff = critic_info.aux_critics - base_q
            q_aux_std = (q_aux_diff**2).mean(dim=2).sqrt()
            opt_weights = q_aux_std
            if self._use_total_std_norm_ctw:
                opt_weights = opt_weights / (q_total_std + 1e-6)
            opt_weights = opt_weights.detach() ** self._critic_training_weight
            opt_weights = opt_weights * opt_weights.numel() / opt_weights.sum()
            # reweight training samples w.r.t. optimization uncertainty
            # critic_loss *= opt_weights
            if self._debug_summaries and alf.summary.should_record_summaries():
                with alf.summary.scope(self._name):
                    safe_mean_hist_summary("total_critic_std", q_total_std)
                    safe_mean_hist_summary("aux_critic_std", q_aux_std)
                    safe_mean_hist_summary("critic_opt_priority", opt_weights)

        if self._calculate_priority:
            if self._num_aux_critics > 0:
                priority = opt_weights
            else:
                valid_masks = (info.step_type != StepType.LAST).to(torch.float32)
                valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
                priority = (
                    (critic_loss * valid_masks).sum(dim=0) / valid_n).sqrt()
        else:
            priority = ()

        return LossInfo(
            loss=critic_loss,
            priority=priority,
            extra=critic_loss / float(self._num_critic_replicas))

    def _trainable_attributes_to_ignore(self):
        ignored = super()._trainable_attributes_to_ignore()
        if self._num_aux_critics > 0:
            ignored.append('_target_aux_critic_networks')
        return ignored
