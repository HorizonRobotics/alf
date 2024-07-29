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
import numpy as np
import functools
from enum import Enum

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable, Optional, Union

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks import QNetwork, QRNNNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops
from alf.utils.normalizers import ScalarAdaptiveNormalizer
from alf.utils.schedulers import Scheduler

ActionType = Enum('ActionType', ('Discrete', 'Continuous', 'Mixed'))

SacActionState = namedtuple(
    "SacActionState", ["actor_network", "critic"], default_value=())

SacCriticState = namedtuple("SacCriticState", ["critics", "target_critics"])

SacState = namedtuple(
    "SacState", ["action", "actor", "critic", "repr", "target_repr"],
    default_value=())

SacCriticInfo = namedtuple("SacCriticInfo", ["critics", "target_critic"])

SacActorInfo = namedtuple(
    "SacActorInfo", ["actor_loss", "neg_entropy"], default_value=())

SacInfo = namedtuple(
    "SacInfo", [
        "reward", "step_type", "discount", "action", "action_distribution",
        "actor", "critic", "alpha", "log_pi", "discounted_return", "repr"
    ],
    default_value=())

SacLossInfo = namedtuple(
    'SacLossInfo', ('actor', 'critic', 'alpha', 'repr'), default_value=())


def _set_target_entropy(name, target_entropy, flat_action_spec):
    """A helper function for computing the target entropy under different
    scenarios of ``target_entropy``.

    Args:
        name (str): the name of the algorithm that calls this function.
        target_entropy (float|Callable|None): If a floating value, it will return
            as it is. If a callable function, then it will be called on the action
            spec to calculate a target entropy. If ``None``, a default entropy will
            be calculated.
        flat_action_spec (list[TensorSpec]): a flattened list of action specs.
    """
    if target_entropy is None or callable(target_entropy):
        if target_entropy is None:
            target_entropy = dist_utils.calc_default_target_entropy
        target_entropy = np.sum(list(map(target_entropy, flat_action_spec)))
        logging.info("Target entropy is calculated for {}: {}.".format(
            name, target_entropy))
    else:
        logging.info("User-supplied target entropy for {}: {}".format(
            name, target_entropy))
    return target_entropy


@alf.configurable
class SacAlgorithm(OffPolicyAlgorithm):
    r"""Soft Actor Critic algorithm, described in:

    ::

        Haarnoja et al "Soft Actor-Critic Algorithms and Applications", arXiv:1812.05905v2

    There are 3 points different with ``tf_agents.agents.sac.sac_agent``:

    1. To reduce computation, here we sample actions only once for calculating
    actor, critic, and alpha loss while ``tf_agents.agents.sac.sac_agent``
    samples actions for each loss. This difference has little influence on
    the training performance.

    2. We calculate losses for every sampled steps.
    :math:`(s_t, a_t), (s_{t+1}, a_{t+1})` in sampled transition are used
    to calculate actor, critic and alpha loss while
    ``tf_agents.agents.sac.sac_agent`` only uses :math:`(s_t, a_t)` and
    critic loss for :math:`s_{t+1}` is 0. You should handle this carefully,
    it is equivalent to applying a coefficient of 0.5 on the critic loss.

    3. We mask out ``StepType.LAST`` steps when calculating losses but
    ``tf_agents.agents.sac.sac_agent`` does not. We believe the correct
    implementation should mask out ``LAST`` steps. And this may make different
    performance on same tasks.

    In addition to continuous actions addressed by the original paper, this
    algorithm also supports discrete actions and a mixture of discrete and
    continuous actions. The networks for computing Q values :math:`Q(s,a)` and
    sampling actions can be divided into 3 cases according to action types:

    1. Discrete only: a ``QNetwork`` is used for estimating Q values. There will
       be no actor network to learn because actions can be directly sampled from
       the Q values: :math:`p(a|s) \propto \exp(\frac{Q(s,a)}{\alpha})`.
    2. Continuous only: a ``CriticNetwork`` is used for estimating Q values. An
       ``ActorDistributionNetwork`` for sampling actions will be learned according
       to Q values.
    3. Mixed: a ``QNetwork`` is used for estimating Q values. The input of this
       particular ``QNetwork`` (dubbed as "Universal Q Network") is augmented
       with all continuous actions as ``(observation, continuous_action)``,
       while the output heads correspond to discrete actions. So a Q value
       :math:`Q(s, a_{cont}, a_{disc}=k)` is estimated by the :math:`k`-th output
       head of the network given :math:`a_{cont}` as the augmented input to
       :math:`s`. Still only an ``ActorDistributionNetwork`` is needed for first
       sampling continuous actions, and then a discrete action is sampled from Q
       values conditioned on the continuous actions. See
       ``alf/docs/notes/sac_with_hybrid_action_types.rst`` for training details.

    In addition to the entropy regularization described in the SAC paper, we
    also support KL-Divergence regularization if a prior actor is provided.
    In this case, the training objective is:
        :math:`E_\pi(\sum_t \gamma^t(r_t - \alpha D_{\rm KL}(\pi(\cdot)|s_t)||\pi^0(\cdot)|s_t)))`
    where :math:`pi^0` is the prior actor.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 q_network_cls=QNetwork,
                 repr_alg_ctor: Optional[Callable] = None,
                 reward_weights=None,
                 train_eps_greedy=1.0,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 normalize_entropy_reward=False,
                 calculate_priority=False,
                 num_critic_replicas=2,
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
                 alpha_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 reproduce_locomotion=False,
                 name="SacAlgorithm"):
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
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network will be called
                to sample continuous actions. All of its output specs must be
                continuous. Note that we don't need a discrete actor network
                because a discrete action can simply be sampled from the Q values.
            critic_network_cls (None or Callable): is used to construct critic network.
                for estimating ``Q(s,a)`` given that the action is continuous.  Note that
                if the algorithm is constructed for evaluation or deployment only, the
                critic_network_cls can be set to None and the network will not be
                constructed at all.
            q_network (Callable): is used to construct QNetwork for estimating ``Q(s,a)``
                given that the action is discrete. Its output spec must be consistent with
                the discrete action in ``action_spec``.
            repr_alg_ctor: if provided, it will be called as ``repr_alg_ctor(
                observation_spec, action_spec, reward_spec, config=config)`` to
                construct a representation learning algorithm. The output of the
                representation learning algorithm is used as the input of the
                actor and critic networks. Different from using representation_learner_cls
                in ``Agent``, a target model of the representation learning algorithm
                will be maintained and the representation calculated by the target
                representation learning algorithm will be used for computing
                target critics.
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor if reward_weights is not None.
                Otherwise, the sum of the q values is used.
            train_eps_greedy (float): a floating value in [0,1], representing the
                chance of taking sampled action during training.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            use_entropy_reward (bool): whether to include entropy as reward
            normalize_entropy_reward (bool): if True, normalize entropy reward
                to reduce bias in episodic cases. Only used if
                ``use_entropy_reward==True``.
            calculate_priority (bool): whether to calculate priority. This is
                only useful if priority replay is enabled.
            num_critic_replicas (int): number of critics to be used. Default is 2.
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
            max_log_alpha (float|None): if not None, ``log_alpha`` will be
                capped at this value.
            target_entropy (float|Callable|None): If a floating value, it's the
                target average policy entropy, for updating ``alpha``. If a
                callable function, then it will be called on the action spec to
                calculate a target entropy. If ``None``, a default entropy will
                be calculated. For the mixed action type, discrete action and
                continuous action will have separate alphas and target entropies,
                so this argument can be a 2-element list/tuple, where the first
                is for discrete action and the second for continuous action.
            prior_actor_ctor (Callable): If provided, it will be called using
                ``prior_actor_ctor(observation_spec, action_spec, debug_summaries=debug_summaries)``
                to constructor a prior actor. The output of the prior actor is
                the distribution of the next action. Two prior actors are implemented:
                ``alf.algorithms.prior_actor.SameActionPriorActor`` and
                ``alf.algorithms.prior_actor.UniformPriorActor``.
            target_kld_per_dim (float): ``alpha`` is dynamically adjusted so that
                the KLD is about ``target_kld_per_dim * dim``.
            target_update_tau: Factor for soft update of the target
                networks.
            target_update_period: Period in terms of gradient updates for soft update of
                the target networks.
            parameter_reset_period: Period in terms of iterations for resetting the value
                of learnable parameters. If negative, no reset is done.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between
                ``[-dqda_clipping, dqda_clipping]``. Will not perform clipping if
                ``dqda_clipping == 0``.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
            debug_summaries (bool): True if debug summaries should be created.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            reproduce_locomotion (bool): if True, some slight tweaks are added
                to the original SAC to roughly reproducing its reported results
                on MuJoCo locomotion tasks. These include uniform action sampling
                in the beginning and different masks for actor and critic losses.
            name (str): The name of this algorithm.
        """
        self._num_critic_replicas = num_critic_replicas
        self._calculate_priority = calculate_priority
        self._train_eps_greedy = train_eps_greedy
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy

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

        critic_networks, actor_network, self._act_type = self._make_networks(
            observation_spec, action_spec, reward_spec, actor_network_cls,
            critic_network_cls, q_network_cls)

        self._use_entropy_reward = use_entropy_reward

        if reward_spec.numel > 1:
            assert self._act_type != ActionType.Mixed, (
                "Only continuous/discrete action is supported for multidimensional reward"
            )

        def _init_log_alpha():
            return nn.Parameter(torch.tensor(float(initial_log_alpha)))

        if self._act_type == ActionType.Mixed:
            # separate alphas for discrete and continuous actions
            log_alpha = type(action_spec)((_init_log_alpha(),
                                           _init_log_alpha()))
        else:
            log_alpha = _init_log_alpha()

        action_state_spec = SacActionState(
            actor_network=(() if self._act_type == ActionType.Discrete else
                           actor_network.state_spec),
            critic=(() if self._act_type == ActionType.Continuous
                    or critic_network_cls is None else
                    critic_networks.state_spec))
        train_state_spec = SacState(
            action=action_state_spec,
            actor=(() if self._act_type != ActionType.Continuous or
                   critic_network_cls is None else critic_networks.state_spec),
            critic=SacCriticState(
                critics=critic_networks.state_spec if critic_network_cls else
                (),
                target_critics=critic_networks.state_spec
                if critic_network_cls else ()),
            repr=repr_alg.train_state_spec if repr_alg else (),
            target_repr=target_repr_alg.predict_state_spec
            if target_repr_alg else ())

        super().__init__(
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

        if not self._is_eval and self._act_type != ActionType.Discrete:
            assert critic_networks is not None, (
                "critic_networks must be provided for training continuous SAC")

        if actor_optimizer is not None and actor_network is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None and critic_networks is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
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
        self._target_critic_networks = None
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

        self._prior_actor = None
        if prior_actor_ctor is not None:
            assert self._act_type == ActionType.Continuous, (
                "Only continuous action is supported when using prior_actor")
            self._prior_actor = prior_actor_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                debug_summaries=debug_summaries)
            total_action_dims = sum(
                [spec.numel for spec in alf.nest.flatten(action_spec)])
            self._target_entropy = -target_kld_per_dim * total_action_dims
        else:
            if self._act_type == ActionType.Mixed:
                if not isinstance(target_entropy, (tuple, list)):
                    target_entropy = nest.map_structure_up_to(
                        nest.nest_top_level(
                            self._action_spec), lambda _: target_entropy,
                        self._action_spec)
                # separate target entropies for discrete and continuous actions
                self._target_entropy = nest.map_structure_up_to(
                    target_entropy, lambda spec, t: _set_target_entropy(
                        self.name, t, nest.flatten(spec)), self._action_spec,
                    target_entropy)
            else:
                self._target_entropy = _set_target_entropy(
                    self.name, target_entropy, nest.flatten(self._action_spec))

        self._dqda_clipping = dqda_clipping

        self._training_started = False
        self._reproduce_locomotion = reproduce_locomotion

        self._entropy_normalizer = None
        if normalize_entropy_reward:
            self._entropy_normalizer = ScalarAdaptiveNormalizer(unit_std=True)

        self._repr_alg = repr_alg
        self._target_repr_alg = target_repr_alg

        def _filter(x):
            return list(filter(lambda x: x is not None, x))

        def _create_target_updater():
            self._update_target = common.TargetUpdater(
                models=_filter([self._critic_networks, repr_alg]),
                target_models=_filter(
                    [self._target_critic_networks, target_repr_alg]),
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
                       continuous_actor_network_cls, critic_network_cls,
                       q_network_cls):
        def _make_parallel(net):
            return net.make_parallel(
                self._num_critic_replicas * reward_spec.numel)

        def _check_spec_equal(spec1, spec2):
            assert nest.flatten(spec1) == nest.flatten(spec2), (
                "Unmatched action specs: {} vs. {}".format(spec1, spec2))

        discrete_action_spec = [
            spec for spec in nest.flatten(action_spec) if spec.is_discrete
        ]
        continuous_action_spec = [
            spec for spec in nest.flatten(action_spec) if spec.is_continuous
        ]

        if discrete_action_spec and continuous_action_spec:
            # When there are both continuous and discrete actions, we require
            # that acition_spec is a tuple/list ``(discrete, continuous)``.
            assert (isinstance(
                action_spec, (tuple, list)) and len(action_spec) == 2), (
                    "In the mixed case, the action spec must be a tuple/list"
                    " (discrete_action_spec, continuous_action_spec)!")
            _check_spec_equal(action_spec[0], discrete_action_spec)
            _check_spec_equal(action_spec[1], continuous_action_spec)
            discrete_action_spec = action_spec[0]
            continuous_action_spec = action_spec[1]
        elif discrete_action_spec:
            discrete_action_spec = action_spec
        elif continuous_action_spec:
            continuous_action_spec = action_spec

        actor_network = None
        critic_networks = None
        if continuous_action_spec:
            assert continuous_actor_network_cls is not None, (
                "If there are continuous actions, then a ActorDistributionNetwork "
                "must be provided for sampling continuous actions!")
            actor_network = continuous_actor_network_cls(
                input_tensor_spec=observation_spec,
                action_spec=continuous_action_spec)
            if not discrete_action_spec:
                act_type = ActionType.Continuous
                if critic_network_cls is not None:
                    critic_network = critic_network_cls(
                        input_tensor_spec=(observation_spec,
                                           continuous_action_spec))
                    critic_networks = _make_parallel(critic_network)

        if discrete_action_spec:
            act_type = ActionType.Discrete
            assert len(alf.nest.flatten(discrete_action_spec)) == 1, (
                "Only support at most one discrete action currently! "
                "Discrete action spec: {}".format(discrete_action_spec))
            assert q_network_cls is not None, (
                "If there exists a discrete action, then QNetwork must "
                "be provided!")
            if continuous_action_spec:
                act_type = ActionType.Mixed
                q_network = q_network_cls(
                    input_tensor_spec=(observation_spec,
                                       continuous_action_spec),
                    action_spec=discrete_action_spec)
            else:
                q_network = q_network_cls(
                    input_tensor_spec=observation_spec,
                    action_spec=action_spec)
            critic_networks = _make_parallel(q_network)

        return critic_networks, actor_network, act_type

    def _predict_action(self,
                        observation,
                        state: SacActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        rollout=False):
        """The reason why we want to do action sampling inside this function
        instead of outside is that for the mixed case, once a continuous action
        is sampled here, we should pair it with the discrete action sampled from
        the Q value. If we just return two distributions and sample outside, then
        the actions will not match.
        """
        new_state = SacActionState()
        if self._act_type != ActionType.Discrete:
            continuous_action_dist, actor_network_state = self._actor_network(
                observation, state=state.actor_network)
            new_state = new_state._replace(actor_network=actor_network_state)
            if eps_greedy_sampling:
                continuous_action = dist_utils.epsilon_greedy_sample(
                    continuous_action_dist, epsilon_greedy)
            else:
                continuous_action = dist_utils.rsample_action_distribution(
                    continuous_action_dist)

        critic_network_inputs = (observation, None)
        if self._act_type == ActionType.Mixed:
            critic_network_inputs = (observation, (None, continuous_action))

        q_values = None
        if self._act_type != ActionType.Continuous:
            q_values, critic_state = self._compute_critics(
                self._critic_networks, *critic_network_inputs, state.critic)

            new_state = new_state._replace(critic=critic_state)
            if self._act_type == ActionType.Discrete:
                alpha = torch.exp(self._log_alpha).detach()
            else:
                alpha = torch.exp(self._log_alpha[0]).detach()
            # p(a|s) = exp(Q(s,a)/alpha) / Z;
            logits = q_values / alpha
            discrete_action_dist = td.Categorical(logits=logits)
            if eps_greedy_sampling:
                discrete_action = dist_utils.epsilon_greedy_sample(
                    discrete_action_dist, epsilon_greedy)
            else:
                discrete_action = dist_utils.sample_action_distribution(
                    discrete_action_dist)

        if self._act_type == ActionType.Mixed:
            # Note that in this case ``action_dist`` is not the valid joint
            # action distribution because ``discrete_action_dist`` is conditioned
            # on a particular continuous action sampled above. So DO NOT use this
            # ``action_dist`` to directly sample an action pair with an arbitrary
            # continuous action anywhere else!
            # However, for computing the log probability of *this* sampled
            # ``action``, it's still valid. It can also be used for summary
            # purpose because of the expectation taken over the continuous action
            # when summarizing.
            action_dist = type(self._action_spec)((discrete_action_dist,
                                                   continuous_action_dist))
            action = type(self._action_spec)((discrete_action,
                                              continuous_action))
        elif self._act_type == ActionType.Discrete:
            action_dist = discrete_action_dist
            action = discrete_action
        else:
            action_dist = continuous_action_dist
            action = continuous_action

        if (self._reproduce_locomotion and rollout
                and not self._training_started):
            # get batch size with ``get_outer_rank`` and ``get_nest_shape``
            # since the observation can be a nest in the general case
            outer_rank = nest_utils.get_outer_rank(observation,
                                                   self._observation_spec)
            outer_dims = alf.nest.get_nest_shape(observation)[:outer_rank]
            # This uniform sampling seems important because for a squashed Gaussian,
            # even with a large scale, a random policy is not nearly uniform.
            action = alf.nest.map_structure(
                lambda spec: spec.sample(outer_dims=outer_dims),
                self._action_spec)

        return action_dist, action, q_values, new_state

    def _repr_step(self, mode, inputs: TimeStep, state: SacState, *args):
        """
        Args:
            mode (str): 'predict' or 'rollout' or 'train'
            *args: for rollout_info when mode is 'train'
        Returns:
            tuple:
            - observation
            - SacState: new_state
            - SacInfo: info
        """
        if self._repr_alg is None:
            return inputs.observation, SacState(), SacInfo()
        else:
            step_func = getattr(self._repr_alg, mode + '_step')
            repr_step = step_func(inputs, state.repr, *args)
            return repr_step.output, SacState(repr=repr_step.state), SacInfo(
                repr=repr_step.info)

    def predict_step(self, inputs: TimeStep, state: SacState):
        observation, new_state, info = self._repr_step("predict", inputs,
                                                       state)
        action_dist, action, _, action_state = self._predict_action(
            observation,
            state=state.action,
            epsilon_greedy=self._epsilon_greedy,
            eps_greedy_sampling=True)
        return AlgStep(
            output=action,
            state=new_state._replace(action=action_state),
            info=info._replace(action_distribution=action_dist))

    def rollout_step(self, inputs: TimeStep, state: SacState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        assert not self._is_eval
        observation, new_state, info = self._repr_step("rollout", inputs,
                                                       state)
        action_dist, action, _, action_state = self._predict_action(
            observation,
            state=state.action,
            epsilon_greedy=self._train_eps_greedy,
            eps_greedy_sampling=True,
            rollout=True)

        # By default use the old target_repr state
        new_state = new_state._replace(target_repr=state.target_repr)

        if self.need_full_rollout_state():
            _, critics_state = self._compute_critics(self._critic_networks,
                                                     observation, action,
                                                     state.critic.critics)
            if self._target_repr_alg is not None:
                tgt_repr_step = self._target_repr_alg.predict_step(
                    inputs, state.target_repr)
                target_observation = tgt_repr_step.output
                new_state = new_state._replace(target_repr=tgt_repr_step.state)
            else:
                target_observation = observation
            _, target_critics_state = self._compute_critics(
                self._target_critic_networks, target_observation, action,
                state.critic.target_critics)
            critic_state = SacCriticState(
                critics=critics_state, target_critics=target_critics_state)
            if self._act_type == ActionType.Continuous:
                # During unroll, the computations of ``critics_state`` and
                # ``actor_state`` are the same.
                actor_state = critics_state
            else:
                actor_state = ()
        else:
            actor_state = state.actor
            critic_state = state.critic

        new_state = new_state._replace(
            action=action_state, actor=actor_state, critic=critic_state)
        return AlgStep(
            output=action,
            state=new_state,
            info=info._replace(action=action, action_distribution=action_dist))

    def _apply_reward_weights(self, critics):
        critics = critics * self.reward_weights
        critics = critics.sum(dim=-1)
        return critics

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         critics_state,
                         replica_min=True,
                         apply_reward_weights=True):
        if self._act_type == ActionType.Continuous:
            observation = (observation, action)
        elif self._act_type == ActionType.Mixed:
            observation = (observation, action[1])  # continuous action
        # discrete/mixed: critics shape [B, replicas, num_actions]
        # continuous: critics shape [B, replicas]
        critics, critics_state = critic_net(observation, state=critics_state)

        # For multi-dim reward, do
        # continuous: [B, replicas * reward_dim] -> [B, replicas, reward_dim]
        # discrete: [B, replicas * reward_dim, num_actions]
        #        -> [B, replicas, reward_dim, num_actions]
        # For scalar reward, do nothing
        if self.has_multidim_reward():
            remaining_shape = critics.shape[2:]
            critics = critics.reshape(-1, self._num_critic_replicas,
                                      *self._reward_spec.shape,
                                      *remaining_shape)
            if self._act_type == ActionType.Discrete:
                # permute: [B, replicas, reward_dim, num_actions]
                #       -> [B, replicas, num_actions, reward_dim]
                order = [0, 1, -1] + list(
                    range(2, 2 + len(self._reward_spec.shape)))
                critics = critics.permute(*order)

        if replica_min:
            if self.has_multidim_reward():
                sign = self.reward_weights.sign()
                critics = (critics * sign).min(dim=1)[0] * sign
            else:
                critics = critics.min(dim=1)[0]

        if apply_reward_weights and self.has_multidim_reward():
            critics = self._apply_reward_weights(critics)

        # The returns have the following shapes in different circumstances:
        # [replica_min=True, apply_reward_weights=True]
        #   discrete/mixed: critics shape [B, num_actions]
        #   continuous: critics shape [B]
        # [replica_min=True, apply_reward_weights=False]
        #   discrete/mixed: critics shape [B, num_actions, reward_dim]
        #   continuous: critics shape [B, reward_dim]
        # [replica_min=False, apply_reward_weights=False]
        #   discrete/mixed: critics shape [B, replicas, num_actions, reward_dim]
        #   continuous: critics shape [B, replicas, reward_dim]
        return critics, critics_state

    def _actor_train_step(self, observation, state, action, critics, log_pi,
                          action_distribution):
        neg_entropy = sum(nest.flatten(log_pi))

        if self._act_type == ActionType.Discrete:
            # Pure discrete case doesn't need to learn an actor network
            return (), LossInfo(extra=SacActorInfo(neg_entropy=neg_entropy))

        if self._act_type == ActionType.Continuous:
            q_value, critics_state = self._compute_critics(
                self._critic_networks, observation, action, state)
            continuous_log_pi = log_pi
            cont_alpha = torch.exp(self._log_alpha).detach()
        else:
            # use the critics computed during action prediction for Mixed type
            # ``critics``` is already after min over replicas
            critics_state = ()
            discrete_act_dist = action_distribution[0]
            q_value = discrete_act_dist.probs.detach() * critics
            action, continuous_log_pi = action[1], log_pi[1]
            cont_alpha = torch.exp(self._log_alpha[1]).detach()

        # This sum() will reduce all dims so q_value can be any rank
        dqda = nest_utils.grad(action, q_value.sum())

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))
        actor_info = LossInfo(
            loss=actor_loss + cont_alpha * continuous_log_pi,
            extra=SacActorInfo(actor_loss=actor_loss, neg_entropy=neg_entropy))
        return critics_state, actor_info

    def _select_q_value(self, action, q_values):
        """Use ``action`` to index and select Q values.
        Args:
            action (Tensor): discrete actions with shape ``[batch_size]``.
            q_values (Tensor): Q values with shape
                ``[batch_size, replicas, num_actions, reward_dim]``, where
                ``reward_dim`` is optional for multi-dim reward.
        Returns:
            Tensor: selected Q values with shape
                ``[batch_size, replicas, reward_dim]``.
        """
        ones = [1] * len(self._reward_spec.shape)
        # [batch_size] -> [batch_size, 1, 1, ...]
        action = action.view(q_values.shape[0], 1, 1, *ones)
        # [batch_size, 1, 1, ...] -> [batch_size, n, 1, reward_dim]
        action = action.expand(-1, q_values.shape[1], -1,
                               *self._reward_spec.shape).long()
        return q_values.gather(2, action).squeeze(2)

    def _critic_train_step(self, observation, target_observation,
                           state: SacCriticState, rollout_info: SacInfo,
                           action, action_distribution):
        critics, critics_state = self._compute_critics(
            self._critic_networks,
            observation,
            rollout_info.action,
            state.critics,
            replica_min=False,
            apply_reward_weights=False)

        with torch.no_grad():
            target_critics, target_critics_state = self._compute_critics(
                self._target_critic_networks,
                target_observation,
                action,
                state.target_critics,
                apply_reward_weights=False)

        if self._act_type == ActionType.Discrete:
            critics = self._select_q_value(rollout_info.action, critics)
            # [B, num_actions] -> [B, num_actions, reward_dim]
            probs = common.expand_dims_as(action_distribution.probs,
                                          target_critics)
            # [B, reward_dim]
            target_critics = torch.sum(probs * target_critics, dim=1)
        elif self._act_type == ActionType.Mixed:
            critics = self._select_q_value(rollout_info.action[0], critics)
            discrete_act_dist = action_distribution[0]
            target_critics = torch.sum(
                discrete_act_dist.probs * target_critics, dim=-1)

        target_critic = target_critics.reshape(target_critics.shape[0],
                                               *self._reward_spec.shape)

        target_critic = target_critic.detach()

        state = SacCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = SacCriticInfo(critics=critics, target_critic=target_critic)

        return state, info

    def _alpha_train_step(self, log_pi):
        # ``log_pi`` should either be a scalar or a pair (mixed action case),
        # so is ``self._target_entropy``
        alpha_loss = nest.map_structure(
            lambda la, lp, t: la * (-lp - t).detach(), self._log_alpha, log_pi,
            self._target_entropy)
        return sum(nest.flatten(alpha_loss))

    def train_step(self, inputs: TimeStep, state: SacState,
                   rollout_info: SacInfo):
        assert not self._is_eval
        self._training_started = True
        if self._target_repr_alg is not None:
            # We calculate the target observation first so that the peak memory
            # usage can be reduced because its computation graph will not be kept.
            with torch.no_grad():
                tgt_repr_step = self._target_repr_alg.predict_step(
                    inputs, state.target_repr)
                target_observation = tgt_repr_step.output
                target_repr_state = tgt_repr_step.state
        else:
            target_observation = inputs.observation
            target_repr_state = ()
        observation, new_state, info = self._repr_step("train", inputs, state,
                                                       rollout_info.repr)
        new_state = new_state._replace(target_repr=target_repr_state)

        (action_distribution, action, critics,
         action_state) = self._predict_action(
             observation, state=state.action)

        log_pi = nest.map_structure(lambda dist, a: dist.log_prob(a),
                                    action_distribution, action)

        if self._act_type == ActionType.Mixed:
            # For mixed type, add log_pi separately
            log_pi = type(self._action_spec)((sum(nest.flatten(log_pi[0])),
                                              sum(nest.flatten(log_pi[1]))))
        else:
            log_pi = sum(nest.flatten(log_pi))

        if self._prior_actor is not None:
            prior_step = self._prior_actor.train_step(inputs, ())
            log_prior = dist_utils.compute_log_probability(
                prior_step.output, action)
            log_pi = log_pi - log_prior

        actor_state, actor_loss = self._actor_train_step(
            observation, state.actor, action, critics, log_pi,
            action_distribution)
        critic_state, critic_info = self._critic_train_step(
            observation, target_observation, state.critic, rollout_info,
            action, action_distribution)
        alpha_loss = self._alpha_train_step(log_pi)

        new_state = new_state._replace(
            action=action_state, actor=actor_state, critic=critic_state)
        info = info._replace(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            action_distribution=action_distribution,
            actor=actor_loss,
            critic=critic_info,
            alpha=alpha_loss,
            log_pi=log_pi,
            discounted_return=rollout_info.discounted_return)
        return AlgStep(action, new_state, info)

    def after_update(self, root_inputs, info: SacInfo):
        self._update_target()
        if self._repr_alg is not None:
            self._repr_alg.after_update(root_inputs, info.repr)
        if self._max_log_alpha is not None:
            nest.map_structure(
                lambda la: la.data.copy_(torch.min(la, self._max_log_alpha)),
                self._log_alpha)

    def after_train_iter(self, inputs: TimeStep, info: SacInfo):
        self._periodic_reset()

    def calc_loss(self, info: SacInfo):
        assert not self._is_eval
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_loss = info.actor

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                if self._act_type == ActionType.Mixed:
                    alf.summary.scalar("alpha/discrete",
                                       self._log_alpha[0].exp())
                    alf.summary.scalar("alpha/continuous",
                                       self._log_alpha[1].exp())
                else:
                    alf.summary.scalar("alpha", self._log_alpha.exp())

        if self._reproduce_locomotion:
            policy_l = math_ops.add_ignore_empty(actor_loss.loss, alpha_loss)
            policy_mask = torch.ones_like(policy_l)
            policy_mask[0, :] = 0.
            critic_l = critic_loss.loss
            critic_mask = torch.ones_like(critic_l)
            critic_mask[-1, :] = 0.
            loss = critic_l * critic_mask + policy_l * policy_mask
        else:
            loss = math_ops.add_ignore_empty(actor_loss.loss, critic_loss.loss)
            loss = math_ops.add_ignore_empty(loss, alpha_loss)

        if self._repr_alg is not None:
            repr_loss = self._repr_alg.calc_loss(info.repr)
            loss = math_ops.add_ignore_empty(loss, repr_loss.loss)
        else:
            repr_loss = LossInfo(loss=0., extra=())

        return LossInfo(
            loss=loss,
            priority=critic_loss.priority,
            extra=SacLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                repr=repr_loss.extra,
                alpha=alpha_loss))

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
            critic_losses.append(
                l(info=info,
                  value=critic_info.critics[:, :, i, ...],
                  target_value=critic_info.target_critic).loss)

        critic_loss = math_ops.add_n(critic_losses)

        if self._calculate_priority:
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
        return ['_target_critic_networks', '_target_repr_alg']
