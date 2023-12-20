# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import copy
from enum import Enum
from typing import Any, Callable, List, NamedTuple, Optional, Union
import torch
from torch.optim.optimizer import Optimizer

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.mpo_loss import MPOLoss, MPOInfo
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.utils import common, dist_utils, math_ops
from alf.utils.schedulers import Scheduler
from alf import nest
from alf.data_structures import AlgStep
from alf.data_structures import TimeStep
from alf.environments.alf_environment import AlfEnvironment


class MPOState(NamedTuple):
    repr: Any = ()
    target_repr: Any = ()


# Q means the network is called as critic_net(observation) to estimate
# all the q values for a given observation.
# Critic means the network is called as critic_net((observation, action))
# to estimate the q value for a given observation and action.
CriticType = Enum('CriticType', ('Q', 'Critic'))


@alf.configurable
class MPOAlgorithm(OffPolicyAlgorithm):
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
        critic_network_ctor (Callable): called as ``critic_network_ctor((observation_spec, action_spec))``
            to construct the CriticNetwork for estimating Q(s,a)
        q_network_ctor (Callable): called as ``q_network_ctor(observation_spec, action_spec)`
            to construct the QNetwork for estimating all the Q values for given observation.
            Note that one and only one of ``critic_network_ctor`` and ``q_network_ctor``
            should be provided.
        repr_alg_ctor: if provided, it will be called as ``repr_alg_ctor(
            observation_spec, action_spec, reward_spec, config=config)`` to
            construct a representation learning algorithm. The output of the
            representation learning algorithm is used as the input of the
            actor and critic networks. Different from using representation_learner_cls
            in ``Agent``, a target model of the representation learning algorithm
            will be maintained and the representation calculated by the target
            representation learning algorithm will be used for computing
            target critics.
        reward_weights: this is only used when the reward is
            multidimensional. In that case, the weighted sum of the q values
            is used for training the actor if reward_weights is not None.
            Otherwise, the sum of the q values is used.
        epsilon_greedy: a floating value in [0,1], representing the
            chance of action sampling instead of taking argmax. This can
            help prevent a dead loop in some deterministic environment like
            Breakout. Only used for evaluation. If None, its value is taken
            from ``config.epsilon_greedy`` and then
            ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
        num_critic_replicas (int): number of critics to be used. Default is 2.
        loss_ctor: mpo loss constructor.
        env: The environment to interact with. ``env`` is a
            batched environment, which means that it runs multiple simulations
            simultateously. ``env` only needs to be provided to the root
            algorithm.
        config: config for training. It only needs to be
            provided to the algorithm which performs ``train_iter()`` by
            itself.
        target_update_tau: Factor for soft update of the target
            networks.
        target_update_period: Period for soft update of the target
            networks.
        checkpoint: a string in the format of "prefix@path",
            where the "prefix" is the multi-step path to the contents in the
            checkpoint to be loaded. "path" is the full path to the checkpoint
            file saved by ALF. Refer to ``Algorithm`` for more details.
        debug_summaries: True if debug summaries should be created.
        name: The name of this algorithm.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_ctor=ActorDistributionNetwork,
                 critic_network_ctor=CriticNetwork,
                 q_network_ctor: Optional[Callable] = None,
                 repr_alg_ctor: Optional[Callable] = None,
                 reward_weights: Optional[List[float]] = None,
                 epsilon_greedy: float = None,
                 num_critic_replicas=2,
                 loss_ctor: Callable = MPOLoss,
                 env: Optional[AlfEnvironment] = None,
                 config: Optional[TrainerConfig] = None,
                 target_update_tau: Union[float, Scheduler] = 0.05,
                 target_update_period: Union[float, Scheduler] = 1,
                 actor_optimizer: Optional[Optimizer] = None,
                 critic_optimizer: Optional[Optimizer] = None,
                 num_candidate_actions: int = 20,
                 checkpoint: Optional[str] = None,
                 debug_summaries: bool = False,
                 name: str = "MPOAlgorithm"):
        assert q_network_ctor is not None or critic_network_ctor is not None, (
            "Either q_network_ctor or critic_network_ctor must be provided")
        assert q_network_ctor is None or critic_network_ctor is None, (
            "Only one of q_network_ctor and critic_network_ctor can be provided"
        )

        self._num_candidate_actions = num_candidate_actions
        self._sample_candidate_actions = False
        self._num_replicas = num_critic_replicas
        self._reward_dim = reward_spec.numel
        if nest.is_nested(
                action_spec
        ) or action_spec.is_continuous or action_spec.numel > 1:
            self._sample_candidate_actions = True
            assert num_candidate_actions is not None, (
                "num_candidate_actions needs "
                "to be provided for continuous actions or multi-dimensional "
                f"discrete actions: action_spec={action_spec}")
        elif not action_spec.is_continuous:
            num_actions = action_spec.maximum - action_spec.minimum + 1
            if num_candidate_actions is not None:
                assert num_candidate_actions < num_actions, (
                    "For scalar discrete action"
                    "num_candidate_actions should be smaller than num_actions. Got"
                    "num_candidate_actions=%s, num_actions=%s" %
                    (num_candidate_actions, num_actions))

        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy

        self._num_critic_replicas = num_critic_replicas
        original_observation_spec = observation_spec
        if repr_alg_ctor is not None:
            repr_alg = repr_alg_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                reward_spec=reward_spec,
                config=config)
            target_repr_alg = repr_alg_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                reward_spec=reward_spec,
                config=config)
            assert hasattr(repr_alg,
                           'output_spec'), "repr_alg must have output_spec"
            observation_spec = repr_alg.output_spec
        else:
            repr_alg = None
            target_repr_alg = None

        critic_networks, actor_network, self._critic_type = self._make_networks(
            observation_spec, action_spec, reward_spec, actor_network_ctor,
            critic_network_ctor, q_network_ctor)

        if not self._sample_candidate_actions:
            assert self._critic_type == CriticType.Q, (
                "For discrete action, critic_networks must be QNetworks")

        assert len(alf.nest.flatten(critic_networks.state_spec)) == 0, (
            "critic_networks should not have state")
        assert len(alf.nest.flatten(actor_network.state_spec)) == 0, (
            "actor_network should not have state")

        train_state_spec = MPOState(
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
            predict_state_spec=MPOState(
                repr=repr_alg.predict_state_spec if repr_alg else ()),
            reward_weights=reward_weights,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._critic_networks = critic_networks
        self._target_critic_networks = critic_networks.copy(
            name='target_critic_networks')
        self._actor_network = actor_network
        self._target_actor_network = actor_network.copy(
            name='target_actor_network')
        self._repr_alg = repr_alg
        self._target_repr_alg = target_repr_alg

        def _filter(x):
            return list(filter(lambda x: x is not None, x))

        self._update_target = common.TargetUpdater(
            models=_filter(
                [self._actor_network, self._critic_networks, repr_alg]),
            target_models=_filter([
                self._target_actor_network, self._target_critic_networks,
                target_repr_alg
            ]),
            tau=target_update_tau,
            period=target_update_period)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])

        self._loss = loss_ctor(reward_dim=reward_spec.numel)

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       actor_network_ctor, critic_network_ctor,
                       q_network_ctor):
        def _make_parallel(net):
            return net.make_parallel(
                self._num_critic_replicas * reward_spec.numel)

        actor_network = actor_network_ctor(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        if critic_network_ctor is not None:
            critic_network = critic_network_ctor(
                input_tensor_spec=(observation_spec, action_spec))
            critic_type = CriticType.Critic
        else:
            critic_network = q_network_ctor(
                input_tensor_spec=observation_spec, action_spec=action_spec)
            critic_type = CriticType.Q

        critic_networks = _make_parallel(critic_network)

        return critic_networks, actor_network, critic_type

    def _trainable_attributes_to_ignore(self):
        return [
            '_target_critic_networks', '_target_repr_alg',
            '_target_actor_network'
        ]

    def _repr_step(self, mode, time_step: TimeStep, state: MPOState, *args):
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
            return time_step.observation, (), ()
        else:
            step_func = getattr(self._repr_alg, mode + '_step')
            repr_step = step_func(time_step, state.repr, *args)
            return repr_step.output, repr_step.state, repr_step.info

    def _predict_action(self,
                        observation,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False):
        action_dist = self._actor_network(observation)[0]
        if eps_greedy_sampling:
            action = dist_utils.epsilon_greedy_sample(action_dist,
                                                      epsilon_greedy)
        else:
            action = dist_utils.rsample_action_distribution(action_dist)

        return action_dist, action

    def predict_step(self, time_step: TimeStep, state: MPOState):
        observation, repr_state, repr_info = self._repr_step(
            "predict", time_step, state)
        action_dist, action = self._predict_action(
            observation,
            epsilon_greedy=self._epsilon_greedy,
            eps_greedy_sampling=True)
        return AlgStep(
            output=action,
            state=MPOState(repr=repr_state),
            info=MPOInfo(repr_info=repr_info, action_distribution=action_dist))

    def rollout_step(self, time_step: TimeStep, state: MPOState):
        observation, repr_state, repr_info = self._repr_step(
            "rollout", time_step, state)
        action_dist, action = self._predict_action(
            observation, epsilon_greedy=1.0, eps_greedy_sampling=True)

        if self.need_full_rollout_state(
        ) and self._target_repr_alg is not None:
            tgt_repr_step = self._target_repr_alg.predict_step(
                time_step, state.target_repr)
            target_repr_state = tgt_repr_step.state
        else:
            target_repr_state = ()

        return AlgStep(
            output=action,
            state=MPOState(repr=repr_state, target_repr=target_repr_state),
            info=MPOInfo(
                repr_info=repr_info,
                action=action,
                action_distribution=action_dist))

    def train_step(self, time_step: TimeStep, state: MPOState,
                   rollout_info: MPOInfo):
        if self._target_repr_alg is not None:
            # We calculate the target observation first so that the peak memory
            # usage can be reduced because its computation graph will not be kept.
            with torch.no_grad():
                tgt_repr_step = self._target_repr_alg.predict_step(
                    time_step, rollout_info.repr)
                target_observation = tgt_repr_step.output
                target_repr_state = tgt_repr_step.state
        else:
            target_observation = time_step.observation
            target_repr_state = ()
        observation, repr_state, repr_info = self._repr_step(
            "train", time_step, state, rollout_info.repr_info)
        action_distribution, action = self._predict_action(observation)
        critics_dist = self._calc_critics_dist(observation,
                                               rollout_info.action)

        with torch.no_grad():
            action_dist = self._target_actor_network(target_observation)[0]
            candidate_actions, candidate_action_weights = self._sample_actions(
                action_dist)
            target_critics = self._calc_target_critics(target_observation,
                                                       candidate_actions)

        new_state = MPOState(repr=repr_state, target_repr=target_repr_state)
        info = MPOInfo(
            repr_info=repr_info,
            action=rollout_info.action,
            reward=time_step.reward,
            discount=time_step.discount,
            step_type=time_step.step_type,
            action_distribution=action_distribution,
            candidate_actions=candidate_actions,
            candidate_action_weights=candidate_action_weights,
            critics_dist=critics_dist,
            target_critics=target_critics,
        )
        return AlgStep(output=action, state=new_state, info=info)

    def _calc_critics_dist(self, observation, action):
        if self._critic_type == CriticType.Critic:
            # [B, replicas * reward_dim, num_quantiles]
            critics_dist = self._critic_networks((observation, action))[0]
            # [B, replicas, reward_dim, num_quantiles]
            critics_dist = critics_dist.reshape(-1, self._num_replicas,
                                                self._reward_dim,
                                                *critics_dist.shape[2:])
        else:
            # [B, replicas * reward_dim, num_actions, num_quantiles]
            critics_dist = self._critic_networks(observation)[0]
            # [B, replicas, reward_dim, num_actions, num_quantiles]
            critics_dist = critics_dist.reshape(-1, self._num_replicas,
                                                self._reward_dim,
                                                *critics_dist.shape[2:])
            B = torch.arange(critics_dist.shape[0])
            # [B, replicas, reward_dim, num_quantiles]
            critics_dist = critics_dist[B, :, :, action, ...]
        return critics_dist

    def _calc_target_critics(self, target_observation, candidate_actions):
        if self._critic_type == CriticType.Critic:
            # [B * num_candidate_actions, ...]
            expanded_target_observation = nest.map_structure(
                lambda x: x.repeat_interleave(
                    self._num_candidate_actions, dim=0), target_observation)
            # [B * num_candidate_actions, ...]
            expanded_candidate_actions = nest.map_structure(
                lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]),
                candidate_actions)
            # [B * num_candidate_actions, replicas * reward_dim, num_quantiles]
            target_critics_dist = self._target_critic_networks(
                (expanded_target_observation, expanded_candidate_actions))[0]

            # [B, num_candidate_actions, replicas, reward_dim, num_quantiles]
            target_critics_dist = target_critics_dist.reshape(
                -1, self._num_candidate_actions, self._num_replicas,
                self._reward_dim, *target_critics_dist.shape[2:])
            # [B, replicas, reward_dim, num_candidate_actions, num_quantiles]
            target_critics_dist = target_critics_dist.transpose(1,
                                                                2).transpose(
                                                                    2, 3)
        else:
            # [B, replicas * reward_dim, num_actions, num_quantiles]
            target_critics_dist = self._target_critic_networks(
                target_observation)[0]
            target_critics_dist = target_critics_dist.reshape(
                -1, self._num_replicas, self._reward_dim,
                *target_critics_dist.shape[2:])

        # [B, replicas, reward_dim, num_candidate_actions]
        target_critics = self._loss.calc_value_expectation(target_critics_dist)
        return target_critics

    def _sample_actions(self, action_distribution):
        if self._sample_candidate_actions:
            # [num_sampled_actions, B, ...]
            actions = action_distribution.rsample(
                (self._num_candidate_actions, ))
            # [B, num_sampled_actions, ...]
            actions = actions.transpose(0, 1)
            # According to the following paper, we should use 1/num_candidate_actions
            # as weight for sampled actions.
            # Hubert et. al. Learning and Planning in Complex Action Spaces, 2021
            action_weights = torch.ones(
                actions.shape[:2]) / self._num_candidate_actions
        else:
            action_probs = action_distribution.probs
            if self._num_candidate_actions is None:
                actions = ()
                action_weights = action_probs
            else:
                action_weights, actions = action_probs.topk(
                    self._num_candidate_actions, sorted=False)
                action_weights = action_weights / action_weights.sum(
                    dim=-1, keepdim=True)
        return actions, action_weights

    def calc_loss(self, info: MPOInfo):
        loss = self._loss(info)
        if self._repr_alg is not None:
            repr_loss = self._repr_alg.calc_loss(info.repr_info)
            extra = copy.copy(loss.extra)
            extra['repr'] = repr_loss.extra
            loss = loss._replace(
                loss=math_ops.add_ignore_empty(loss.loss, repr_loss.loss),
                extra=extra)
        return loss

    @torch.no_grad()
    def set_reward_weights(self, reward_weights):
        """Update reward weights; this function can be called at any step during
        training. Once called, the updated reward weights are expected to be used
        by the algorithm in the next.

        Args:
            reward_weights (Tensor): a tensor that is compatible with
                ``self._reward_spec``.
        """
        self._loss.set_reward_weights(reward_weights)

    def after_update(self, root_inputs, info: MPOInfo):
        self._update_target()
        if self._repr_alg is not None:
            self._repr_alg.after_update(root_inputs, info.repr)
