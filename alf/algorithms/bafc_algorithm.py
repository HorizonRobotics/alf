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
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorFCNetwork, FuncCriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops
from alf.utils.normalizers import ScalarAdaptiveNormalizer
from alf.utils.schedulers import Scheduler
from alf.networks.neural_graphs.actor_graph import ActorGraph
from alf.networks.neural_graphs.graph_network import GraphNetwork

BafcActionState = namedtuple(
    "BafcActionState", ["actor_network", "critic"], default_value=())

BafcCriticState = namedtuple("BafcCriticState", ["critic", "target_critic"])

BafcState = namedtuple(
    "BafcState", ["action", "actor", "critic"],
    default_value=())

BafcCriticInfo = namedtuple(
    "BafcCriticInfo", ["critic", "target_critic"], default_value=())

BafcActorInfo = namedtuple(
    "BafcActorInfo", ["action_loss", "param_loss"], default_value=())

BafcInfo = namedtuple(
    "BafcInfo", [
        "reward", "step_type", "discount", "action", "actor", "critic", 
        "discounted_return"
    ],
    default_value=())

BafcLossInfo = namedtuple(
    'BafcLossInfo', ('actor', 'critic'), default_value=())


def batch_to_graphs(
    weights,  # list of shape [bs, out, in] tensors
    biases,   # list of shape [bs, out] tensors
    weights_mean=None,
    weights_std=None,
    biases_mean=None,
    biases_std=None,
):
    device = weights[0].device
    bsz = weights[0].shape[0]
    num_nodes = weights[0].shape[2] + sum(w.shape[1] for w in weights)

    node_features = torch.zeros(bsz, num_nodes, device=device)
    edge_features = torch.zeros(
        bsz, num_nodes, num_nodes, device=device
    )

    row_offset = 0
    col_offset = weights[0].shape[2]  # no edge to input nodes
    for i, w in enumerate(weights):
        _, num_out, num_in = w.shape
        w_mean = weights_mean[i] if weights_mean is not None else 0
        w_std = weights_std[i] if weights_std is not None else 1
        edge_features[
            :, col_offset:col_offset + num_out, row_offset:row_offset + num_in
        ] = (w - w_mean) / w_std
        row_offset += num_in
        col_offset += num_out

    row_offset = weights[0].shape[2]  # no bias in input nodes
    for i, b in enumerate(biases):
        _, num_out = b.shape
        b_mean = biases_mean[i] if biases_mean is not None else 0
        b_std = biases_std[i] if biases_std is not None else 1
        node_features[:, row_offset:row_offset + num_out] = (b - b_mean) / b_std
        row_offset += num_out

    return node_features, edge_features


@alf.configurable
class BafcAlgorithm(OffPolicyAlgorithm):
    r"""Boostrapped Actor and Functional Critic algorithm, 

    ::

        Bai et al "Bootstrapped Actors and Functional Critic", arXiv, 2025

    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorFCNetwork,
                 critic_network_cls=FuncCriticNetwork,
                 reward_weights=None,
                 calculate_priority=False,
                 num_bootstrapped_actors=10,
                 num_actor_eval_samples=256,
                 actor_graph_node_dim=64,
                 actor_graph_edge_dim=32,
                 actor_encoding_dim=128,
                 obs_action_encoding_dim=64,
                 actor_utd: Optional[int] = None,
                 critic_utd: Optional[int] = None,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_update_tau: Union[float, Scheduler] = 0.05,
                 target_update_period: Union[int, Scheduler] = 1,
                 parameter_reset_period: Union[int, Scheduler] = -1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 reproduce_locomotion=False,
                 name="BafAlgorithm"):
        """
        Args:
        """
        if actor_utd is None and critic_utd is None:
            self._train_mode = TrainMode.standard
        else:
            total_utd = config.num_updates_per_train_iter
            if actor_utd is None:
                assert critic_utd < total_utd, (
                    "critic_utd should be less than num_updates_per_train_iter "
                    "if actor_utd is not provided.")
                actor_utd = total_utd - critic_utd
            elif critic_utd is None:
                assert actor_utd < total_utd, (
                    "actor_utd should be less than num_updates_per_train_iter "
                    "if critic_utd is not provided.")
                critic_utd = total_utd - actor_utd
            assert actor_utd <= critic_utd, (
                f"actor_utd {actor_utd} should not be greater than critic_utd {critic_utd}"
            )
            self._train_mode = TrainMode.critic
            self._actor_utd = actor_utd
            self._critic_utd = critic_utd

        self._num_bootstrapped_actors = num_bootstrapped_actors
        actor_networks = actor_network_cls(
            input_tensor_spec=observation_spec,
            action_spec=action_spec,
            n_groups=num_bootstrapped_actors)
        actor_eval_samples = 2 * torch.rand(
            num_actor_eval_samples, observation_spec.shape[0]) - 1

        # extract weight_param spec and bias_param spec from actor_networks
        actor_weight_spec = nest.map_structure(
            lambda tensor: TensorSpec.from_tensor(tensor, from_dim=1), 
            actor_networks.weight_params)
        actor_bias_spec = nest.map_structure(
            lambda tensor: TensorSpec.from_tensor(tensor, from_dim=1), 
            actor_networks.bias_params)
        actor_layer_layout = [
            actor_weight_spec[0].shape[1]] + [b.shape[0] for b in actor_bias_spec] 

        actor_graph = ActorGraph(layer_layout=actor_layer_layout,
                                 d_node=actor_graph_node_dim,
                                 d_edge=actor_graph_edge_dim,
                                 num_eval_samples=num_actor_eval_samples)
        graph_network = GraphNetwork(layer_layout=actor_layer_layout,
                                     d_node=actor_graph_node_dim,
                                     d_edge=actor_graph_edge_dim,
                                     d_out=actor_encoding_dim) 

        actor_spec = TensorSpec(shape=(actor_encoding_dim,))
        obs_action_spec = (observation_spec, action_spec)
        critic_network = critic_network_cls(
            input_tensor_spec=(actor_spec, obs_action_spec), 
            obs_action_encoding_dim=obs_action_encoding_dim)

        action_state_spec = BafcActionState(
            actor_network=actor_networks.state_spec,
            critic=critic_network.state_spec)
        train_state_spec = BafcState(
            action=action_state_spec,
            actor=critic_network.state_spec,
            critic=BafcCriticState(
                critic=critic_network.state_spec,
                target_critic=critic_network.state_spec))

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            rollout_state_spec=train_state_spec,
            predict_state_spec=BafcState(action=action_state_spec),
            reward_weights=reward_weights,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None and actor_networks is not None:
            self.add_optimizer(actor_optimizer, [actor_networks])
        if critic_optimizer is not None and critic_network is not None:
            self.add_optimizer(critic_optimizer, [critic_network])

        self._actor_networks = actor_networks
        self._actor_eval_samples = nn.Parameter(actor_eval_samples)
        self._actor_graph = actor_graph
        self._graph_network = graph_network
        self._critic_network = critic_network
        self._target_critic_network = critic_network.copy(
            name='target_critic_network')
        self._target_critic_network.set_obs_action_batch_dominate(True)

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        self._critic_loss = critic_loss_ctor(debug_summaries=debug_summaries)

        self._rollout_actor_id = 0
        self._actor_update_counter = 0
        self._critic_update_counter = 0
        self._dqda_clipping = dqda_clipping
        self._training_started = False
        self._do_critic_summary = False

        def _filter(x):
            return list(filter(lambda x: x is not None, x))

        def _create_target_updater():
            self._update_target = common.TargetUpdater(
                models=_filter([self._critic_network]),
                target_models=_filter(
                    [self._target_critic_network]),
                tau=target_update_tau,
                period=target_update_period)

        _create_target_updater()

    def _predict_action(self,
                        observation,
                        state: BafcActionState,
                        train=False):
        if not self._training_started:
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
            return action, state

        if train:
            action, state = self._actor_networks(
                observation, state=state.actor_network)
        else:
            action, state = self._actor_networks(
                observation,
                id=self._rollout_actor_id,
                state=state.actor_network)
        new_state = BafcActionState(actor_network=state)

        return action, new_state

    def predict_step(self, inputs: TimeStep, state: BafcState):
        action, action_state = self._predict_action(
            inputs.observation,
            state=state.action)
        return AlgStep(
            output=action,
            state=state._replace(action=action_state),
            info=BafcInfo(action=action))

    def rollout_step(self, inputs: TimeStep, state: BafcState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        assert not self._is_eval
        if inputs.step_type == StepType.FIRST:
            self._rollout_actor_id = torch.randint(self._num_bootstrapped_actors, ())

        action, action_state = self._predict_action(
            inputs.observation,
            state=state.action)
        return AlgStep(
            output=action,
            state=state._replace(action=action_state),
            info=BafcInfo(action=action))

    def _encode_actor(self):
        eval_out = self._actor_networks(
            self._actor_eval_samples, full_neurons=True)[0]
        node_features, edge_features = batch_to_graphs(
            self._actor_networks.weight_params,
            self._actor_networks.bias_params
        ) 
        inputs = (node_features, edge_features, eval_out)
        graph_inputs = self._actor_graph(inputs)
        return self._graph_network(graph_inputs)

    def _actor_train_step(self, observation, action, actor_encoding, state):
        """Compute the full off-policy policy gradient from the functional critic,
        which consists of two terms, 

        1. the gradient w.r.t. input action, as in standard actor-critic algorithms.

        2. the gradient w.r.t. input actor encoding, obtainable only with functional 
           critic.
        """
        # [B * n_actor, d]
        critic_observation = observation.repeat_interleave(
            self._num_bootstrapped_actors, dim=0)
        q_value, critic_state = self._critic_network(
            (actor_encoding, (critic_observation, action)), state) 

        # This sum() will reduce all dims so q_value can be any rank
        action_grad = nest_utils.grad(action, q_value.sum(), retain_graph=True)
        encoding_grad = nest_utils.grad(actor_encoding, q_value.sum()) #, retain_graph=True)

        def critic_input_loss_fn(dqda, a_in):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + a_in).detach(), a_in)
            return loss.sum(list(range(1, loss.ndim)))

        # loss corresponding to input action
        action_loss = nest.map_structure(critic_input_loss_fn, action_grad, action)
        action_loss = action_loss.reshape(
            -1, self._num_bootstrapped_actors).sum(-1)
        # action_loss = math_ops.add_n(nest.flatten(action_loss))

        # loss corresponding to input actor encoding
        encoding_loss = nest.map_structure(
            critic_input_loss_fn, encoding_grad, actor_encoding)
        # ALF workaround: reduce to scalar_loss and repeat to [T*B]
        # Will be averaged to a scalar_loss in calc_loss
        encoding_loss = encoding_loss.sum().repeat(
            action_loss.shape[0])

        actor_info = LossInfo(
            loss=action_loss,
            extra=BafcActorInfo(action_loss=action_loss, 
                                param_loss=encoding_loss))
        return critic_state, actor_info

    def _critic_train_step(self, observation, state: BafcCriticState, 
                           rollout_info: BafcInfo, action, actor_encoding):
        critics, critic_state = self._critic_network(
            (actor_encoding, (observation, rollout_info.action)), state.critic)
        # [T*B, n_actor]
        critics = critics.reshape(-1, self._num_bootstrapped_actors)

        with torch.no_grad():
            target_observation = observation.repeat_interleave(
                self._num_bootstrapped_actors, dim=0)
            target_critics, target_critic_state = self._target_critic_network(
                (actor_encoding, (target_observation, action)), state.target_critic)

        # [T*B, n_actor]
        target_critics = target_critics.reshape(-1, self._num_bootstrapped_actors)
        target_critics = target_critics.detach()

        state = BafcCriticState(
            critic=critic_state, target_critic=target_critic_state)
        info = BafcCriticInfo(critic=critics, target_critic=target_critics)

        return state, info

    def _update_train_mode(self):
        if self._train_mode == TrainMode.actor:
            if self._actor_update_counter % self._actor_utd == 0:
                self._train_mode = TrainMode.critic
                self._critic_network.set_obs_action_batch_dominate(False)
        elif self._train_mode == TrainMode.critic:
            if self._critic_update_counter % self._critic_utd == 0:
                self._train_mode = TrainMode.actor
                self._critic_network.set_obs_action_batch_dominate(True)

    def train_step(self, inputs: TimeStep, state: BafcState,
                   rollout_info: BafcInfo):
        assert not self._is_eval
        self._training_started = True
        # [B, n_actor, d_a]
        action, action_state = self._predict_action(
            inputs.observation, state=state.action, train=True)
        action = action.reshape(-1, action.shape[-1])  # [B*n_actor, d_a]

        new_state = BafcState(action=action_state,
                              actor=state.actor,
                              critic=state.critic)

        # actor_params = (self._actor_networks.weight_params, 
        #                 self._actor_networks.bias_params) 
        actor_encoding = self._encode_actor()

        if self._train_mode == TrainMode.standard or (
                self._critic_update_counter == 0
                and self._actor_update_counter == 0):
            actor_state, actor_info = self._actor_train_step(
                inputs.observation, action, actor_encoding, state.actor)
            critic_state, critic_info = self._critic_train_step(
                inputs.observation, state.critic, rollout_info, 
                action, actor_encoding)
            new_state = new_state._replace(actor=actor_state,
                                           critic=critic_state)
            self._critic_update_counter += 1
        else:
            if self._train_mode == TrainMode.actor:
                actor_state, actor_info = self._actor_train_step(
                    inputs.observation, action, actor_encoding, state.actor)
                critic_info = BafcCriticInfo()
                new_state = new_state._replace(actor=actor_state)
                self._actor_update_counter += 1
            else:
                critic_state, critic_info = self._critic_train_step(
                    inputs.observation, state.critic, rollout_info, 
                    action, actor_encoding)
                actor_info = LossInfo(extra=BafcActorInfo())
                # actor_info = LossInfo()
                new_state = new_state._replace(critic=critic_state)
                self._critic_update_counter += 1

        if self._debug_summaries and alf.summary.should_record_summaries():
            self._do_critic_summary = True

        info = BafcInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            actor=actor_info,
            critic=critic_info,
            discounted_return=rollout_info.discounted_return)
        return AlgStep(action, new_state, info)

    def calc_loss(self, info: BafcInfo):
        assert not self._is_eval
        actor_loss = info.actor
        scalar_loss = actor_loss.extra.param_loss
        if isinstance(scalar_loss, torch.Tensor):
            scalar_loss = scalar_loss.mean()
        if self._train_mode == TrainMode.actor:
            critic_loss = LossInfo()
        else:
            critic_loss = self._calc_critic_loss(info)

        loss = math_ops.add_ignore_empty(actor_loss.loss, critic_loss.loss)

        return LossInfo(
            loss=loss,
            scalar_loss=scalar_loss,
            extra=BafcLossInfo(
                actor=actor_loss.extra, critic=critic_loss.extra))

    def _calc_critic_loss(self, info: BafcInfo):
        with alf.summary.record_if(lambda: self._do_critic_summary):
            critic_info = info.critic
            critic_losses = []
            # [T, B, n_actor]
            critic_loss = self._critic_loss(
                info=info,
                value=critic_info.critic,
                target_value=critic_info.target_critic).loss

        self._do_critic_summary = False

        return LossInfo(
            loss=critic_loss,
            extra=critic_loss)

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_network']

    def after_update(self, root_inputs, info: BafcInfo):
        self._update_train_mode()
        self._update_target()
