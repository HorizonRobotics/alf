# Copyright (c) 2021 Horizon Robotics. All Rights Reserved.
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

from absl import logging
from enum import Enum
import functools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.data_structures import Experience, LossInfo, namedtuple, TimeStep
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
import alf.summary.render as render
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import (common, dist_utils, losses, math_ops, spec_utils,
                       tensor_utils, value_ops, summary_utils)
from alf.utils.conditional_ops import conditional_update
from alf.utils.summary_utils import safe_mean_hist_summary, summarize_action
from alf.algorithms.tasac_algorithm import TASACTDLoss
from alf.summary.render import _rendering_wrapper, _convert_to_image

TrajState = namedtuple(
    "TrajState",
    [
        "a",  # The current action value
        "v",  # The current first derivative of action
        "u"  # The current second derivative of action
    ],
    default_value=None)

Distributions = namedtuple("Distributions", ["beta_dist", "b1_a_dist"])

Actions = namedtuple("Actions", ["b", "b0_u", "b0_v", "b1_u", "b1_v", "b1_a"])

ActPredOutput = namedtuple(
    "ActPredOutput", ["dists", "actions", "traj_states", "q_values2"],
    default_value=())

PtpState = namedtuple(
    "PtpState",
    [
        "traj",  # TrajState
        "repeats",  # How many steps the current trajectory has been used
        "action_history",  # for RENDER
        "b_history"  # for RENDER
    ],
    default_value=())

PtpCriticInfo = namedtuple(
    "PtpCriticInfo", ["critics", "target_critic", "value_loss"],
    default_value=())

PtpActorInfo = namedtuple(
    "PtpActorInfo", [
        "actor_loss", "u_entropy", "b1_a_entropy", "beta_entropy", "adv",
        "value_loss"
    ],
    default_value=())

PtpInfo = namedtuple(
    "PtpInfo",
    [
        "action_distribution",
        "action",
        "actor",
        "critic",
        "alpha",
        "repeats",
        "traj",  # for critic training step
    ],
    default_value=())

PtpLossInfo = namedtuple('PtpLossInfo', ('actor', 'critic', 'alpha'))

Mode = Enum('AlgorithmMode', ('predict', 'rollout', 'train'))


@_rendering_wrapper
def _render_action_trajectory(name,
                              data,
                              b_history,
                              y_range=None,
                              img_height=256,
                              img_width=256,
                              dpi=300,
                              figsize=(2, 2),
                              linewidth=1):

    assert len(data.shape) == 2, "Must be rank-2 (history_len, action_dim)!"
    assert len(b_history.shape) == 1, "Must be rank-1 (history_len,)!"
    action = data.cpu().numpy()
    b = b_history.cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    T, N = action.shape

    #          111.. color,  000.. colors
    colors = [
        ["lightcoral", "brown"],
        ["gray", "black"],
        ["gold", "orange"],
        ["limegreen", "forestgreen"],
        ["skyblue", "deepskyblue"],
        ["magenta", "purple"],
    ]

    def _bridge_plotting_gap(ax, plotted_sofar, action, x0, n):
        if x0 > plotted_sofar:
            assert x0 == plotted_sofar + 1
            xx = [plotted_sofar, x0]
            ax.plot(xx, action[xx, n], linewidth=linewidth, color=colors[n][0])

    last_1 = -1  # the time step of last b==1
    last_0 = -1  # the time step of last b==0
    plotted_sofar = 0  # the time step plotted so far
    for t in range(T):
        if b[t] == 1:
            # At least one 0 between two 1s: plotting the pattern '1000...'
            if t - last_1 > 1:
                x = range(max(last_1, 0), t)
                for n in range(N):
                    array = action[x, n]
                    ax.plot(
                        x,
                        array,
                        linewidth=2 * linewidth,
                        color=colors[n][1],
                        marker='o')
                    _bridge_plotting_gap(ax, plotted_sofar, action, x[0], n)
                plotted_sofar = x[-1]
            # else '11' pattern; do nothing
            last_1 = t
        else:  # b[t] == 0
            # At least two 1s between two 0s: plotting the pattern '111...'
            if t - last_0 > 2:
                x = range(max(last_0, 0), t - 1)
                for n in range(N):
                    array = action[x, n]
                    ax.plot(x, array, linewidth=linewidth, color=colors[n][0])
                    _bridge_plotting_gap(ax, plotted_sofar, action, x[0], n)
                plotted_sofar = x[-1]
            last_0 = t

    # plot the remaining
    x = range(plotted_sofar, T)
    for n in range(N):
        array = action[x, n]
        ax.plot(x, array, linewidth=linewidth, color=colors[n][0])

    if y_range:
        ax.set_ylim(y_range)

    return _convert_to_image(name, fig, dpi, img_height, img_width)


@alf.configurable
class PtpAlgorithm(OffPolicyAlgorithm):
    r"""Ptp: Piecewise trajectory prior for efficient continuous control.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=None,
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 num_critic_replicas=2,
                 env=None,
                 config: TrainerConfig = None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 critic_loss_ctor=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 debug_summaries=False,
                 b1_advantage_clipping=None,
                 target_entropy=None,
                 name="PtpAlgorithm"):

        assert len(
            nest.flatten(action_spec)) == 1 and action_spec.is_continuous, (
                "Only support a single continuous action!")
        assert (
            np.all(action_spec.minimum == -1)
            and np.all(action_spec.maximum == 1)
        ), ("Only support actions in [-1, 1]! Consider using env wrappers to "
            "scale your action space first.")

        b_spec = BoundedTensorSpec(shape=(), dtype='int64', maximum=1)

        self._inv_squash_clip = 0.999
        self._inv_squash_range = self._atanh(
            torch.tensor(self._inv_squash_clip))

        self._num_critic_replicas = num_critic_replicas

        critic_networks, actor_network = self._make_networks(
            observation_spec, action_spec, actor_network_cls,
            critic_network_cls)

        log_alpha = (nn.Parameter(torch.zeros(())),
                     nn.Parameter(torch.zeros(())))

        train_state_spec = PtpState(
            traj=TrajState(a=action_spec, v=action_spec, u=action_spec),
            repeats=TensorSpec(shape=(), dtype=torch.int64))

        # for rendering
        predict_state_spec = train_state_spec._replace(
            action_history=TensorSpec(shape=(100, ) + action_spec.shape),
            b_history=TensorSpec(
                shape=(100, ) + b_spec.shape, dtype=torch.int64))

        super().__init__(
            observation_spec,
            action_spec,
            # Just for logging the action repeats statistics
            train_state_spec=train_state_spec,
            rollout_state_spec=train_state_spec,
            predict_state_spec=predict_state_spec,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, list(log_alpha))

        self._log_alpha = log_alpha
        self._log_alpha_paralist = nn.ParameterList(list(log_alpha))
        self._actor_network = actor_network
        self._critic_networks = critic_networks
        self._target_critic_networks = self._critic_networks.copy(
            name='target_critic_networks')

        if critic_loss_ctor is None:
            critic_loss_ctor = TASACTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        # Have different names to separate their summary curves
        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))
        self._gamma = self._critic_losses[0]._gamma

        # separate target entropies for discrete and continuous actions
        if not isinstance(target_entropy, tuple):
            target_entropy = (target_entropy, ) * 2
        self._target_entropy = nest.map_structure(
            lambda spec, t: _set_target_entropy(self.name, t, [spec]),
            (b_spec, action_spec), target_entropy)

        self._b_spec = b_spec
        self._b1_advantage_clipping = b1_advantage_clipping

        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

    def _atanh(self, yy):
        # clip to prevent infy gradients
        y = torch.clamp(yy, -self._inv_squash_clip, self._inv_squash_clip)
        return 0.5 * torch.log((1 + y) / (1 - y))

    def _make_networks(self, observation_spec, action_spec, actor_network_cls,
                       critic_network_cls):
        def _make_parallel(net):
            return net.make_parallel(self._num_critic_replicas)

        traj_state_spec = nest.map_structure(lambda _: action_spec,
                                             TrajState())
        traj_detach = nest.map_structure(lambda _: alf.layers.Detach(),
                                         TrajState())

        def _normalize(x, s=1.):
            return x / self._inv_squash_range / s

        traj_norm = TrajState(
            a=alf.layers.Lambda(_normalize),
            v=alf.layers.Lambda(functools.partial(_normalize, s=2.)),
            u=alf.layers.Lambda(functools.partial(_normalize, s=2.)))

        traj_tanh = nest.map_structure(lambda _: alf.layers.Lambda(torch.tanh),
                                       TrajState())

        # Because traj state are quantities in linear space, we first squash them
        # with tanh to (-1,1) for input normalization for the actor and critic networks
        actor_network = actor_network_cls(
            input_tensor_spec=(observation_spec, traj_state_spec),
            input_preprocessors=(alf.layers.Detach(), traj_tanh),
            preprocessing_combiner=nest_utils.NestConcat(),
            action_spec=action_spec)  # u
        critic_network = critic_network_cls(
            input_tensor_spec=(observation_spec,
                               traj_state_spec),  # traj as action
            action_input_processors=traj_tanh,
            action_preprocessing_combiner=nest_utils.NestConcat())
        critic_networks = _make_parallel(critic_network)

        return critic_networks, actor_network

    @torch.no_grad()
    def _update_traj_state(self, traj_state):
        """Compute next action on a quadratic curve specified by the triplet of
        action, action derivative, and action second derivative.
        """
        # normal update
        v_ = traj_state.v + traj_state.u
        da = (v_ + traj_state.v) / 2.
        a_ = traj_state.a + da

        # compute the reflection
        def _compute_reflection(b, traj):
            a, v, u = traj.a, traj.v, traj.u
            tmp = v**2 + 2 * u * (b - a)
            #print(tmp, v, u, a, b)
            assert torch.all(tmp >= 0.)
            dt1 = (-v - torch.sqrt(tmp)) / u
            dt2 = (-v + torch.sqrt(tmp)) / u
            dt = torch.where(((b - a) * u < 0) ^ (u > 0), dt2, dt1)

            vr = -(v + u * dt)
            v_ = vr + (1 - dt) * u
            a_ = b + (vr + v_) / 2. * (1 - dt)
            return a_, v_

        """ def _compute_reflection(b, traj):
            #a = torch.where(clipped, traj_state.a, a_)
            #v = torch.where(clipped, -traj_state.v, v_)
            return traj.a, -traj.v """

        clipped = (a_ > self._inv_squash_range) | (a_ <
                                                   -self._inv_squash_range)
        b = self._inv_squash_range * (
            2 * (a_ > self._inv_squash_range).to(torch.float32) - 1)

        shape = a_.shape
        # conditional_update only accepts 1d cond bool tensor
        a_, v_ = conditional_update(
            target=(a_.reshape(-1), v_.reshape(-1)),
            cond=clipped.reshape(-1),
            func=_compute_reflection,
            b=b.reshape(-1),
            traj=nest.map_structure(lambda s: s.reshape(-1), traj_state))

        return TrajState(
            a=a_.reshape(*shape), v=v_.reshape(*shape), u=traj_state.u)

    def _calc_new_traj_state(self, a, traj_state):
        """Given a new target action ``a``, compute the second derivative ``u``
        if the previous ``a`` and ``v`` are kept the same.
        # backward to compute the desired new ``u``
        """
        if True:
            v_ = torch.zeros_like(traj_state.v)
        else:
            v_ = traj_state.v

        v = 2 * (a - traj_state.a) - v_
        # clip v within 2 times the range of a
        #v = self._atanh((v / 2.).tanh())[0] * 2.
        u = v - v_
        return TrajState(a=a, v=v, u=u)

    def _build_beta_dist(self, q_values2):
        def _safe_categorical(logits, alpha):
            r"""A numerically stable implementation of categorical distribution
            :math:`exp(\frac{Q}{\alpha})`.
            """
            logits = logits / torch.clamp(alpha, min=1e-10)
            # logits are equivalent after subtracting a common number
            logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
            return td.Categorical(logits=logits)

        # compute beta dist *conditioned* on ``action``
        with torch.no_grad():
            beta_alpha = self._log_alpha[0].exp().detach()
            if self._b1_advantage_clipping is None:
                beta_dist = _safe_categorical(q_values2, beta_alpha)
            else:
                clip_min, clip_max = self._b1_advantage_clipping
                clipped_q_values2 = (q_values2 - q_values2[..., :1]).clamp(
                    min=clip_min, max=clip_max)
                beta_dist = _safe_categorical(clipped_q_values2, beta_alpha)

        return beta_dist

    def _compute_beta_and_traj_state(self, observation, state, epsilon_greedy,
                                     mode):

        b1_a_dist, _ = self._actor_network((observation, state.traj))

        if mode == Mode.predict:
            b1_a = dist_utils.epsilon_greedy_sample(b1_a_dist, epsilon_greedy)
        else:
            b1_a = dist_utils.rsample_action_distribution(b1_a_dist)

        # Maintain the traj state
        b0_traj_state = self._update_traj_state(state.traj)
        b1_traj_state = self._calc_new_traj_state(
            self._atanh(b1_a), state.traj)

        # compute Q(s, b0_u) and Q(s, u)
        with torch.no_grad():
            # What's the Q value if following the current trajectory?
            q_0 = self._compute_critics(self._critic_networks, observation,
                                        b0_traj_state)
        # What's the Q value of following a new trajectory?
        if mode == Mode.train:
            assert np.all(
                nest.flatten(
                    nest.map_structure(lambda x: x.requires_grad,
                                       b1_traj_state)))
        q_1 = self._compute_critics(self._critic_networks, observation,
                                    b1_traj_state)

        q_values2 = torch.stack([q_0, q_1], dim=-1)
        beta_dist = self._build_beta_dist(q_values2)

        if mode == Mode.predict:
            b = dist_utils.epsilon_greedy_sample(beta_dist, epsilon_greedy)
        else:
            b = dist_utils.sample_action_distribution(beta_dist)

        dists = Distributions(beta_dist=beta_dist, b1_a_dist=b1_a_dist)
        actions = Actions(
            b=b,
            b0_u=state.traj.u,
            b0_v=state.traj.v,
            b1_u=b1_traj_state.u,
            b1_v=b1_traj_state.v,
            b1_a=b1_a)
        return ActPredOutput(
            dists=dists,
            actions=actions,
            traj_states=(b0_traj_state, b1_traj_state),
            q_values2=q_values2)

    def _predict_action(self,
                        time_step_or_exp,
                        state: PtpState,
                        epsilon_greedy=None,
                        mode=Mode.rollout):

        ap_out = self._compute_beta_and_traj_state(
            time_step_or_exp.observation, state, epsilon_greedy, mode)

        b0_traj_state, b1_traj_state = ap_out.traj_states

        # By default follow the current trajectory
        new_state = state._replace(traj=b0_traj_state)

        def _b1_decision(b1_traj_state, new_state):
            return new_state._replace(
                repeats=torch.zeros_like(new_state.repeats),
                traj=b1_traj_state)

        # If b=1, follow the new trajectory and reset the repetition counter
        new_state = conditional_update(
            target=new_state,
            cond=ap_out.actions.b.to(torch.bool),
            func=_b1_decision,
            b1_traj_state=b1_traj_state,
            new_state=new_state)

        # For either choice we need to increment the repetition counter
        new_state = new_state._replace(repeats=new_state.repeats + 1)

        return ap_out, new_state

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         replica_min=True):
        """Compute Q(s,a)"""
        observation = (observation, action)
        critics, _ = critic_net(observation)  # [B, replicas]
        if replica_min:
            critics = critics.min(dim=1)[0]
        return critics

    def _calc_critic_loss(self, experience, train_info: PtpInfo):
        critic_info = train_info.critic
        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            kwargs = dict(
                experience=experience,
                value=critic_info.critics[:, :, i, ...],
                target_value=critic_info.target_critic)
            if isinstance(l, TASACTDLoss):
                kwargs["train_b"] = train_info.action.b
                kwargs["rollout_b"] = experience.rollout_info.action.b

            critic_losses.append(l(**kwargs).loss)

        critic_loss = math_ops.add_n(critic_losses)
        return LossInfo(
            loss=critic_loss,
            extra=critic_loss / float(self._num_critic_replicas))

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']

    def _render(self, actions, action_dists, traj_states, info, new_state):
        if render.is_rendering_enabled():
            with alf.summary.scope(self._name):
                action_history = torch.cat(
                    (new_state.action_history[:, 1:, ...],
                     new_state.traj.a.unsqueeze(1)),
                    dim=1)  # [B=1, 100, ...]
                b_history = torch.cat(
                    (new_state.b_history[:, 1:], actions.b.unsqueeze(1)),
                    dim=1)  # [B=1, 100]
                new_state = new_state._replace(
                    action_history=action_history, b_history=b_history)

                history_action_img = _render_action_trajectory(
                    name="history_action",
                    data=action_history.squeeze(0),
                    b_history=b_history.squeeze(0),
                    y_range=(np.min(self._action_spec.minimum),
                             np.max(self._action_spec.maximum)),
                    linewidth=1,
                    figsize=(4, 4),
                    img_width=1024,
                    img_height=512)
                info['history_action'] = history_action_img

                b_dist_img = render.render_action_distribution(
                    name="beta_dist",
                    act_dist=action_dists.beta_dist,
                    action_spec=self._b_spec)
                info['beta_dist'] = b_dist_img

                b1_action_dist_img = render.render_action_distribution(
                    name="b1_action_dist",
                    act_dist=action_dists.b1_a_dist,
                    action_spec=self._action_spec)
                info['b1_action_dist'] = b1_action_dist_img

                b0_traj_state_img = render.render_bar(
                    name="b0_traj_state",
                    data=torch.cat(
                        [traj_states[0].a, traj_states[0].v, traj_states[0].u],
                        dim=-1)  # [a,v,u]
                )
                info['b0_traj_state'] = b0_traj_state_img

                b1_traj_state_img = render.render_bar(
                    name="b1_traj_state",
                    data=torch.cat(
                        [traj_states[1].a, traj_states[1].v, traj_states[1].u],
                        dim=-1)  # [a,v,u]
                )
                info['b1_traj_state'] = b1_traj_state_img
        return new_state

    def predict_step(self,
                     time_step: TimeStep,
                     state: PtpState,
                     epsilon_greedy=0.):
        ap_out, new_state = self._predict_action(
            time_step, state, epsilon_greedy=epsilon_greedy, mode=Mode.predict)

        info = dict(
            info=PtpInfo(
                action_distribution=ap_out.dists, action=ap_out.actions))

        new_state = self._render(ap_out.actions, ap_out.dists,
                                 ap_out.traj_states, info, new_state)

        return AlgStep(
            output=new_state.traj.a.tanh(), state=new_state, info=info)

    def rollout_step(self, time_step: TimeStep, state):
        ap_out, new_state = self._predict_action(
            time_step, state, mode=Mode.rollout)
        return AlgStep(
            output=new_state.traj.a.tanh(),
            state=new_state,
            info=PtpInfo(
                action_distribution=ap_out.dists,
                action=ap_out.actions,
                traj=new_state.traj,  # later for critic training
                repeats=state.repeats))

    def _actor_train_step(self, a, action_entropy, beta_dist, q_values2):
        alpha = self._log_alpha[1].exp().detach()
        q_a = beta_dist.probs[:, 1].detach() * q_values2[:, 1]

        dqda = nest_utils.grad(a, q_a.sum())

        def actor_loss_fn(dqda, action):
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        actor_loss = nest.map_structure(actor_loss_fn, dqda, a)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))
        actor_loss -= alpha * action_entropy

        return LossInfo(
            loss=actor_loss,
            extra=PtpActorInfo(
                actor_loss=actor_loss, adv=q_values2[:, 1] - q_values2[:, 0]))

    def _critic_train_step(self, exp: Experience, b0_traj, b1_traj, beta_dist,
                           state: PtpState):

        with torch.no_grad():
            target_q_0 = self._compute_critics(self._target_critic_networks,
                                               exp.observation, b0_traj)
            target_q_1 = self._compute_critics(self._target_critic_networks,
                                               exp.observation, b1_traj)

            beta_probs = beta_dist.probs
            target_critic = (beta_probs[..., 0] * target_q_0 +
                             beta_probs[..., 1] * target_q_1)

        critics = self._compute_critics(
            self._critic_networks,
            exp.observation,
            exp.rollout_info.traj,
            replica_min=False)
        return PtpCriticInfo(critics=critics, target_critic=target_critic)

    def _alpha_train_step(self, beta_entropy, action_entropy):
        alpha_loss = (self._log_alpha[1] *
                      (action_entropy - self._target_entropy[1]).detach())
        alpha_loss += (self._log_alpha[0] *
                       (beta_entropy - self._target_entropy[0]).detach())
        return alpha_loss

    def train_step(self, exp: Experience, state):

        ap_out, new_state = self._predict_action(
            exp, state=state, mode=Mode.train)

        beta_dist, b1_a_dist = ap_out.dists.beta_dist, ap_out.dists.b1_a_dist
        b1_a = ap_out.actions.b1_a

        b1_a_entropy = -dist_utils.compute_log_probability(b1_a_dist, b1_a)
        beta_entropy = beta_dist.entropy()

        actor_loss = self._actor_train_step(b1_a, b1_a_entropy, beta_dist,
                                            ap_out.q_values2)
        actor_loss = actor_loss._replace(
            extra=actor_loss.extra._replace(
                b1_a_entropy=b1_a_entropy, beta_entropy=beta_entropy))

        critic_info = self._critic_train_step(exp, ap_out.traj_states[0],
                                              ap_out.traj_states[1], beta_dist,
                                              state)
        alpha_loss = self._alpha_train_step(beta_entropy, b1_a_entropy)

        info = PtpInfo(
            action_distribution=ap_out.dists,
            action=ap_out.actions,
            actor=actor_loss,
            critic=critic_info,
            alpha=alpha_loss,
            repeats=state.repeats)
        return AlgStep(
            output=new_state.traj.a.tanh(), state=new_state, info=info)

    def after_update(self, experience, train_info: PtpInfo):
        self._update_target()

    def calc_loss(self, experience, train_info: PtpInfo):
        critic_loss = self._calc_critic_loss(experience, train_info)
        alpha_loss = train_info.alpha
        actor_loss = train_info.actor
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha/beta", self._log_alpha[0].exp())
                alf.summary.scalar("alpha/action", self._log_alpha[1].exp())
                alf.summary.scalar("resample_advantage",
                                   torch.mean(actor_loss.extra.adv))
                p_beta0 = train_info.action_distribution.beta_dist.probs[...,
                                                                         0]
                alf.summary.histogram("P_beta_0/value", p_beta0)
                alf.summary.scalar("P_beta_0/mean", p_beta0.mean())
                alf.summary.scalar("P_beta_0/std", p_beta0.std())
                repeats = train_info.repeats
                alf.summary.scalar("train_repeats/mean",
                                   torch.mean(repeats.to(torch.float32)))
                alf.summary.histogram("train_repeats/value",
                                      repeats.to(torch.float32))

                summary_utils.summarize_action(train_info.action.b1_a,
                                               self._action_spec, "b1_a")

                alf.summary.scalar("b0_u/mean", train_info.action.b0_u.mean())
                alf.summary.histogram("b0_u/value", train_info.action.b0_u)
                alf.summary.scalar("b0_v/mean", train_info.action.b0_v.mean())
                alf.summary.histogram("b0_v/value", train_info.action.b0_v)

                alf.summary.scalar("b1_u/mean", train_info.action.b1_u.mean())
                alf.summary.histogram("b1_u/value", train_info.action.b1_u)
                alf.summary.scalar("b1_v/mean", train_info.action.b1_v.mean())
                alf.summary.histogram("b1_v/value", train_info.action.b1_v)

        return LossInfo(
            loss=actor_loss.loss + alpha_loss + critic_loss.loss,
            extra=PtpLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss))

    def summarize_rollout(self, experience):
        info = experience.rollout_info
        repeats = info.repeats.reshape(-1)
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                # if rollout batch size=1, hist won't show
                alf.summary.histogram("rollout_repeats/value", repeats)
                alf.summary.scalar("rollout_repeats/mean",
                                   torch.mean(repeats.to(torch.float32)))
