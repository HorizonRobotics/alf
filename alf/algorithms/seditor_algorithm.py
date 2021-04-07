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

from enum import Enum
import functools

import torch
import torch.nn as nn

import alf
from alf.algorithms.config import TrainerConfig
from alf.data_structures import LossInfo, namedtuple, TimeStep, AlgStep
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.td_loss import TDLoss
import alf.nest as nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork, ValueNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
import alf.summary.render as render
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, dist_utils, losses, math_ops, spec_utils

SEditorInfo = namedtuple(
    "SEditorInfo",
    ["ap_out", "reward", "step_type", "discount", "actor", "critic", "alpha"],
    default_value=())

SEditorActorInfo = namedtuple(
    "SEditorActorInfo", [
        "a_loss", "da_loss", "a_entropy", "da_entropy", "change_a_loss",
        "dqda_cos"
    ],
    default_value=())

SEditorCriticInfo = namedtuple(
    "SEditorCriticInfo", ["critics", "target_critic"], default_value=())

SEditorLossInfo = namedtuple(
    "SEditorLossInfo", ["actor", "critic", "alpha"], default_value=())

ActPredOutput = namedtuple(
    "ActPredOutput", ["a", "da", "a_dist", "da_dist", "output", "a_before"],
    default_value=())

Mode = Enum('AlgorithmMode', ('predict', 'rollout', 'train'))


@alf.configurable
class SEditorAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec,
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 reward_weights=None,
                 num_critic_replicas=2,
                 initial_alpha=1.,
                 communicate_steps=1,
                 train_da_alpha=True,
                 epsilon_greedy=None,
                 env=None,
                 config: TrainerConfig = None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 critic_loss_ctor=None,
                 optimizer=None,
                 debug_summaries=False,
                 target_entropy=None,
                 name="SEditorAlgorithm"):

        assert len(
            nest.flatten(action_spec)) == 1 and action_spec.is_continuous, (
                "Only support a single continuous action!")
        # d0 is utility and d1 is constraint
        assert reward_spec.numel >= 2

        self._num_critic_replicas = num_critic_replicas
        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value(
                'TrainerConfig.epsilon_greedy')
        self._epsilon_greedy = epsilon_greedy

        self._communicate_steps = communicate_steps

        (critic_networks,
         actor_network, d_actor_network) = self._make_networks(
             observation_spec, action_spec, reward_spec, actor_network_cls,
             critic_network_cls)

        if not isinstance(initial_alpha, (tuple, list)):
            initial_alpha = (initial_alpha, ) * 2
        log_alpha = nest.map_structure(
            lambda ia: nn.Parameter(torch.tensor(ia).log()), initial_alpha)

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=(),
            reward_spec=reward_spec,
            reward_weights=reward_weights,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if optimizer is not None:
            self.add_optimizer(optimizer, [
                actor_network,
                d_actor_network,
                critic_networks,
            ] + log_alpha)

        self._log_alpha = log_alpha
        self._log_alpha_paralist = nn.ParameterList(log_alpha)

        self._train_da_alpha = train_da_alpha

        self._actor_network = actor_network
        self._actor_network_opt_ignored = actor_network.copy(
            name='actor_network_opt_ignored')
        self._d_actor_network = d_actor_network
        self._d_actor_network_opt_ignored = d_actor_network.copy(
            name='d_actor_network_opt_ignored')

        self._critic_networks = critic_networks
        self._target_critic_networks = self._critic_networks.copy(
            name='target_critic_networks')

        if critic_loss_ctor is None:
            critic_loss_ctor = TDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)

        # Have different names to separate their summary curves
        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))
        self._gamma = self._critic_losses[0]._gamma

        if not isinstance(target_entropy, tuple):
            target_entropy = (target_entropy, ) * 2
        self._target_entropy = nest.map_structure(
            lambda spec, t: _set_target_entropy(self.name, t, [spec]),
            (action_spec, action_spec), target_entropy)

        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

        self._actor_networks_copy = common.get_target_updater(
            models=[self._actor_network, self._d_actor_network],
            target_models=[
                self._actor_network_opt_ignored,
                self._d_actor_network_opt_ignored
            ],
            tau=1,
            period=1)

    def _trainable_attributes_to_ignore(self):
        return [
            '_target_critic_networks', '_actor_network_opt_ignored',
            '_d_actor_network_opt_ignored'
        ]

    def after_update(self, root_inputs, info: SEditorInfo):
        self._update_target()
        self._actor_networks_copy()

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       actor_network_cls, critic_network_cls):
        def _make_parallel(net):
            return net.make_parallel(
                self._num_critic_replicas * reward_spec.numel)

        step_spec = BoundedTensorSpec((),
                                      maximum=self._communicate_steps - 1,
                                      dtype=torch.int64)
        step_processor = EmbeddingPreprocessor(
            input_tensor_spec=step_spec, embedding_dim=action_spec.numel)

        obs_action_spec = (observation_spec, action_spec)
        actor_network = actor_network_cls(
            input_tensor_spec=(obs_action_spec, step_spec),
            input_preprocessors=((alf.layers.Detach(), None), step_processor),
            preprocessing_combiner=nest_utils.NestConcat(),
            action_spec=action_spec)
        d_actor_network = actor_network_cls(
            input_tensor_spec=(obs_action_spec, step_spec),
            input_preprocessors=((alf.layers.Detach(), None), step_processor),
            preprocessing_combiner=nest_utils.NestConcat(),
            action_spec=action_spec)
        critic_network = critic_network_cls(
            input_tensor_spec=obs_action_spec,
            action_preprocessing_combiner=nest_utils.NestConcat())
        critic_networks = _make_parallel(critic_network)

        return critic_networks, actor_network, d_actor_network

    def _communicate(self, time_step, epsilon_greedy, mode, opt_ignore='none'):
        # Only let value learning updates observation encoder's parameters
        obs = common.detach(time_step.observation)

        def _actor(net, in_a, step):
            dist, _ = net(((obs, in_a), step))
            if mode == Mode.predict:
                a = dist_utils.epsilon_greedy_sample(dist, epsilon_greedy)
            else:
                a = dist_utils.rsample_action_distribution(dist)
            return a, dist

        a_seq, da_seq = [], []
        a_dist_seq, da_dist_seq = [], []
        da = torch.zeros_like(time_step.prev_action)

        a_net = (self._actor_network_opt_ignored
                 if opt_ignore == 'a' else self._actor_network)
        da_net = (self._d_actor_network_opt_ignored
                  if opt_ignore == 'da' else self._d_actor_network)

        for i in range(self._communicate_steps):
            step = torch.ones_like(time_step.step_type).to(torch.int64) * i

            a, a_dist = _actor(a_net, da, step)
            a_seq.append(a)
            a_dist_seq.append(a_dist)

            #a = self._safe_action(da, a)

            da, da_dist = _actor(da_net, a, step)
            da_seq.append(da)
            da_dist_seq.append(da_dist)

            da = self._safe_action(a, da)

        return ActPredOutput(
            a=a_seq,
            a_dist=a_dist_seq,
            da=da_seq,
            da_dist=da_dist_seq,
            output=da,
            a_before=a)

    def _predict_action(self,
                        time_step,
                        state,
                        epsilon_greedy=None,
                        mode=Mode.rollout):

        if mode == Mode.train:
            ap_out0 = self._communicate(
                time_step, epsilon_greedy, mode, opt_ignore='a')
            ap_out1 = self._communicate(
                time_step, epsilon_greedy, mode, opt_ignore='da')
            return ActPredOutput(
                a=ap_out1.a,
                a_dist=ap_out1.a_dist,
                da=ap_out0.da,
                da_dist=ap_out0.da_dist,
                output=(
                    ap_out1.output,
                    ap_out0.output,
                    # how much the safety policy changes the action
                    ap_out0.a_before))
        else:
            return self._communicate(
                time_step, epsilon_greedy, mode, opt_ignore='none')

    def _safe_action(self, a, da):
        #return math_ops.softclip(
        #    a + 2 * da,
        #    low=float(self._action_spec.minimum),
        #    high=float(self._action_spec.maximum),
        #    hinge_softness=0.1)
        return spec_utils.clip_to_spec(a + 2 * da, self._action_spec)

    def predict_step(self, inputs: TimeStep, state):
        ap_out = self._predict_action(inputs, state, self._epsilon_greedy,
                                      Mode.predict)

        a_img, da_img, out_a_img = None, None, None
        if render.is_rendering_enabled():
            a_img = render.render_action(
                name="a", action=ap_out.a[-1], action_spec=self._action_spec)
            da_img = render.render_action(
                name="da", action=ap_out.da[-1], action_spec=self._action_spec)
            out_a_img = render.render_action(
                name="out_a",
                action=ap_out.output,
                action_spec=self._action_spec)

        return AlgStep(
            output=ap_out.output,
            state=state,
            info=dict(
                ap_out=ap_out, a_img=a_img, da_img=da_img,
                out_a_img=out_a_img))

    def rollout_step(self, inputs: TimeStep, state):
        ap_out = self._predict_action(inputs, state, mode=Mode.rollout)
        return AlgStep(output=ap_out.output, state=state, info=ap_out)

    def train_step(self, inputs: TimeStep, state, rollout_info: SEditorInfo):
        def _calc_seq_entropy(dist, a):
            a_entropy = []
            for i in range(self._communicate_steps):
                ent = -dist_utils.compute_log_probability(dist[i], a[i])
                a_entropy.append(ent)
            return sum(a_entropy[-1:])

        ap_out = self._predict_action(inputs, state, mode=Mode.train)

        a_entropy = _calc_seq_entropy(ap_out.a_dist, ap_out.a)
        da_entropy = _calc_seq_entropy(ap_out.da_dist, ap_out.da)

        actor_loss = self._actor_train_step(inputs, ap_out, a_entropy)
        da_loss, change_a_loss, dqda_cos = self._d_actor_train_step(
            inputs, ap_out, da_entropy)
        actor_info = SEditorActorInfo(
            a_loss=actor_loss,
            da_loss=da_loss,
            a_entropy=a_entropy,
            da_entropy=da_entropy,
            dqda_cos=dqda_cos,
            change_a_loss=change_a_loss)

        critic_info = self._critic_train_step(inputs, ap_out, rollout_info)

        alpha_loss = self._alpha_train_step(a_entropy, da_entropy)

        info = SEditorInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            actor=actor_info,
            critic=critic_info,
            alpha=alpha_loss)

        return AlgStep(output=ap_out.output[0], state=state, info=info)

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         replica_min=True):
        """Compute Q(s,a)"""
        observation = (observation, action)
        critics, _ = critic_net(observation)  # [B, replicas * reward_dim]
        critics = critics.reshape(  # [B, replicas, reward_dim]
            -1, self._num_critic_replicas, *self._reward_spec.shape)
        if replica_min:
            if self.has_multidim_reward():
                sign = self.reward_weights.sign()
                critics = (critics * sign).min(dim=1)[0] * sign
            else:
                critics = critics.min(dim=1)[0]

        return critics

    def _actor_loss_fn(self, dqda, action):
        loss = 0.5 * losses.element_wise_squared_loss(
            (dqda + action).detach(), action)
        return loss.sum(list(range(1, loss.ndim)))

    def _actor_train_step(self, inputs, ap_out, a_entropy):
        alpha = self._log_alpha[0].exp().detach()

        critics = self._compute_critics(self._critic_networks,
                                        inputs.observation, ap_out.output[0])
        # only maximize the utility Q value
        q = critics[..., 0].sum()

        dqda = nest_utils.grad(ap_out.a[-1], q)
        actor_loss = self._actor_loss_fn(dqda, ap_out.a[-1])
        actor_loss -= alpha * a_entropy
        return actor_loss

    def _d_actor_train_step(self, inputs, ap_out, da_entropy):
        alpha = self._log_alpha[1].exp().detach()

        critics = self._compute_critics(self._critic_networks,
                                        inputs.observation, ap_out.output[1])

        # only maximize the constraint Q value
        q = critics[..., 1]
        # Take mean so that the loss magnitude is invariant to action dimension
        #change_a_loss = ((ap_out.output[1] - ap_out.output[2])**2).mean(dim=-1)

        a_critics = self._compute_critics(self._critic_networks,
                                          inputs.observation, ap_out.output[2])
        change_a_loss = (a_critics[..., 0] - critics[..., 0])**2

        if False:
            q2 = torch.einsum("bd,d->b",
                              torch.stack((-change_a_loss, q), dim=-1),
                              self.reward_weights).sum()
            dqda = nest_utils.grad(ap_out.da[-1], q2)
        else:
            q2 = torch.stack((-change_a_loss, q), dim=-1) * self.reward_weights
            dqda1 = nest_utils.grad(
                ap_out.da[-1], q2[..., 0].sum(), retain_graph=True)
            dqda2 = nest_utils.grad(ap_out.da[-1], q2[..., 1].sum())
            cos = torch.nn.CosineSimilarity(dim=-1)
            dqda_cos = cos(dqda1, dqda2)
            dqda = dqda1 + dqda2

        actor_loss = self._actor_loss_fn(dqda, ap_out.da[-1])
        actor_loss -= alpha * da_entropy
        return actor_loss, change_a_loss, dqda_cos

    def _critic_train_step(self, inputs: TimeStep, ap_out,
                           rollout_info: SEditorInfo):

        with torch.no_grad():
            target_critics = self._compute_critics(
                self._target_critic_networks, inputs.observation,
                ap_out.output[0])

        critics = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_info.output,
            replica_min=False)
        return SEditorCriticInfo(critics=critics, target_critic=target_critics)

    def _alpha_train_step(self, a_entropy, da_entropy):
        #alpha_loss = (self._log_alpha[0] *
        #              (a_entropy - self._target_entropy[0]).detach())
        alpha_loss = torch.zeros(())
        if self._train_da_alpha:
            alpha_loss = alpha_loss + (self._log_alpha[1] * (
                da_entropy - self._target_entropy[1]).detach())
        return alpha_loss

    def _calc_critic_loss(self, info: SEditorInfo):
        critic_info = info.critic
        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            kwargs = dict(
                info=info,
                value=critic_info.critics[:, :, i, ...],
                target_value=critic_info.target_critic)
            critic_losses.append(l(**kwargs).loss)

        critic_loss = math_ops.add_n(critic_losses)
        return LossInfo(
            loss=critic_loss,
            extra=critic_loss / float(self._num_critic_replicas))

    def calc_loss(self, info: SEditorInfo):
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_info = info.actor
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha/a", self._log_alpha[0].exp())
                alf.summary.scalar("alpha/da", self._log_alpha[1].exp())

        return LossInfo(
            loss=actor_info.a_loss + actor_info.da_loss + alpha_loss +
            critic_loss.loss,
            extra=SEditorLossInfo(
                actor=actor_info, critic=critic_loss.extra, alpha=alpha_loss))
