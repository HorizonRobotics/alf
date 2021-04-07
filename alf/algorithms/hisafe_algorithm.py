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
import torch.distributions as td

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
from alf.utils.summary_utils import safe_mean_hist_summary

HiSafeInfo = namedtuple(
    "HiSafeInfo",
    ["reward", "step_type", "discount", "actor", "critic", "alpha"],
    default_value=())

HiSafeActorInfo = namedtuple(
    "HiSafeActorInfo",
    ["a_loss", "safe_a_loss", "a_entropy", "safe_a_entropy", "b_entropy"],
    default_value=())

HiSafeCriticInfo = namedtuple(
    "HiSafeCriticInfo", ["critics", "target_critic"], default_value=())

HiSafeLossInfo = namedtuple(
    "HiSafeLossInfo", ["actor", "critic", "alpha"], default_value=())

ActPredOutput = namedtuple(
    "ActPredOutput", [
        "a", "safe_a", "a_dist", "safe_a_dist", "b", "b_dist", "a_critics",
        "safe_a_critics", "output"
    ],
    default_value=())

Mode = Enum('AlgorithmMode', ('predict', 'rollout', 'train'))


def _safe_categorical(logits, alpha):
    r"""A numerically stable implementation of categorical distribution given
    the logits: :math:`exp(\frac{Q}{\alpha})`.
    """
    logits = logits / torch.clamp(alpha, min=1e-10)
    # logits are equivalent after subtracting a common number
    logits = logits - torch.mean(logits, dim=-1, keepdim=True)[0]
    return td.Categorical(logits=logits)


@alf.configurable
class HiSafeAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec,
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 reward_weights=None,
                 num_critic_replicas=2,
                 initial_alpha=1.,
                 epsilon_greedy=None,
                 env=None,
                 config: TrainerConfig = None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 train_b_entropy=True,
                 critic_loss_ctor=None,
                 optimizer=None,
                 debug_summaries=False,
                 target_entropy=None,
                 name="HiSafeAlgorithm"):

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

        (critic_networks,
         actor_network, safe_actor_network) = self._make_networks(
             observation_spec, action_spec, reward_spec, actor_network_cls,
             critic_network_cls)

        # [a_alpha, safe_a_alpha, b_alpha]
        if not isinstance(initial_alpha, (tuple, list)):
            initial_alpha = (initial_alpha, ) * 3
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
                safe_actor_network,
                critic_networks,
            ] + log_alpha)

        self._log_alpha = log_alpha
        self._log_alpha_paralist = nn.ParameterList(log_alpha)

        self._actor_network = actor_network
        self._safe_actor_network = safe_actor_network

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
            target_entropy = (target_entropy, ) * 3
        self._b_spec = BoundedTensorSpec(
            shape=(), maximum=1, dtype=torch.int64)
        self._target_entropy = nest.map_structure(
            lambda spec, t: _set_target_entropy(self.name, t, [spec]),
            (action_spec, action_spec, self._b_spec), target_entropy)

        self._train_b_entropy = train_b_entropy

        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

    def summarize_rollout(self, experience):
        ap_out = experience.rollout_info
        if self._debug_summaries:
            p_b0 = ap_out.b_dist.probs[..., 0]
            with alf.summary.scope(self._name):
                alf.summary.scalar("P_b0/mean", p_b0.mean())
                alf.summary.scalar("P_b0/std", p_b0.std())

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']

    def after_update(self, root_inputs, info: HiSafeInfo):
        self._update_target()

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       actor_network_cls, critic_network_cls):
        def _make_parallel(net):
            return net.make_parallel(
                self._num_critic_replicas * reward_spec.numel)

        actor_network = actor_network_cls(
            # should detach observation in the conf file
            input_tensor_spec=observation_spec,
            action_spec=action_spec)
        safe_actor_network = actor_network.copy()

        critic_network = critic_network_cls(
            input_tensor_spec=(observation_spec, action_spec),
            action_preprocessing_combiner=nest_utils.NestConcat())
        critic_networks = _make_parallel(critic_network)

        return critic_networks, actor_network, safe_actor_network

    def _predict_action(self,
                        inputs,
                        state,
                        epsilon_greedy=None,
                        mode=Mode.rollout):
        def _sample(dist, rsample=True):
            if mode == Mode.predict:
                a = dist_utils.epsilon_greedy_sample(dist, epsilon_greedy)
            else:
                if rsample:
                    a = dist_utils.rsample_action_distribution(dist)
                else:
                    a = dist_utils.sample_action_distribution(dist)
            return a, dist

        safe_a, safe_a_dist = _sample(
            self._safe_actor_network(inputs.observation)[0])

        a, a_dist = _sample(self._actor_network(inputs.observation)[0])

        safe_a_critics = self._compute_critics(self._critic_networks,
                                               inputs.observation, safe_a)
        a_critics = self._compute_critics(self._critic_networks,
                                          inputs.observation, a)

        # only use constraint Q to determine ``b``
        q_b0 = a_critics[..., 1] * self.reward_weights[1]

        a_diff_sqr = ((safe_a - a.detach())**2).mean(-1)
        q_b1 = torch.einsum(
            "bd,d->b",
            torch.stack((-a_diff_sqr, safe_a_critics[..., 1]), dim=-1),
            self.reward_weights)

        beta_alpha = self._log_alpha[2].exp().detach()
        b_dist = _safe_categorical(
            torch.stack((q_b0, q_b1), dim=-1), beta_alpha)
        b = _sample(b_dist, rsample=False)[0]

        # select which action to output
        out_a = torch.where(b.to(torch.bool).unsqueeze(-1), safe_a, a)

        return ActPredOutput(
            a=a,
            a_dist=a_dist,
            safe_a=safe_a,
            safe_a_dist=safe_a_dist,
            b=b,
            b_dist=b_dist,
            a_critics=a_critics,
            safe_a_critics=safe_a_critics,
            output=out_a)

    def predict_step(self, inputs: TimeStep, state):
        ap_out = self._predict_action(inputs, state, self._epsilon_greedy,
                                      Mode.predict)

        a_img, b_img, safe_a_img, out_a_img = None, None, None, None
        a_qc_img, safe_a_qc_img = None, None

        if render.is_rendering_enabled():
            a_img = render.render_action(
                name="a", action=ap_out.a, action_spec=self._action_spec)
            safe_a_img = render.render_action(
                name="safe_a",
                action=ap_out.safe_a,
                action_spec=self._action_spec)
            b_img = render.render_action(
                name="b", action=ap_out.b, action_spec=self._b_spec)
            out_a_img = render.render_action(
                name="out_a",
                action=ap_out.output,
                action_spec=self._action_spec)
            a_qc_img = render.render_bar(
                name="a_qc", data=ap_out.a_critics[..., 1])
            safe_a_qc_img = render.render_bar(
                name="safe_a_qc", data=ap_out.safe_a_critics[..., 1])

        return AlgStep(
            output=ap_out.output,
            state=state,
            info=dict(
                ap_out=ap_out,
                a_img=a_img,
                safe_a_img=safe_a_img,
                b_img=b_img,
                out_a_img=out_a_img,
                a_qc_img=a_qc_img,
                safe_a_qc_img=safe_a_qc_img))

    def rollout_step(self, inputs: TimeStep, state):
        ap_out = self._predict_action(inputs, state, mode=Mode.rollout)
        return AlgStep(output=ap_out.output, state=state, info=ap_out)

    def train_step(self, inputs: TimeStep, state, rollout_info):
        ap_out = self._predict_action(inputs, state, mode=Mode.train)

        a_entropy = -dist_utils.compute_log_probability(
            ap_out.a_dist, ap_out.a)
        safe_a_entropy = -dist_utils.compute_log_probability(
            ap_out.safe_a_dist, ap_out.safe_a)
        b_entropy = ap_out.b_dist.entropy()  # Categorical

        actor_loss = self._actor_train_step(inputs, ap_out, a_entropy)
        safe_actor_loss = self._safe_actor_train_step(inputs, ap_out,
                                                      safe_a_entropy)

        actor_info = HiSafeActorInfo(
            a_loss=actor_loss,
            safe_a_loss=safe_actor_loss,
            a_entropy=a_entropy,  # for summary
            safe_a_entropy=safe_a_entropy,  # for summary
            b_entropy=b_entropy)  # for summary

        critic_info = self._critic_train_step(inputs, ap_out, rollout_info)

        alpha_loss = self._alpha_train_step(a_entropy, safe_a_entropy,
                                            b_entropy)

        info = HiSafeInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            actor=actor_info,
            critic=critic_info,
            alpha=alpha_loss)

        return AlgStep(output=ap_out.output, state=state, info=info)

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

        # only maximize the utility Q value
        critics = torch.stack(
            (ap_out.a_critics[..., 0], ap_out.safe_a_critics[..., 0]), dim=-1)
        q = critics * ap_out.b_dist.probs

        dqda = nest_utils.grad(ap_out.a, q.sum(), retain_graph=True)
        actor_loss = self._actor_loss_fn(dqda, ap_out.a)
        actor_loss -= alpha * a_entropy
        return actor_loss

    def _safe_actor_train_step(self, inputs, ap_out, safe_a_entropy):
        alpha = self._log_alpha[1].exp().detach()

        # only maximize the constraint Q value
        critics = torch.stack(
            (ap_out.a_critics[..., 1], ap_out.safe_a_critics[..., 1]), dim=-1)
        q = critics * ap_out.b_dist.probs

        dqda = nest_utils.grad(ap_out.safe_a, q.sum())
        actor_loss = self._actor_loss_fn(dqda, ap_out.safe_a)
        actor_loss -= alpha * safe_a_entropy
        return actor_loss

    def _critic_train_step(self, inputs: TimeStep, ap_out, rollout_info):

        with torch.no_grad():
            target_critics = self._compute_critics(
                self._target_critic_networks, inputs.observation,
                ap_out.output)

        critics = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_info.output,
            replica_min=False)
        return HiSafeCriticInfo(critics=critics, target_critic=target_critics)

    def _alpha_train_step(self, a_entropy, safe_a_entropy, b_entropy):
        alpha_loss = (self._log_alpha[0] *
                      (a_entropy - self._target_entropy[0]).detach())
        alpha_loss += (self._log_alpha[1] *
                       (safe_a_entropy - self._target_entropy[1]).detach())
        if self._train_b_entropy:
            alpha_loss += (self._log_alpha[2] *
                           (b_entropy - self._target_entropy[2]).detach())
        return alpha_loss

    def _calc_critic_loss(self, info: HiSafeInfo):
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

    def calc_loss(self, info: HiSafeInfo):
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_info = info.actor
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha/a", self._log_alpha[0].exp())
                alf.summary.scalar("alpha/safe_a", self._log_alpha[1].exp())
                alf.summary.scalar("alpha/b", self._log_alpha[2].exp())

        return LossInfo(
            loss=actor_info.a_loss + actor_info.safe_a_loss + alpha_loss +
            critic_loss.loss,
            extra=HiSafeLossInfo(
                actor=actor_info, critic=critic_loss.extra, alpha=alpha_loss))
