# Copyright (c) 2023 Horizon Robotics. All Rights Reserved.
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
from typing import Callable, List, Union, Tuple

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
    [
        "ap_out",  # an instance of ``ActPredOutput``
        "reward",
        "step_type",
        "discount",
        "actor",
        "critic",
        "alpha"
    ],
    default_value=())

SEditorActorInfo = namedtuple(
    "SEditorActorInfo",
    [
        "a_loss",  # loss for UM policy (Eq 7a)
        "da_loss",  # loss for SE policy (Eq 7b)
        "a_entropy",  # entropy of UM policy
        "da_entropy",  # entropy of SE policy
        "change_a_loss",  # action editing loss :math:`-d(a,\hat{a})`
        "a_l2"  # :math:`|a-\hat{a}|^2`
    ],
    default_value=())

SEditorCriticInfo = namedtuple(
    "SEditorCriticInfo", ["critics", "target_critic"], default_value=())

SEditorLossInfo = namedtuple(
    "SEditorLossInfo", ["actor", "critic", "alpha"], default_value=())

ActPredOutput = namedtuple(
    "ActPredOutput",
    [
        "a",  # action proposal by UM
        "da",  # :math:`\delta a` by SE
        "a_dist",  # action proposal distribution
        "da_dist",  # delta distribution
        "output"  # final output after action editing
    ],
    default_value=())

Actions = namedtuple(
    "Actions", ["out_a_UM_detached", "out_a_SE_detached", "a_UM_detached"],
    default_value=())

Mode = Enum('AlgorithmMode', ('predict', 'rollout', 'train'))


@alf.configurable
class SEditorAlgorithm(OffPolicyAlgorithm):
    r"""Implements the Safe Editor Policy algorithm in the paper:

    "Towards Safe Reinforcement Learning with a Safety Editor Policy", Yu et al. 2022.

    SEditor consists of two policies UM and SE. UM proposes a preliminary action
    :math:`\hat{a}` that aims at maximizing the utility reward. Then SE outputs
    an adjustment :math:`\delta a`, to ensure a low constraint violation rate.

    Notation: Note that the notation 'a' in this file actually refers
    to :math:`\hat{a}` in the paper. For simplicity, we directly write as 'a'.
    'da' means the delta action by SE. The final output action is stored in
    ``ActPredOutput.output``.
    """

    def __init__(self,
                 observation_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 reward_spec: TensorSpec,
                 actor_network_ctor: Callable = ActorDistributionNetwork,
                 critic_network_ctor: Callable = CriticNetwork,
                 reward_weights: List[float] = None,
                 num_critic_replicas: int = 2,
                 initial_alpha: Union[float, Tuple[float, float]] = 1.,
                 train_alpha: bool = True,
                 epsilon_greedy: float = None,
                 soft_clipping: bool = False,
                 hinge_softness: float = 0.1,
                 regularize_action_diff: bool = False,
                 env=None,
                 config: TrainerConfig = None,
                 target_update_tau: float = 0.05,
                 target_update_period: float = 1,
                 critic_loss_ctor: Callable = None,
                 optimizer: torch.optim.Optimizer = None,
                 debug_summaries: bool = False,
                 target_entropy: Callable = None,
                 name="SEditorAlgorithm"):
        """
        Args:
            reward_spec: a two-dim reward spec. The first dim should be the utility
                reward and the second dim should be the (negative) constraint reward.
                Both rewards are the higher the better.
            actor_network_ctor: the same actor distribution network ctor for
                creating UM and SE.
            critic_network_ctor: the same critic network ctor for creating utility
                Q and constraint Q networks.
            reward_weights: passed to ``RLAlgorithm``. Weights for the reward dims.
            num_critic_replicas: SAC style duplicate critic values
            initial_alpha: initial entropy weight(s) for UM and SE. The two policies
                will have independent alpha training losses.
            train_alpha: whether to learn the entropy weights or not
            epsilon_greedy: epsilon greedy for prediction step
            soft_clipping: if True, then after applying :math:`\delta a` to
                :math:`\hat{a}`, we use a ``math_ops.softclip`` function to clip
                the modified action in range. Otherwise, the modified action is
                hard clipped.
            hinge_softness: parameter to ``math_ops.softclip()`` controlling how
                soft the clipping is
            regularize_action_diff: if True, then the action dist function ``d``
                is selected to be MSE error; otherwise it's the hinge loss of
                the utility state-action values (Eq 8). The paper uses False.
            target_update_tau: similar to that of SAC
            target_update_period: similar to that of SAC
            critic_loss_ctor: similar to that of SAC
            optimizer: if provided, will be used for optimizing the parameters
                of this alg
            target_entropy: either a single callable to compute the same target entropy
                for UM and SE, or a tuple of callabls to compute different target
                entropies for UM and SE.
        """

        assert len(
            nest.flatten(action_spec)) == 1 and action_spec.is_continuous, (
                "Only support a single continuous action!")
        # d0 is utility and d1 is constraint
        assert reward_spec.numel == 2

        self._num_critic_replicas = num_critic_replicas
        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value(
                'TrainerConfig.epsilon_greedy')
        self._epsilon_greedy = epsilon_greedy

        (critic_networks,
         actor_network, d_actor_network) = self._make_networks(
             observation_spec, action_spec, reward_spec, actor_network_ctor,
             critic_network_ctor)

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

        self._train_alpha = train_alpha
        self._soft_clipping = soft_clipping

        self._actor_network = actor_network
        self._d_actor_network = d_actor_network

        # Here we make a copy of ``actor_network`` and ``d_actor_network`` but
        # refrain from training them, so that it's easier for us to compute
        # :math:`dqd\hat{a}` and :math:`dqd\delta a` without being interfered by
        # the other network (Blue and orange paths in Figure 2).
        # After each gradient update, these two copies will get updated.
        self._actor_network_opt_ignored = actor_network.copy(
            name='actor_network_opt_ignored')
        self._d_actor_network_opt_ignored = d_actor_network.copy(
            name='d_actor_network_opt_ignored')

        self._critic_networks = critic_networks
        self._target_critic_networks = self._critic_networks.copy(
            name='target_critic_networks')

        self._hinge_softness = hinge_softness
        self._regularize_action_diff = regularize_action_diff

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

        self._update_target = common.TargetUpdater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

        self._actor_networks_copy = common.TargetUpdater(
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
                       actor_network_ctor, critic_network_ctor):
        def _make_parallel(net):
            return net.make_parallel(
                self._num_critic_replicas * reward_spec.numel)

        obs_action_spec = (observation_spec, action_spec)
        actor_network = actor_network_ctor(
            input_tensor_spec=observation_spec,
            input_preprocessors=alf.layers.Detach(),
            action_spec=action_spec)
        d_actor_network = actor_network_ctor(
            input_tensor_spec=obs_action_spec,
            input_preprocessors=(alf.layers.Detach(), None),
            preprocessing_combiner=nest_utils.NestConcat(),
            action_spec=action_spec)
        critic_network = critic_network_ctor(input_tensor_spec=obs_action_spec)
        critic_networks = _make_parallel(critic_network)

        return critic_networks, actor_network, d_actor_network

    def _forward(self,
                 time_step: TimeStep,
                 epsilon_greedy: float,
                 mode: Mode,
                 opt_ignore: str = 'none') -> ActPredOutput:
        r"""
        Doing one inference of SEditor. We first compute :math:`\hat{a}` using UM and
        then compute :math:`\delta a` using SE. Finally we using the action editing
        function to output the edited action.

        Args:
            time_step: current time step
            epsilon_greedy: only used by prediction
            mode: Mode.train, Mode.rollout, or Mode.predict
            opt_ignore: which policy network to ignore for optimization following
                this forward. Should be either 'a' (UM) or 'da' (SE). 'a' corresponds
                to the orange path in Figure 2, and 'da' corresponds to the blue
                path in Figure 2.

        Returns:
            ActPredOutput:
            - a: sampled action proposal
            - a_dist: action proposal distribution
            - da: sampled delta action
            - da_dist: delta action distribution
            - output: edited action
        """

        def _actor(net, in_a=None):
            if in_a is None:
                dist, _ = net(time_step.observation)
            else:
                dist, _ = net((time_step.observation, in_a))
            if mode == Mode.predict:
                a = dist_utils.epsilon_greedy_sample(dist, epsilon_greedy)
            else:
                a = dist_utils.rsample_action_distribution(dist)
            return a, dist

        a_net = (self._actor_network_opt_ignored
                 if opt_ignore == 'a' else self._actor_network)
        da_net = (self._d_actor_network_opt_ignored
                  if opt_ignore == 'da' else self._d_actor_network)

        # sample `a` according to current actor network
        a, a_dist = _actor(a_net)

        da, da_dist = _actor(da_net, a)

        return ActPredOutput(
            a=a,
            a_dist=a_dist,
            da=da,
            da_dist=da_dist,
            output=self._safe_action(a, da))

    def _predict_action(self,
                        time_step,
                        state,
                        epsilon_greedy=None,
                        mode=Mode.rollout,
                        rollout_info=None):

        if mode == Mode.train:
            ap_out0 = self._forward(
                time_step, epsilon_greedy, mode, opt_ignore='a')
            ap_out1 = self._forward(
                time_step, epsilon_greedy, mode, opt_ignore='da')
            return ActPredOutput(
                a=ap_out1.a,
                a_dist=ap_out1.a_dist,
                da=ap_out0.da,
                da_dist=ap_out0.da_dist,
                output=Actions(
                    out_a_SE_detached=ap_out1.output,
                    out_a_UM_detached=ap_out0.output,
                    # for computing how much the safety policy changes the action
                    a_UM_detached=ap_out0.a))
        else:
            return self._forward(
                time_step, epsilon_greedy, mode, opt_ignore='none')

    def _safe_action(self, a, da):
        r"""Perform action editing.

        Depending on ``self._soft_clipping``, we use either ``softclip`` or
        ``clip_to_spec``.

        Args:
            a: the proposed action (:math:`\hat{a}`) by UM.
            da: the delta action by SE.

        Output:
            torch.Tensor: the edited action to the env
        """
        if self._soft_clipping:
            return math_ops.softclip(
                a + 2 * da,
                low=float(self._action_spec.minimum),
                high=float(self._action_spec.maximum),
                hinge_softness=self._hinge_softness)
        else:
            return spec_utils.clip_to_spec(a + 2 * da, self._action_spec)

    def predict_step(self, inputs: TimeStep, state):
        ap_out = self._predict_action(inputs, state, self._epsilon_greedy,
                                      Mode.predict)

        imgs = {}
        if render.is_rendering_enabled():
            q_a = self._compute_critics(self._critic_networks,
                                        inputs.observation, ap_out.a)[:, 0]
            q_out_a = self._compute_critics(
                self._critic_networks, inputs.observation, ap_out.output)[:, 0]
            hinge_loss = torch.relu(q_a - q_out_a)
            l2 = ((ap_out.a - ap_out.output)**2).mean(dim=-1)

            imgs['hinge_loss'] = render.render_bar(
                name="hinge_loss", data=hinge_loss)
            imgs['l2'] = render.render_bar(name="action_l2", data=l2)
            imgs['a'] = render.render_action(
                name="a", action=ap_out.a, action_spec=self._action_spec)
            imgs['out_a'] = render.render_action(
                name="out_a",
                action=ap_out.output,
                action_spec=self._action_spec)

        return AlgStep(
            output=ap_out.output, state=state, info={
                'ap_out': ap_out,
                **imgs
            })

    def rollout_step(self, inputs: TimeStep, state):
        ap_out = self._predict_action(inputs, state, mode=Mode.rollout)
        return AlgStep(output=ap_out.output, state=state, info=ap_out)

    def train_step(self, inputs: TimeStep, state, rollout_info: ActPredOutput):
        def _calc_entropy(dist, a):
            return -dist_utils.compute_log_probability(dist, a)

        ap_out = self._predict_action(
            inputs, state, mode=Mode.train, rollout_info=rollout_info)

        a_entropy = _calc_entropy(ap_out.a_dist, ap_out.a)
        da_entropy = _calc_entropy(ap_out.da_dist, ap_out.da)

        actor_loss = self._actor_train_step(inputs, ap_out, a_entropy)
        da_loss, change_a_loss, a_l2 = self._d_actor_train_step(
            inputs, ap_out, da_entropy)
        actor_info = SEditorActorInfo(
            a_loss=actor_loss,
            da_loss=da_loss,
            a_entropy=a_entropy,
            da_entropy=da_entropy,
            change_a_loss=change_a_loss,
            a_l2=a_l2)

        critic_info = self._critic_train_step(inputs, ap_out, rollout_info)

        alpha_loss = self._alpha_train_step(a_entropy, da_entropy)

        info = SEditorInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            actor=actor_info,
            critic=critic_info,
            alpha=alpha_loss)

        return AlgStep(
            output=ap_out.output.out_a_SE_detached, state=state, info=info)

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
        """Train UM policy according to Eq 7a, to maximize the utility Q.
        """
        alpha = self._log_alpha[0].exp().detach()

        critics = self._compute_critics(self._critic_networks,
                                        inputs.observation,
                                        ap_out.output.out_a_SE_detached)
        # only maximize the utility Q value
        q = critics[..., 0].sum()

        dqda = nest_utils.grad(ap_out.a, q)
        actor_loss = self._actor_loss_fn(dqda, ap_out.a)
        actor_loss -= alpha * a_entropy
        return actor_loss

    def _d_actor_train_step(self, inputs, ap_out, da_entropy):
        """Train SE policy according to Eq 7b, to maximize the constraint Q while
        minimizing the action editing loss.
        """
        alpha = self._log_alpha[1].exp().detach()

        critics = self._compute_critics(self._critic_networks,
                                        inputs.observation,
                                        ap_out.output.out_a_UM_detached)

        # only maximize the constraint Q value
        q = critics[..., 1]

        a_l2 = ((ap_out.output.out_a_UM_detached - ap_out.output.a_UM_detached)
                **2).mean(dim=-1)

        if self._regularize_action_diff:
            # Take mean so that the loss magnitude is invariant to action dimension
            change_a_loss = a_l2
        else:
            with torch.no_grad():
                a_critics = self._compute_critics(self._critic_networks,
                                                  inputs.observation,
                                                  ap_out.output.a_UM_detached)
            change_a_loss = torch.relu(a_critics[..., 0] - critics[..., 0])

        q2 = torch.stack((-change_a_loss, q), dim=-1)
        q2 = torch.matmul(q2, self.reward_weights).sum()
        dqda = nest_utils.grad(ap_out.da, q2)

        actor_loss = self._actor_loss_fn(dqda, ap_out.da)
        actor_loss -= alpha * da_entropy
        return actor_loss, change_a_loss, a_l2

    def _critic_train_step(self, inputs: TimeStep, ap_out,
                           rollout_info: ActPredOutput):
        """Typical TD learning as seen in SAC. The utility Q and constraint Q
        are learned in parallel.
        """

        with torch.no_grad():
            target_critics = self._compute_critics(
                self._target_critic_networks, inputs.observation,
                ap_out.output.out_a_SE_detached)

        critics = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_info.output,
            replica_min=False)
        return SEditorCriticInfo(critics=critics, target_critic=target_critics)

    def _alpha_train_step(self, a_entropy, da_entropy):
        if self._train_alpha:
            alpha_loss = (self._log_alpha[0] *
                          (a_entropy - self._target_entropy[0]).detach())
            alpha_loss = alpha_loss + (self._log_alpha[1] * (
                da_entropy - self._target_entropy[1]).detach())
        else:
            alpha_loss = torch.zeros_like(a_entropy)
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
            loss=(actor_info.a_loss + actor_info.da_loss + alpha_loss +
                  critic_loss.loss),
            extra=SEditorLossInfo(
                actor=actor_info, critic=critic_loss.extra, alpha=alpha_loss))
