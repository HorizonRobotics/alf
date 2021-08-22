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

import functools

import torch
import torch.nn as nn

import alf
from alf.data_structures import LossInfo, namedtuple, TimeStep, AlgStep
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.td_loss import TDLoss
import alf.nest.utils as nest_utils
from alf.tensor_specs import TensorSpec
from alf.networks import ActorDistributionNetwork, CriticNetwork
import alf.summary.render as render
from alf.utils import common, dist_utils, losses, math_ops, tensor_utils
from alf.utils.averager import EMAverager

ChebySafetyInfo = namedtuple(
    "ChebySafetyInfo",
    ["reward", "step_type", "discount", "actor", "critic", "alpha"],
    default_value=())

ChebySafetyActorInfo = namedtuple(
    "ChebySafetyActorInfo", ["a", "a_loss", "a_entropy", "q_select", "z"],
    default_value=())

ChebySafetyCriticInfo = namedtuple(
    "ChebySafetyCriticInfo", ["critics", "target_critic"], default_value=())

ChebySafetyLossInfo = namedtuple(
    "ChebySafetyLossInfo", ["actor", "critic", "alpha"], default_value=())


@alf.configurable
class ChebySafetyAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec,
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 reward_weights=None,
                 num_critic_replicas=2,
                 initial_alpha=1.,
                 epsilon_greedy=None,
                 env=None,
                 config=None,
                 z=None,
                 p=3,
                 target_update_tau=0.05,
                 target_update_period=1,
                 critic_loss_ctor=None,
                 optimizer=None,
                 debug_summaries=False,
                 target_entropy=None,
                 name="ChebySafetyAlgorithm"):

        assert len(alf.nest.flatten(
            action_spec)) == 1 and action_spec.is_continuous, (
                "Only support a single continuous action!")
        # d0 is utility and d1 is constraint
        assert reward_spec.numel == 2

        self._num_critic_replicas = num_critic_replicas
        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value(
                'TrainerConfig.epsilon_greedy')
        self._epsilon_greedy = epsilon_greedy

        (critic_networks, a_network) = self._make_networks(
            observation_spec, action_spec, reward_spec, actor_network_cls,
            critic_network_cls)

        log_alpha = nn.Parameter(torch.tensor(initial_alpha).log())

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
                a_network,
                critic_networks,
            ] + [log_alpha])

        self._log_alpha = log_alpha

        self._a_network = a_network

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

        self._target_entropy = _set_target_entropy(self.name, target_entropy,
                                                   [action_spec])

        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

        self._z = torch.tensor(z)
        self._p = float(p)

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']

    def after_update(self, root_inputs, info: ChebySafetyInfo):
        self._update_target()

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       actor_network_cls, critic_network_cls):
        def _make_parallel(net):
            return net.make_parallel(
                self._num_critic_replicas * reward_spec.numel)

        a_network = actor_network_cls(
            input_tensor_spec=observation_spec,
            input_preprocessors=alf.layers.Detach(),
            action_spec=action_spec)

        critic_network = critic_network_cls(
            input_tensor_spec=(observation_spec, action_spec),
            action_preprocessing_combiner=nest_utils.NestConcat())
        critic_networks = _make_parallel(critic_network)

        return critic_networks, a_network

    def predict_step(self, inputs: TimeStep, state):
        a_dist, _ = self._a_network(inputs.observation)
        a = dist_utils.epsilon_greedy_sample(a_dist, self._epsilon_greedy)
        if render.is_rendering_enabled():
            a_img = render.render_action(
                name="a", action=a, action_spec=self._action_spec)
        return AlgStep(output=a, state=state, info=())

    def rollout_step(self, inputs: TimeStep, state):
        a_dist, _ = self._a_network(inputs.observation)
        a = dist_utils.sample_action_distribution(a_dist)
        return AlgStep(output=a, state=state, info=a)

    def train_step(self, inputs: TimeStep, state,
                   rollout_info: ChebySafetyInfo):

        actor_info = self._actor_train_step(inputs)

        critic_info = self._critic_train_step(inputs, rollout_info, actor_info)

        alpha_loss = self._alpha_train_step(actor_info)

        a = actor_info.a
        actor_info = actor_info._replace(a=())  # only scalars are allowed

        info = ChebySafetyInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            actor=actor_info,
            critic=critic_info,
            alpha=alpha_loss)

        return AlgStep(output=a, state=state, info=info)

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         replica_min=True):
        """Compute Q(s,a)"""
        observation = (observation, action)
        critics, _ = critic_net(observation)  # [B, actions_n * replicas]
        critics = critics.reshape(  # [B, replicas, reward_dim]
            -1, self._num_critic_replicas, self._reward_spec.numel)
        if replica_min:
            critics = critics.min(dim=1)[0]
        return critics

    def _actor_loss_fn(self, dqda, action):
        loss = 0.5 * losses.element_wise_squared_loss(
            (dqda + action).detach(), action)
        return loss.sum(list(range(1, loss.ndim)))

    def _actor_train_step(self, inputs):

        a_dist, _ = self._a_network(inputs.observation)
        a = dist_utils.rsample_action_distribution(a_dist)  # [B,A]
        a_entropy = -dist_utils.compute_log_probability(a_dist, a)

        B = a.shape[0]

        critics = self._compute_critics(self._critic_networks,
                                        inputs.observation, a)

        f = (self._z - critics) * self.reward_weights
        f, q_select = f.max(dim=-1)

        #q = critics.mean(dim=0)
        #z = critics.max(dim=0)[0]
        #f = (self._z - q) * self.reward_weights

        #f, q_select = f.max(dim=0)
        q_select = q_select.to(torch.float32)

        #zq, _ = list(critics.max(dim=0)[0])

        #f = (2. - q_) * self.reward_weights[0]
        #fc = -qc_ * self.reward_weights[1]

        #if f > fc:
        #    q = critics[..., 0] * self.reward_weights[0]
        #else:
        #    q = critics[..., 1] * self.reward_weights[1]
        #q_select = torch.ones_like(q) * (f < fc).to(torch.float32)

        dqda = nest_utils.grad(a, -f.sum())
        a_loss = self._actor_loss_fn(dqda, a)
        a_loss -= self._log_alpha.exp().detach() * a_entropy

        return ChebySafetyActorInfo(
            a=a,
            a_loss=a_loss,
            a_entropy=a_entropy,
            q_select=q_select,
            #z=torch.tile(z, (B, 1))
        )

    def _critic_train_step(self, inputs: TimeStep, rollout_a, actor_info):

        with torch.no_grad():
            target_critics = self._compute_critics(
                self._target_critic_networks, inputs.observation, actor_info.a)

        critics = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_a,
            replica_min=False)
        return ChebySafetyCriticInfo(
            critics=critics, target_critic=target_critics)

    def _alpha_train_step(self, actor_info):
        alpha_loss = (self._log_alpha *
                      (actor_info.a_entropy - self._target_entropy).detach())
        return alpha_loss

    def _calc_critic_loss(self, info: ChebySafetyInfo):
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

    def calc_loss(self, info: ChebySafetyInfo):
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_info = info.actor
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha/a", self._log_alpha.exp())
            #for i in range(self._reward_spec.numel):
            #    alf.summary.scalar(f"z/{i}", actor_info.z[..., i].mean())
        actor_info = actor_info._replace(z=())

        return LossInfo(
            loss=(actor_info.a_loss + alpha_loss + critic_loss.loss),
            extra=ChebySafetyLossInfo(
                actor=actor_info, critic=critic_loss.extra, alpha=alpha_loss))
