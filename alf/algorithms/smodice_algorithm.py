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

import torch
import torch.nn.functional as F

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.data_structures import AlgStep
from alf.networks import EncodingNetwork, ActorNetwork, ValueNetwork, CriticNetwork, Network
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import dist_utils
from alf.data_structures import AlgStep, StepType
from alf.utils import losses, tensor_utils
from alf.algorithms.algorithm import Algorithm

SmoState = namedtuple("SmoState", ["actor"], default_value=())

SmoInfo = namedtuple(
    "SmoInfo",
    [
        "actor",
        "value",
        "discriminator_loss",
        "reward",
        "discount",
        # online
        "action",
        "action_distribution"
    ],
    default_value=())

SmoCriticInfo = namedtuple("SmoCriticInfo",
                           ["values", "initial_v_values", "is_first"])

SmoLossInfo = namedtuple("SmoLossInfo", ["actor"], default_value=())


# -> algorithm
class Discriminator_SA(Algorithm):
    def __init__(self, observation_spec, action_spec):
        super().__init__(observation_spec=observation_spec)

        disc_net = CriticNetwork((observation_spec, action_spec))
        self._disc_net = disc_net

    def forward(self, inputs, state=()):
        return self._disc_net(inputs, state)

    def compute_grad_pen(self, expert_state, offline_state, lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = expert_state
        offline_data = offline_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * offline_data
        mixup_data.requires_grad = True

        disc = self(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, offline_loader):
        self.train()

        loss = 0
        n = 0
        for expert_state, offline_state in zip(expert_loader, offline_loader):

            expert_state = expert_state[0].to(self.device)
            offline_state = offline_state[0][:expert_state.shape[0]].to(
                self.device)

            policy_d = self(offline_state)
            expert_d = self(expert_state)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, offline_state)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state):
        with torch.no_grad():
            self.eval()
            d = self(state)
            s = torch.sigmoid(d)
            # log(d^E/d^O)
            # reward  = - (1/s-1).log()
            reward = s.log() - (1 - s).log()
            return reward


@alf.configurable
class SmodiceAlgorithm(OffPolicyAlgorithm):
    r"""SMODICE algorithm.
    SMODICE is an offline imitation approach to learn a policy
    :math:`\pi_{\theta}(a|s)`, which is a function that maps an input
    observation :math:`s` to an action :math:`a`. The paramerates (:math:`\theta`)
    of this policy is learned by maximizing the weighted probability of dataset actions
    on the training data :math:`D`:
    :math:`\max_{\theta} E_{(s,a)~D} w(s, s') \log \pi_{\theta}(a|s)`,
    where (s, s') is a state transition pair.

    Reference:
    ::
        Jason Ma SMODICE: Versatile Offline Imitation Learning via State Occupancy Matching,
        ICML 2022.
    """

    def __init__(
            self,
            observation_spec,
            action_spec: BoundedTensorSpec,
            reward_spec=TensorSpec(()),
            actor_network_cls=ActorNetwork,
            v_network_cls=ValueNetwork,
            discriminator_network_cls=None,
            actor_optimizer=None,
            value_optimizer=None,
            discriminator_optimizer=None,
            #=====new params
            gamma: float = 0.99,
            v_l2_reg: float = 0.001,
            env=None,
            config: TrainerConfig = None,
            checkpoint=None,
            debug_summaries=False,
            epsilon_greedy=None,
            f="chi",
            name="SmodiceAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                be a mixture of discrete and continuous actions. The number of
                continuous actions can be arbitrary while only one discrete
                action is allowed currently. If it's a mixture, then it must be
                a tuple/list ``(discrete_action_spec, continuous_action_spec)``.
            reward_spec (Callable): a rank-1 or rank-0 tensor spec representing
                the reward(s). For interface compatiblity purpose. Not actually
                used in SmodiceAlgorithm.
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network is a determinstic network and
                will be used to generate continuous actions.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries (bool): True if debug summaries should be created.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            name (str): The name of this algorithm.
        """

        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy

        actor_network = actor_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        value_network = v_network_cls(input_tensor_spec=observation_spec)

        discriminator_net = discriminator_network_cls(
            input_tensor_spec=(observation_spec, action_spec))

        action_state_spec = actor_network.state_spec
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=SmoState(actor=action_state_spec),
            predict_state_spec=SmoState(actor=action_state_spec),
            reward_weights=None,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network
        self._value_network = value_network
        self._discriminator_net = discriminator_net

        assert actor_optimizer is not None
        if actor_optimizer is not None and actor_network is not None:
            self.add_optimizer(actor_optimizer, [actor_network])

        assert value_optimizer is not None
        if value_optimizer is not None and value_network is not None:
            self.add_optimizer(value_optimizer, [value_network])

        assert discriminator_optimizer is not None
        if discriminator_optimizer is not None and discriminator_net is not None:
            self.add_optimizer(discriminator_optimizer, [discriminator_net])

        self._actor_optimizer = actor_optimizer
        self._value_optimizer = value_optimizer
        self._v_l2_reg = v_l2_reg
        self._gamma = gamma
        self._f = f
        assert f == "chi", "only support chi form"

        # f-divergence functions
        if self._f == 'chi':
            self._f_fn = lambda x: 0.5 * (x - 1)**2
            self._f_star_prime = lambda x: torch.relu(x + 1)
            self._f_star = lambda x: 0.5 * x**2 + x
        elif self._f == 'kl':
            self._f_fn = lambda x: x * torch.log(x + 1e-10)
            self._f_star_prime = lambda x: torch.exp(x - 1)

    def _predict_action(self, observation, state):
        action_dist, actor_network_state = self._actor_network(
            observation, state=state)

        return action_dist, actor_network_state

    def rollout_step(self, inputs: TimeStep, state: SmoState):
        action_dist, new_state = self._predict_action(
            inputs.observation, state=state.actor)
        action = dist_utils.epsilon_greedy_sample(action_dist,
                                                  self._epsilon_greedy)

        info = SmoInfo()
        return AlgStep(
            output=action,
            state=SmoState(actor=new_state),
            info=info._replace(action=action, action_distribution=action_dist))

    def predict_step(self, inputs: TimeStep, state: SmoState):
        action_dist, new_state = self._predict_action(
            inputs.observation, state=state.actor)
        action = dist_utils.epsilon_greedy_sample(action_dist,
                                                  self._epsilon_greedy)

        return AlgStep(output=action, state=SmoState(actor=new_state))

    def _actor_train_step_imitation(self, inputs: TimeStep, rollout_info,
                                    action_dist):

        exp_action = rollout_info.action
        im_loss = -action_dist.log_prob(exp_action)

        actor_info = LossInfo(loss=im_loss, extra=SmoLossInfo(actor=im_loss))

        return actor_info

    def predict_reward(self, inputs, rollout_info, state=()):
        with torch.no_grad():
            observation = inputs.observation
            action = rollout_info.action
            expert_logits, _ = self._discriminator_net((observation, action),
                                                       state)
            # self._discriminator_net.eval()
            s = torch.sigmoid(expert_logits)
            # log(d^E/d^O)
            # reward  = - (1/s-1).log()
            reward = s.log() - (1 - s).log()
            return reward

    def _discriminator_train_step(self, inputs: TimeStep, state, rollout_info,
                                  is_expert):
        """train discriminator with offline data (expert)
        """
        observation = inputs.observation
        action = rollout_info.action
        expert_logits, _ = self._discriminator_net((observation, action),
                                                   state)

        if is_expert:
            label = torch.ones(expert_logits.size())
        else:
            label = torch.zeros(expert_logits.size())

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, label, reduction='none')

        return LossInfo(loss=expert_loss, extra=SmoLossInfo(actor=expert_loss))

    def value_train_step(self, inputs: TimeStep, state, rollout_info):
        # initial_v_values, e_v, result={}
        observation = inputs.observation

        # extract initial observation from batch, or prepare a batch
        initial_observation = observation

        # Shared network values
        # mini_batch_length
        initial_v_values, _ = self._value_network(initial_observation)

        # mini-batch len
        v_values, _ = self._value_network(observation)
        info = SmoCriticInfo(
            initial_v_values=initial_v_values,
            values=v_values,
            is_first=inputs.is_first())
        return info

    def train_step(self,
                   inputs: TimeStep,
                   state,
                   rollout_info,
                   pre_train=False):

        action_dist, new_state = self._predict_action(
            inputs.observation, state=state.actor)

        actor_loss = self._actor_train_step_imitation(inputs, rollout_info,
                                                      action_dist)

        value_info = self.value_train_step(
            inputs, state=(), rollout_info=rollout_info)

        expert_disc_loss = self._discriminator_train_step(
            inputs, state, rollout_info, is_expert=False)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("imitation_loss_online",
                                   actor_loss.loss.mean())
                alf.summary.scalar("discriminator_loss_online",
                                   expert_disc_loss.loss.mean())

        # use predicted reward
        reward = self.predict_reward(inputs, rollout_info)

        info = SmoInfo(
            actor=actor_loss,
            value=value_info,
            discriminator_loss=expert_disc_loss,
            reward=reward,
            discount=inputs.discount)

        return AlgStep(
            rollout_info.action, state=SmoState(actor=new_state), info=info)

    def train_step_offline(self,
                           inputs: TimeStep,
                           state,
                           rollout_info,
                           pre_train=False):

        action_dist, new_state = self._predict_action(
            inputs.observation, state=state.actor)

        actor_loss = self._actor_train_step_imitation(inputs, rollout_info,
                                                      action_dist)

        value_info = self.value_train_step(
            inputs, state=(), rollout_info=rollout_info)

        expert_disc_loss = self._discriminator_train_step(
            inputs, state, rollout_info, is_expert=True)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("imitation_loss_offline",
                                   actor_loss.loss.mean())
                alf.summary.scalar("discriminator_loss_offline",
                                   expert_disc_loss.loss.mean())

        # use predicted reward
        reward = self.predict_reward(inputs, rollout_info)

        info = SmoInfo(
            actor=actor_loss,
            value=value_info,
            discriminator_loss=expert_disc_loss,
            reward=reward,
            discount=inputs.discount)

        return AlgStep(
            rollout_info.action, state=SmoState(actor=new_state), info=info)

    def calc_loss(
            self,
            info,
    ):

        # [mini_batch_len, batch_size]
        values = info.value.values
        initial_v_values = info.value.initial_v_values
        is_first = info.value.is_first

        reward = info.reward[1:]
        v = values[:-1]
        v_next = values[1:]
        discount = info.discount[1:]

        e_v = reward + (1 - discount) * self._gamma * v_next - v

        v_loss0 = (1 - self._gamma) * initial_v_values * is_first

        if self._f == 'kl':
            v_loss1 = torch.log(torch.mean(
                torch.exp(e_v))).unsqueeze(-1).expand(e_v.shape[0], 1)
        else:
            v_loss1 = (self._f_star(e_v))

        v_loss1 = tensor_utils.tensor_extend_zero(v_loss1)

        v_loss = v_loss0 + v_loss1

        # weighted policy loss
        # # extracting importance weight (Equation 21 in the paper)
        if self._f == 'kl':
            w_e = torch.exp(e_v)
        else:
            w_e = self._f_star_prime(e_v)

        actor_loss_tensor = info.actor.loss

        # [T, B]
        discriminator_loss = info.discriminator_loss.loss

        # detach the weight from policy loss
        w_a_loss = actor_loss_tensor[:-1] * w_e.detach()
        w_a_loss = tensor_utils.tensor_extend_zero(w_a_loss)

        return LossInfo(
            loss=w_a_loss + v_loss + discriminator_loss,
            extra=SmoLossInfo(actor=info.actor.extra))
