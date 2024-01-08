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

SmoLossInfo = namedtuple(
    "SmoLossInfo", ["actor", "grad_penalty"], default_value=())


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

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorNetwork,
                 v_network_cls=ValueNetwork,
                 discriminator_network_cls=None,
                 actor_optimizer=None,
                 value_optimizer=None,
                 discriminator_optimizer=None,
                 gamma: float = 0.99,
                 f: str = "chi",
                 gradient_penalty_weight: float = 1,
                 env=None,
                 config: TrainerConfig = None,
                 checkpoint=None,
                 debug_summaries=False,
                 epsilon_greedy=None,
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
            v_network_cls (Callable): is used to construct the value network.
            discriminator_network_cls (Callable): is used to construct the discriminatr.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            value_optimizer (torch.optim.optimizer): The optimizer for value network.
            discriminator_optimizer (torch.optim.optimizer): The optimizer for discriminator.
            gamma (float): the discount factor.
            f: the function form for f-divergence. Currently support 'chi' and 'kl'
            gradient_penalty_weight: the weight for discriminator gradient penalty
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
        self._gradient_penalty_weight = gradient_penalty_weight

        assert actor_optimizer is not None
        if actor_optimizer is not None and actor_network is not None:
            self.add_optimizer(actor_optimizer, [actor_network])

        assert value_optimizer is not None
        if value_optimizer is not None and value_network is not None:
            self.add_optimizer(value_optimizer, [value_network])

        assert discriminator_optimizer is not None
        if discriminator_optimizer is not None and discriminator_net is not None:
            self.add_optimizer(discriminator_optimizer, [discriminator_net])

        self._gamma = gamma
        self._f = f
        assert f in ["chi", "kl"], "only support chi or kl form"

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

        discriminator_inputs = (observation, action)

        if is_expert:
            # turn on input gradient for gradient penalty in the case of expert data
            for e in discriminator_inputs:
                e.requires_grad = True

        expert_logits, _ = self._discriminator_net(discriminator_inputs, state)

        if is_expert:
            grads = torch.autograd.grad(
                outputs=expert_logits,
                inputs=discriminator_inputs,
                grad_outputs=torch.ones_like(expert_logits),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)

            grad_pen = 0
            for g in grads:
                grad_pen += self._gradient_penalty_weight * (
                    g.norm(2, dim=1) - 1).pow(2)

            label = torch.ones(expert_logits.size())
            # turn on input gradient for gradient penalty in the case of expert data
            for e in discriminator_inputs:
                e.requires_grad = True
        else:
            label = torch.zeros(expert_logits.size())
            grad_pen = ()

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, label, reduction='none')

        return LossInfo(
            loss=expert_loss if grad_pen == () else expert_loss + grad_pen,
            extra=SmoLossInfo(actor=expert_loss, grad_penalty=grad_pen))

    def value_train_step(self, inputs: TimeStep, state, rollout_info):
        observation = inputs.observation
        initial_observation = observation
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
                                   expert_disc_loss.extra.actor.mean())

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
                alf.summary.scalar("grad_penalty",
                                   expert_disc_loss.extra.grad_penalty.mean())
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
