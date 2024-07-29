# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Causal Behavior Cloning Algorithm."""

import torch
import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.data_structures import AlgStep
from alf.networks import ActorNetwork, EncodingNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import dist_utils, tensor_utils

BcState = namedtuple("BcState", ["actor"], default_value=())

BcInfo = namedtuple(
    "BcInfo", ["actor", "discriminator", "target"], default_value=())

BcLossInfo = namedtuple(
    "LossInfo", ["actor", "discriminator"], default_value=())


@alf.configurable
class CausalBcAlgorithm(OffPolicyAlgorithm):
    r"""Causal behavior cloning algorithm.
    This is the implementation of ResiduIL algorithm proposed in the following
    paper:
    ::
        Swamy et al. Causal Imitation Learning under Temporally Correlated Noise,
        ICML 2022
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorNetwork,
                 discriminator_network_cls=EncodingNetwork,
                 actor_optimizer=None,
                 discriminator_optimizer=None,
                 f_norm_penalty_weight=1e-3,
                 bc_regulatization_weight=5e-2,
                 env=None,
                 config: TrainerConfig = None,
                 checkpoint=None,
                 debug_summaries=False,
                 epsilon_greedy=None,
                 name="CausalBcAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                be a mixture of discrete and continuous actions. The number of
                continuous actions can be arbitrary while only one discrete
                action is allowed currently. If it's a mixture, then it must be
                a tuple/list ``(discrete_action_spec, continuous_action_spec)``.
            reward_spec (Callable): a rank-1 or rank-0 tensor spec representing
                the reward(s). For interface compatibility purpose. Not actually
                used in CausalBcAlgorithm.
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network is a deterministic network and
                will be used to generate continuous actions.
            discriminator_network_cls (Callable): is used to construct the
                discriminator network. The discrimonator is trained in a way
                that is adversarial to the training of the policy, to help with
                the learning of a robust policy. It takes the observation from
                the previous time step to generate the lagrange multiplier
                for the current step.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            discriminator_optimizer (torch.optim.optimizer): the optimizer for
                discriminator.
            f_norm_penalty_weight (float): penalty weight for the output of
                the discriminator.
            bc_regulatization_weight (float): weight for the squared prediction
                error based regularization term.
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

        discriminator_network = discriminator_network_cls(
            input_tensor_spec=observation_spec)

        action_state_spec = actor_network.state_spec
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=BcState(actor=action_state_spec),
            predict_state_spec=BcState(actor=action_state_spec),
            reward_weights=None,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network
        self._discriminator_network = discriminator_network

        if actor_optimizer is not None and actor_network is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        self._actor_optimizer = actor_optimizer

        if discriminator_optimizer is not None and discriminator_network is not None:
            self.add_optimizer(discriminator_optimizer,
                               [discriminator_network])
        self._discriminator_optimizer = discriminator_optimizer

        self._bc_regulatization_weight = bc_regulatization_weight
        self._f_norm_penalty_weight = f_norm_penalty_weight

    def _predict_action(self, observation, state):
        action_dist, actor_network_state = self._actor_network(
            observation, state=state)

        return action_dist, actor_network_state

    def predict_step(self, inputs: TimeStep, state: BcState):
        action_dist, new_state = self._predict_action(
            inputs.observation, state=state.actor)
        action = dist_utils.epsilon_greedy_sample(action_dist,
                                                  self._epsilon_greedy)

        return AlgStep(output=action, state=BcState(actor=new_state))

    def residuIL_loss(self, targets, predictions, pred_residuals):

        # train policy (detach discriminator)
        target_prediction_differences = targets - predictions

        # bc_regularization is optionally used according to Appendix of the
        # paper (Table 4 and 5)
        policy_loss = (
            2 * (target_prediction_differences) * pred_residuals.detach()
        ).mean(-1) + self._bc_regulatization_weight * (
            torch.square(target_prediction_differences)).mean(-1)

        # train discriminator (detach policy)
        discriminator_loss = -(
            2 * (target_prediction_differences).detach() * pred_residuals -
            pred_residuals * pred_residuals).mean(-1)

        # f_norm_penalty is used according to Appendix of the paper (Table 4 and 5)
        discriminator_loss = (
            discriminator_loss +
            self._f_norm_penalty_weight * torch.linalg.norm(pred_residuals))

        return policy_loss, discriminator_loss

    def train_step_offline(self,
                           inputs: TimeStep,
                           state,
                           rollout_info,
                           pre_train=False):

        action_dist, new_state = self._predict_action(
            inputs.observation, state=state.actor)

        predictions = dist_utils.get_rmode(action_dist)
        pred_residuals, _ = self._discriminator_network(inputs.observation)

        info = BcInfo(
            actor=predictions,
            discriminator=pred_residuals,
            target=rollout_info.action)
        return AlgStep(
            rollout_info.action, state=BcState(actor=new_state), info=info)

    def calc_loss_offline(self, info, pre_train=False):

        #[T, B, action_dim]
        predictions = info.actor
        pred_residuals = info.discriminator
        targets = info.target

        actor_loss, discriminator_loss = self.residuIL_loss(
            targets[1:], predictions[1:], pred_residuals[:-1])

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("actor_loss", actor_loss.mean())
                alf.summary.scalar("discriminator_loss",
                                   discriminator_loss.mean())

        loss = actor_loss + discriminator_loss
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(
            loss=loss,
            extra=BcLossInfo(
                actor=tensor_utils.tensor_extend_zero(actor_loss),
                discriminator=tensor_utils.tensor_extend_zero(
                    discriminator_loss)))
