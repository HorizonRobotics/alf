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
"""Behavior Cloning (BC) Algorithm."""

import torch
from typing import Any, Callable, Optional

import alf
from alf.algorithms.bc_algorithm import BcState
from alf.algorithms.config import TrainerConfig
from alf.algorithms.dynamics_learning_algorithm import DynamicsLearningAlgorithm
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.data_structures import AlgStep
from alf.networks import ActorNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import dist_utils
from alf.utils.math_ops import add_ignore_empty, jacobian_trace

VfbcState = namedtuple("VfbcState", ["actor", "dynamics"], default_value=())

VfbcInfo = namedtuple("VfbcInfo", ["actor", "dynamics"], default_value=())

VfbcRolloutInfo = namedtuple("VfbcRolloutInfo", ["basic", "feature"],
                             default_value=())

BcLossInfo = namedtuple("LossInfo", ["actor"], default_value=())


@alf.configurable
class VfbcAlgorithm(OffPolicyAlgorithm):
    """Vector Field Divergence Regularized Behavior cloning algorithm.
    Behavior cloning algorithm with vector field regularization
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorNetwork,
                 dynamics_module_ctor: Optional[
                     Callable[[Any, Any], DynamicsLearningAlgorithm]] = None,
                 actor_optimizer=None,
                 dynamics_optimizer=None,
                 env=None,
                 config: TrainerConfig = None,
                 checkpoint=None,
                 debug_summaries=False,
                 epsilon_greedy=None,
                 name="VfbcAlgorithm"):
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
                used in BcAlgorithm.
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network is a determinstic network and
                will be used to generate continuous actions.
            dynamics_module_ctor: used to construct the module for learning to
                predict the difference between the next feature the previous feature 
                based on the previous feature and action. It should accept input 
                with spec [feature_spec, encoded_action_spec] and output a tensor 
                of shape feature_spec.
                For discrete action, encoded_action is an one-hot representation
                of the action. For continuous action, encoded action is same as
                the original action.
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

        action_state_spec = actor_network.state_spec

        dynamics_module = None
        if dynamics_module_ctor is not None:
            dynamics_module = dynamics_module_ctor(
                action_spec=action_spec, feature_spec=observation_spec)

        train_state_spec = VfbcState(
            actor=action_state_spec,
            dynamics=dynamics_module.train_state_spec
            if dynamics_module is not None else ())

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            # train_state_spec=BcState(actor=action_state_spec),
            predict_state_spec=BcState(actor=action_state_spec),
            reward_weights=None,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network
        self._dynamics_module = dynamics_module

        if actor_optimizer is not None and actor_network is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if dynamics_optimizer is not None:
            self.add_optimizer(dynamics_optimizer, [dynamics_module])
        self._actor_optimizer = actor_optimizer
        self._dynamics_optimizer = dynamics_optimizer

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

    def _actor_train_step_imitation(self, inputs: TimeStep, rollout_info,
                                    action_dist):

        exp_action = rollout_info.action
        im_loss = -action_dist.log_prob(exp_action)

        actor_info = LossInfo(loss=im_loss, extra=BcLossInfo(actor=im_loss))

        return actor_info

    def preprocess_experience(self, root_inputs: TimeStep, 
                              rollout_info, batch_info):
        """Fill experience.rollout_info with prev observations.

        Note that the shape of experience is [B, T, ...].

        The target is a Tensor (or a nest of Tensors) when there is only one
        decoder. When there are multiple decorders, the target is a list,
        and each of its element is a Tensor (or a nest of Tensors), which is
        used as the target for the corresponding decoder.
        """
        observation = root_inputs.observation
        ind = torch.arange(-1, observation.shape[1] - 1)
        feature = observation[:, ind, ...]

        return root_inputs, VfbcRolloutInfo(
            basic=rollout_info, feature=feature)

    def _actor_vf_loss(self, observation, state):
        observation.requires_grad_()
        action_dist, actor_state = self._predict_action(observation, state.actor)
        dynamics_step = self._dynamics_module.predict_step(
            observation, action_dist.base_dist.mean, state.dynamics)
        obs_vf = dynamics_step.output
        vf_loss = jacobian_trace(obs_vf, observation)

        return vf_loss, VfbcState(actor=actor_state, dynamics=dynamics_step.state)


    def train_step_offline(self,
                           inputs: TimeStep,
                           state,
                           rollout_info,
                           pre_train=False):

        action_dist, actor_state = self._predict_action(
            inputs.observation, state=state.actor)

        actor_imitation_info = self._actor_train_step_imitation(inputs, rollout_info.basic,
                                                      action_dist)
        actor_vf_loss, state = self._actor_vf_loss(
            inputs.observation, state)
        actor_loss = actor_imitation_info.loss + actor_vf_loss

        dynamics_step = self._dynamics_module.train_step(
            inputs, state.dynamics, rollout_info.feature)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("imitation_loss", actor_imitation_info.loss.mean())
                alf.summary.scalar("vf_loss", actor_vf_loss.mean())

        info = VfbcInfo(actor=actor_loss, dynamics=dynamics_step.info)
        return AlgStep(
            rollout_info.basic.action, 
            state=VfbcState(actor=actor_state,
                            dynamics=dynamics_step.state), 
            info=info)

    def calc_loss_offline(self, info, pre_train=False):
        loss_dynamics = self._dynamics_module.calc_loss(info.dynamics)
        loss = loss_dynamics.loss
        loss = add_ignore_empty(loss, info.actor)

        return LossInfo(loss=loss, scalar_loss=loss_dynamics.scalar_loss)
