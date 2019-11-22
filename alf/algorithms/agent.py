# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Agent for integrating multiple algorithms."""

from typing import Callable

import gin
import tensorflow as tf

from tf_agents.networks.network import Network
from tf_agents.trajectories.policy_step import PolicyStep

from alf.algorithms.algorithm import Algorithm
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.entropy_target_algorithm import EntropyTargetAlgorithm
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.algorithms.misc_algorithm import MISCAlgorithm
from alf.algorithms.on_policy_algorithm import Experience, OnPolicyAlgorithm, RLAlgorithm
from alf.algorithms.rl_algorithm import ActionTimeStep, TrainingInfo, LossInfo, namedtuple
from alf.utils.common import cast_transformer
from alf.utils.math_ops import add_ignore_empty

AgentState = namedtuple("AgentState", ["rl", "icm"], default_value=())

AgentInfo = namedtuple(
    "AgentInfo", ["rl", "icm", "entropy_target"], default_value=())

AgentLossInfo = namedtuple(
    "AgentLossInfo", ["rl", "icm", "entropy_target"], default_value=())


@gin.configurable
class Agent(OnPolicyAlgorithm):
    """Agent

    Agent is a master algorithm that integrates different algorithms together.
    """

    def __init__(self,
                 action_spec,
                 rl_algorithm_cls=ActorCriticAlgorithm,
                 encoding_network: Network = None,
                 intrinsic_curiosity_module=None,
                 intrinsic_reward_coef=1.0,
                 extrinsic_reward_coef=1.0,
                 enforce_entropy_target=False,
                 optimizer=None,
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 reward_shaping_fn: Callable = None,
                 observation_transformer=cast_transformer,
                 debug_summaries=False,
                 name="AgentAlgorithm"):
        """Create an Agent

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            rl_algorithm_cls (type): The algorithm class for learning the policy.
            encoding_network (Network): A function that encodes the observation
            intrinsic_curiosity_module (Algorithm): an algorithm whose outputs
                is a scalar intrinsic reward
            intrinsic_reward_coef (float): Coefficient for intrinsic reward
            extrinsic_reward_coef (float): Coefficient for extrinsic reward
            enforce_entropy_target (bool): If True, use EntropyTargetAlgorithm
                to dynamically adjust entropy regularization so that entropy is
                not smaller than `entropy_target` supplied for constructing
                EntropyTargetAlgorithm. If this is enabled, make sure you don't
                use entropy_regularization for loss (see ActorCriticLoss or
                PPOLoss).
            optimizer (tf.optimizers.Optimizer): The optimizer for training
            gradient_clipping (float): If not None, serve as a positive threshold
                for clipping gradient norms
            clip_by_global_norm (bool): If True, use tf.clip_by_global_norm to
                clip gradient. If False, use tf.clip_by_norm for each grad.
            reward_shaping_fn (Callable): a function that transforms extrinsic
                immediate rewards
            observation_transformer (Callable | list[Callable]): transformation(s)
                applied to `time_step.observation`
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
            """
        rl_algorithm = rl_algorithm_cls(
            action_spec=action_spec, debug_summaries=debug_summaries)
        train_state_spec = AgentState(rl=rl_algorithm.train_state_spec)
        predict_state_spec = AgentState(rl=rl_algorithm.predict_state_spec)

        if intrinsic_curiosity_module is not None:
            train_state_spec = train_state_spec._replace(
                icm=intrinsic_curiosity_module.train_state_spec)

        entropy_target_algorithm = None
        if enforce_entropy_target:
            entropy_target_algorithm = EntropyTargetAlgorithm(
                action_spec, debug_summaries=debug_summaries)

        super(Agent, self).__init__(
            action_spec=action_spec,
            predict_state_spec=predict_state_spec,
            train_state_spec=train_state_spec,
            action_distribution_spec=rl_algorithm.action_distribution_spec,
            optimizer=[optimizer],
            trainable_module_sets=[[encoding_network]],
            gradient_clipping=gradient_clipping,
            clip_by_global_norm=clip_by_global_norm,
            reward_shaping_fn=reward_shaping_fn,
            observation_transformer=observation_transformer,
            debug_summaries=debug_summaries,
            name=name)

        self._rl_algorithm = rl_algorithm
        self._entropy_target_algorithm = entropy_target_algorithm
        self._encoding_network = encoding_network
        self._intrinsic_reward_coef = intrinsic_reward_coef
        self._extrinsic_reward_coef = extrinsic_reward_coef
        self._icm = intrinsic_curiosity_module

    def _encode(self, time_step: ActionTimeStep):
        observation = time_step.observation
        if self._encoding_network is not None:
            observation, _ = self._encoding_network(observation)
        return observation

    def predict(self, time_step: ActionTimeStep, state: AgentState):
        """Predict for one step."""
        observation = self._encode(time_step)

        new_state = AgentState()

        rl_step = self._rl_algorithm.predict(
            time_step._replace(observation=observation), state.rl)
        new_state = new_state._replace(rl=rl_step.state)

        return PolicyStep(action=rl_step.action, state=new_state, info=())

    def greedy_predict(self, time_step: ActionTimeStep, state=None, eps=0.1):
        observation = self._encode(time_step)

        new_state = AgentState()

        rl_step = self._rl_algorithm.greedy_predict(
            time_step._replace(observation=observation), state.rl)
        new_state = new_state._replace(rl=rl_step.state)

        return PolicyStep(action=rl_step.action, state=new_state, info=())

    def rollout(self, time_step: ActionTimeStep, state: AgentState):
        """Rollout for one step."""
        new_state = AgentState()
        info = AgentInfo()
        observation = self._encode(time_step)
        if self._icm is not None:
            icm_step = self._icm.train_step(
                (observation, time_step.prev_action), state=state.icm)
            info = info._replace(icm=icm_step.info)
            new_state = new_state._replace(icm=icm_step.state)

        rl_step = self._rl_algorithm.rollout(
            time_step._replace(observation=observation), state.rl)

        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        # TODO
        # avoid computing this when rollout (off policy train)
        if self._entropy_target_algorithm:
            et_step = self._entropy_target_algorithm.train_step(
                rl_step.action, step_type=time_step.step_type)
            info = info._replace(entropy_target=et_step.info)

        return PolicyStep(action=rl_step.action, state=new_state, info=info)

    def train_step(self, exp: Experience, state):
        new_state = AgentState()
        info = AgentInfo()
        observation = self._encode(exp)

        if self._icm is not None:
            icm_step = self._icm.train_step((observation, exp.prev_action),
                                            state=state.icm,
                                            calc_intrinsic_reward=False)
            info = info._replace(icm=icm_step.info)
            new_state = new_state._replace(icm=icm_step.state)

        rl_step = self._rl_algorithm.train_step(
            exp._replace(observation=observation), state.rl)

        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        if self._entropy_target_algorithm:
            et_step = self._entropy_target_algorithm.train_step(
                rl_step.action, step_type=exp.step_type)
            info = info._replace(entropy_target=et_step.info)

        return PolicyStep(action=rl_step.action, state=new_state, info=info)

    def calc_training_reward(self, external_reward, info: AgentInfo):
        """Calculate the reward actually used for training.

        The training_reward includes both intrinsic reward (if there's any) and
        the external reward.
        Args:
            external_reward (Tensor): reward from environment
            info (ActorCriticInfo): (batched) policy_step.info from train_step()
        Returns:
            reward used for training.
        """
        # record shaped extrinsic rewards actually used for training
        self.add_reward_summary("reward/extrinsic", external_reward)

        reward = external_reward
        if self._extrinsic_reward_coef != 1.0:
            reward *= self._extrinsic_reward_coef

        if self._icm is not None:
            self.add_reward_summary("reward/icm", info.icm.reward)
            reward += self._intrinsic_reward_coef * info.icm.reward

        if id(reward) != id(external_reward):
            self.add_reward_summary("reward/overall", reward)

        return reward

    def calc_loss(self, training_info):
        """Calculate loss."""
        if training_info.collect_info == ():
            training_info = training_info._replace(
                reward=self.calc_training_reward(training_info.reward,
                                                 training_info.info))

        def _update_loss(loss_info, training_info, name, algorithm):
            if algorithm is None:
                return loss_info
            new_loss_info = algorithm.calc_loss(
                getattr(training_info.info, name))
            return LossInfo(
                loss=add_ignore_empty(loss_info.loss, new_loss_info.loss),
                scalar_loss=add_ignore_empty(loss_info.scalar_loss,
                                             new_loss_info.scalar_loss),
                extra=loss_info.extra._replace(**{name: new_loss_info.extra}))

        rl_loss_info = self._rl_algorithm.calc_loss(
            training_info._replace(info=training_info.info.rl))
        loss_info = rl_loss_info._replace(
            extra=AgentLossInfo(rl=rl_loss_info.extra))
        loss_info = _update_loss(loss_info, training_info, 'icm', self._icm)
        loss_info = _update_loss(loss_info, training_info, 'entropy_target',
                                 self._entropy_target_algorithm)

        return loss_info

    def after_train(self, training_info):
        self._rl_algorithm.after_train(
            training_info._replace(info=training_info.info.rl))

    def preprocess_experience(self, exp: Experience):
        reward = self.calc_training_reward(exp.reward, exp.info)
        return self._rl_algorithm.preprocess_experience(
            exp._replace(reward=reward, info=exp.info.rl))
