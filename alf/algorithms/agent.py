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

import torch

from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.agent_helpers import AgentStateSpecs
from alf.algorithms.agent_helpers import accumulate_algortihm_rewards
from alf.algorithms.agent_helpers import accumulate_loss_info, after_update
from alf.algorithms.config import TrainerConfig
from alf.algorithms.entropy_target_algorithm import EntropyTargetAlgorithm
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import AlgStep, Experience
from alf.data_structures import TimeStep, TrainingInfo, namedtuple
from alf.utils.common import cast_transformer

AgentState = namedtuple(
    "AgentState", ["rl", "irm", "goal_generator"], default_value=())

AgentInfo = namedtuple(
    "AgentInfo", ["rl", "irm", "goal_generator", "entropy_target"],
    default_value=())


@gin.configurable
class Agent(OnPolicyAlgorithm):
    """Agent

    Agent is a master algorithm that integrates different algorithms together.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 env=None,
                 config: TrainerConfig = None,
                 goal_generator=None,
                 rl_algorithm_cls=ActorCriticAlgorithm,
                 intrinsic_reward_module=None,
                 intrinsic_reward_coef=1.0,
                 extrinsic_reward_coef=1.0,
                 enforce_entropy_target=False,
                 entropy_target_cls=None,
                 optimizer=None,
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 reward_shaping_fn: Callable = None,
                 observation_transformer=cast_transformer,
                 debug_summaries=False,
                 name="AgentAlgorithm"):
        """Create an Agent

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            env (Environment): The environment to interact with. `env` is a
                batched environment, which means that it runs multiple
                simulations simultaneously. Running multiple environments in
                parallel is crucial to on-policy algorithms as it increases the
                diversity of data and decreases temporal correlation. `env` only
                needs to be provided to the root `Algorithm`.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs `train_iter()` by
                itself.
            rl_algorithm_cls (type): The algorithm class for learning the policy.
            intrinsic_reward_module (Algorithm): an algorithm whose outputs
                is a scalar intrinsic reward.
            goal_generator (Algorithm): an algorithm with output a goal vector
            intrinsic_reward_coef (float): Coefficient for intrinsic reward
            extrinsic_reward_coef (float): Coefficient for extrinsic reward
            enforce_entropy_target (bool): If True, use EntropyTargetAlgorithm
                to dynamically adjust entropy regularization so that entropy is
                not smaller than `entropy_target` supplied for constructing
                EntropyTargetAlgorithm. If this is enabled, make sure you don't
                use entropy_regularization for loss (see ActorCriticLoss or
                PPOLoss). In order to use this, The PolicyStep.info from
                rl_algorithm_cls.train_step() and rl_algorithm_cls.rollout()
                needs to contain `action_distribution`.
            entropy_target_cls (type): If provided, will be used to dynamically
                adjust entropy regularization.
            optimizer (tf.optimizers.Optimizer): The optimizer for training
            gradient_clipping (float): If not None, serve as a positive threshold
                for clipping gradient norms
            clip_by_global_norm (bool): If True, use `tensor_utils.clip_by_global_norm`
                to clip gradients. If False, use `tensor_utils.clip_by_norm` for
                each grad.
            reward_shaping_fn (Callable): a function that transforms extrinsic
                immediate rewards
            observation_transformer (Callable | list[Callable]): transformation(s)
                applied to `time_step.observation`
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
            """
        state_specs = AgentStateSpecs(AgentState)

        ## 1. goal generator
        rl_observation_spec = observation_spec
        if goal_generator is not None:
            state_specs.collect_algorithm_state_specs(goal_generator,
                                                      "goal_generator")
            rl_observation_spec = [
                observation_spec, goal_generator.action_spec
            ]

        ## 2. rl algorithm
        rl_algorithm = rl_algorithm_cls(
            observation_spec=rl_observation_spec,
            action_spec=action_spec,
            debug_summaries=debug_summaries)
        state_specs.collect_algorithm_state_specs(rl_algorithm, "rl")
        # Whether the agent is on-policy or not depends on its rl algorithm.
        self._is_on_policy = rl_algorithm.is_on_policy()

        ## 3. intrinsic motivation module
        if intrinsic_reward_module is not None:
            state_specs.collect_algorithm_state_specs(intrinsic_reward_module,
                                                      "irm")

        ## 4. entropy target
        entropy_target_algorithm = None
        if entropy_target_cls or enforce_entropy_target:
            if entropy_target_cls is None:
                entropy_target_cls = EntropyTargetAlgorithm
            entropy_target_algorithm = entropy_target_cls(
                action_spec, debug_summaries=debug_summaries)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            optimizer=optimizer,
            env=env,
            config=config,
            gradient_clipping=gradient_clipping,
            clip_by_global_norm=clip_by_global_norm,
            reward_shaping_fn=reward_shaping_fn,
            observation_transformer=observation_transformer,
            debug_summaries=debug_summaries,
            name=name,
            **vars(state_specs))

        self._rl_algorithm = rl_algorithm
        self._entropy_target_algorithm = entropy_target_algorithm
        self._intrinsic_reward_coef = intrinsic_reward_coef
        self._extrinsic_reward_coef = extrinsic_reward_coef
        self._irm = intrinsic_reward_module
        self._goal_generator = goal_generator

    def is_on_policy(self):
        return self._is_on_policy

    def predict_step(self, time_step: TimeStep, state: AgentState,
                     epsilon_greedy):
        """Predict for one step."""
        new_state = AgentState()
        observation = time_step.observation
        if self._goal_generator is not None:
            goal_step = self._goal_generator.predict_step(
                time_step, state.goal_generator, epsilon_greedy)
            new_state = new_state._replace(goal_generator=goal_step.state)
            observation = [observation, goal_step.output]

        rl_step = self._rl_algorithm.predict_step(
            time_step._replace(observation=observation), state.rl,
            epsilon_greedy)
        new_state = new_state._replace(rl=rl_step.state)

        return AlgStep(output=rl_step.output, state=new_state, info=())

    def rollout_step(self, time_step: TimeStep, state: AgentState):
        """Rollout for one step."""
        new_state = AgentState()
        info = AgentInfo()
        observation = time_step.observation

        if self._goal_generator is not None:
            goal_step = self._goal_generator.rollout_step(
                time_step, state.goal_generator)
            new_state = new_state._replace(goal_generator=goal_step.state)
            info = info._replace(goal_generator=goal_step.info)
            observation = [observation, goal_step.output]

        rl_step = self._rl_algorithm.rollout_step(
            time_step._replace(observation=observation), state.rl)
        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        if self._irm is not None:
            irm_step = self._irm.rollout_step(
                time_step._replace(observation=observation), state=state.irm)
            info = info._replace(irm=irm_step.info)
            new_state = new_state._replace(irm=irm_step.state)

        if self._entropy_target_algorithm:
            assert 'action_distribution' in rl_step.info._fields, (
                "AlgStep from rl_algorithm.rollout() does not contain "
                "`action_distribution`, which is required by "
                "`enforce_entropy_target`")
            et_step = self._entropy_target_algorithm.rollout_step(
                rl_step.info.action_distribution,
                step_type=time_step.step_type,
                on_policy_training=self.is_on_policy())
            info = info._replace(entropy_target=et_step.info)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def train_step(self, exp: Experience, state):
        new_state = AgentState()
        info = AgentInfo()
        observation = exp.observation

        if self._goal_generator is not None:
            goal_step = self._goal_generator.train_step(
                exp._replace(rollout_info=exp.rollout_info.goal_generator),
                state.goal_generator)
            info = info._replace(goal_generator=goal_step.info)
            new_state = new_state._replace(goal_generator=goal_step.state)
            observation = [observation, goal_step.output]

        if self._irm is not None:
            irm_step = self._irm.train_step(
                exp._replace(observation=observation), state=state.irm)
            info = info._replace(irm=irm_step.info)
            new_state = new_state._replace(irm=irm_step.state)

        rl_step = self._rl_algorithm.train_step(
            exp._replace(
                observation=observation, rollout_info=exp.rollout_info.rl),
            state.rl)

        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        if self._entropy_target_algorithm:
            assert 'action_distribution' in rl_step.info._fields, (
                "PolicyStep from rl_algorithm.train_step() does not contain "
                "`action_distribution`, which is required by "
                "`enforce_entropy_target`")
            et_step = self._entropy_target_algorithm.train_step(
                rl_step.info.action_distribution, step_type=exp.step_type)
            info = info._replace(entropy_target=et_step.info)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

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
        rewards = [(external_reward, self._extrinsic_reward_coef, "extrinsic")]
        if self._irm:
            rewards.append((info.irm.reward, self._intrinsic_reward_coef,
                            "irm"))
        if self._goal_generator and 'reward' in info.goal_generator._fields:
            rewards.append((info.goal_generator.reward, 1., "goal_generator"))

        return accumulate_algortihm_rewards(
            *zip(*rewards),
            summary_prefix="reward",
            summarize_fn=self.summarize_reward)

    def calc_loss(self, training_info):
        """Calculate loss."""
        if training_info.rollout_info == ():
            training_info = training_info._replace(
                reward=self.calc_training_reward(training_info.reward,
                                                 training_info.info))
        # (algorithm, name)
        algorithms = [(self._rl_algorithm, 'rl')]
        if self._irm:
            algorithms.append((self._irm, 'irm'))
        if self._goal_generator:
            algorithms.append((self._goal_generator, 'goal_generator'))
        if self._entropy_target_algorithm:
            algorithms.append((self._entropy_target_algorithm,
                               'entropy_target'))
        return accumulate_loss_info(*zip(*algorithms), training_info)

    def after_update(self, training_info):
        algorithms = [(self._rl_algorithm, "rl")]
        if self._goal_generator:
            algorithms.append((self._goal_generator, "goal_generator"))
        after_update(*zip(*algorithms), training_info)

    def preprocess_experience(self, exp: Experience):
        reward = self.calc_training_reward(exp.reward, exp.rollout_info)
        new_exp = self._rl_algorithm.preprocess_experience(
            exp._replace(reward=reward, rollout_info=exp.rollout_info.rl))
        return new_exp._replace(
            rollout_info=exp.rollout_info._replace(rl=new_exp.rollout_info))
