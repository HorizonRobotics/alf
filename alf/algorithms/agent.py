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

import copy
from typing import Callable

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.agent_helpers import AgentHelper
from alf.algorithms.config import TrainerConfig
from alf.algorithms.entropy_target_algorithm import (
    EntropyTargetAlgorithm, NestedEntropyTargetAlgorithm)
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.algorithms.mbrl_algorithm import LatentMbrlAlgorithm
from alf.algorithms.predictive_representation_learner import \
                            PredictiveRepresentationLearner
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import AlgStep, Experience
from alf.data_structures import TimeStep, namedtuple
from alf.tensor_specs import TensorSpec

AgentState = namedtuple(
    "AgentState", ["rl", "irm", "goal_generator", "repr", "rw"],
    default_value=())

AgentInfo = namedtuple(
    "AgentInfo",
    ["rl", "irm", "goal_generator", "entropy_target", "repr", "rw", "rewards"],
    default_value=())


@alf.configurable
class Agent(RLAlgorithm):
    """Agent is a master algorithm that integrates different algorithms together.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=TensorSpec(()),
                 env=None,
                 config: TrainerConfig = None,
                 rl_algorithm_cls=ActorCriticAlgorithm,
                 reward_weight_algorithm_cls=None,
                 representation_learner_cls=None,
                 representation_use_rl_state: bool = False,
                 goal_generator=None,
                 intrinsic_reward_module=None,
                 intrinsic_reward_coef=1.0,
                 extrinsic_reward_coef=1.0,
                 enforce_entropy_target=False,
                 entropy_target_cls=None,
                 optimizer=None,
                 debug_summaries=False,
                 name="AgentAlgorithm"):
        """Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple
                simulations simultaneously. Running multiple environments in
                parallel is crucial to on-policy algorithms as it increases the
                diversity of data and decreases temporal correlation. ``env`` only
                needs to be provided to the root ``Algorithm``.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            rl_algorithm_cls (type): The algorithm class for learning the policy.
                It will be called as ``rl_algorithm_cls(observation_spec=?,
                action_spec=?, reward_spec=?, config=?, debug_summaries=?)``.
            reward_weight_algorithm_cls (type): The algorithm class for adjusting
                reward weights when multi-dim rewards are used. If provided, the
                the default ``reward_weights`` of ``rl_algorithm`` will be
                overwritten by this algorithm.
            representation_learner_cls (type): The algorithm class for learning
                the representation. If provided, the constructed learner will
                calculate the representation from the original observation as
                the observation for downstream algorithms such as
                ``rl_algorithm``. Similar to rl_algorithm_cls, it will be called
                as ``rl_algorithm_cls(observation_spec=?, action_spec=?,
                reward_spec=?, config=?, debug_summaries=?)``.
            representation_use_rl_state: When set to True, representation learner
                will receive (previous) state from the RL algorithm as input instead
                of its own state for ``rollout_step()`` and ``predict_step()``. This
                is particularly useful for algorithm such as MuZero representation
                learner, whose reanalyze component requires access to the RL
                algorithm's state.
            intrinsic_reward_module (Algorithm): an algorithm whose outputs
                is a scalar intrinsic reward.
            goal_generator (Algorithm): an algorithm which outputs a tuple of goal
                vector and a reward. The reward can be ``()`` if no reward is given.
            intrinsic_reward_coef (float): Coefficient for intrinsic reward
            extrinsic_reward_coef (float): Coefficient for extrinsic reward
            enforce_entropy_target (bool): If True, use ``(Nested)EntropyTargetAlgorithm``
                to dynamically adjust entropy regularization so that entropy is
                not smaller than ``entropy_target`` supplied for constructing
                ``(Nested)EntropyTargetAlgorithm``. If this is enabled, make sure you don't
                use ``entropy_regularization`` for loss (see ``ActorCriticLoss`` or
                ``PPOLoss``). In order to use this, The ``AlgStep.info`` from
                ``rl_algorithm_cls.train_step()`` and ``rl_algorithm_cls.rollout_step()``
                needs to contain ``action_distribution``.
            entropy_target_cls (type): If provided, will be used to dynamically
                adjust entropy regularization.
            optimizer (optimizer): The optimizer for training
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.

        """
        agent_helper = AgentHelper(AgentState)

        rl_observation_spec = observation_spec

        ## 0. representation learner
        representation_learner = None
        if representation_learner_cls is not None:
            representation_learner = representation_learner_cls(
                observation_spec=rl_observation_spec,
                action_spec=action_spec,
                reward_spec=reward_spec,
                config=config,
                debug_summaries=debug_summaries)
            assert hasattr(representation_learner, 'output_spec'), (
                "representation_learner must have output_spec")
            rl_observation_spec = representation_learner.output_spec
            agent_helper.register_algorithm(representation_learner, "repr")
        self._representation_use_rl_state = representation_use_rl_state

        ## 1. goal generator
        if goal_generator is not None:
            agent_helper.register_algorithm(goal_generator, "goal_generator")
            rl_observation_spec = [
                rl_observation_spec, goal_generator.action_spec
            ]

        ## 2. rl algorithm
        rl_algorithm = rl_algorithm_cls(
            observation_spec=rl_observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            config=config,
            debug_summaries=debug_summaries)
        agent_helper.register_algorithm(rl_algorithm, "rl")

        if isinstance(rl_algorithm, LatentMbrlAlgorithm):
            assert isinstance(representation_learner,
                              PredictiveRepresentationLearner), (
                                  "need to use "
                                  "PredictiveRepresentationLearner")
            rl_algorithm.set_latent_predictive_representation_module(
                representation_learner)

        ## 3. intrinsic motivation module
        if intrinsic_reward_module is not None:
            agent_helper.register_algorithm(intrinsic_reward_module, "irm")

        ## 4. entropy target
        entropy_target_algorithm = None
        if entropy_target_cls or enforce_entropy_target:
            if entropy_target_cls is None:
                if alf.nest.is_nested(action_spec):
                    entropy_target_cls = NestedEntropyTargetAlgorithm
                else:
                    entropy_target_cls = EntropyTargetAlgorithm
            entropy_target_algorithm = entropy_target_cls(
                action_spec, debug_summaries=debug_summaries)
            agent_helper.register_algorithm(entropy_target_algorithm,
                                            "entropy_target")

        # 5. reward weight algorithm
        reward_weight_algorithm = None
        if reward_weight_algorithm_cls is not None:
            reward_weight_algorithm = reward_weight_algorithm_cls(
                reward_spec=reward_spec, debug_summaries=debug_summaries)
            agent_helper.register_algorithm(reward_weight_algorithm, "rw")
            # Initialize the reward weights of the rl algorithm
            rl_algorithm.set_reward_weights(
                reward_weight_algorithm.reward_weights)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            optimizer=optimizer,
            is_on_policy=rl_algorithm.on_policy,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name,
            **agent_helper.state_specs())

        for alg in (representation_learner, goal_generator,
                    intrinsic_reward_module, entropy_target_algorithm,
                    reward_weight_algorithm):
            if alg is not None:
                alg.set_on_policy(self.on_policy)
        self._representation_learner = representation_learner
        self._rl_algorithm = rl_algorithm
        self._reward_weight_algorithm = reward_weight_algorithm
        self._entropy_target_algorithm = entropy_target_algorithm
        self._intrinsic_reward_coef = intrinsic_reward_coef
        self._extrinsic_reward_coef = extrinsic_reward_coef
        self._irm = intrinsic_reward_module
        self._goal_generator = goal_generator
        self._agent_helper = agent_helper
        # Set ``use_rollout_state``` for all submodules using the setter.
        # Need to make sure that no submodules use ``self._use_rollout_state``
        # before this line.
        self.use_rollout_state = self.use_rollout_state

    def set_path(self, path):
        super().set_path(path)
        self._agent_helper.set_path(path)

    def predict_step(self, time_step: TimeStep, state: AgentState):
        """Predict for one step."""
        new_state = AgentState()
        observation = time_step.observation
        info = AgentInfo()

        if self._representation_learner is not None:
            input_state = state.rl if self._representation_use_rl_state else state.repr
            repr_step = self._representation_learner.predict_step(
                time_step, input_state)
            new_state = new_state._replace(repr=repr_step.state)
            info = info._replace(repr=repr_step.info)
            observation = repr_step.output

        if self._goal_generator is not None:
            goal_step = self._goal_generator.predict_step(
                time_step._replace(observation=observation),
                state.goal_generator)
            goal, goal_reward = goal_step.output
            new_state = new_state._replace(goal_generator=goal_step.state)
            info = info._replace(goal_generator=goal_step.info)
            observation = [observation, goal]

        rl_step = self._rl_algorithm.predict_step(
            time_step._replace(observation=observation), state.rl)
        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def rollout_step(self, time_step: TimeStep, state: AgentState):
        """Rollout for one step."""
        new_state = AgentState()
        info = AgentInfo()
        observation = time_step.observation

        if self._representation_learner is not None:
            input_state = state.rl if self._representation_use_rl_state else state.repr
            repr_step = self._representation_learner.rollout_step(
                time_step, input_state)
            new_state = new_state._replace(repr=repr_step.state)
            info = info._replace(repr=repr_step.info)
            observation = repr_step.output

        rewards = {}

        if self._goal_generator is not None:
            goal_step = self._goal_generator.rollout_step(
                time_step._replace(observation=observation),
                state.goal_generator)
            new_state = new_state._replace(goal_generator=goal_step.state)
            info = info._replace(goal_generator=goal_step.info)
            goal, goal_reward = goal_step.output
            observation = [observation, goal]
            if goal_reward != ():
                rewards['goal_generator'] = goal_reward

        if self._irm is not None:
            irm_step = self._irm.rollout_step(
                time_step._replace(observation=observation), state=state.irm)
            info = info._replace(irm=irm_step.info)
            new_state = new_state._replace(irm=irm_step.state)
            rewards['irm'] = irm_step.output

        if rewards:
            info = info._replace(rewards=rewards)
            overall_reward = self._calc_overall_reward(time_step.reward,
                                                       rewards)
        else:
            overall_reward = time_step.reward

        rl_time_step = time_step._replace(
            observation=observation, reward=overall_reward)
        rl_step = self._rl_algorithm.rollout_step(rl_time_step, state.rl)
        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        if self._entropy_target_algorithm:
            assert 'action_distribution' in rl_step.info._fields, (
                "AlgStep from rl_algorithm.rollout() does not contain "
                "`action_distribution`, which is required by "
                "`enforce_entropy_target`")
            et_step = self._entropy_target_algorithm.rollout_step(
                (rl_step.info.action_distribution, time_step.step_type))
            info = info._replace(entropy_target=et_step.info)

        if self._reward_weight_algorithm:
            rw_step = self._reward_weight_algorithm.rollout_step(
                time_step, state.rw)
            new_state = new_state._replace(rw=rw_step.state)
            info = info._replace(rw=rw_step.info)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def train_step(self, time_step: TimeStep, state, rollout_info):
        new_state = AgentState()
        info = AgentInfo(rewards=rollout_info.rewards)
        observation = time_step.observation

        if self._representation_learner is not None:
            repr_step = self._representation_learner.train_step(
                time_step, state.repr, rollout_info.repr)
            new_state = new_state._replace(repr=repr_step.state)
            info = info._replace(repr=repr_step.info)
            observation = repr_step.output

        if self._goal_generator is not None:
            goal_step = self._goal_generator.train_step(
                time_step._replace(observation=observation),
                state.goal_generator, rollout_info.goal_generator)
            goal, goal_reward = goal_step.output
            info = info._replace(goal_generator=goal_step.info)
            new_state = new_state._replace(goal_generator=goal_step.state)
            observation = [observation, goal]

        if self._irm is not None:
            irm_step = self._irm.train_step(
                time_step._replace(observation=observation), state=state.irm)
            info = info._replace(irm=irm_step.info)
            new_state = new_state._replace(irm=irm_step.state)

        rl_step = self._rl_algorithm.train_step(
            time_step._replace(observation=observation), state.rl,
            rollout_info.rl)

        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        if self._entropy_target_algorithm:
            assert 'action_distribution' in rl_step.info._fields, (
                "PolicyStep from rl_algorithm.train_step() does not contain "
                "`action_distribution`, which is required by "
                "`enforce_entropy_target`")
            et_step = self._entropy_target_algorithm.train_step(
                (rl_step.info.action_distribution, time_step.step_type))
            info = info._replace(entropy_target=et_step.info)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def train_step_offline(self, time_step: TimeStep, state, rollout_info,
                           pre_train):
        new_state = AgentState()
        info = AgentInfo(rewards=rollout_info.rewards)
        observation = time_step.observation

        if self._representation_learner is not None:
            repr_step = self._representation_learner.train_step_offline(
                time_step, state.repr, rollout_info.repr)
            new_state = new_state._replace(repr=repr_step.state)
            info = info._replace(repr=repr_step.info)
            observation = repr_step.output

        if self._goal_generator is not None:
            goal_step = self._goal_generator.train_step_offline(
                time_step._replace(observation=observation),
                state.goal_generator, rollout_info.goal_generator)
            goal, goal_reward = goal_step.output
            info = info._replace(goal_generator=goal_step.info)
            new_state = new_state._replace(goal_generator=goal_step.state)
            observation = [observation, goal]

        if self._irm is not None:
            irm_step = self._irm.train_step_offline(
                time_step._replace(observation=observation), state=state.irm)
            info = info._replace(irm=irm_step.info)
            new_state = new_state._replace(irm=irm_step.state)

        rl_step = self._rl_algorithm.train_step_offline(
            time_step._replace(observation=observation), state.rl,
            rollout_info.rl, pre_train)

        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        if self._entropy_target_algorithm:
            assert 'action_distribution' in rl_step.info._fields, (
                "PolicyStep from rl_algorithm.train_step() does not contain "
                "`action_distribution`, which is required by "
                "`enforce_entropy_target`")
            et_step = self._entropy_target_algorithm.train_step_offline(
                (rl_step.info.action_distribution, time_step.step_type))
            info = info._replace(entropy_target=et_step.info)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def _calc_overall_reward(self, extrinsic_reward, intrinsic_rewards):
        overall_reward = extrinsic_reward
        if self._extrinsic_reward_coef != 1:
            overall_reward *= self._extrinsic_reward_coef
        if 'irm' in intrinsic_rewards:
            overall_reward += self._intrinsic_reward_coef * intrinsic_rewards[
                'irm']
        if 'goal_generator' in intrinsic_rewards:
            overall_reward += intrinsic_rewards['goal_generator']
        return overall_reward

    def calc_loss(self, info: AgentInfo):
        """Calculate loss."""

        if info.rewards != ():
            for name, reward in info.rewards.items():
                self.summarize_reward("reward/%s" % name, reward)

        algorithms = [
            self._representation_learner, self._rl_algorithm, self._irm,
            self._goal_generator, self._entropy_target_algorithm
        ]
        algorithms = list(filter(lambda a: a is not None, algorithms))
        return self._agent_helper.accumulate_loss_info(algorithms, info)

    def calc_loss_offline(self, info, pre_train):
        """Calculate loss for the offline RL branch."""
        if info.rewards != ():
            for name, reward in info.rewards.items():
                self.summarize_reward("reward_offline/%s" % name, reward)

        algorithms = [
            self._representation_learner, self._rl_algorithm, self._irm,
            self._goal_generator, self._entropy_target_algorithm
        ]
        algorithms = list(filter(lambda a: a is not None, algorithms))
        return self._agent_helper.accumulate_loss_info(algorithms, info, True,
                                                       pre_train)

    def after_update(self, experience, train_info: AgentInfo):
        """Call ``after_update()`` of the RL algorithm and goal generator,
        respectively.
        """
        algorithms = [
            self._rl_algorithm, self._representation_learner,
            self._goal_generator
        ]
        algorithms = list(filter(lambda a: a is not None, algorithms))
        self._agent_helper.after_update(algorithms, experience, train_info)

    def after_train_iter(self, experience, info: AgentInfo):
        """Call ``after_train_iter()`` of the RL algorithm and goal generator,
        respectively.
        """
        algorithms = [
            self._rl_algorithm, self._representation_learner,
            self._goal_generator, self._reward_weight_algorithm
        ]
        algorithms = list(filter(lambda a: a is not None, algorithms))
        self._agent_helper.after_train_iter(algorithms, experience, info)

        if self._reward_weight_algorithm:
            self._rl_algorithm.set_reward_weights(
                self._reward_weight_algorithm.reward_weights)

    def preprocess_experience(self, root_inputs, rollout_info, batch_info):
        """Add intrinsic rewards to extrinsic rewards if there is an intrinsic
        reward module. Also call ``preprocess_experience()`` of the rl
        algorithm.
        """
        exp = root_inputs
        rewards = rollout_info.rewards
        if rewards != ():
            rewards = copy.copy(rewards)
            rewards['overall'] = self._calc_overall_reward(
                root_inputs.reward, rewards)
            exp = exp._replace(reward=rewards['overall'])

        if self._representation_learner:
            exp, repr_info = self._representation_learner.preprocess_experience(
                exp, rollout_info.repr, batch_info)
            rollout_info = rollout_info._replace(repr=repr_info)

        exp, rl_info = self._rl_algorithm.preprocess_experience(
            exp, rollout_info.rl, batch_info)

        # Expand discounted_return in batch_info to the correct shape, and
        # populate to rl_info.
        if hasattr(rl_info,
                   "discounted_return") and batch_info.discounted_return != ():
            discounted_return = batch_info.discounted_return.unsqueeze(
                1).expand(exp.reward.shape[:2])
            rl_info = rl_info._replace(discounted_return=discounted_return)

        return exp, rollout_info._replace(rl=rl_info)

    def summarize_rollout(self, experience):
        """First call ``RLAlgorithm.summarize_rollout()`` to summarize basic
        rollout statisics. If the rl algorithm has overridden this function,
        then also call its customized version.
        """
        super(Agent, self).summarize_rollout(experience)
        if (hasattr(self._rl_algorithm, "summarize_rollout")
                and super(Agent, self).summarize_rollout.__func__ !=
                self._rl_algorithm.summarize_rollout.__func__):
            self._rl_algorithm.summarize_rollout(
                experience._replace(rollout_info=experience.rollout_info.rl))
