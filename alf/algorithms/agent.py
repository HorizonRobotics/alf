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

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.agent_helpers import AgentHelper
from alf.algorithms.config import TrainerConfig
from alf.algorithms.entropy_target_algorithm import EntropyTargetAlgorithm
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.algorithms.mbrl_algorithm import LatentMbrlAlgorithm
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.predictive_representation_learner import \
                            PredictiveRepresentationLearner
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import AlgStep, Experience
from alf.data_structures import TimeStep, namedtuple
from alf.utils import math_ops
from alf.tensor_specs import TensorSpec

AgentState = namedtuple(
    "AgentState", ["obs_trans", "rl", "irm", "goal_generator", "repr", "rw"],
    default_value=())

AgentInfo = namedtuple(
    "AgentInfo",
    ["rl", "irm", "goal_generator", "entropy_target", "repr", "rw"],
    default_value=())


@alf.configurable
class Agent(OnPolicyAlgorithm):
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
                 goal_generator=None,
                 intrinsic_reward_module=None,
                 intrinsic_reward_coef=1.0,
                 extrinsic_reward_coef=1.0,
                 enforce_entropy_target=False,
                 entropy_target_cls=None,
                 optimizer=None,
                 debug_summaries=False,
                 name="AgentAlgorithm"):
        """
        Args:
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
            reward_weight_algorithm_cls (type): The algorithm class for adjusting
                reward weights when multi-dim rewards are used. If provided, the
                the default ``reward_weights`` of ``rl_algorithm`` will be
                overwritten by this algorithm.
            representation_learner_cls (type): The algorithm class for learning
                the representation. If provided, the constructed learner will
                calculate the representation from the original observation as
                the observation for downstream algorithms such as ``rl_algorithm``.
            intrinsic_reward_module (Algorithm): an algorithm whose outputs
                is a scalar intrinsic reward.
            goal_generator (Algorithm): an algorithm with output a goal vector
            intrinsic_reward_coef (float): Coefficient for intrinsic reward
            extrinsic_reward_coef (float): Coefficient for extrinsic reward
            enforce_entropy_target (bool): If True, use ``EntropyTargetAlgorithm``
                to dynamically adjust entropy regularization so that entropy is
                not smaller than ``entropy_target`` supplied for constructing
                ``EntropyTargetAlgorithm``. If this is enabled, make sure you don't
                use ``entropy_regularization`` for loss (see ``ActorCriticLoss`` or
                ``PPOLoss``). In order to use this, The ``PolicyStep.info`` from
                ``rl_algorithm_cls.train_step()`` and ``rl_algorithm_cls.rollout()``
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
                debug_summaries=debug_summaries)
            rl_observation_spec = representation_learner.output_spec
            agent_helper.register_algorithm(representation_learner, "repr")

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
            debug_summaries=debug_summaries)
        agent_helper.register_algorithm(rl_algorithm, "rl")
        # Whether the agent is on-policy or not depends on its rl algorithm.
        self._is_on_policy = rl_algorithm.is_on_policy()

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

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            optimizer=optimizer,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name,
            **agent_helper.state_specs())

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

    def is_on_policy(self):
        return self._is_on_policy

    def predict_step(self, time_step: TimeStep, state: AgentState,
                     epsilon_greedy):
        """Predict for one step."""
        new_state = AgentState()
        observation = time_step.observation
        info = AgentInfo()

        if self._representation_learner is not None:
            repr_step = self._representation_learner.predict_step(
                time_step, state.repr)
            new_state = new_state._replace(repr=repr_step.state)
            info = info._replace(repr=repr_step.info)
            observation = repr_step.output

        if self._goal_generator is not None:
            goal_step = self._goal_generator.predict_step(
                time_step._replace(observation=observation),
                state.goal_generator, epsilon_greedy)
            new_state = new_state._replace(goal_generator=goal_step.state)
            info = info._replace(goal_generator=goal_step.info)
            observation = [observation, goal_step.output]

        rl_step = self._rl_algorithm.predict_step(
            time_step._replace(observation=observation), state.rl,
            epsilon_greedy)
        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def rollout_step(self, time_step: TimeStep, state: AgentState):
        """Rollout for one step."""
        new_state = AgentState()
        info = AgentInfo()
        observation = time_step.observation

        if self._representation_learner is not None:
            repr_step = self._representation_learner.rollout_step(
                time_step, state.repr)
            new_state = new_state._replace(repr=repr_step.state)
            info = info._replace(repr=repr_step.info)
            observation = repr_step.output

        if self._goal_generator is not None:
            goal_step = self._goal_generator.rollout_step(
                time_step._replace(observation=observation),
                state.goal_generator)
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

        if self._reward_weight_algorithm:
            rw_step = self._reward_weight_algorithm.rollout_step(
                time_step, state.rw)
            info = info._replace(rw=rw_step.info)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def train_step(self, exp: Experience, state):
        new_state = AgentState()
        info = AgentInfo()
        observation = exp.observation

        if self._representation_learner is not None:
            repr_step = self._representation_learner.train_step(
                exp._replace(rollout_info=exp.rollout_info.repr), state.repr)
            new_state = new_state._replace(repr=repr_step.state)
            info = info._replace(repr=repr_step.info)
            observation = repr_step.output

        if self._goal_generator is not None:
            goal_step = self._goal_generator.train_step(
                exp._replace(
                    observation=observation,
                    rollout_info=exp.rollout_info.goal_generator),
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
            info (ActorCriticInfo): (batched) ``policy_step.info`` from ``train_step()``
        Returns:
            reward used for training.
        """
        rewards = [(external_reward, self._extrinsic_reward_coef, "extrinsic")]
        if self._irm:
            rewards.append((info.irm.reward, self._intrinsic_reward_coef,
                            "irm"))
        if self._goal_generator and 'reward' in info.goal_generator._fields:
            rewards.append((info.goal_generator.reward, 1., "goal_generator"))

        return self._agent_helper.accumulate_algorithm_rewards(
            *zip(*rewards),
            summary_prefix="reward",
            summarize_fn=self.summarize_reward)

    def calc_loss(self, experience, train_info: AgentInfo):
        """Calculate loss."""
        if experience.rollout_info is ():
            experience = experience._replace(
                reward=self.calc_training_reward(experience.reward,
                                                 train_info))
        algorithms = [
            self._representation_learner, self._rl_algorithm, self._irm,
            self._goal_generator, self._entropy_target_algorithm
        ]
        algorithms = list(filter(lambda a: a is not None, algorithms))
        return self._agent_helper.accumulate_loss_info(algorithms, experience,
                                                       train_info)

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

    def after_train_iter(self, experience, train_info: AgentInfo = None):
        """Call ``after_train_iter()`` of the RL algorithm and goal generator,
        respectively.
        """
        algorithms = [
            self._rl_algorithm, self._representation_learner,
            self._goal_generator, self._reward_weight_algorithm
        ]
        algorithms = list(filter(lambda a: a is not None, algorithms))
        self._agent_helper.after_train_iter(algorithms, experience, train_info)

        if self._reward_weight_algorithm:
            self._rl_algorithm.set_reward_weights(
                self._reward_weight_algorithm.reward_weights)

    def preprocess_experience(self, exp: Experience):
        """Add intrinsic rewards to extrinsic rewards if there is an intrinsic
        reward module. Also call ``preprocess_experience()`` of the rl
        algorithm.
        """
        reward = self.calc_training_reward(exp.reward, exp.rollout_info)
        exp = exp._replace(reward=reward)

        if self._representation_learner:
            new_exp = self._representation_learner.preprocess_experience(
                exp._replace(
                    rollout_info=exp.rollout_info.repr,
                    rollout_info_field=exp.rollout_info_field + '.repr'))
            exp = new_exp._replace(
                rollout_info=exp.rollout_info._replace(
                    repr=new_exp.rollout_info))

        new_exp = self._rl_algorithm.preprocess_experience(
            exp._replace(
                rollout_info=exp.rollout_info.rl,
                rollout_info_field=exp.rollout_info_field + '.rl'))
        return new_exp._replace(
            rollout_info=exp.rollout_info._replace(rl=new_exp.rollout_info),
            rollout_info_field=exp.rollout_info_field)

    def summarize_rollout(self, experience):
        """First call ``RLAlgorithm.summarize_rollout()`` to summarize basic
        rollout statisics. If the rl algorithm has overridden this function,
        then also call its customized version.
        """
        super(Agent, self).summarize_rollout(experience)
        if (super(Agent, self).summarize_rollout.__func__ !=
                self._rl_algorithm.summarize_rollout.__func__):
            self._rl_algorithm.summarize_rollout(
                experience._replace(rollout_info=experience.rollout_info.rl))
