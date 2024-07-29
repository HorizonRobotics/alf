# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Model-based RL Algorithm."""

from functools import partial

import torch
from typing import Any, Callable, Optional

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.data_structures import (AlgStep, Experience, LossInfo, namedtuple,
                                 TimeStep)
from alf.nest import nest
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils.math_ops import add_ignore_empty

from alf.algorithms.dynamics_learning_algorithm import DynamicsLearningAlgorithm
from alf.algorithms.reward_learning_algorithm import RewardEstimationAlgorithm
from alf.algorithms.planning_algorithm import PlanAlgorithm
from alf.algorithms.predictive_representation_learner import \
                                    PredictiveRepresentationLearner

MbrlState = namedtuple("MbrlState", ["dynamics", "reward", "planner"])
MbrlInfo = namedtuple(
    "MbrlInfo", ["dynamics", "reward", "planner"], default_value=())


@alf.configurable
class MbrlAlgorithm(OffPolicyAlgorithm):
    """Model-based RL algorithm
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_module: RewardEstimationAlgorithm,
                 planner_module_ctor: Callable[[Any, Any], PlanAlgorithm],
                 feature_spec: Optional[TensorSpec] = None,
                 dynamics_module_ctor: Optional[
                     Callable[[Any, Any], DynamicsLearningAlgorithm]] = None,
                 reward_spec=TensorSpec(()),
                 particles_per_replica=1,
                 epsilon_greedy=None,
                 env=None,
                 config: TrainerConfig = None,
                 dynamics_optimizer=None,
                 reward_optimizer=None,
                 planner_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 name="MbrlAlgorithm"):
        """Create an MbrlAlgorithm.
        The MbrlAlgorithm takes as input the following set of modules for
        making decisions on actions based on the current observation:
        1) learnable/fixed dynamics module
        2) learnable/fixed reward module
        3) learnable/fixed planner module

        Args:
            action_spec (BoundedTensorSpec): representing the actions.
            dynamics_module_ctor: used to construct the module for learning to
                predict the next feature based on the previous feature and
                action. It should accept input with spec [feature_spec,
                encoded_action_spec] and output a tensor of shape feature_spec.
                For discrete action, encoded_action is an one-hot representation
                of the action. For continuous action, encoded action is same as
                the original action.
            reward_module (RewardEstimationAlgorithm): module for calculating
                the reward, i.e.,  evaluating the reward for a (s, a) pair
            planner_module_ctor:: used to construct the module for generating
                planned action based on specified reward function and dynamics
                function
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            particles_per_replica (int): number of particles for each replica
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs `train_iter()` by
                itself.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.

        """
        if feature_spec is None:
            feature_spec = observation_spec
        dynamics_module = None
        if dynamics_module_ctor is not None:
            dynamics_module = dynamics_module_ctor(
                feature_spec=feature_spec, action_spec=action_spec)
        planner_module = planner_module_ctor(
            feature_spec=feature_spec, action_spec=action_spec)
        train_state_spec = MbrlState(
            dynamics=dynamics_module.train_state_spec
            if dynamics_module is not None else (),
            reward=reward_module.train_state_spec
            if reward_module is not None else (),
            planner=planner_module.train_state_spec
            if planner_module is not None else ())
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy

        super().__init__(
            feature_spec,
            action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        flat_action_spec = nest.flatten(action_spec)
        action_spec = flat_action_spec[0]

        assert action_spec.is_continuous, "only support \
                                                    continuous control"

        num_actions = action_spec.shape[-1]

        flat_feature_spec = nest.flatten(feature_spec)
        assert len(flat_feature_spec) == 1, "Mbrl doesn't support nested \
                                             feature_spec"

        self._action_spec = action_spec
        self._num_actions = num_actions

        if dynamics_optimizer is not None:
            self.add_optimizer(dynamics_optimizer, [dynamics_module])

        if planner_optimizer is not None:
            self.add_optimizer(planner_optimizer, [planner_module])

        if reward_optimizer is not None:
            self.add_optimizer(reward_optimizer, [reward_module])

        self._dynamics_module = dynamics_module
        self._reward_module = reward_module
        self._planner_module = planner_module
        self._planner_module.set_action_sequence_cost_func(
            self._predict_multi_step_cost)
        if dynamics_module is not None:
            self._num_dynamics_replicas = dynamics_module.num_replicas
        self._particles_per_replica = particles_per_replica

    def _predict_next_step(self, time_step, dynamics_state):
        """Predict the next step (observation and state) based on the current
            time step and state
        Args:
            time_step (TimeStep): input data for next step prediction
            dynamics_state: input dynamics state next step prediction
        Returns:
            next_time_step (TimeStep): updated time_step with observation
                predicted from the dynamics module
            next_dynamic_state: updated dynamics state from the dynamics module
        """
        with torch.no_grad():
            dynamics_step = self._dynamics_module.predict_step(
                time_step, dynamics_state)
            pred_obs = dynamics_step.output
            next_time_step = time_step._replace(observation=pred_obs)
            next_dynamic_state = dynamics_step.state
        return next_time_step, next_dynamic_state

    def _expand_to_population(self, data, population_size):
        """Expand the input tensor to a population of replications
        Args:
            data (Tensor): input data with shape [batch_size, ...]
        Returns:
            data_population (Tensor) with shape
                                    [batch_size * population_size, ...].
            For example data tensor [[a, b], [c, d]] and a population_size of 2,
            we have the following data_population tensor as output
                                    [[a, b], [a, b], [c, d], [c, d]]
        """
        data_population = torch.repeat_interleave(data, population_size, dim=0)
        return data_population

    def _expand_to_particles(self, inputs):
        """Expand the inputs of shape [B, ...] to [B*p, n, ...] if n > 1,
            or to [B*p, ...] if n = 1, where n is the number of replicas
            and p is the number of particles per replica.
        """
        # [B, ...] -> [B*p, ...]
        inputs = torch.repeat_interleave(
            inputs, self._particles_per_replica, dim=0)
        if self._num_dynamics_replicas > 1:
            # [B*p, ...] -> [B*p, n, ...]
            inputs = inputs.unsqueeze(1).expand(
                -1, self._num_dynamics_replicas, *inputs.shape[1:])

        return inputs

    @torch.no_grad()
    def _predict_multi_step_cost(self, observation, actions):
        """Compute the total cost by unrolling multiple steps according to
            the given initial observation and multi-step actions.
        Args:
            observation: the current observation for predicting quantities of
                future time steps
            actions (Tensor): a set of action sequences to
                shape [B, population, unroll_steps, action_dim]
        Returns:
            cost (Tensor): negation of accumulated predicted reward, with
                the shape of [B, population]
        """
        batch_size, population_size, num_unroll_steps = actions.shape[0:3]

        state = self.get_initial_predict_state(batch_size)
        time_step = TimeStep()
        dyn_state = state.dynamics._replace(feature=observation)
        dyn_state = nest.map_structure(
            partial(
                self._expand_to_population, population_size=population_size),
            dyn_state)

        # expand to particles
        dyn_state = nest.map_structure(self._expand_to_particles, dyn_state)
        reward_state = state.reward
        reward = 0
        for i in range(num_unroll_steps):
            action = actions[:, :, i, ...].view(-1, actions.shape[3])
            action = self._expand_to_particles(action)
            time_step = time_step._replace(prev_action=action)
            time_step, dyn_state = self._predict_next_step(
                time_step, dyn_state)
            next_obs = time_step.observation
            # Note: currently using (next_obs, action), might need to
            # consider (obs, action) in order to be more compatible
            # with the conventional definition of the reward function
            reward_step, reward_state = self._calc_step_reward(
                next_obs, action, reward_state)
            reward = reward + reward_step
        cost = -reward
        # reshape cost
        # [B*par, n] -> [B, par*n]
        cost = cost.reshape(
            -1, self._particles_per_replica * self._num_dynamics_replicas)
        cost = cost.mean(-1)

        # reshape cost back to [batch size, population_size]
        cost = torch.reshape(cost, [batch_size, -1])

        return cost

    def _calc_step_reward(self, obs, action, reward_state):
        """Calculate the step reward based on the given observation, action
            and state.
        Args:
            obs (Tensor): observation
            action (Tensor): action
            state: state for reward calculation
        Returns:
            reward (Tensor): compuated reward for the given input
            updated_state: updated state from the reward module
        """
        reward, reward_state = self._reward_module.compute_reward(
            obs, action, reward_state)
        return reward, reward_state

    def _predict_with_planning(self, time_step: TimeStep, state: MbrlState,
                               epsilon_greedy):

        action, planner_state = self._planner_module.predict_plan(
            time_step, state.planner, epsilon_greedy)

        dynamics_state = self._dynamics_module.update_state(
            time_step, state.dynamics)

        return AlgStep(
            output=action,
            state=state._replace(
                dynamics=dynamics_state, planner=planner_state),
            info=MbrlInfo())

    def predict_step(self, time_step: TimeStep, state):
        return self._predict_with_planning(
            time_step, state, epsilon_greedy=self._epsilon_greedy)

    def rollout_step(self, time_step: TimeStep, state):
        # note epsilon_greedy
        # 0.1 for random exploration
        return self._predict_with_planning(
            time_step, state, epsilon_greedy=0.0)

    def train_step(self, inputs: TimeStep, state: MbrlState,
                   rollout_info=None):
        dynamics_step = self._dynamics_module.train_step(
            inputs, state.dynamics)
        reward_step = self._reward_module.train_step(inputs, state.reward)
        plan_step = self._planner_module.train_step(inputs, state.planner)
        state = MbrlState(
            dynamics=dynamics_step.state,
            reward=reward_step.state,
            planner=plan_step.state)
        info = MbrlInfo(
            dynamics=dynamics_step.info,
            reward=reward_step.info,
            planner=plan_step.info)
        return AlgStep((), state, info)

    def calc_loss(self, training_info):
        loss_dynamics = self._dynamics_module.calc_loss(training_info.dynamics)
        loss = loss_dynamics.loss
        loss = add_ignore_empty(loss, training_info.reward)
        loss = add_ignore_empty(loss, training_info.planner)
        return LossInfo(loss=loss, scalar_loss=loss_dynamics.scalar_loss)

    def after_update(self, root_inputs, training_info):
        self._planner_module.after_update(
            root_inputs, training_info._replace(planner=training_info.planner))


@alf.configurable
class LatentMbrlAlgorithm(MbrlAlgorithm):
    """Model-based RL algorithm in a latent space.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 planner_module_ctor: Callable[[Any, Any], PlanAlgorithm],
                 reward_spec=TensorSpec(()),
                 env=None,
                 config: TrainerConfig = None,
                 planner_optimizer=None,
                 debug_summaries=False,
                 name="LatentMbrlAlgorithm"):
        """Create an LatentMbrlAlgorithm.
        The LatentMbrlAlgorithm takes as input a planner module for
        making decisions on actions based on the latent representation of the
        current observation as well as a latent dynamics model.

        The latent representation as well as the latent dynamics is provided by
        a latent predictive representation module, which is an instance of
        ``PredictiveRepresentationLearner``. It is set through the
        ``set_latent_predictive_representation_module()`` function. The latent
        predictive representation module should have a function
        ``predict_multi_step`` for performing multi-step imagined rollout.
        Currently it is assumed that the training of the latent representation
        module is outside of the ``LatentMbrlAlgorithm``, although the
        ``LatentMbrlAlgorithm`` can also contribute to its training by using
        the latent representation in loss calculation.

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (BoundedTensorSpec): representing the actions.
            planner_module_ctor: used to construct module for generating planned
                action based on specified reward function and dynamics function
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs `train_iter()` by
                itself.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.

        """

        super().__init__(
            observation_spec,
            feature_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            dynamics_module_ctor=None,
            reward_module=None,
            planner_module_ctor=planner_module_ctor,
            planner_optimizer=planner_optimizer,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        flat_action_spec = nest.flatten(action_spec)
        action_spec = flat_action_spec[0]

        assert action_spec.is_continuous, "only support \
                                                    continuous control"

        num_actions = action_spec.shape[-1]

        self._action_spec = action_spec
        self._num_actions = num_actions

        self._latent_pred_rep_module = None  # set it later

    def set_latent_predictive_representation_module(
            self, latent_pred_rep_module: PredictiveRepresentationLearner):
        self._latent_pred_rep_module = latent_pred_rep_module

    def _trainable_attributes_to_ignore(self):
        return ['_latent_pred_rep_module']

    @torch.no_grad()
    def _predict_multi_step_cost(self, init_rep, actions):
        """Compute the total cost by unrolling multiple steps according to
            the given initial observation and multi-step actions.
        Args:
            init_rep: the current observation for predicting quantities of
                future time steps of shape [B, d]
            actions (Tensor): a set of action sequences to
                shape [B, population, unroll_steps, action_dim]
        Returns:
            cost (Tensor): negation of accumulated predicted reward, with
                the shape of [B, population]
        """
        batch_size, population_size, num_unroll_steps = actions.shape[0:3]

        init_rep = self._expand_to_population(init_rep, population_size)

        # merge batch with population
        # [B, population, unroll_steps, ...] -> [B*population, unroll_steps, ...]
        actions = torch.reshape(actions, (-1, *actions.shape[2:]))

        pred_rewards = self._latent_pred_rep_module.predict_multi_step(
            init_rep, actions, target_field="reward")

        pred_rewards = pred_rewards.view(num_unroll_steps + 1, batch_size,
                                         population_size, -1)
        # [B, population, unroll_steps, reward_dim]
        # here we remove the predicted reward of the current step,
        # which is irrelevant to the optimization of future actions
        pred_rewards = pred_rewards[1:].permute(1, 2, 0, 3)

        # currently assume the first dimension is the overall reward
        # [B, population, unroll_steps]
        pred_rewards = pred_rewards[..., 0]
        cost = -pred_rewards
        cost = cost.sum(2)
        return cost

    def _predict_with_planning(self, time_step: TimeStep, state,
                               epsilon_greedy):
        action, planner_state = self._planner_module.predict_plan(
            time_step, state.planner, epsilon_greedy)

        return AlgStep(
            output=action,
            state=state._replace(planner=planner_state),
            info=MbrlInfo())

    def train_step(self, exp: Experience, state: MbrlState, rollout_info=None):
        # overwrite the behavior of base class ``train_step``
        return AlgStep(output=(), state=state, info=MbrlInfo())

    def calc_loss(self, training_info: MbrlInfo):
        # overwrite the behavior of base class ``calc_loss``
        return LossInfo()
