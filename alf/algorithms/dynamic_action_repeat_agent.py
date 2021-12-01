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

import torch

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.data_transformer import RewardNormalizer
from alf.data_structures import TimeStep, Experience, namedtuple, AlgStep
from alf.data_structures import make_experience
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils.conditional_ops import conditional_update
from alf.utils import common, summary_utils

ActionRepeatState = namedtuple(
    "ActionRepeatState", [
        "rl", "action", "steps", "k", "rl_discount", "rl_reward",
        "sample_rewards", "repr"
    ],
    default_value=())


@alf.configurable
class DynamicActionRepeatAgent(OffPolicyAlgorithm):
    """Create an agent which learns a variable action repetition duration.
    At each decision step, the agent outputs both the action to repeat and
    the number of steps to repeat. These two quantities together constitute the
    action of the agent. We use SAC with mixed action type for training.

    The core idea is similar to `Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning <http://arxiv.org/abs/1702.06054>`_.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=TensorSpec(()),
                 env=None,
                 config: TrainerConfig = None,
                 K=5,
                 rl_algorithm_cls=SacAlgorithm,
                 representation_learner_cls=None,
                 reward_normalizer_ctor=None,
                 gamma=0.99,
                 optimizer=None,
                 debug_summaries=False,
                 name="DynamicActionRepeatAgent"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                only be continuous actions for now.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. ``config`` only needs to
                be provided to the algorithm which performs a training iteration
                by itself.
            K (int): the maximal repeating times for an action.
            rl_algorithm_cls (Callable): creates an RL algorithm to be augmented
                by this dynamic action repeating ability.
            representation_learner_cls (type): The algorithm class for learning
                the representation. If provided, the constructed learner will
                calculate the representation from the original observation as
                the observation for downstream algorithms such as ``rl_algorithm``.
                We assume that the representation is trained by ``rl_algorithm``.
            reward_normalizer_ctor (Callable): if not None, it must be
                ``RewardNormalizer`` and environment rewards will be normalized
                for training.
            gamma (float): the reward discount to be applied when accumulating
                ``k`` steps' rewards for a repeated action. Note that this value
                should be equal to the gamma used by the critic loss for target
                values.
            optimizer (None|Optimizer): The default optimizer for
                training. See comments above for detail.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this agent.
        """
        assert action_spec.is_continuous, (
            "Only support continuous actions for now!")

        rl_observation_spec = observation_spec

        repr_learner = None
        if representation_learner_cls is not None:
            repr_learner = representation_learner_cls(
                observation_spec=observation_spec,
                action_spec=action_spec,
                debug_summaries=debug_summaries)
            rl_observation_spec = repr_learner.output_spec

        self._rl_action_spec = (BoundedTensorSpec(
            shape=(), dtype='int64', maximum=K - 1), action_spec)
        rl = rl_algorithm_cls(
            observation_spec=rl_observation_spec,
            action_spec=self._rl_action_spec,
            debug_summaries=debug_summaries)

        self._action_spec = action_spec
        self._observation_spec = observation_spec
        self._gamma = gamma

        predict_state_spec = ActionRepeatState(
            rl=rl.predict_state_spec,
            action=action_spec,
            steps=TensorSpec(shape=(), dtype='int64'))

        rollout_state_spec = predict_state_spec._replace(
            rl=rl.rollout_state_spec,
            rl_discount=TensorSpec(()),
            rl_reward=TensorSpec(()),
            k=TensorSpec((), dtype='int64'),
            sample_rewards=TensorSpec(()))

        train_state_spec = ActionRepeatState(rl=rl.train_state_spec)

        if repr_learner is not None:
            predict_state_spec = predict_state_spec._replace(
                repr=repr_learner.predict_state_spec)
            rollout_state_spec = rollout_state_spec._replace(
                repr=repr_learner.rollout_state_spec)
            train_state_spec = train_state_spec._replace(
                repr=repr_learner.train_state_spec)

        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec,
            predict_state_spec=predict_state_spec,
            env=env,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._repr_learner = repr_learner
        self._reward_normalizer = None
        if reward_normalizer_ctor is not None:
            self._reward_normalizer = reward_normalizer_ctor(
                observation_spec=())
        self._rl = rl
        self._K = K

    def observe_for_replay(self, exp):
        # Do not observe data at every time step; customized observing
        pass

    def _should_switch_action(self, time_step: TimeStep, state):
        repeat_last_step = (state.steps == 0)
        return repeat_last_step | time_step.is_first() | time_step.is_last()

    def predict_step(self, time_step: TimeStep, state):
        switch_action = self._should_switch_action(time_step, state)

        @torch.no_grad()
        def _generate_new_action(time_step, state):
            repr_state = ()
            if self._repr_learner is not None:
                repr_step = self._repr_learner.predict_step(
                    time_step, state.repr)
                time_step = time_step._replace(observation=repr_step.output)
                repr_state = repr_step.state

            rl_step = self._rl.predict_step(time_step, state.rl)
            steps, action = rl_step.output
            return ActionRepeatState(
                action=action,
                steps=steps + 1,  # [0, K-1] -> [1, K]
                rl=rl_step.state,
                repr=repr_state)

        new_state = conditional_update(
            target=state,
            cond=switch_action,
            func=_generate_new_action,
            time_step=time_step,
            state=state)
        new_state = new_state._replace(steps=new_state.steps - 1)

        return AlgStep(
            output=new_state.action,
            state=new_state,
            # plot steps and action when rendering video
            info=dict(action=(new_state.action, new_state.steps)))

    def rollout_step(self, time_step: TimeStep, state: ActionRepeatState):
        switch_action = self._should_switch_action(time_step, state)

        # state.k is the current step index over K steps
        state = state._replace(
            rl_reward=state.rl_reward + torch.pow(
                self._gamma, state.k.to(torch.float32)) * time_step.reward,
            rl_discount=state.rl_discount * time_step.discount * self._gamma,
            k=state.k + 1)

        if self._reward_normalizer is not None:
            # The probability of a reward at step k being kept till K steps is:
            # 1/k * k/(k+1) * .. * (K-1)/K = 1/K. This provides enough randomness
            # to make the normalizer unbiased.
            state = state._replace(
                sample_rewards=torch.where((
                    torch.rand_like(state.sample_rewards) < 1. /
                    state.k.to(torch.float32)
                ), time_step.reward, state.sample_rewards))

        @torch.no_grad()
        def _generate_new_action(time_step, state):
            rl_time_step = time_step._replace(
                reward=state.rl_reward,
                # To keep consistent with other algorithms, we choose to multiply
                # discount with gamma once more in td_loss.py
                discount=state.rl_discount / self._gamma)

            observation, repr_state = rl_time_step.observation, ()
            if self._repr_learner is not None:
                repr_step = self._repr_learner.rollout_step(
                    time_step, state.repr)
                observation = repr_step.output
                repr_state = repr_step.state

            rl_step = self._rl.rollout_step(
                rl_time_step._replace(observation=observation), state.rl)
            rl_step = rl_step._replace(
                info=(rl_step.info, state.k, state.sample_rewards))
            # Store to replay buffer.
            super(DynamicActionRepeatAgent, self).observe_for_replay(
                make_experience(
                    rl_time_step._replace(
                        # Store the untransformed observation so that later it will
                        # be transformed again during training
                        observation=rl_time_step.untransformed.observation),
                    rl_step,
                    state))
            steps, action = rl_step.output
            return ActionRepeatState(
                action=action,
                steps=steps + 1,  # [0, K-1] -> [1, K]
                k=torch.zeros_like(state.k),
                repr=repr_state,
                rl=rl_step.state,
                rl_reward=torch.zeros_like(state.rl_reward),
                sample_rewards=torch.zeros_like(state.sample_rewards),
                rl_discount=torch.ones_like(state.rl_discount))

        new_state = conditional_update(
            target=state,
            cond=switch_action,
            func=_generate_new_action,
            time_step=time_step,
            state=state)

        new_state = new_state._replace(steps=new_state.steps - 1)

        return AlgStep(output=new_state.action, state=new_state)

    def train_step(self, inputs: TimeStep, state: ActionRepeatState,
                   rollout_info):
        """Train the underlying RL algorithm ``self._rl``. Because in
        ``self.rollout_step()`` the replay buffer only stores info related to
        ``self._rl``, here we can directly call ``self._rl.train_step()``.

        Args:
            rl_exp (Experience): experiences that have been transformed to be
                learned by ``self._rl``.
            state (ActionRepeatState):
        """
        repr_state = ()
        if self._repr_learner is not None:
            repr_step = self._repr_learner.train_step(inputs, state.repr)
            inputs = inputs._replace(observation=repr_step.output)
            repr_state = repr_step.state

        rl_step = self._rl.train_step(inputs, state.rl, rollout_info)
        new_state = ActionRepeatState(rl=rl_step.state, repr=repr_state)
        return rl_step._replace(state=new_state)

    def calc_loss(self, info):
        """Calculate the loss for training ``self._rl``."""
        return self._rl.calc_loss(info)

    def after_update(self, root_inputs, info):
        """Call ``self._rl.after_update()``."""
        self._rl.after_update(root_inputs, info)

    def summarize_train(self, experience, train_info, loss_info, params):
        """Overwrite the function because the training action spec is
        different from the rollout action spec.
        """
        Algorithm.summarize_train(self, experience, train_info, loss_info,
                                  params)

        if self._debug_summaries:
            summary_utils.summarize_action(experience.action,
                                           self._rl_action_spec)
            self.summarize_reward("training_reward", experience.reward)

        if self._config.summarize_action_distributions:
            field = alf.nest.find_field(train_info, 'action_distribution')
            if len(field) == 1:
                summary_utils.summarize_distribution("action_dist", field[0])

    def preprocess_experience(self, root_inputs, rollout_info, batch_info):
        """Normalize training rewards if a reward normalizer is provided. Shape
        of ``rl_exp`` is ``[B, T, ...]``. The statistics of the normalizer is
        updated by random sample rewards.
        """
        reward = root_inputs.reward
        rl_info, repeats, sample_rewards = rollout_info

        if self._reward_normalizer is not None:
            normalizer = self._reward_normalizer.normalizer
            normalizer.update(sample_rewards)

            # compute current variance
            m = normalizer._mean_averager.get()
            m2 = normalizer._m2_averager.get()
            var = torch.relu(m2 - m**2)

            # compute accumulated mean over ``repeats`` steps
            acc_mean = ((1 - torch.pow(self._gamma, repeats.to(torch.float32)))
                        / (1 - self._gamma) * m)

            reward -= acc_mean
            reward = alf.layers.normalize_along_batch_dims(
                reward,
                torch.zeros_like(var),
                var,
                variance_epsilon=normalizer._variance_epsilon)

            clip = self._reward_normalizer.clip_value
            if clip > 0:
                # The clip value is for single-step rewards, so we need to multiply
                # it with the repeated steps.
                clip = clip * repeats
                reward = torch.max(torch.min(clip, reward), -clip)

        root_inputs = root_inputs._replace(reward=reward)
        return self._rl.preprocess_experience(root_inputs, rl_info, batch_info)
