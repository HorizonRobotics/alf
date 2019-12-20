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
"""Base class for RL algorithms."""

from abc import abstractmethod
import os
import psutil
from typing import Callable
from collections import Iterable

import gin.tf

import tensorflow as tf

from tf_agents.metrics import tf_metrics
from tf_agents.utils import eager_utils

from alf.algorithms.algorithm import Algorithm
from alf.data_structures import ActionTimeStep, Experience, make_experience, PolicyStep, StepType, TrainingInfo
from alf.experience_replayers.experience_replay import OnetimeExperienceReplayer
from alf.experience_replayers.experience_replay import SyncUniformExperienceReplayer
import alf.utils
from alf.utils import common, nest_utils
from alf.utils.common import cast_transformer


@gin.configurable
class RLAlgorithm(Algorithm):
    """Abstract base class for  RL Algorithms.

    RLAlgorithm provide basic functions and generic interface for rl algorithms.

    The key interface functions are:
    1. predict(): one step of computation of action for evaluation.
       The subclass can choose to implement predict_action_distribution() instead
       of predict(). In that case, RLAlgorithm.predict() will sample an action
       from the action distribution
    2. rollout(): one step of comutation for rollout. Besides action, it also
       needs to compute other information necessary for training.
       The subclass can choose to implement rollout_action_distribution() instead
       of rollout(). In that case, RLAlgorithm.rollout() will sample an action
       from the action distribution.
    3. train_step(): only used for off-policy training.
    4. train_complete(): Complete one training iteration based on the
       information collected from rollout() and/or train_step()

    See OnPolicyAlgorithm and OffPolicyAlgorithm for detail.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 train_state_spec,
                 predict_state_spec=None,
                 optimizer=None,
                 trainable_module_sets=None,
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 reward_shaping_fn: Callable = None,
                 observation_transformer=cast_transformer,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 summarize_action_distributions=False,
                 name="RLAlgorithm"):
        """Create a RLAlgorithm.

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            train_state_spec (nested TensorSpec): for the network state of
                `rollout()`
            action_distribution_spec (nested DistributionSpec): for the action
                distributions.
            predict_state_spec (nested TensorSpec): for the network state of
                `predict()`. If None, it's assumed to be the same as train_state_spec
            optimizer (tf.optimizers.Optimizer | list[Optimizer]): The
                optimizer(s) for training.
            reward_shaping_fn (Callable): a function that transforms extrinsic
                immediate rewards
            observation_transformer (Callable | list[Callable]): transformation(s)
                applied to `time_step.observation`
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
            summarize_grads_and_vars (bool): If True, gradient and network
                variable summaries will be written during training.
            summarize_action_distributions (bool): If True, generate summaris
                for the action distributions.
        """
        super(RLAlgorithm, self).__init__(
            train_state_spec=train_state_spec,
            predict_state_spec=predict_state_spec,
            optimizer=optimizer,
            trainable_module_sets=trainable_module_sets,
            gradient_clipping=gradient_clipping,
            clip_by_global_norm=clip_by_global_norm,
            debug_summaries=debug_summaries,
            name=name)

        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._reward_shaping_fn = reward_shaping_fn
        if isinstance(observation_transformer, Iterable):
            observation_transformers = list(observation_transformer)
        else:
            observation_transformers = [observation_transformer]
        self._observation_transformers = observation_transformers
        self._exp_observers = []
        self._proc = psutil.Process(os.getpid())
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._summarize_action_distributions = summarize_action_distributions
        self._use_rollout_state = False

        self._rollout_info_spec = None
        self._train_step_info_spec = None
        self._processed_experience_spec = None

    @property
    def use_rollout_state(self):
        return self._use_rollout_state

    @use_rollout_state.setter
    def use_rollout_state(self, flag):
        self._use_rollout_state = flag

    def need_full_rollout_state(self):
        """Whether PolicyStep.state from rollout should be full.

        If True, it means that rollout() should return the complete state
        for train_step().
        """
        return self._is_rnn and self._use_rollout_state

    @property
    def observation_spec(self):
        """Return the observation spec."""
        return self._observation_spec

    @property
    def train_step_info_spec(self):
        """The spec for the PolicyInfo.info returned from train_step()."""
        if self._train_step_info_spec is not None:
            return self._train_step_info_spec
        batch_size = 4
        processed_exp = common.zeros_from_spec(self.processed_experience_spec,
                                               batch_size)
        state = common.zeros_from_spec(self.train_state_spec, batch_size)
        policy_step = self.train_step(processed_exp, state)
        self._train_step_info_spec = common.extract_spec(policy_step.info)
        return self._train_step_info_spec

    @property
    def rollout_info_spec(self):
        """The spec for the PolicyInfo.info returned from rollout()."""
        if self._rollout_info_spec is not None:
            return self._rollout_info_spec
        batch_size = 4
        time_step = common.zeros_from_spec(self.time_step_spec, batch_size)
        state = common.zeros_from_spec(self.train_state_spec, batch_size)
        policy_step = self.rollout(
            self.transform_timestep(time_step), state,
            RLAlgorithm.PREPARE_SPEC)
        self._rollout_info_spec = common.extract_spec(policy_step.info)
        return self._rollout_info_spec

    @property
    def experience_spec(self):
        """Spec for experience."""
        policy_step_spec = PolicyStep(
            action=self.action_spec,
            state=self.train_state_spec,
            info=self.rollout_info_spec)
        exp_spec = make_experience(self.time_step_spec, policy_step_spec)
        if not self._use_rollout_state:
            exp_spec = exp_spec._replace(state=())
        return exp_spec

    @property
    def processed_experience_spec(self):
        """Spec for processed experience.

        Returns:
            Spec for the experience returned by preprocess_experience().
        """
        if self._processed_experience_spec is not None:
            return self._processed_experience_spec
        batch_size = 4
        exp = common.zeros_from_spec(self.experience_spec, batch_size)
        transformed_exp = self.transform_timestep(exp)
        processed_exp = self.preprocess_experience(transformed_exp)
        self._processed_experience_spec = self.experience_spec._replace(
            observation=common.extract_spec(processed_exp.observation),
            rollout_info=common.extract_spec(processed_exp.rollout_info))
        if not self._use_rollout_state:
            self._procesed_experience_spec = \
                self._processed_experience_spec._replace(state=())
        return self._processed_experience_spec

    @property
    def time_step_spec(self):
        """Return spec for ActionTimeStep."""
        return ActionTimeStep(
            step_type=tf.TensorSpec((), tf.int32),
            reward=tf.TensorSpec((), tf.float32),
            discount=tf.TensorSpec((), tf.float32),
            observation=self.observation_spec,
            prev_action=self.action_spec,
            env_id=tf.TensorSpec((), tf.int32))

    @property
    def action_spec(self):
        """Return the action spec."""
        return self._action_spec

    @property
    def action_distribution_spec(self):
        """Return the action distribution spec for the action distributions."""
        return self._action_distribution_spec

    @property
    def exp_observers(self):
        """Return experience observers."""
        return self._exp_observers

    def set_metrics(self, metrics=[]):
        """Set metrics.

        metrics (list[TFStepMetric]): An optional list of metrics
            len(metrics) >= 2 as required by calling "self._metrics[:2]" in training_summary()
        """
        self._metrics = metrics

    def set_summary_settings(self,
                             summarize_grads_and_vars=False,
                             summarize_action_distributions=False):
        """Set summary flags."""
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._summarize_action_distributions = summarize_action_distributions

    def add_reward_summary(self, name, rewards):
        if self._debug_summaries:
            tf.summary.histogram(name + "/value", rewards)
            tf.summary.scalar(name + "/mean", tf.reduce_mean(rewards))

    def add_experience_observer(self, observer: Callable):
        """Add an observer to receive experience.

        Args:
            observer (Callable): callable which accept Experience as argument.
        """
        self._exp_observers.append(observer)

    def set_exp_replayer(self, exp_replayer: str, num_envs):
        """Set experience replayer.

        Args:
            exp_replayer (str): type of experience replayer. One of ("one_time",
                "uniform")
            num_envs (int): the total number of environments from all batched
                environments.
        """
        if exp_replayer == "one_time":
            self._exp_replayer = OnetimeExperienceReplayer()
        elif exp_replayer == "uniform":
            exp_spec = nest_utils.to_distribution_param_spec(
                self.experience_spec)
            self._exp_replayer = SyncUniformExperienceReplayer(
                exp_spec, num_envs)
        else:
            raise ValueError("invalid experience replayer name")
        self.add_experience_observer(self._exp_replayer.observe)

    def observe(self, exp: Experience):
        """An algorithm can override to manipulate experience.

        Args:
            exp (Experience): The shapes can be either [Q, T, B, ...] or
                [B, ...], where Q is `learn_queue_cap` in `AsyncOffPolicyDriver`,
                T is the sequence length, and B is the batch size of the batched
                environment.
        """
        if not self._use_rollout_state:
            exp = exp._replace(state=())
        for observer in self._exp_observers:
            observer(exp)

    def training_summary(self, training_info, loss_info, grads_and_vars):
        """Generate summaries for training & loss info."""

        if self._summarize_grads_and_vars:
            alf.utils.summary_utils.add_variables_summaries(grads_and_vars)
            alf.utils.summary_utils.add_gradients_summaries(grads_and_vars)
        if self._debug_summaries:
            alf.utils.common.add_action_summaries(training_info.action,
                                                  self._action_spec)
            alf.utils.common.add_loss_summaries(loss_info)

        if self._summarize_action_distributions:
            alf.utils.summary_utils.summarize_action_dist(
                training_info.action_distribution, self._action_spec)
            if training_info.collect_action_distribution:
                alf.utils.summary_utils.summarize_action_dist(
                    action_distributions=training_info.
                    collect_action_distribution,
                    action_specs=self._action_spec,
                    name="collect_action_dist")

    def metric_summary(self):
        """Generate summaries for metrics `AverageEpisodeLength`, `AverageReturn`..."""

        if self._metrics:
            for metric in self._metrics:
                metric.tf_summaries(
                    train_step=common.get_global_counter(),
                    step_metrics=self._metrics[:2])

        mem = tf.py_function(
            lambda: self._proc.memory_info().rss // 1e6, [],
            tf.float32,
            name='memory_usage')
        if not tf.executing_eagerly():
            mem.set_shape(())
        tf.summary.scalar(name='memory_usage', data=mem)

    def greedy_predict(self, time_step: ActionTimeStep, state=None, eps=0.1):
        """Predict for one step of observation.

        Generate greedy action that maximizes the action probability).

        Args:
            time_step (ActionTimeStep): Current observation and other inputs
                for computing action.
            state (nested Tensor): should be consistent with predict_state_spec
            eps (float): a floating value in [0,1], representing the chance of
                action sampling instead of taking argmax. This can help prevent
                a dead loop in some deterministic environment like Breakout.

        Returns:
            policy_step (PolicyStep):
              policy_step.action is nested tf.distribution which consistent with
                `action_distribution_spec`
              policy_step.state should be consistent with `predict_state_spec`
        """

        def dist_fn(dist):
            try:
                greedy_action = tf.cond(
                    tf.less(tf.random.uniform((), 0, 1), eps), dist.sample,
                    dist.mode)
            except NotImplementedError:
                raise ValueError(
                    "Your network's distribution does not implement mode "
                    "making it incompatible with a greedy policy.")

            return greedy_action

        policy_step = self.predict_action_distribution(time_step, state)
        action = tf.nest.map_structure(dist_fn, policy_step.action)
        return policy_step._replace(action=action)

    # Subclass may override predict() to allow more efficient implementation
    def predict(self, time_step: ActionTimeStep, state=None):
        """Predict for one step of observation.

        This only used for evaluation. So it only need to perform compuations
        for generating action distribution.

        Args:
            time_step (ActionTimeStep): Current observation and other inputs
                for computing action.
            state (nested Tensor): should be consistent with predict_state_spec
        Returns:
            policy_step (PolicyStep):
              policy_step.action is nested tf.distribution which consistent with
                `action_distribution_spec`
              policy_step.state should be consistent with `predict_state_spec`
        """
        if self.__class__.predict_action_distribution != RLAlgorithm.predict_action_distribution:
            # predict_action_distribution is overridden, use it.
            policy_step = self.predict_action_distribution(time_step, state)
            action = common.sample_action_distribution(policy_step.action)
            return policy_step._replace(action=action)
        else:
            # predict_action_distribution is not overridden. Use rollout()
            policy_step = self.rollout(time_step, state)
            return policy_step._replace(info=())

    # Subclass may override predict_action_distribution()
    def predict_action_distribution(self,
                                    time_step: ActionTimeStep,
                                    state=None):
        """Predict for one step of observation.

        This only used for evaluation. So it only need to perform compuations
        for generating action distribution.

        Args:
            time_step (ActionTimeStep): Current observation and other inputs
                for computing action.
            state (nested Tensor): should be consistent with predict_state_spec
        Returns:
            policy_step (PolicyStep):
              policy_step.action is nested Distribution which consistent with
                `action_distribution_spec`
              policy_step.state should be consistent with `predict_state_spec`
        """
        policy_step = self.rollout_action_distribution(time_step, state)
        return policy_step._replace(info=())

    ON_POLICY_TRAINING = 0
    OFF_POLICY_TRAINING = 1
    ROLLOUT = 2
    PREPARE_SPEC = 3

    @abstractmethod
    def rollout(self,
                time_step: ActionTimeStep,
                state=None,
                mode=ON_POLICY_TRAINING):
        """Perform one step of rollout.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec
            mode (int): one of (ON_POLICY_TRAINING, OFF_POLICY_TRAINING, ROLLOUT).
                ON_POLICY_TRAINING: called during on-policy training
                OFF_POLICY_TRAINING: called during the training phase off-policy
                    training
                ROLLOUT: called during the rollout phase of off-policy training
                PREPARE_SPEC: called using fake data for preparing various specs.
                    rollout() should not make any side effect during this, such
                    as making changes to Variable using the provided time_step.
        Returns:
            policy_step (PolicyStep):
              action (nested Tensor): should be consistent with
                `action_distribution_spec`
              state (nested Tensor): should be consistent with `train_state_spec`
              info (nested Tensor): everything necessary for training. Note that
                ("action", "reward", "discount", "is_last") are automatically
                collected by OnPolicyDriver. So the user only need to put other
                stuff (e.g. value estimation) into `policy_step.info`
        """
        policy_step = self.rollout_action_distribution(time_step, state, mode)
        action = common.sample_action_distribution(policy_step.action)
        return policy_step._replace(action=action)

    def rollout_action_distribution(self,
                                    time_step: ActionTimeStep,
                                    state=None,
                                    mode=ON_POLICY_TRAINING):
        """Perform one step of rollout.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec
            mode (int): one of (ON_POLICY_TRAINING, OFF_POLICY_TRAINING, ROLLOUT).
        Returns:
            policy_step (PolicyStep):
              action (nested Distribution): should be consistent with
                `action_distribution_spec`
              state (nested Tensor): should be consistent with `train_state_spec`
              info (nested Tensor): everything necessary for training. Note that
                ("action", "reward", "discount", "is_last") are automatically
                collected by OnPolicyDriver. So the user only need to put other
                stuff (e.g. value estimation) into `policy_step.info`
        """
        raise NotImplementedError(
            "rollout_action_distribution is not implemented")

    def transform_timestep(self, time_step):
        """Transform time_step.

        `transform_timestep` is called by driver for all raw time_step got from
        the environment before passing to `predict`, 'rollout`. For off-policy
        algorithms, the replay buffer stores the raw time_step. So when
        experiences are retrieved from the replay buffer, they are tranformed by
        `transform_timestep` in OffPolicyDriver before passing to `train_step`.

        It includes tranforming observation and reward and should be stateless.

        Args:
            time_step (ActionTimeStep | Experience): time step
        Returns:
            ActionTimeStep | Experience: transformed time step
        """
        if self._reward_shaping_fn is not None:
            time_step = time_step._replace(
                reward=self._reward_shaping_fn(time_step.reward))
        if self._observation_transformers is not None:
            for observation_transformer in self._observation_transformers:
                time_step = time_step._replace(
                    observation=observation_transformer(time_step.observation))
        return time_step

    def preprocess_experience(self, experience: Experience):
        """Preprocess experience.

        preprocess_experience is called for the experiences got from replay
        buffer. An example is to calculate advantages and returns in PPOAlgorithm.

        The shapes of tensors in experience are assumed to be (B, T, ...)

        Args:
            experience (Experience): original experience
        Returns:
            processed experience
        """
        return experience

    @abstractmethod
    def train_step(self, experience: Experience, state):
        """Perform one step of training computation.

        Args:
            experience (Experience):
            state (nested Tensor): should be consistent with train_state_spec

        Returns (PolicyStep):
            action (nested tf.distribution): should be consistent with
                `action_distribution_spec`
            state (nested Tensor): should be consistent with `train_state_spec`
            info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OffPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        pass

    # Subclass may override train_complete() to allow customized training
    def train_complete(self,
                       tape: tf.GradientTape,
                       training_info: TrainingInfo,
                       weight=1.0):
        """Complete one iteration of training.

        `train_complete` should calculate gradients and update parameters using
        those gradients.

        Args:
            tape (tf.GradientTape): the tape which are used for calculating
                gradient. All the previous `train_interval` `train_step()` for
                are called under the context of this tape.
            training_info (TrainingInfo): information collected for training.
                training_info.info are the batched from each policy_step.info
                returned by train_step()
            weight (float): weight for this batch. Loss will be multiplied with
                this weight before calculating gradient
        Returns:
            a tuple of the following:
            loss_info (LossInfo): loss information
            grads_and_vars (list[tuple]): list of gradient and variable tuples
        """
        valid_masks = tf.cast(
            tf.not_equal(training_info.step_type, StepType.LAST), tf.float32)

        return super().train_complete(tape, training_info, valid_masks, weight)

    @abstractmethod
    def calc_loss(self, training_info: TrainingInfo):
        """Calculate the loss for each step.

        `calc_loss()` does not need to mask out the loss at invalid steps as
        train_complete() will apply the mask automatically.

        Args:
            training_info (TrainingInfo): information collected for training.
                training_info.info are the batched from each policy_step.info
                returned by train_step(). Note that training_info.next_discount
                is 0 if the next step is the last step in an episode.

        Returns (LossInfo):
            loss at each time step for each sample in the batch. The shapes of
            the tensors in loss_info should be (T, B)
        """
        pass
