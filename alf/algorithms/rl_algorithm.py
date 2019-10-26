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

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.trajectories.time_step import StepType

from alf.algorithms.algorithm import Algorithm
from alf.utils.common import ActionTimeStep, namedtuple, LossInfo, make_action_time_step
from tf_agents.utils import eager_utils
from tf_agents.metrics import tf_metrics

import alf.utils

TrainingInfo = namedtuple("TrainingInfo", [
    "action_distribution", "action", "step_type", "reward", "discount", "info",
    "collect_info", "collect_action_distribution"
])


def make_training_info(action_distribution=(),
                       action=(),
                       step_type=(),
                       reward=(),
                       discount=(),
                       info=(),
                       collect_info=(),
                       collect_action_distribution=()):
    """Create an instance of TrainingInfo."""
    return TrainingInfo(
        action_distribution=action_distribution,
        action=action,
        step_type=step_type,
        reward=reward,
        discount=discount,
        info=info,
        collect_info=collect_info,
        collect_action_distribution=collect_action_distribution)


class RLAlgorithm(Algorithm):
    """Abstract base class for  RL Algorithms.

    RLAlgorithm provide basic functions and generic interface for rl algorithms.

    The key interface functions are:
    1. predict(): one step of computation of action for evaluation.
    2. rollout(): one step of comutation for rollout. Besides action, it also
       needs to compute other information necessary for training.
    3. train_step(): only used for off-policy training.
    4. train_complete(): Complete one training iteration based on the
       information collected from rollout() and/or train_step()

    See OnPolicyAlgorithm and OffPolicyAlgorithm for detail.
    """

    def __init__(self,
                 action_spec,
                 train_state_spec,
                 action_distribution_spec,
                 predict_state_spec=None,
                 optimizer=None,
                 trainable_module_sets=None,
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 reward_shaping_fn: Callable = None,
                 observation_transformer: Callable = None,
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
                `predict()`. If None, it's assume to be same as train_state_spec
            optimizer (tf.optimizers.Optimizer | list[Optimizer]): The
                optimizer(s) for training.
            reward_shaping_fn (Callable): a function that transforms extrinsic
                immediate rewards
            observation_transformer (Callable): transformation applied to
                `time_step.observation`
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

        self._action_spec = action_spec
        self._action_distribution_spec = action_distribution_spec
        self._reward_shaping_fn = reward_shaping_fn
        self._observation_transformer = observation_transformer
        self._exp_observers = []
        self._proc = psutil.Process(os.getpid())
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._summarize_action_distributions = summarize_action_distributions
        self._use_rollout_state = False

    @property
    def use_rollout_state(self):
        return self._use_rollout_state

    @use_rollout_state.setter
    def use_rollout_state(self, flag):
        self._use_rollout_state = flag

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

        if self._metrics:
            for metric in self._metrics:
                metric.tf_summaries(step_metrics=self._metrics[:2])

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

            return tfp.distributions.Deterministic(loc=greedy_action)

        policy_step = self.predict(time_step, state)
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
        policy_step = self.rollout(time_step, state)
        return policy_step._replace(info=())

    @abstractmethod
    def rollout(self, time_step: ActionTimeStep, state=None):
        """Perform one step of rollout.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec
        Returns:
            policy_step (PolicyStep):
              action (nested tf.distribution): should be consistent with
                `action_distribution_spec`
              state (nested Tensor): should be consistent with `train_state_spec`
              info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OnPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        pass

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
        if self._observation_transformer is not None:
            time_step = time_step._replace(
                observation=self._observation_transformer(time_step.
                                                          observation))
        return time_step

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

        ret = super().train_complete(tape, training_info, valid_masks, weight)
        self.after_train()
        return ret

    def after_train(self):
        """Do things after complete one iteration of training, such as update
        target network.
        """
        pass

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
