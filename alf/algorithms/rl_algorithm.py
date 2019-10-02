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
import collections
import itertools
from typing import Callable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.trajectories.time_step import StepType
from tf_agents.utils import eager_utils

import alf.utils


def namedtuple(typename, field_names, default_value=None, default_values=()):
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (default_value, ) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


TrainingInfo = namedtuple("TrainingInfo", [
    "action_distribution", "action", "step_type", "reward", "discount", "info",
    "collect_info", "collect_action_distribution"
])

LossInfo = namedtuple(
    "LossInfo",
    [
        "loss",  # batch loss shape should be (T, B)
        "scalar_loss",  # shape is ()
        "extra"  # nested batch and/or scalar losses, for summary only
    ],
    default_value=())


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


class ActionTimeStep(
        namedtuple(
            'ActionTimeStep',
            ['step_type', 'reward', 'discount', 'observation', 'prev_action'])
):
    """TimeStep with action."""

    def is_first(self):
        if tf.is_tensor(self.step_type):
            return tf.equal(self.step_type, StepType.FIRST)
        return np.equal(self.step_type, StepType.FIRST)

    def is_mid(self):
        if tf.is_tensor(self.step_type):
            return tf.equal(self.step_type, StepType.MID)
        return np.equal(self.step_type, StepType.MID)

    def is_last(self):
        if tf.is_tensor(self.step_type):
            return tf.equal(self.step_type, StepType.LAST)
        return np.equal(self.step_type, StepType.LAST)


def make_action_time_step(time_step, prev_action):
    return ActionTimeStep(
        step_type=time_step.step_type,
        reward=time_step.reward,
        discount=time_step.discount,
        observation=time_step.observation,
        prev_action=prev_action)


class RLAlgorithm(tf.Module):
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
                 get_trainable_variables_func=None,
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 reward_shaping_fn: Callable = None,
                 observation_transformer: Callable = None,
                 train_step_counter=None,
                 debug_summaries=False,
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
            get_trainable_variables_func (Callable | list[Callable]): each one
                corresponds to one optimizer in `optimizer`. When called, it
                should return the variables for the corresponding optimizer. If
                there is only one optimizer, this can be None and
                `self.trainable_variables` will be used.
            gradient_clipping (float): If not None, serve as a positive threshold
            clip_by_global_norm (bool): If True, use tf.clip_by_global_norm to
                clip gradient. If False, use tf.clip_by_norm for each grad.
            reward_shaping_fn (Callable): a function that transforms extrinsic
                immediate rewards
            observation_transformer (Callable): transformation applied to
                `time_step.observation`
            train_step_counter (tf.Variable): An optional counter to increment
                every time the a new iteration is started. If None, it will use
                tf.summary.experimental.get_step(). If this is still None, a
                counter will be created.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """
        super(RLAlgorithm, self).__init__(name=name)

        self._action_spec = action_spec
        self._train_state_spec = train_state_spec
        self._is_rnn = len(tf.nest.flatten(train_state_spec)) > 0
        if predict_state_spec is None:
            predict_state_spec = train_state_spec
        self._predict_state_spec = predict_state_spec
        self._action_distribution_spec = action_distribution_spec
        self._optimizers = alf.utils.common.as_list(optimizer)
        if get_trainable_variables_func is None:
            get_trainable_variables_func = lambda: super(RLAlgorithm, self
                                                         ).trainable_variables
        self._get_trainable_variables_funcs = alf.utils.common.as_list(
            get_trainable_variables_func)
        if optimizer:
            assert (len(self._optimizers) == len(
                self._get_trainable_variables_funcs)), (
                    "`optimizer` and `get_trainable_variables_func`"
                    "should have same length")

        self._gradient_clipping = gradient_clipping
        self._clip_by_global_norm = clip_by_global_norm
        self._reward_shaping_fn = reward_shaping_fn
        self._observation_transformer = observation_transformer
        self._train_step_counter = alf.utils.common.get_global_counter(
            train_step_counter)
        self._debug_summaries = debug_summaries
        self._cached_var_sets = None
        self._use_rollout_state = False

    @property
    def optimizer(self):
        """Return the optimizer for this algorithm."""
        if len(self._optimizers) == 0:
            return None
        if len(self._optimizers) == 1:
            return self._optimizers[0]
        raise ValueError("There are multiple optimzers")

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
    def predict_state_spec(self):
        """Return the RNN state spec for predict()."""
        return self._predict_state_spec

    @property
    def train_state_spec(self):
        """Return the RNN state spec for train_step()."""
        return self._train_state_spec

    @property
    def trainable_variables(self):
        return list(itertools.chain(*self._get_cached_var_sets()))

    def _get_cached_var_sets(self):
        if self._cached_var_sets is None:
            # Cache it because trainable_variables is an expensive operation
            # according to the documentation.
            self._cached_var_sets = []
            for f in self._get_trainable_variables_funcs:
                self._cached_var_sets.append(f())
        return self._cached_var_sets

    def add_reward_summary(self, name, rewards):
        if self._debug_summaries:
            step = self._train_step_counter
            tf.summary.histogram(name + "/value", rewards, step)
            tf.summary.scalar(name + "/mean", tf.reduce_mean(rewards), step)

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
        Including tranforming observation and reward.

        Args:
            time_step (ActionTimeStep | Experience): time step
        Returns:
            ActionTimeStep | Experience: transformed time step
        """
        # reward shaping
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

        # record shaped extrinsic rewards actually used for training
        self.add_reward_summary("reward/extrinsic", training_info.reward)

        with tape:
            loss_info = self.calc_loss(training_info)
            loss_info = tf.nest.map_structure(
                lambda l: tf.reduce_mean(l * valid_masks)
                if len(l.shape) == 2 else l, loss_info)
            if isinstance(loss_info.scalar_loss, tf.Tensor):
                assert len(loss_info.scalar_loss.shape) == 0
                loss_info = loss_info._replace(
                    loss=loss_info.loss + loss_info.scalar_loss)
            loss = weight * loss_info.loss

        var_sets = self._get_cached_var_sets()
        all_grads_and_vars = ()
        for i, vars, optimizer in zip(
                range(len(var_sets)), var_sets, self._optimizers):
            grads = tape.gradient(loss, vars)
            grads_and_vars = tuple(zip(grads, vars))
            all_grads_and_vars = all_grads_and_vars + grads_and_vars
            if self._gradient_clipping is not None:
                if self._clip_by_global_norm:
                    grads, global_norm = tf.clip_by_global_norm(
                        grads, self._gradient_clipping)
                    grads_and_vars = tuple(zip(grads, vars))
                    alf.utils.common.run_if(
                        alf.utils.common.should_record_summaries(), lambda: tf.
                        summary.scalar("global_grad_norm/%s" % i, global_norm))
                else:
                    grads_and_vars = eager_utils.clip_gradient_norms(
                        grads_and_vars, self._gradient_clipping)

            optimizer.apply_gradients(grads_and_vars)

        return loss_info, all_grads_and_vars

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
