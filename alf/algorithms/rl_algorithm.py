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
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import StepType
from tf_agents.utils import eager_utils

import alf.utils.common as common

TrainingInfo = namedtuple("TrainingInfo", [
    "action_distribution", "action", "step_type", "reward", "discount", "info",
    "collect_info"
])


def make_training_info(action_distribution=(),
                       action=(),
                       step_type=(),
                       reward=(),
                       discount=(),
                       info=(),
                       collect_info=()):
    """Create an instance of TrainingInfo."""
    return TrainingInfo(
        action_distribution=action_distribution,
        action=action,
        step_type=step_type,
        reward=reward,
        discount=discount,
        info=info,
        collect_info=collect_info)


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
    """The base class of RL algorithm."""

    def __init__(self,
                 action_spec,
                 train_state_spec,
                 action_distribution_spec,
                 predict_state_spec=None,
                 optimizer=None,
                 get_trainable_variables_func=None,
                 gradient_clipping=None,
                 train_step_counter=None,
                 debug_summaries=False,
                 name="RLAlgorithm"):
        """Create a RLAlgorithm.

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            train_state_spec (nested TensorSpec): for the network state of
                `train_step()`
            action_distribution_spec (nested DistributionSpec): for the action
                distributions.
            predict_state_spec (nested TensorSpec): for the network state of
                `train_step()`. If None, it's assume to be same as
                 train_state_spec
            optimizer (tf.optimizers.Optimizer | list[Optimizer]): The
                optimizer(s) for training.
            get_trainable_variables_func (Callable | list[Callable]): each one
                corresponds to one optimizer in `optimizer`. When called, it
                should return the variables for the correponding optimizer. If
                there is only one optimizer, this can be None and
                `self.trainable_variables` will be used. 
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
        if predict_state_spec is None:
            predict_state_spec = train_state_spec
        self._predict_state_spec = predict_state_spec
        self._action_distribution_spec = action_distribution_spec
        self._optimizers = common.as_list(optimizer)
        if optimizer is not None and get_trainable_variables_func is None:
            get_trainable_variables_func = lambda: self.trainable_variables
        self._get_trainable_variables_funcs = common.as_list(
            get_trainable_variables_func)
        assert (len(self._optimizers) == len(
            self._get_trainable_variables_funcs)), (
                "`optimizer` and `get_trainable_variables_func`"
                "should have same length")

        self._gradient_clipping = gradient_clipping
        self._train_step_counter = common.get_global_counter(
            train_step_counter)
        self._debug_summaries = debug_summaries
        self._cached_var_sets = None

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

    def add_reward_summary(self, name, rewards):
        if self._debug_summaries:
            step = self._train_step_counter
            tf.summary.histogram(name + "/value", rewards, step)
            tf.summary.scalar(name + "/mean", tf.reduce_mean(rewards), step)

    def greedy_predict(self, time_step: ActionTimeStep, state=None):
        """Predict for one step of observation.

        Generate greedy action that maximizes the action probablity).

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec

        Returns:
            policy_step (PolicyStep):
              policy_step.action is nested tf.distribution which consistent with 
                `action_distribution_spec`
              policy_step.state should be consistent with `predict_state_spec`
        """

        def dist_fn(dist):
            try:
                greedy_action = dist.mode()
            except NotImplementedError:
                raise ValueError(
                    "Your network's distribution does not implement mode "
                    "making it incompatible with a greedy policy.")

            return tfp.distributions.Deterministic(loc=greedy_action)

        policy_step = self.predict(time_step, state)
        action = tf.nest.map_structure(dist_fn, policy_step.action)
        return policy_step._replace(action=action)

    @abstractmethod
    def predict(self, time_step: ActionTimeStep, state=None):
        """Predict for one step of observation.

        Returns:
            policy_step (PolicyStep):
              policy_step.action is nested tf.distribution which consistent with
                `action_distribution_spec`
              policy_step.state should be consistent with `predict_state_spec`
        """
        pass

    # Subclass may override train_complete() to allow customized training
    def train_complete(self, tape: tf.GradientTape,
                       training_info: TrainingInfo,
                       final_time_step: ActionTimeStep, final_info):
        """Complete one iteration of training.

        `train_complete` should calcuate gradients and update parameters using
        those gradients.

        Args:
            tape (tf.GradientTape): the tape which are used for calculating 
                gradient. All the previous `train_interval` `train_step()` for
                are called under the context of this tape.
            training_info (TrainingInfo): information collected for training.
                training_info.info are the batched from each policy_step.info
                returned by train_step()
            final_time_step (ActionTimeStep): the additional time_step
            final_info (nested Tensor): `info` from additional
                `train_step` evaluated from final_time_step or final_experience.
                This final_info is NOT calculated under the context of `tape`

        Returns:
            a tuple of the following:
            loss_info (LossInfo): loss information
            grads_and_vars (list[tuple]): list of gradient and variable tuples
        """
        valid_masks = tf.cast(
            tf.not_equal(training_info.step_type, StepType.LAST), tf.float32)
        with tape:
            loss_info = self.calc_loss(training_info, final_time_step,
                                       final_info)
            loss_info = tf.nest.map_structure(
                lambda l: tf.reduce_mean(l * valid_masks), loss_info)

        if self._cached_var_sets is None:
            # Cache it because trainable_variables is an expensive operation
            # according to the documentation.
            self._cached_var_sets = []
            for f in self._get_trainable_variables_funcs:
                self._cached_var_sets.append(f())

        var_sets = self._cached_var_sets
        all_grads_and_vars = ()
        for vars, optimizer in zip(var_sets, self._optimizers):
            grads = tape.gradient(loss_info.loss, vars)
            grads_and_vars = tuple(zip(grads, vars))
            all_grads_and_vars = all_grads_and_vars + grads_and_vars
            if self._gradient_clipping is not None:
                grads_and_vars = eager_utils.clip_gradient_norms(
                    grads_and_vars, self._gradient_clipping)
            optimizer.apply_gradients(grads_and_vars)

        return loss_info, all_grads_and_vars

    @abstractmethod
    def calc_loss(self, training_info: TrainingInfo,
                  final_time_step: ActionTimeStep, final_info):
        """Calculate the loss for each step.

        `calc_loss()` does not need to mask out the loss at invalid steps as
        train_complete() will apply the mask automatically.

        Args:
            training_info (TrainingInfo): information collected for training.
                training_info.info are the batched from each policy_step.info
                returned by train_step(). Note that training_info.next_discount
                is 0 if the next step is the last step in an episode.
            final_time_step (ActionTimeStep): the additional time_step
            final_info (nested Tensor): `info` from additional
                `train_step` evaluated from final_time_step or final_experience.
                This final_info is NOT calculated under the context of `tape`

        Returns (LossInfo):
            loss at each time step for each sample in the batch. The shapes of
            the tensors in loss_info should be (T, B)
        """
        pass
