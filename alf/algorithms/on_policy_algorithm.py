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
"""Base class for on-policy RL algorithms."""

from abc import abstractmethod
from collections import namedtuple

import tensorflow as tf

from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import StepType
from tf_agents.utils import eager_utils
from alf.algorithms import policy_algorithm
from alf.drivers.policy_driver import ActionTimeStep

TrainingInfo = namedtuple("TrainingInfo", [
    "action_distribution", "action", "step_type", "reward", "discount", "info"
])


def make_training_info(action_distribution=None,
                       action=None,
                       step_type=None,
                       reward=None,
                       discount=None,
                       info=None,
                       collect_info=None):
    return TrainingInfo(
        action_distribution=action_distribution,
        action=action,
        step_type=step_type,
        reward=reward,
        discount=discount,
        info=info,
        collect_info=collect_info)


class OnPolicyAlgorithm(policy_algorithm.PolicyAlgorithm):
    """
    OnPolicyAlgorithm works with alf.policies.TrainingPolicy to do training
    at the time of policy rollout.

    User needs to implement train_step() and train_complete().
    
    train_step() is called to generate actions for every environment step.
    It also needs to generate necessary information for training.

    train_complete() is called every `train_interval` steps (specified in
    TrainingPolicy). All the training information collected at each previous
    train_step() are batched and provided as arguments for train_complete().

    The following is the pseudo code to illustrate how OnPolicyAlgoirhtm is used
    by TrainingPolicy:

    ```python
    tape = tf.GradientTape()
    training_info = []
    
    while training not ends:
        if len(training_info) == train_intervel:
            old_tape = tape
            tape = tf.GradientTape()
        with tape:
            policy_step = train_step(time_step, policy_step.state)
        if len(training_info) == train_intervel:
            with old_tape:
                get batched_training_info from training_info
            train_complete(tape, batched_training_info, time_step, policy_step)
            training_info = []
        action = sample action from policy_step.action
        collect necessary information and policy_step.info into training_info
        time_step = env.step(action)
    ```
    """

    def __init__(self,
                 action_spec,
                 train_state_spec,
                 action_distribution_spec,
                 predict_state_spec=None,
                 optimizer=None,
                 gradient_clipping=None,
                 train_step_counter=None,
                 debug_summaries=False,
                 name="OnPolicyAlgorithm"):
        """Create an OnPolicyAlgorithm

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            train_state_spec (nested TensorSpec): for the network state of 
                `train_step()`
            action_distribution_spec (nested DistributionSpec): for the action
                distributions.
            predict_state_spec (nested TensorSpec): for the network state of 
                `train_step()`. If None, it's assume to be same as
                 train_state_spec
            optimizer (tf.optimizers.Optimizer): The optimizer for training.
            train_step_counter (tf.Variable): An optional counter to increment
                every time the a new iteration is started. If None, it will use 
                tf.summary.experimental.get_step(). If this is still None, a
                counter will be created.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """

        super(OnPolicyAlgorithm, self).__init__(
            action_spec,
            train_state_spec,
            action_distribution_spec,
            predict_state_spec,
            optimizer,
            gradient_clipping,
            train_step_counter,
            debug_summaries,
            name=name)

    @abstractmethod
    def train_step(self, time_step: ActionTimeStep = None, state=None):
        """Perform one step of action and training computation.
        
        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistant with train_state_spec

        Returns (PolicyStep):
            info: everything necessary for training. Note that 
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by TrainingPolicy. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        pass

    # Subclass may override train_complete() to allow customized training
    def train_complete(self,
                       tape: tf.GradientTape = None,
                       training_info: TrainingInfo = None,
                       final_time_step: ActionTimeStep = None,
                       final_policy_step: PolicyStep = None):
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
            final_policy_step (PolicyStep): the additional policy_step evaluated
                from final_time_step. This final_policy_step is NOT calculated
                under the context of `tape`

        Returns:
            a tuple of the following:
            loss_info (LossInfo): loss information
            grads_and_vars (list[tuple]): list of gradient and variable tuples
            
        """
        valid_masks = tf.cast(
            tf.not_equal(training_info.step_type, StepType.LAST), tf.float32)
        with tape:
            loss_info = self.calc_loss(training_info, final_time_step,
                                       final_policy_step)
            loss_info = tf.nest.map_structure(
                lambda l: tf.reduce_mean(l * valid_masks), loss_info)

        if self._cached_vars is None:
            # Cache it because trainable_variables is an expensive operation
            # according to the documentation.
            self._cached_vars = self.trainable_variables
        vars = self._cached_vars
        grads = tape.gradient(loss_info.loss, vars)
        grads_and_vars = tuple(zip(grads, vars))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(
                grads_and_vars, self._gradient_clipping)
        self._optimizer.apply_gradients(grads_and_vars)
        return loss_info, grads_and_vars

    @abstractmethod
    def calc_loss(self, training_info: TrainingInfo,
                  final_time_step: ActionTimeStep,
                  final_policy_step: PolicyStep):
        """Calculate the loss for each step.

        `calc_loss()` does not need to mask out the loss at invalid steps as
        train_complete() will apply the mask automatically.

        Args:
            training_info (TrainingInfo): information collected for training.
                training_info.info are the batched from each policy_step.info
                returned by train_step(). Note that training_info.next_discount
                is 0 if the next step is the last step in an episode.
            final_time_step (ActionTimeStep): the additional time_step
                final_policy_step (PolicyStep): the additional policy_step
                evaluated from final_time_step. This final_policy_step is NOT
                calculated under the context of `tape`

        Returns (LossInfo):
            loss at each time step for each sample in the batch. The shapes of
            the tensors in loss_info should be (T, B)
        """
        pass
