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

from abc import abstractmethod

import tensorflow as tf
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from alf.policies.policy_training_info import TrainingInfo


class OnPolicyAlgorithm(object):
    """
    OnPolicyAlgorithm works with alf.policies.TrainingPolicy to do training
    at the time of policy rollout.

    User needs to implement train_step() and train_complete().
    
    train_step() is called to generate actions for every environment step.
    It also needs to generate necessary information for training.

    train_complete() is called every train_interval steps (specified in
    TrainingPolicy). All the training information collected at each previous
    train_step() are batched and provided as arguments for train_complete().

    The following is the pseudo code to illustrate how OnPolicyAlgoirhtm is used
    by TrainingPolicy:

    ```python
    new_tape = tf.GradientTape()
    
    while training not ends:
        tape = new_tape
        training_info = []
        
        for i in range(train_interval):
            with tape:
                policy_step = train_step(time_step, policy_step.state)
            action = sample action from policy_step.action
            collect training_info into training_info
            time_step = env.step(action)
        
        new_tape = tf.GradientTape()
        with new_tape:        
            policy_step = train_step(time_step, policy_step.state)
        
        with tape:
            get batched_training_info from training_info
        
        train_complete(tape, batched_training_info, time_step, policy_step)
    ```
    """

    def __init__(self,
                 action_spec,
                 train_state_spec,
                 action_distribution_spec,
                 predict_state_spec=None):
        """Create an OnPolicyAlgorithm

        Args:
          action_spec: A nest of BoundedTensorSpec representing the actions.
          train_state_spec: nested TensorSpec for the network state of 
            `train_step()`
          action_distribution_spec: nested DistributionSpec for the action
            distributions.
          predict_state_spec: nested TensorSpec for the network state of 
            `train_step()`. If None, it's assume to be same as train_state_spec
        """

        self._action_spec = action_spec
        self._train_state_spec = train_state_spec
        if predict_state_spec is None:
            predict_state_spec = train_state_spec
        self._predict_state_spec = predict_state_spec
        self._action_distribution_spec = action_distribution_spec

    @property
    def action_spec(self):
        """Returns the action spec
        """
        return self._action_spec

    @property
    def predict_state_spec(self):
        """Returns the RNN state spec for predict()
        """
        return self._predict_state_spec

    @property
    def train_state_spec(self):
        """Returns the RNN state spec for train_step()
        """
        return self._train_state_spec

    @property
    def action_distribution_spec(self):
        """Returns the action distribution spec for the action distributions
        """
        return self._action_distribution_spec

    # Use may override predict() to allow more efficient implementation
    def predict(self, time_step: TimeStep, state=None):
        """Predict for one step of observation
        Returns:
            policy_step (PolicyStep):
              policy_step.action is nested tf.distribution which consistent with 
                action_distribution_spec
              policy_step.state should be consistent with (None | nested tf.Tensor): RNN state
        """
        policy_step = self.train_step(time_step, state)
        return policy_step._replace(info=())

    #------------- User need to implement the following functions -------
    @abstractmethod
    def train_step(self, time_step: TimeStep, state=None):
        """Perform one step of action and training computation.
        
        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
          time_step (TimeStep):
          state: nested tensors consistent train_state_spec
        Returns:
          policy_step (PolicyStep): everything necessary for training should be
            put into policy_step.info. Note that ("action_distribution",
            "action", "reward", "discount", "is_last") are automatically
            collected by TrainingPolicy. So the user only need to put other
            stuff (e.g. value estimation) into `policy_step.info`
        """
        pass

    @abstractmethod
    def train_complete(self, tape: tf.GradientTape,
                       training_info: TrainingInfo, final_time_step: TimeStep,
                       final_policy_step: PolicyStep):
        """Complte one iteration of training

        `train_complete` should calcuate gradients and update parameters using
        those gradients.

        Args:
          tape (tf.GradientTape): the tape which are used for calculating 
            gradient. All the previous `train_interval` `train_step()` for
            are called under the context of this tape.
          training_info (TrainingInfo): information collected for training.
            training_info.info are the batched from each policy_step.info
            returned by train_step()
          final_time_step (TimeStep): the additional time_step
          final_policy_step (PolicyStep): the additional policy_step evaluated
            from final_time_step. This final_policy_step is NOT calculated under
            the context of `tape`
        Returns:
          loss_info (LossInfo): loss information
          grads_and_vars (list[tuple]): list of gradient and variable tuples
        """
        pass
