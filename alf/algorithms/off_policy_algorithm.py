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

import abc
from collections import namedtuple

import tensorflow as tf

from tf_agents.trajectories.time_step import StepType

from alf.algorithms import policy_algorithm

Experience = namedtupple("Experience", [
    'step_type', 'reward', 'discount'
    'observation', 'prev_action', 'action', 'info'
])


class OffPolicyAlgorithm(policy_algorithm.PolicyAlgorithm):
    def __init__(self,
                 action_spec,
                 train_state_spec,
                 action_distribution_spec,
                 predict_state_spec=None,
                 optimizer=None,
                 gradient_clipping=None,
                 train_step_counter=None,
                 debug_summaries=False,
                 name="OffPolicyAlgorithm"):
        super().__init__(action_spec, train_state_spec,
                         action_distribution_spec, predict_state_spec,
                         optimizer, gradient_clipping, train_step_counter,
                         debug_summaries, name)

    @abc.abstractmethod
    def train_step(self, experience, state):
        pass

    def train_complete(self, tape: tf.GradientTape,
                       training_info: TrainingInfo,
                       final_time_step: ActionTimeStep, final_info):

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
