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
"""A Driver for on-policy training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from typing import Callable

import gin.tf
import tensorflow as tf

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.specs.distribution_spec import nested_distributions_from_specs

import alf.utils.common as common
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.rl_algorithm import TrainingInfo
from alf.drivers import policy_driver


@gin.configurable
class OnPolicyDriver(policy_driver.PolicyDriver):
    """Driver for on-policy training.

    OnPolicyDriver runs the environment with the algorithm for on-policy training.
    Training consists of multiple iterations. Each iteration performs the
    following computation:

    ```python
    with GradientTape as tape:
        for _ in range(train_interval):
            policy_step = algorithm.rollout(time_step, policy_step.state)
            action = sample action from policy_step.action
            collect necessary information and policy_step.info into training_info
            time_step = env.step(action)
    final_policy_step = algorithm.rollout(training_info) ***
    collect necessary information and final_policy_step.info into training_info ***
    algorithm.train_complete(tape, training_info)
    ```

    There are two modes of handling the final_policy_step from the above code:
    * FINAL_STEP_REDO: discard final_policy_step. Hence the algorithm will
        redo the policy_step for the same time_step in the next iteration.
        This requires that the algorithm can correctly generate policy_step with
        repeated train_step() call.
    * FINAL_STEP_SKIP: use final_policy_step for the next env.step(). Hence this
        environment step will be skipped for training because it's not performed
        with GradientTape() context.
    * FINAL_STEP_NO: do not include final_step for training_info (
        skip step (***) in the above pseudocode)
    """
    FINAL_STEP_REDO = 0  # redo the final step for training
    FINAL_STEP_SKIP = 1  # skip the final step for training
    FINAL_STEP_NO = 2  # do not include final step for training_info

    def __init__(self,
                 env: TFEnvironment,
                 algorithm: OnPolicyAlgorithm,
                 observers=[],
                 metrics=[],
                 training=True,
                 greedy_predict=False,
                 train_interval=20,
                 final_step_mode=FINAL_STEP_REDO):
        """Create an OnPolicyDriver.

        Args:
            env (TFEnvironment): A TFEnvironment
            algorithm (OnPolicyAlgorithm): The algorithm for training
            observers (list[Callable]): An optional list of observers that are
                updated after every step in the environment. Each observer is a
                callable(time_step.Trajectory).
            metrics (list[TFStepMetric]): An optional list of metrics.
            training (bool): True for training, false for evaluating
            greedy_predict (bool): use greedy action for evaluation (i.e.
                training==False).
            train_interval (int):
            final_step_mode (int): FINAL_STEP_REDO for redo the final step for
                training. FINAL_STEP_SKIP for skipping the final step for
                training. See the class comment for explanation.
        """
        super(OnPolicyDriver, self).__init__(
            env=env,
            algorithm=algorithm,
            observers=observers,
            metrics=metrics,
            training=training,
            greedy_predict=greedy_predict)

        if training and not algorithm.need_final_step():
            self._final_step_mode = OnPolicyDriver.FINAL_STEP_NO
        else:
            self._final_step_mode = final_step_mode

        if training:
            algorithm.set_metrics(self._metrics)
            self._prepare_specs(algorithm)
            self._trainable_variables = algorithm.trainable_variables
            self._train_interval = train_interval

    def _prepare_specs(self, algorithm):
        time_step_spec = self._env.time_step_spec()
        self._action_distribution_spec = tf.nest.map_structure(
            common.to_distribution_spec, algorithm.action_distribution_spec)
        action_distribution_param_spec = tf.nest.map_structure(
            lambda spec: spec.input_params_spec,
            self._action_distribution_spec)

        policy_step = algorithm.rollout(
            algorithm.transform_timestep(self.get_initial_time_step()),
            self._initial_state)
        info_spec = tf.nest.map_structure(
            lambda t: tf.TensorSpec(t.shape[1:], t.dtype), policy_step.info)

        self._training_info_spec = TrainingInfo(
            action_distribution=action_distribution_param_spec,
            action=self._env.action_spec(),
            step_type=time_step_spec.step_type,
            reward=time_step_spec.reward,
            discount=time_step_spec.discount,
            info=info_spec,
            env_id=tf.TensorSpec((), tf.int32))

    def _run(self, max_num_steps, time_step, policy_state):
        if self._training:
            return self.train(max_num_steps, time_step, policy_state)
        else:
            return self.predict(max_num_steps, time_step, policy_state)

    def train(self, max_num_steps, time_step, policy_state):
        """Perform on-policy training with `max_num_steps`.

        Args:
            max_num_steps (int): stops after so many environment steps. Is the
                total number of steps from all the individual environment in
                the batched environments including StepType.LAST steps.
            time_step (ActionTimeStep): optional initial time_step. If None, it
                will use self.get_initial_time_step(). Elements should be shape
                [batch_size, ...].
            policy_state (nested Tensor): optional initial state for the policy.
        Returns:
            None
        """
        steps_per_unroll = (self._env.batch_size * (self._train_interval + (
            self._final_step_mode == OnPolicyDriver.FINAL_STEP_SKIP)))
        maximum_iterations = math.ceil(max_num_steps / steps_per_unroll)
        [time_step, policy_state] = tf.while_loop(
            cond=lambda *_: True,
            body=self._iter,
            loop_vars=[time_step, policy_state],
            maximum_iterations=maximum_iterations,
            back_prop=False,
            name="")
        return time_step, policy_state, maximum_iterations * steps_per_unroll

    def _train_loop_body(self, counter, time_step, policy_state,
                         training_info_ta):

        next_time_step, policy_step, action, transformed_time_step = self._step(
            time_step, policy_state)
        action = tf.nest.map_structure(tf.stop_gradient, action)
        action_distribution_param = common.get_distribution_params(
            policy_step.action)

        training_info = TrainingInfo(
            action_distribution=action_distribution_param,
            action=action,
            reward=transformed_time_step.reward,
            discount=transformed_time_step.discount,
            step_type=transformed_time_step.step_type,
            info=policy_step.info,
            env_id=transformed_time_step.env_id)

        training_info_ta = tf.nest.map_structure(
            lambda ta, x: ta.write(counter, x), training_info_ta,
            training_info)

        counter += 1

        return [counter, next_time_step, policy_step.state, training_info_ta]

    def _iter(self, time_step, policy_state):
        """One training iteration."""
        counter = tf.zeros((), tf.int32)
        batch_size = self._env.batch_size
        ta_size = self._train_interval
        ta_size += self._final_step_mode != OnPolicyDriver.FINAL_STEP_NO

        def create_ta(s):
            return tf.TensorArray(
                dtype=s.dtype,
                size=ta_size,
                element_shape=tf.TensorShape([batch_size]).concatenate(
                    s.shape))

        training_info_ta = tf.nest.map_structure(create_ta,
                                                 self._training_info_spec)

        with tf.GradientTape(
                watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(self._trainable_variables)
            [counter, time_step, policy_state,
             training_info_ta] = tf.while_loop(
                 cond=lambda *_: True,
                 body=self._train_loop_body,
                 loop_vars=[
                     counter, time_step, policy_state, training_info_ta
                 ],
                 back_prop=True,
                 parallel_iterations=1,
                 maximum_iterations=self._train_interval,
                 name='iter_loop')

        if self._final_step_mode == OnPolicyDriver.FINAL_STEP_SKIP:
            next_time_step, policy_step, action, transformed_time_step = self._step(
                time_step, policy_state)
            next_state = policy_step.state
        elif self._final_step_mode == OnPolicyDriver.FINAL_STEP_REDO:
            transformed_time_step = self._algorithm.transform_timestep(
                time_step)
            policy_step = common.algorithm_step(
                self._algorithm.rollout, transformed_time_step, policy_state)
            action = common.sample_action_distribution(policy_step.action)
            next_time_step = time_step
            next_state = policy_state
        else:
            next_time_step = time_step
            next_state = policy_state

        if self._final_step_mode != OnPolicyDriver.FINAL_STEP_NO:
            action_distribution_param = common.get_distribution_params(
                policy_step.action)

            final_training_info = TrainingInfo(
                action_distribution=action_distribution_param,
                action=action,
                reward=transformed_time_step.reward,
                discount=transformed_time_step.discount,
                step_type=transformed_time_step.step_type,
                info=policy_step.info,
                env_id=transformed_time_step.env_id)

        with tape:
            if self._final_step_mode != OnPolicyDriver.FINAL_STEP_NO:
                training_info_ta = tf.nest.map_structure(
                    lambda ta, x: ta.write(counter, x), training_info_ta,
                    final_training_info)
            training_info = tf.nest.map_structure(lambda ta: ta.stack(),
                                                  training_info_ta)

            action_distribution = nested_distributions_from_specs(
                self._action_distribution_spec,
                training_info.action_distribution)

            training_info = training_info._replace(
                action_distribution=action_distribution)

        loss_info, grads_and_vars = self._algorithm.train_complete(
            tape, training_info)

        del tape

        self._algorithm.training_summary(training_info, loss_info,
                                         grads_and_vars)
        self._algorithm.metric_summary()

        common.get_global_counter().assign_add(1)

        return [next_time_step, next_state]
