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

import math
from typing import Callable

import gin.tf
import tensorflow as tf

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.specs.distribution_spec import nested_distributions_from_specs

import alf.utils.common as common
import alf.data_structures as ds
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.drivers import policy_driver
from alf.utils import nest_utils


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
    algorithm.train_complete(tape, training_info)
    ```
    """

    def __init__(self,
                 env: TFEnvironment,
                 algorithm: OnPolicyAlgorithm,
                 observers=[],
                 metrics=[],
                 training=True,
                 epsilon_greedy=0.1,
                 train_interval=20):
        """Create an OnPolicyDriver.

        Args:
            env (TFEnvironment): A TFEnvironment
            algorithm (OnPolicyAlgorithm): The algorithm for training
            observers (list[Callable]): An optional list of observers that are
                updated after every step in the environment. Each observer is a
                callable(time_step.Trajectory).
            metrics (list[TFStepMetric]): An optional list of metrics.
            training (bool): True for training, false for evaluating
            epsilon_greedy (float):  a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for training=False
            train_interval (int):
        """
        super(OnPolicyDriver, self).__init__(
            env=env,
            algorithm=algorithm,
            observers=observers,
            metrics=metrics,
            epsilon_greedy=epsilon_greedy,
            mode=self.ON_POLICY_TRAINING if training else self.PREDICT)

        if training:
            algorithm.set_metrics(self._metrics)
            self._prepare_specs(algorithm)
            self._trainable_variables = algorithm.trainable_variables
            self._train_interval = train_interval

    def _prepare_specs(self, algorithm):
        time_step_spec = algorithm.time_step_spec
        self._training_info_spec = ds.TrainingInfo(
            action=algorithm.action_spec,
            step_type=time_step_spec.step_type,
            reward=time_step_spec.reward,
            discount=time_step_spec.discount,
            info=algorithm.rollout_info_spec,
            env_id=time_step_spec.env_id)

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
        steps_per_unroll = self._env.batch_size * self._train_interval
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

        next_time_step, policy_step, transformed_time_step = self._step(
            time_step, policy_state)
        action = tf.nest.map_structure(tf.stop_gradient, policy_step.action)

        training_info = ds.TrainingInfo(
            action=action,
            reward=transformed_time_step.reward,
            discount=transformed_time_step.discount,
            step_type=transformed_time_step.step_type,
            info=nest_utils.distributions_to_params(policy_step.info),
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

        def create_ta(s):
            return tf.TensorArray(
                dtype=s.dtype,
                size=self._train_interval,
                element_shape=tf.TensorShape([batch_size]).concatenate(
                    s.shape))

        training_info_ta = tf.nest.map_structure(
            create_ta,
            self._training_info_spec._replace(
                info=nest_utils.to_distribution_param_spec(
                    self._training_info_spec.info)))

        with tf.GradientTape(
                watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(self._trainable_variables)
            [counter, next_time_step, next_state,
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

            training_info = tf.nest.map_structure(lambda ta: ta.stack(),
                                                  training_info_ta)

            training_info = nest_utils.params_to_distributions(
                training_info, self._training_info_spec)

        loss_info, grads_and_vars = self._algorithm.train_complete(
            tape, training_info)

        del tape

        self._algorithm.summarize_train(training_info, loss_info,
                                        grads_and_vars)
        self._algorithm.summarize_metrics()

        common.get_global_counter().assign_add(1)

        return [next_time_step, next_state]
