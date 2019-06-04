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
import os
import psutil
from typing import Callable

import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.drivers import driver
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.metrics import tf_metrics
from tf_agents.specs.distribution_spec import nested_distributions_from_specs
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.trajectory import from_transition
from tf_agents.utils import eager_utils

import alf.utils.common as common
from alf.algorithms.on_policy_algorithm import ActionTimeStep
from alf.algorithms.on_policy_algorithm import make_action_time_step
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.on_policy_algorithm import TrainingInfo


@gin.configurable
class OnPolicyDriver(driver.Driver):
    """Driver for on-policy training.

    OnPolicyDriver runs the eviroment with the algorithm for on-policy training.
    Training consists of multiple iterations. Each iteration performs the
    following computation:
    
    ```python
    with GradientTape as tape:
        for _ in range(train_interval):
            policy_step = algorithm.train_step(time_step, policy_step.state)
            action = sample action from policy_step.action
            collect necessary information and policy_step.info into training_info
            time_step = env.step(action)
    final_policy_step = algorithm.train_step(training_info)
    algorithm.train_complete(tape, training_info, time_step, final_policy_step)
    ```

    There are two modes of handling the final_policy_step from the above code:
    * FINAL_STEP_REDO: discard final_policy_step. Hence the algorithm will
        redo the policy_step for the same time_step in the next iteration.
        This requires that the algorithm can correctly generate policy_step with
        repeated train_step() call.
    * FINAL_STEP_SKIP: use final_policy_step for one more env.step(). Hence this 
        environment step will be skipped for training because it's not performed
        with GradientTape() context.
    """
    FINAL_STEP_REDO = 0  # redo the final step for training
    FINAL_STEP_SKIP = 1  # skip the final step for training

    def __init__(self,
                 env: TFEnvironment,
                 algorithm: OnPolicyAlgorithm,
                 observation_transformer: Callable = None,
                 observers=[],
                 metrics=[],
                 training=True,
                 train_interval=20,
                 final_step_mode=FINAL_STEP_REDO,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None):
        """Create an OnPolicyDriver.

        Args:
            env (TFEnvironment): A TFEnvoronmnet
            algorithm (OnPolicyAlgorith): The algorithm for training
            observers (list[Callable]): An optional list of observers that are
                updated after every step in the environment. Each observer is a
                callable(time_step.Trajectory).
            metrics (list[TFStepMetric]): An optiotional list of metrics.
            training (bool): True for training, false for evaluating
            train_interval (int):
            final_step_mode (int): FINAL_STEP_REDO for redo the final step for
                training. FINAL_STEP_SKIP for skipping the final step for
                training. See the class comment for explanation.
            debug_summaries (bool): A bool to gather debug summaries.
            summarize_grads_and_vars (bool): If True, gradient and network
                variable summaries will be written during training.
            train_step_counter (tf.Variable): An optional counter to increment
                every time the a new iteration is started. If None, it will use 
                tf.summary.experimental.get_step(). If this is still None, a
                counter will be created.
        """
        metric_buf_size = max(10, env.batch_size)
        standard_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=metric_buf_size),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=metric_buf_size),
        ]
        metrics = standard_metrics + metrics

        super(OnPolicyDriver, self).__init__(env, None, observers + metrics)

        self._algorithm = algorithm
        self._training = training
        self._train_interval = train_interval
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._final_step_mode = final_step_mode
        self._metrics = metrics
        self._observation_transformer = observation_transformer

        if training:
            self._policy_state_spec = algorithm.train_state_spec
            time_step_spec = env.time_step_spec()
            action_distribution_param_spec = tf.nest.map_structure(
                lambda spec: spec.input_params_spec,
                algorithm.action_distribution_spec)

            policy_step = algorithm.train_step(self.get_initial_time_step(),
                                               self.get_initial_state())
            info_spec = tf.nest.map_structure(
                lambda t: tf.TensorSpec(t.shape[1:], t.dtype),
                policy_step.info)

            self._training_info_spec = TrainingInfo(
                action_distribution=action_distribution_param_spec,
                action=env.action_spec(),
                step_type=time_step_spec.step_type,
                reward=time_step_spec.reward,
                discount=time_step_spec.discount,
                info=info_spec)
            self._trainable_variables = algorithm.trainable_variables
        else:
            self._policy_state_spec = algorithm.predict_state_spec

        self._train_step_counter = common.get_global_counter(
            train_step_counter)

        self._initial_state = self.get_initial_state()
        self._proc = psutil.Process(os.getpid())

    def run(self, max_num_steps, time_step=None, policy_state=None):
        """Take steps in the environment for max_num_steps.

        If in training mode, algorithm.train_step() and 
        algorithm.train_complete() will be called.
        If not in training mode, algorith.predict() will be called.

        The observers will also be called for every environment step.

        Args:
            max_num_steps (int): stops after so many environment steps. Is the
                total number of steps from all the individual environment in
                the bached enviroments including StepType.LAST steps.
            time_step (ActionTimeStep): optional initial time_step. If None, it
                will use self.get_initial_time_step(). Elements should be shape
                [batch_size, ...].
            policy_state (nested Tensor): optional initial state for the policy.

        Returns:
            time_step (ActionTimeStep): named tuple with final observation,
                reward, etc.
            policy_state (nested Tensor): final step policy state.
        """
        if time_step is None:
            time_step = self.get_initial_time_step()
        if policy_state is None:
            policy_state = self._initial_state
        return self._run(
            time_step=time_step,
            policy_state=policy_state,
            max_num_steps=max_num_steps)

    def get_initial_time_step(self):
        """Returns the initial action_time_step."""
        time_step = self.env.current_time_step()
        action = common.zero_tensor_from_nested_spec(self.env.action_spec(),
                                                     self.env.batch_size)
        return make_action_time_step(time_step, action)

    def get_initial_state(self):
        """Returns an initial state usable by the algorithm.

        Returns:
            A nested object of type `policy_state` containing properly
            initialized Tensors.
        """
        return common.zero_tensor_from_nested_spec(self._policy_state_spec,
                                                   self.env.batch_size)

    def get_metrics(self):
        """Returns the metrics monitored by this driver.

        Returns:
            list[TFStepMetric]
        """
        return self._metrics

    def _algorithm_step(self, time_step, state):
        if self._observation_transformer is not None:
            time_step = time_step._replace(
                observation=self._observation_transformer(time_step.
                                                          observation))

        if self._training:
            return self._algorithm.train_step(time_step, state)
        else:
            return self._algorithm.predict(time_step, state)

    def _run(self, time_step, policy_state, max_num_steps):
        if self._training:
            maximum_iterations = math.ceil(
                max_num_steps /
                (self._env.batch_size *
                 (self._train_interval +
                  (self._final_step_mode == OnPolicyDriver.FINAL_STEP_SKIP))))
            [time_step, policy_state] = tf.while_loop(
                cond=lambda *_: True,
                body=self._iter,
                loop_vars=[time_step, policy_state],
                maximum_iterations=maximum_iterations,
                back_prop=False,
                name="driver_loop")
        else:
            maximum_iterations = math.ceil(
                max_num_steps / self._env.batch_size)
            [time_step, policy_state] = tf.while_loop(
                cond=lambda *_: True,
                body=self._eval_loop_body,
                loop_vars=[time_step, policy_state],
                maximum_iterations=maximum_iterations,
                back_prop=False,
                name="driver_loop")

        return time_step, policy_state

    def _sample_action_distribution(self, action_distribution):
        seed_stream = tfp.distributions.SeedStream(seed=None, salt='driver')
        return tf.nest.map_structure(lambda d: d.sample(seed=seed_stream()),
                                     action_distribution)

    def _env_step(self, action):
        time_step = self.env.step(action)
        return make_action_time_step(time_step, action)

    def _step(self, time_step, policy_state):
        policy_state = common.reset_state_if_necessary(policy_state,
                                                       self._initial_state,
                                                       time_step.is_first())

        policy_step = self._algorithm_step(time_step, state=policy_state)

        action = self._sample_action_distribution(policy_step.action)

        next_time_step = self._env_step(action)

        if self._observers:
            traj = from_transition(time_step,
                                   policy_step._replace(action=action),
                                   next_time_step)
            for observer in self._observers:
                observer(traj)

        return next_time_step, policy_step, action

    def _train_loop_body(self, counter, time_step, policy_state,
                         training_info_ta):

        next_time_step, policy_step, action = self._step(
            time_step, policy_state)
        action_distribution_param = common.get_distribution_params(
            policy_step.action)

        training_info = TrainingInfo(
            action_distribution=action_distribution_param,
            action=action,
            reward=time_step.reward,
            discount=time_step.discount,
            step_type=time_step.step_type,
            info=policy_step.info)

        training_info_ta = tf.nest.map_structure(
            lambda ta, x: ta.write(counter, x), training_info_ta,
            training_info)

        counter += 1

        return [counter, next_time_step, policy_step.state, training_info_ta]

    def _eval_loop_body(self, time_step, policy_state):
        next_time_step, policy_step, _ = self._step(time_step, policy_state)

        return [next_time_step, policy_step.state]

    def _iter(self, time_step, policy_state):
        """One training iteration."""
        counter = tf.zeros((), tf.int32)
        batch_size = self.env.batch_size

        def create_ta(s):
            return tf.TensorArray(
                dtype=s.dtype,
                size=self._train_interval,
                element_shape=tf.TensorShape([batch_size]).concatenate(
                    s.shape))

        training_info_ta = tf.nest.map_structure(create_ta,
                                                 self._training_info_spec)

        with tf.GradientTape() as tape:
            [_, time_step, policy_state, training_info_ta] = tf.while_loop(
                cond=lambda counter, *_: tf.less(counter, self._train_interval),
                body=self._train_loop_body,
                loop_vars=[counter, time_step, policy_state, training_info_ta],
                back_prop=True,
                parallel_iterations=1,
                name='iter_loop')

            training_info = tf.nest.map_structure(lambda ta: ta.stack(),
                                                  training_info_ta)

            action_distribution = nested_distributions_from_specs(
                self._algorithm.action_distribution_spec,
                training_info.action_distribution)

            training_info = training_info._replace(
                action_distribution=action_distribution)

        if self._final_step_mode == OnPolicyDriver.FINAL_STEP_SKIP:
            next_time_step, policy_step, _ = self._step(
                time_step, policy_state)
            next_state = policy_step.state
        else:
            policy_step = self._algorithm_step(time_step, policy_state)
            next_time_step = time_step
            next_state = policy_state

        loss_info, grads_and_vars = self._algorithm.train_complete(
            tape, training_info, time_step, policy_step)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self._train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self._train_step_counter)
        if self._debug_summaries:
            common.add_loss_summaries(loss_info)
            common.add_action_summaries(training_info.action,
                                        self._training_info_spec.action)

        for metric in self._metrics:
            metric.tf_summaries(
                train_step=self._train_step_counter,
                step_metrics=self._metrics[:2])

        mem = tf.py_function(
            lambda: self._proc.memory_info().rss // 1e6, [],
            tf.float32,
            name='memory_usage')
        if not tf.executing_eagerly():
            mem.set_shape(())
        tf.summary.scalar(name='memory_usage', data=mem)

        self._train_step_counter.assign_add(1)

        return next_time_step, next_state
