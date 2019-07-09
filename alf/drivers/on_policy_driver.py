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

from tf_agents.metrics import tf_metrics
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.trajectories.trajectory import from_transition
from tf_agents.specs.distribution_spec import nested_distributions_from_specs

import alf.utils.common as common
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.off_policy_algorithm import make_experience
from alf.algorithms.rl_algorithm import make_action_time_step
from alf.algorithms.rl_algorithm import make_training_info
from alf.drivers import policy_driver


@gin.configurable
class OnPolicyDriver(policy_driver.PolicyDriver):
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
    collect necessary information and final_policy_step.info into training_info
    algorithm.train_complete(tape, training_info)
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
                 observers=[],
                 metrics=[],
                 training=True,
                 greedy_predict=False,
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
            greedy_predict (bool): use greedy action for evaluation (i.e.
                training==False).
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
        self._metrics = standard_metrics + metrics
        self._exp_observers = []

        super(OnPolicyDriver, self).__init__(
            env=env,
            algorithm=algorithm,
            observers=observers + self._metrics,
            training=training,
            greedy_predict=greedy_predict,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)

        self._final_step_mode = final_step_mode

        if training:
            self._policy_state_spec = algorithm.train_state_spec
            self._prepare_specs(algorithm)
            self._trainable_variables = algorithm.trainable_variables
            self._train_interval = train_interval
        else:
            self._policy_state_spec = algorithm.predict_state_spec

        self._initial_state = common.get_initial_policy_state(
            self._env.batch_size, self._policy_state_spec)

    def _prepare_specs(self, algorithm):
        time_step_spec = self._env.time_step_spec()
        action_distribution_param_spec = tf.nest.map_structure(
            lambda spec: spec.input_params_spec,
            algorithm.action_distribution_spec)

        policy_step = algorithm.train_step(
            common.get_initial_time_step(self._env),
            common.get_initial_policy_state(self._env.batch_size,
                                            self._policy_state_spec))
        info_spec = tf.nest.map_structure(
            lambda t: tf.TensorSpec(t.shape[1:], t.dtype), policy_step.info)

        self._training_info_spec = make_training_info(
            action_distribution=action_distribution_param_spec,
            action=self._env.action_spec(),
            step_type=time_step_spec.step_type,
            reward=time_step_spec.reward,
            discount=time_step_spec.discount,
            info=info_spec)

    def add_experience_observer(self, observer: Callable):
        """Add an observer to receive experience.

        Args:
            observer (Callable): callable which accept Experience as argument.
        """
        self._exp_observers.append(observer)

    def get_step_metrics(self):
        """See PolicyDriver.get_step_metrics()"""
        return self._metrics[:2]

    def get_metrics(self):
        """See PolicyDriver.get_metrics()"""
        return self._metrics

    def get_initial_time_step(self):
        return common.get_initial_time_step(self._env)

    def get_initial_policy_state(self):
        return common.get_initial_policy_state(self._env.batch_size,
                                               self._policy_state_spec)

    def predict(self, max_num_steps, time_step, policy_state):
        maximum_iterations = math.ceil(max_num_steps / self._env.batch_size)
        [time_step, policy_state] = tf.while_loop(
            cond=lambda *_: True,
            body=self._eval_loop_body,
            loop_vars=[time_step, policy_state],
            maximum_iterations=maximum_iterations,
            back_prop=False,
            name="predict_loop")
        return time_step, policy_state

    def _eval_loop_body(self, time_step, policy_state):
        next_time_step, policy_step, _ = self._step(time_step, policy_state)
        return [next_time_step, policy_step.state]

    def _step(self, time_step, policy_state):
        policy_state = common.reset_state_if_necessary(policy_state,
                                                       self._initial_state,
                                                       time_step.is_first())
        policy_step = common.algorithm_step(
            self._algorithm,
            self._observation_transformer,
            time_step,
            state=policy_state,
            training=self._training,
            greedy_predict=self._greedy_predict)
        action = common.sample_action_distribution(policy_step.action)
        next_time_step = self._env_step(action)
        if self._observers:
            traj = from_transition(time_step,
                                   policy_step._replace(action=action),
                                   next_time_step)
            for observer in self._observers:
                observer(traj)
        if self._exp_observers:
            action_distribution_param = common.get_distribution_params(
                policy_step.action)
            exp = make_experience(
                time_step,
                policy_step._replace(action=action),
                action_distribution=action_distribution_param)
            for observer in self._exp_observers:
                observer(exp)

        return next_time_step, policy_step, action

    def _env_step(self, action):
        time_step = self._env.step(action)
        return make_action_time_step(time_step, action)

    @tf.function
    def run(self, max_num_steps=None, time_step=None, policy_state=None):
        """
        Take steps in the environment for max_num_steps.

        If in training mode, algorithm.train_step() and
        algorithm.train_complete() will be called.
        If not in training mode, algorithm.predict() will be called.

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
            time_step = common.get_initial_time_step(self._env)
        if policy_state is None:
            policy_state = self._initial_state
        if self._training:
            return self.train(max_num_steps, time_step, policy_state)
        else:
            return self.predict(max_num_steps, time_step, policy_state)

    def train(self, max_num_steps, time_step, policy_state):
        """Perform on-policy training with `max_num_steps`.

        Args:
            max_num_steps (int): stops after so many environment steps. Is the
                total number of steps from all the individual environment in
                the bached enviroments including StepType.LAST steps.
            time_step (ActionTimeStep): optional initial time_step. If None, it
                will use self.get_initial_time_step(). Elements should be shape
                [batch_size, ...].
            policy_state (nested Tensor): optional initial state for the policy.
        Returns:
            None
        """
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
        return time_step, policy_state

    def _train_loop_body(self, counter, time_step, policy_state,
                         training_info_ta):

        next_time_step, policy_step, action = self._step(
            time_step, policy_state)
        action = tf.nest.map_structure(tf.stop_gradient, action)
        action_distribution_param = common.get_distribution_params(
            policy_step.action)

        training_info = make_training_info(
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

    def _iter(self, time_step, policy_state):
        """One training iteration."""
        counter = tf.zeros((), tf.int32)
        batch_size = self._env.batch_size

        def create_ta(s):
            return tf.TensorArray(
                dtype=s.dtype,
                size=self._train_interval + 1,
                element_shape=tf.TensorShape([batch_size]).concatenate(
                    s.shape))

        training_info_ta = tf.nest.map_structure(create_ta,
                                                 self._training_info_spec)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
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
            next_time_step, policy_step, action = self._step(
                time_step, policy_state)
            next_state = policy_step.state
        else:
            policy_step = common.algorithm_step(
                self._algorithm,
                self._observation_transformer,
                time_step,
                policy_state,
                training=self._training,
                greedy_predict=self._greedy_predict)
            action = common.sample_action_distribution(policy_step.action)
            next_time_step = time_step
            next_state = policy_state

        action_distribution_param = common.get_distribution_params(
            policy_step.action)

        final_training_info = make_training_info(
            action_distribution=action_distribution_param,
            action=action,
            reward=time_step.reward,
            discount=time_step.discount,
            step_type=time_step.step_type,
            info=policy_step.info)

        with tape:
            training_info_ta = tf.nest.map_structure(
                lambda ta, x: ta.write(counter, x), training_info_ta,
                final_training_info)
            training_info = tf.nest.map_structure(lambda ta: ta.stack(),
                                                  training_info_ta)

            action_distribution = nested_distributions_from_specs(
                self._algorithm.action_distribution_spec,
                training_info.action_distribution)

            training_info = training_info._replace(
                action_distribution=action_distribution)

        loss_info, grads_and_vars = self._algorithm.train_complete(
            tape, training_info)

        self._training_summary(training_info, loss_info, grads_and_vars)

        self._train_step_counter.assign_add(1)

        return next_time_step, next_state
