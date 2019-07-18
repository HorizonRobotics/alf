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
import os
from typing import Callable

import psutil
import math
import gin.tf
import tensorflow as tf

from tf_agents.drivers import driver
from tf_agents.metrics import tf_metrics
from tf_agents.trajectories.trajectory import from_transition

from alf.algorithms.off_policy_algorithm import make_experience
from alf.utils import common, summary_utils
from alf.algorithms.rl_algorithm import make_action_time_step


@gin.configurable
class PolicyDriver(driver.Driver):
    def __init__(self,
                 env,
                 algorithm,
                 observation_transformer: Callable = None,
                 observers=[],
                 metrics=[],
                 training=True,
                 greedy_predict=False,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None):
        """Create a PolicyDriver.

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

        super(PolicyDriver, self).__init__(env, None,
                                           observers + self._metrics)

        self._algorithm = algorithm
        self._training = training
        self._greedy_predict = greedy_predict
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._observation_transformer = observation_transformer
        self._train_step_counter = common.get_global_counter(
            train_step_counter)
        self._proc = psutil.Process(os.getpid())
        if training:
            self._policy_state_spec = algorithm.train_state_spec
        else:
            self._policy_state_spec = algorithm.predict_state_spec
        self._initial_state = self.get_initial_policy_state()

    def add_experience_observer(self, observer: Callable):
        """Add an observer to receive experience.

        Args:
            observer (Callable): callable which accept Experience as argument.
        """
        self._exp_observers.append(observer)

    def _training_summary(self, training_info, loss_info, grads_and_vars):
        if self._summarize_grads_and_vars:
            summary_utils.add_variables_summaries(grads_and_vars,
                                                  self._train_step_counter)
            summary_utils.add_gradients_summaries(grads_and_vars,
                                                  self._train_step_counter)
        if self._debug_summaries:
            common.add_action_summaries(training_info.action,
                                        self.env.action_spec())
            common.add_loss_summaries(loss_info)

        for metric in self.get_metrics():
            metric.tf_summaries(
                train_step=self._train_step_counter,
                step_metrics=self.get_metrics()[:2])

        mem = tf.py_function(
            lambda: self._proc.memory_info().rss // 1e6, [],
            tf.float32,
            name='memory_usage')
        if not tf.executing_eagerly():
            mem.set_shape(())
        tf.summary.scalar(name='memory_usage', data=mem)

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def observation_transformer(self):
        return self._observation_transformer

    @abc.abstractmethod
    def _prepare_specs(self, algorithm):
        pass

    def get_step_metrics(self):
        """Get step metrics that used for generating summaries against

        Returns:
             list[TFStepMetric]: step metrics `EnvironmentSteps` and `NumberOfEpisodes`
        """
        return self._metrics[:2]

    def get_metrics(self):
        """Returns the metrics monitored by this driver.

        Returns:
            list[TFStepMetric]
        """
        return self._metrics

    def get_initial_time_step(self):
        return common.get_initial_time_step(self._env)

    def get_initial_policy_state(self):
        """
        Return can be the train or prediction state spec, depending on self._training
        """
        return common.get_initial_policy_state(self._env.batch_size,
                                               self._policy_state_spec)

    def get_initial_train_state(self, batch_size):
        """Always return the training state spec"""
        return common.zero_tensor_from_nested_spec(
            self._algorithm.train_state_spec, batch_size)

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
            For OnPolicyDriver and SyncOffPolicyDriver:
                time_step (ActionTimeStep): named tuple with final observation,
                    reward, etc.
                policy_state (nested Tensor): final step policy state.
            For AsyncOffPolicyDriver:
                num_steps (int): how many steps have been run
        """
        if time_step is None:
            time_step = self.get_initial_time_step()
        if policy_state is None:
            policy_state = self._initial_state
        return self._run(max_num_steps, time_step, policy_state)

    def _run(self, max_num_steps, time_step, policy_state):
        """Different drivers implement different runs"""
        raise NotImplementedError()
