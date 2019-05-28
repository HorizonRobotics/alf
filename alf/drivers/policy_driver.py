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

import os
from collections import namedtuple

import numpy as np
import psutil
import math
import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.drivers import driver
from tf_agents.metrics import tf_metrics
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.trajectory import from_transition
from tf_agents.utils import eager_utils
from alf.utils import common, common as common


class ActionTimeStep(
    namedtuple(
        'ActionTimeStep',
        ['step_type', 'reward', 'discount', 'observation', 'action'])):
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


def make_action_time_step(time_step, action):
    return ActionTimeStep(
        step_type=time_step.step_type,
        reward=time_step.reward,
        discount=time_step.discount,
        observation=time_step.observation,
        action=action)


@gin.configurable
class PolicyDriver(driver.Driver):
    def __init__(self,
                 env,
                 algorithm,
                 observers=[],
                 metrics=[],
                 training=True,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None):
        metric_buf_size = max(10, env.batch_size)
        standard_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=metric_buf_size),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=metric_buf_size),
        ]
        metrics = standard_metrics + metrics
        super(PolicyDriver, self).__init__(env, None, observers + metrics)

        self._algorithm = algorithm
        self._training = training

        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._metrics = metrics

        if training:
            self._policy_state_spec = algorithm.train_state_spec
            self._algorithm_step = algorithm.train_step
            self._trainable_variables = algorithm.trainable_variables
        else:
            self._policy_state_spec = algorithm.predict_state_spec
            self._algorithm_step = algorithm.predict

        self._train_step_counter = common.get_global_counter(
            train_step_counter)
        self._initial_state = self.get_initial_state()
        self._proc = psutil.Process(os.getpid())

    def get_initial_time_step(self):
        """Returns the initial action_time_step."""
        time_step = self.env.current_time_step()
        action = common.zero_tensor_from_nested_spec(self.env.action_spec(),
                                                     self.env.batch_size)
        return make_action_time_step(time_step, action)

    def run(self, max_num_steps=None, time_step=None, policy_state=None):
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

        if self._training:
            return self.train(max_num_steps, time_step, policy_state)
        else:
            return self.predict(max_num_steps, time_step, policy_state)

    def train(self, max_num_steps, time_step, policy_state):
        raise NotImplemented

    def predict(self, max_num_steps, time_step, policy_state):
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

    def summary(self, loss_info, grads_and_vars):
        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(
                grads_and_vars, self._train_step_counter)
            eager_utils.add_gradients_summaries(
                grads_and_vars, self._train_step_counter)
        if self._debug_summaries:
            common.add_loss_summaries(loss_info)

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

    def _eval_loop_body(self, time_step, policy_state):
        next_time_step, policy_step, _ = self._step(time_step, policy_state)
        return [next_time_step, policy_step.state]

    def _step(self, time_step, policy_state):
        policy_state = common.reset_state_if_necessary(
            policy_state, self._initial_state, time_step.is_first())
        policy_step = self._algorithm_step(time_step, state=policy_state)
        action = self._sample_action_distribution(policy_step.action)
        next_time_step = self._env_step(action)
        if self._observers:
            traj = from_transition(
                time_step, policy_step._replace(action=action), next_time_step)
            for observer in self._observers:
                observer(traj)

        return next_time_step, policy_step, action

    def _env_step(self, action):
        time_step = self.env.step(action)
        return make_action_time_step(time_step, action)

    def _sample_action_distribution(self, actions_or_distributions):
        def _to_distribution(action_or_distribution):
            if isinstance(action_or_distribution, tf.Tensor):
                return tfp.distributions.Deterministic(loc=action_or_distribution)
            return action_or_distribution

        distributions = tf.nest.map_structure(
            _to_distribution, actions_or_distributions)
        seed_stream = tfp.distributions.SeedStream(seed=None, salt='driver')
        return tf.nest.map_structure(
            lambda d: d.sample(seed=seed_stream()), distributions)
