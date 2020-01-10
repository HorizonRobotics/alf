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
import functools
import os
from typing import Callable

import math
import gin.tf
import tensorflow as tf

from tf_agents.drivers import driver
from tf_agents.metrics import tf_metrics
from tf_agents.trajectories.trajectory import from_transition

from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import make_action_time_step, make_experience
from alf.utils import common, nest_utils, summary_utils


@gin.configurable
class PolicyDriver(driver.Driver):
    ON_POLICY_TRAINING = 0
    OFF_POLICY_TRAINING = 1
    PREDICT = 2

    def __init__(self,
                 env,
                 algorithm,
                 observers=[],
                 metrics=[],
                 epsilon_greedy=0.1,
                 mode=PREDICT):
        """Create a PolicyDriver.

        Args:
            env (TFEnvironment): A TFEnvironmnet
            algorithm (RLAlgorithm): The algorithm for training
            observers (list[Callable]): An optional list of observers that are
                updated after every step in the environment. Each observer is a
                callable(time_step.Trajectory).
            metrics (list[TFStepMetric]): An optional list of metrics.
            epsilon_greedy (float):  a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for mode=PREDICT.
            mode (int): One of (ON_POLICY_TRAINING, OFF_POLICY_TRAINING, PREDICT)
        """
        self._mode = mode
        self._epsilon_greedy = epsilon_greedy
        metric_buf_size = max(10, env.batch_size)
        standard_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
        ]
        training = mode in [self.ON_POLICY_TRAINING, self.OFF_POLICY_TRAINING]
        # This is a HACK.
        # Due to tf_agents metric API change:
        # https://github.com/tensorflow/agents/commit/b08a142edf180325b63441ec1b71119c393c4a64,
        # after tensorflow v20190807, these two metrics will cause error during
        # playing of the trained model, when num_parallel_envs > 1.
        # Somehow the restore_specs do not match restored metric tensors in
        # tensorflow_core/python/training/saving/functional_saver.py
        if training:
            standard_metrics += [
                tf_metrics.AverageReturnMetric(
                    batch_size=env.batch_size, buffer_size=metric_buf_size),
                tf_metrics.AverageEpisodeLengthMetric(
                    batch_size=env.batch_size, buffer_size=metric_buf_size),
            ]
        self._metrics = standard_metrics + metrics

        super(PolicyDriver, self).__init__(env, None,
                                           observers + self._metrics)

        self._algorithm = algorithm
        self._training = training
        if training:
            self._policy_state_spec = algorithm.train_state_spec
        else:
            self._policy_state_spec = algorithm.predict_state_spec
        self._initial_state = self.get_initial_policy_state()

    @property
    def algorithm(self):
        return self._algorithm

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
        """Always return the training state spec."""
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
        if self._mode == self.PREDICT:
            step_func = functools.partial(
                self._algorithm.predict, epsilon_greedy=self._epsilon_greedy)
        elif self._mode == self.ON_POLICY_TRAINING:
            step_func = functools.partial(
                self._algorithm.rollout, mode=RLAlgorithm.ON_POLICY_TRAINING)
        elif self._mode == self.OFF_POLICY_TRAINING:
            step_func = functools.partial(
                self._algorithm.rollout, mode=RLAlgorithm.ROLLOUT)
        else:
            raise ValueError()
        transformed_time_step = self._algorithm.transform_timestep(time_step)
        policy_step = step_func(transformed_time_step, policy_state)

        next_time_step = self._env_step(policy_step.action)
        if self._observers:
            traj = from_transition(time_step, policy_step._replace(info=()),
                                   next_time_step)
            for observer in self._observers:
                observer(traj)
        if self._algorithm.exp_observers and self._training:
            policy_step = nest_utils.distributions_to_params(policy_step)
            exp = make_experience(time_step, policy_step, policy_state)
            self._algorithm.observe(exp)

        return next_time_step, policy_step, transformed_time_step

    def _env_step(self, action):
        time_step = self._env.step(action)
        return make_action_time_step(time_step, action)

    @tf.function
    def run(self, max_num_steps=None, time_step=None, policy_state=None):
        """
        Take steps in the environment for max_num_steps.

        If in training mode, algorithm.rollout() will be used.
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
        """Different drivers implement different runs."""
        raise NotImplementedError()
