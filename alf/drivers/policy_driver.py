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
from tf_agents.utils import eager_utils

from alf.utils import common


@gin.configurable
class PolicyDriver(driver.Driver):
    def __init__(self,
                 env,
                 algorithm,
                 observation_transformer: Callable = None,
                 observers=[],
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

        super(PolicyDriver, self).__init__(env, None, observers)

        self._algorithm = algorithm
        self._training = training
        self._greedy_predict = greedy_predict
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._observation_transformer = observation_transformer
        self._train_step_counter = common.get_global_counter(
            train_step_counter)
        self._proc = psutil.Process(os.getpid())

    def _training_summary(self, training_info, loss_info, grads_and_vars):
        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self._train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
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
    def get_step_metrics(self):
        """Get step metrics that used for generating summaries against

        Returns:
             list[TFStepMetric]: step metrics `EnvironmentSteps` and `NumberOfEpisodes`
        """
        pass

    @abc.abstractmethod
    def get_metrics(self):
        """Returns the metrics monitored by this driver.

        Returns:
            list[TFStepMetric]
        """
        pass
