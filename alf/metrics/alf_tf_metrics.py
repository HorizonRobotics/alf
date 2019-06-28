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

import tensorflow as tf
import threading

from tf_agents.metrics import tf_py_metric
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_metric
from alf.metrics import alf_py_metrics


class TFPyMetric(tf_metric.TFStepMetric):
    """
    The difference from tf_metrics.TFPyMetric is that we allow using
    lock here. Other code is just copied.
    """
    def __init__(self, py_metric, name=None, dtype=tf.float32):
        """Creates a TF metric given a py metric to wrap.

        py_metric (alf_py_metrics.StreamingMetric): A batched python metric to wrap.
        name (str): Name of the metric.
        dtype (tf.dtype): Data type of the metric.
        """
        name = name or py_metric.name
        super(TFPyMetric, self).__init__(name=name)
        self._py_metric = py_metric
        self._dtype = dtype
        self._lock = threading.Lock()

    def call(self, trajectory):
        """Update the value of the metric using trajectory.

        The trajectory can be either batched or un-batched depending on
        the expected inputs for the py_metric being wrapped.

        Input:
            trajectory (tf_agents.trajectory.Trajectory):

        Output:
            trajectory (tf_agents.trajectory.Trajectory) :
                the argument itself, for easy chaining.
        """
        def _call(*flattened_trajectories):
            with self._lock:
                flat_sequence = [x.numpy() for x in flattened_trajectories]
                packed_trajectories = tf.nest.pack_sequence_as(
                        structure=(trajectory), flat_sequence=flat_sequence)
                return self._py_metric(packed_trajectories)

        flattened_trajectories = tf.nest.flatten(trajectory)
        metric_op = tf.py_function(
                _call,
                flattened_trajectories,
                [],
                name='metric_call_py_func')

        with tf.control_dependencies([metric_op]):
            return tf.nest.map_structure(tf.identity, trajectory)

    def result(self):
        def _result():
            with self._lock:
                return self._py_metric.result()

        result_value = tf.py_function(
                _result,
                [],
                self._dtype,
                name='metric_result_py_func')
        if not tf.executing_eagerly():
            result_value.set_shape(())
        return result_value

    def reset(self):
        def _reset():
            with self._lock:
                return self._py_metric.reset()

        return tf.py_function(
                _reset, [], [],
                name='metric_reset_py_func')


class AverageReturnMetric(TFPyMetric):
    """
    Metric to compute the average return.
    Use our customized alf_py_metrics.AverageReturnMetric
    """

    def __init__(self, num_envs, env_batch_size, buffer_size=None,
                 name='AverageReturn', dtype=tf.float32):
        py_metric = alf_py_metrics.AverageReturnMetric(
                num_envs, env_batch_size, buffer_size, name)
        super(AverageReturnMetric, self).__init__(
                py_metric=py_metric, name=name, dtype=dtype)


class AverageEpisodeLengthMetric(TFPyMetric):
    """
    Metric to compute the average episode length.
    Use our customized alf_py_metrics.AverageEpisodeLengthMetric.
    """

    def __init__(self,
                 num_envs,
                 env_batch_size,
                 buffer_size=None,
                 name='AverageEpisodeLength',
                 dtype=tf.float32):

        py_metric = alf_py_metrics.AverageEpisodeLengthMetric(
                num_envs, env_batch_size, buffer_size, name)
        super(AverageEpisodeLengthMetric, self).__init__(
                py_metric=py_metric, name=name, dtype=dtype)


class NumberOfEpisodes(tf_metrics.NumberOfEpisodes):
    """
    Overload call() to accept the other arg (env_id) and ignore it.
    """
    def __init__(self, name='NumberOfEpisodes', dtype=tf.int64):
        super(NumberOfEpisodes, self).__init__(name, dtype)

    def call(self, trajectory):
        traj, _ = trajectory
        return super().call(traj)


class EnvironmentSteps(tf_metrics.EnvironmentSteps):
    """
    Overload call() to accept the other arg (env_id) and ignore it.
    """
    def __init__(self, name='EnvironmentSteps', dtype=tf.int64):
        super(EnvironmentSteps, self).__init__(name, dtype)

    def call(self, trajectory):
        traj, _ = trajectory
        return super().call(traj)
