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
"""Customized ALF python metrics"""

import gin
import numpy as np
import itertools
import abc
import six

from tf_agents.metrics import py_metric
from tf_agents.utils import numpy_storage
from tf_agents.metrics import py_metrics
from tf_agents.utils import nest_utils


class StreamingMetric(py_metrics.StreamingMetric):
    """
    The difference between this class with the one defined by tf_agents is that
    for async training we assume `call()` receives both `trajectory` and the corresponding
    `id` so that we know how to align metrics w.r.t. different environments, because
    each time `trajectory` is from one of all the environments.
    """

    def __init__(self, buffer_size, num_envs, name='StreamingMetric'):
        super(StreamingMetric, self).__init__(name, buffer_size, num_envs)

    def call(self, trajectory, id):
        if trajectory.step_type.ndim == 0:
            trajectory = nest_utils.batch_nested_array(trajectory)
        self._batched_call(trajectory, id)


@six.add_metaclass(abc.ABCMeta)
class AsyncStreamingMetric(StreamingMetric):
    def __init__(self, name, num_envs, env_batch_size, buffer_size=None):
        """
        Args:
            name (str): name of the metric
            num_envs (int): number of tf_agents.environments; each environment is
                    a batched environment (contains multiple independent envs)
            env_batch_size (int): the size of each batched environment
            buffer_size (int): the window size of data points we want to average over
        """
        num_envs *= env_batch_size
        self._env_batch_size = env_batch_size
        self._np_state = numpy_storage.NumpyState()
        # Set a dummy value on self._np_state.episode_return so it gets included in
        # the first checkpoint (before metric is first called).
        self._np_state.episode_return = np.float64(0)
        if buffer_size is None:
            buffer_size = max(env_batch_size, 10)
        super(AsyncStreamingMetric, self).__init__(
            buffer_size=buffer_size, num_envs=num_envs, name=name)

    def _batched_call(self, trajectory, id):
        """
        Args:
            trajectory (Trajectory): a nested structure where each leaf has the
                shape (`unroll_length`, `env_batch_size`, ...)
            id (int): indicates which environment generated `trajectory`
        """
        is_first = trajectory.is_first()
        is_last = trajectory.is_last()
        is_boundary = trajectory.is_boundary()
        reward = trajectory.reward
        ids = np.arange(self._env_batch_size) + id * self._env_batch_size
        for t in range(reward.shape[0]):
            self._batched_call_per_step(is_first[t], is_last[t],
                                        is_boundary[t], reward[t], ids)

    @abc.abstractmethod
    def _batched_call_per_step(self, is_first, is_last, is_boundary, reward,
                               ids):
        """Update np storage state given the active `ids` at each time step
        Args:
            is_first (np.array[bool]): if the steps are StepType.FIRST
            is_last (np.array[bool]): if the next steps are StepType.LAST
            is_boundary (np.array[bool]): if the steps are StepType.LAST
            reward (np.array[float32]): the step rewards
            ids (np.array[int]): the indices of the environments that are active for
                                 the current batched call. They constitute a subset of
                                 all env ids.
        """


@gin.configurable
class AverageReturnMetric(AsyncStreamingMetric):
    """
    Computes the average undiscounted reward.
    The difference with py_metrics.AverageReturnMetric is that this metric
    assumes _batched_call on partial trajectory data, i.e., not every
    environment will be summarized for a particular call. This is due to
    the nature of the asynchronous training update.
    """

    def __init__(self,
                 num_envs,
                 env_batch_size,
                 buffer_size=None,
                 name='AverageReturn'):
        super(AverageReturnMetric, self).__init__(name, num_envs,
                                                  env_batch_size, buffer_size)

    def _reset(self, num_envs):
        """Resets stat gathering variables"""
        self._np_state.episode_return = np.zeros(
            shape=(num_envs, ), dtype=np.float64)

    def _batched_call_per_step(self, is_first, is_last, is_boundary, reward,
                               ids):
        episode_return = self._np_state.episode_return
        # reset to 0 where is_first==True
        episode_return[ids] *= ~is_first
        # add rewards
        episode_return[ids] += reward
        # add episodic rewards
        self.add_to_buffer(episode_return[ids][np.where(is_last)])


@gin.configurable
class AverageEpisodeLengthMetric(AsyncStreamingMetric):
    """ Computes the average episode length. """

    def __init__(self,
                 num_envs,
                 env_batch_size,
                 buffer_size=None,
                 name='AverageEpisodeLength'):
        super(AverageEpisodeLengthMetric, self).__init__(
            name, num_envs, env_batch_size, buffer_size)

    def _reset(self, num_envs):
        """Resets stat gathering variables"""
        self._np_state.episode_steps = np.zeros(
            shape=(num_envs, ), dtype=np.float64)

    def _batched_call_per_step(self, is_first, is_last, is_boundary, reward,
                               ids):
        episode_steps = self._np_state.episode_steps
        episode_steps[ids] += 1 - is_boundary
        self.add_to_buffer(episode_steps[ids][np.where(is_last)])
        episode_steps[ids] *= ~is_last
