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

from tf_agents.utils import numpy_storage
from tf_agents.metrics import py_metrics
from tf_agents.utils import nest_utils


class StreamingMetric(py_metrics.StreamingMetric):
    """
    The difference between this class with the one defined by tf_agents is that
    for async training we assume `call()` receives both `traj` and the corresponding
    `ids` so that we know how to align metrics w.r.t. different environments, because
    each time `traj` might be from some but not all environments.
    """
    def __init__(self, buffer_size, num_envs, name='StreamingMetric'):
        super(StreamingMetric, self).__init__(name, buffer_size, num_envs)

    def call(self, trajectory):
        traj, ids = trajectory
        if traj.step_type.ndim == 0:
            traj = nest_utils.batch_nested_array(traj)
        self._batched_call((traj, ids))


@six.add_metaclass(abc.ABCMeta)
class AsyncStreamingMetric(StreamingMetric):
    def __init__(self, name, num_envs, env_batch_size, buffer_size=None):
        """
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

    def _select_time_major(self, arr, indices):
        arr = np.array([arr[k] for k in indices])
        tmp = np.transpose(arr, (1, 0, 2))
        return tmp

    def _batched_call(self, trajectory):
        """
        trajectory (tuple(traj, ids)):
            `traj` is a nested structure where each leaf has the shape (`num_envs`,
            `unroll_length`, `env_batch_size`, ...). `ids` is a vector of length
            `num_envs` indicating which environment generated `traj` along axis=0
        """
        traj, ids = trajectory
        is_first = traj.is_first()
        is_last = traj.is_last()
        is_boundary = traj.is_boundary()
        reward = traj.reward

        # We assume that `traj` might contain duplicate env ids.
        # Note that for every environment, we need to accumulate its statistics exactly
        # following the time order. That's the reason why we perform id grouping below.
        idx_id = enumerate(ids)  # [(idx, env_id)]
        # sort the trajs according to batch_env ids while preserving
        # the orders within an identical id
        idx_id = sorted(idx_id, key=lambda p: p[1])
        id_groups = [(id, [k for k, _ in idx]) \
                     for id, idx in itertools.groupby(idx_id, lambda p: p[1])]
        max_n = max([len(ig[1]) for ig in id_groups])

        # For each n, it is guaranteed that no duplicate batch_env ids are
        # each n is a slice of distinct environments
        for n in range(max_n):
            ids, indices = zip(*[(id, idx[n]) for id, idx in id_groups if n < len(idx)])
            ids = np.array([np.arange(self._env_batch_size) + id * self._env_batch_size
                            for id in ids]).flatten()

            n_is_first = self._select_time_major(is_first, indices)
            n_is_last = self._select_time_major(is_last, indices)
            n_is_boundary = self._select_time_major(is_boundary, indices)
            n_reward = self._select_time_major(reward, indices)

            for t in range(n_reward.shape[0]):
                t_is_first = np.concatenate(n_is_first[t], axis=0)
                t_is_last = np.concatenate(n_is_last[t], axis=0)
                t_is_boundary = np.concatenate(n_is_boundary[t], axis=0)
                t_reward = np.concatenate(n_reward[t], axis=0)
                self._batched_call_per_step(t_is_first, t_is_last, t_is_boundary, t_reward, ids)

    @abc.abstractmethod
    def _batched_call_per_step(self, is_first, is_last, is_boundary, reward, ids):
        """Update np storage state given the active `ids` at each time step
        Input:
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
    def __init__(self, num_envs, env_batch_size, buffer_size=None, name='AverageReturn'):
        super(AverageReturnMetric, self).__init__(
            name, num_envs, env_batch_size, buffer_size)

    def _reset(self, num_envs):
        """Resets stat gathering variables"""
        self._np_state.episode_return = np.zeros(
            shape=(num_envs,), dtype=np.float64)

    def _batched_call_per_step(self, is_first, is_last, is_boundary, reward, ids):
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
    def __init__(self, num_envs, env_batch_size, buffer_size=None, name='AverageEpisodeLength'):
        super(AverageEpisodeLengthMetric, self).__init__(
            name, num_envs, env_batch_size, buffer_size)

    def _reset(self, num_envs):
        """Resets stat gathering variables"""
        self._np_state.episode_steps = np.zeros(
            shape=(num_envs,), dtype=np.float64)

    def _batched_call_per_step(self, is_first, is_last, is_boundary, reward, ids):
        episode_steps = self._np_state.episode_steps
        episode_steps[ids] += 1 - is_boundary
        self.add_to_buffer(episode_steps[ids][np.where(is_last)])
        episode_steps[ids] *= ~is_last




