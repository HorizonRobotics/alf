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

import six
import abc
import tensorflow as tf
import gin.tf
from alf.utils.common import flatten_once
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
"""
NOTE: The APIs in this file are subject to changes when we implement generic replay
buffers for off-policy drivers in the future.
"""


@six.add_metaclass(abc.ABCMeta)
class ExperienceReplayer(object):
    """
    Base class for implementing experience storing and replay. A subclass should
    implement the abstract functions. This class object will be used by OffPolicyDrivers
    after training data are accumulated for a certain amount of time.
    """

    @abc.abstractmethod
    def observe(self, exp, env_ids=None):
        """
        Observe a batch of `exp`, potentially storing it to replay buffers.

        Args:
            exp (Experience): each item has the shape of (`num_envs`, `env_batch_size`,
                `unroll_length`, ...), where `num_envs` is the number of tf_agents
                *batched* environments, each of which contains `env_batch_size`
                independent & parallel single environments.

            env_ids (tf.tensor): if not None, has the shape of (`num_envs`). Each
                element of `env_ids` indicates which batched env the data come from.
        """

    @abc.abstractmethod
    def replay(self, sample_batch_size, mini_batch_length):
        """Replay experiences from buffers

        Args:
            sample_batch_size (int): A batch size to specify the number of items to
                return. A batch of `sample_batch_size` items is returned, where each
                tensor in items will have its first dimension equal to sample_batch_size
                and the rest of the dimensions match the corresponding data_spec.
            mini_batch_length (int): the temporal length of each sample

        Output:
            exp (Experience): each item has the shape (`sample_batch_size`,
                `mini_batch_length`, ...)
        """

    @abc.abstractmethod
    def replay_all(self):
        """Replay all experiences

        Output:
            exp (Experience): each item has the shape (`full_batch_size`,
                `buffer_length`, ...)
        """

    @abc.abstractmethod
    def clear(self):
        """Clear all buffers"""


@gin.configurable
class OnetimeExperienceReplayer(ExperienceReplayer):
    """
    A simple one-time experience replayer. For each incoming `exp`,
    it stores it with a temporary variable which is used for training
    only once.

    Example algorithms: IMPALA, PPO2
    """

    def __init__(self, experience_spec, batch_size):
        self._experience = None

    def observe(self, exp, env_ids):
        # flatten the shape (num_envs, env_batch_size)
        self._experience = tf.nest.map_structure(flatten_once, exp)

    def replay(self, sample_batch_size, mini_batch_length):
        raise NotImplementedError()  # Only supports replaying all!

    def replay_all(self):
        return self._experience

    def clear(self):
        self._experience = None


@gin.configurable
class SyncUniformExperienceReplayer(ExperienceReplayer):
    """
    For synchronous off-policy training.

    Example algorithms: DDPG, SAC
    """

    def __init__(self, experience_spec, batch_size):
        self._buffer = TFUniformReplayBuffer(experience_spec, batch_size)
        self._data_iter = None

    def observe(self, exp, env_ids=None):
        """
        For the sync driver, `exp` has the shape (`env_batch_size`, ...)
        with `num_envs`==1 and `unroll_length`==1. This function always ignores
        `env_ids`.
        """
        self._buffer.add_batch(exp)

    def replay(self, sample_batch_size, mini_batch_length):
        if self._data_iter is None:
            dataset = self._buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=sample_batch_size,
                num_steps=mini_batch_length).prefetch(3)
            self._data_iter = iter(dataset)
        return next(self._data_iter)

    def replay_all(self):
        return self._buffer.gather_all()

    def clear(self):
        self._buffer.clear()
