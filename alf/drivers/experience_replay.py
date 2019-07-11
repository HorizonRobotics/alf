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
from alf.drivers.threads import flatten_once
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer


@six.add_metaclass(abc.ABCMeta)
class ExperienceReplayer(object):
    """
    Base class for implementing experience storing and replay. A subclass
    should implement two abstract functions: `observe` and `replay`.
    This class object will be used by OffPolicyAsyncDriver after training
    data are dequeued from the learning queue.
    """

    @abc.abstractmethod
    def observe(self, exp, env_ids):
        """
        Observe a batch of `exp`, potentially storing it to replay buffers.
        `exp` has the shape of (`num_envs`, `env_batch_size`, `unroll_length`, ...)
        `env_ids` has the shape of (`num_envs`),
        where `num_envs` is the number of tf_agents *batched* environments, each of
        which contains `env_batch_size` independent & parallel single environments.

        Each element of `env_ids` indicates which batched env the data come from.
        """

    @abc.abstractmethod
    def replay(self, sample_batch_size, mini_batch_length):
        """Replay experiences from buffers"""

    @abc.abstractmethod
    def replay_all(self):
        """Replay all experiences"""

    @abc.abstractmethod
    def clear(self):
        """Clear buffers"""


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
    """

    def __init__(self, experience_spec, batch_size):
        self._buffer = TFUniformReplayBuffer(experience_spec, batch_size)

    def observe(self, exp, env_ids=None):
        self._buffer.add_batch(exp)

    def replay(self, sample_batch_size, mini_batch_length):
        return self._buffer.get_next(
            sample_batch_size=sample_batch_size, num_steps=mini_batch_length)

    def replay_all(self):
        return self._buffer.gather_all()

    def clear(self):
        self._buffer.clear()
