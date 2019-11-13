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
from alf.utils.data_buffer import DataBuffer
from alf.utils.nest_utils import get_nest_batch_size
from alf.utils.conditional_ops import run_if


class ReservoirSampler(DataBuffer):
    """
    A reservoir sampler for keeping K samples without replacement in a single
    pass of the time horizon (O(1) complexity at every time step). See wiki for
    its definition details:
    https://en.wikipedia.org/wiki/Reservoir_sampling

    Suppose the maximal time step is T, usually a reservoir sampler will keep
    every item with the same probability of K/T. Here we make some changes so that
    each item t can be kept with a probability roughly in proportional to
    t^(s-1), where s is a positive integer speed defined by the user.

    You can use this sampler as a replay buffer. One big difference between this
    sampler and a uniform replay buffer is that samples overflown by a replay
    buffer will never be sampled again (probability hard cutting-off).

    Probability computation:
    - The probability of the t-th sample being selected in the set is s*K/t
    - The probability of the t-th sample not being replaced by the sample at time
      t+m is (1 - s*K/(t+m) * 1/K) = (t+m-s) / (t+m)
    - The over probability of the t-th sample still being kept in the set at time
      T is

      s*K/t * (t+1-s)/(t+1) * (t+2-s)/(t+2) * ... * (T-s)/T

      which is proportional to t(t-1)..(t-s+1) / ((T-1)(T-2)...(T-s+1)) and is
      roughly proportional to t^(s-1), when t,T >> s*K
    """

    def __init__(self,
                 data_spec: tf.TensorSpec,
                 K,
                 speed=1,
                 name="ReservoirSampler"):
        """
        Create a reservoir sampler.

        Args:
            data_spec (nested TensorSpec): spec for the data item
            K (int): the size of the reservoir set
            speed (int): a bigger speed results in a faster decay of old samples.
                When `speed` is 1, it results in the original reservoir sampling.
            name (str): name of the sampler
        """
        super(ReservoirSampler, self).__init__(
            data_spec=data_spec, capacity=K, name=name)
        assert isinstance(K, int) and K > 0
        assert isinstance(speed, int) and speed > 0
        self._s = speed
        self._K = K
        self._t = tf.Variable(
            tf.ones((), tf.float32), trainable=False, name="t")

    def _add_batch(self, batch):
        """
        Add a batch of samples to the buffer. When the buffer is not full, add
        whatever many samples to the buffer until full. Once it's full, the samples
        in the buffer will be randomly replaced.

        Args:
            batch (nested Tensor): shape should be [batch_size] + data_spec.shape
        """
        batch_size = get_nest_batch_size(batch, tf.int32)
        buffer_space_left = self._capacity - self._current_size
        batch_replacing_size = batch_size - buffer_space_left

        # Split the batch into two: the first one (if exists) will be directly
        # added into the buffer; the second one (if exists) will randomly replace
        # samples in the buffer
        def _fill_buffer():
            data_filling_buffer = tf.nest.map_structure(
                lambda bat: bat[:buffer_space_left], batch)
            super(ReservoirSampler, self).add_batch(data_filling_buffer)

        def _replace_buffer():
            # Sample the slots to be replaced in the set
            replace_indices = tf.random.uniform(
                shape=(batch_replacing_size, ),
                dtype=tf.int32,
                minval=0,
                maxval=self._capacity)
            # replace_indices can contain duplicates; sort the indices and the
            # corresponding batch data
            sort_idx = tf.argsort(replace_indices)
            replace_indices = tf.sort(replace_indices)
            data_replacing_buffer = tf.nest.map_structure(
                lambda bat: tf.gather(
                    bat[-batch_replacing_size:], sort_idx, axis=0), batch)

            # For duplicate indices, we only need to keep the last one
            unique_indices, _, count = tf.unique_with_counts(replace_indices)
            last = tf.cumsum(count) - 1
            unique_indices = tf.expand_dims(unique_indices, axis=-1)

            tf.nest.map_structure(
                lambda buf, bat: buf.scatter_nd_update(
                    unique_indices,
                    tf.stop_gradient(tf.gather(bat, last, axis=0))),
                self._buffer, data_replacing_buffer)

        run_if(buffer_space_left > 0, _fill_buffer)
        run_if(batch_replacing_size > 0, _replace_buffer)

    def add_batch(self, batch):
        """
        Add a batch with certain probabilities to the buffer. For every sample
        in the batch, the probability is computed as:
        p = min(s*K/(t+i), 1), where i is the sample index in the batch.

        Note that the entire batch will advance the time step by batch_size

        Args:
            batch (nest Tensor): shape should be [batch_size] + data_spec.shape
        """
        batch_size = get_nest_batch_size(batch, tf.int32)
        ps = tf.random.uniform(
            shape=(batch_size, ), dtype=tf.float32, maxval=1.0)
        batch_size = tf.cast(batch_size, tf.float32)

        ts = tf.range(self._t, self._t + batch_size, dtype=tf.float32)
        threshold = self._s * self._K / ts
        selected_idx = tf.reshape(tf.where(ps < threshold), [-1])
        # the unselected samples will be discarded
        selected_batch = tf.nest.map_structure(
            lambda bat: tf.gather(bat, selected_idx, axis=0), batch)

        self._add_batch(selected_batch)
        self._t.assign_add(batch_size)

    def clear(self):
        """Reset the sampler status and clear the reservoir set."""
        self._t.assign(1)
        super(ReservoirSampler, self).clear()
