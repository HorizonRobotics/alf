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
"""Various layers."""

import six

import tensorflow as tf


class ListWrapper(list):
    """A wrapper of list.

    It makes it possible to add some attributes to the object which is
    necessary for keras Layer.
    """

    pass


class Split(tf.keras.layers.Layer):
    """Splits a tensor into sub tensors."""

    def __init__(self, num_or_size_splits, axis):
        """Create a Split layer.

        If `num_or_size_splits` is an integer, then `input` is split along
        dimension `axis` into `num_split` smaller tensors. This requires that
        `num_split` evenly divides `value.shape[axis]`. If `num_or_size_splits`
        is a 1-D Tensor (or list), we call it `size_splits` and `input` is split
        into `len(size_splits)` elements. The shape of the `i`-th element has
        the same size as the `value` except along dimension `axis` where the
        size is `size_splits[i]`.

        Args:
            num_or_size_splits (int|list[int]): Either an integer indicating the
                number of splits along split_dim or a 1-D integer `Tensor` or Python
                list containing the sizes of each output tensor along split_dim. If 
                a scalar then it must evenly divide `input.shape[axis]`; otherwise 
                the sum of sizes along the split dimension must match that of the
                `input`.
            axis: An integer or scalar `int32` `Tensor`. The dimension along which
                to split. Must be in the range `[-rank(input), rank(input))`.
        """
        self._num_or_size_splits = num_or_size_splits
        self._axis = axis
        super(Split, self).__init__()

    def call(self, input):
        """Split `input`."""
        ret = ListWrapper(
            tf.split(
                input,
                num_or_size_splits=self._num_or_size_splits,
                axis=self._axis))
        ret._keras_mask = None
        return ret

    def compute_output_shape(self, input_shape):
        """Compute output_shape given the `input_shape`."""
        if isinstance(self._num_or_size_splits, six.integer_types):
            shape = input_shape.as_list()
            shape[self._axis] = shape[self._axis] / self._num_or_size_splits
            shape = tf.TensorShape(shape)
            return [shape] * self._num_or_size_splits
        else:
            shape = input_shape.as_list()
            shapes = []
            for size in self._num_or_size_splits:
                shape[self._axis] = size
                shapes.append(tf.TensorShape(shape))
            return shapes

    def compute_mask(self, inputs, mask=None):
        """Compute output_mask."""
        if isinstance(self._num_or_size_splits, six.integer_types):
            return [None] * self._num_or_size_splits
        else:
            return [None] * len(self._num_or_size_splits)
