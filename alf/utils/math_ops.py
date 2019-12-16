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
"""Various math ops."""

import gin
import tensorflow as tf


@gin.configurable
def clipped_exp(value, clip_value_min=-20, clip_value_max=2):
    """ Clip value to the range [`clip_value_min`, `clip_value_max`]
    then compute exponential

    Args:
         value (Tensor): input tensor.
         clip_value_min (float): The minimum value to clip by.
         clip_value_max (float): The maximum value to clip by.
    """
    value = tf.clip_by_value(value, clip_value_min, clip_value_max)
    return tf.exp(value)


def add_ignore_empty(x, y):
    """Add two Tensors which may be None or ().

     If x or y is None, they are assumed to be zero and the other tensor is
     returned.

     Args:
          x (Tensor|None|()):
          y (Tensor(|None|())):
     Returns:
          x + y
     """

    def _ignore(t):
        return t is None or (isinstance(t, tuple) and len(t) == 0)

    if _ignore(y):
        return x
    elif _ignore(x):
        return y
    else:
        return x + y


@gin.configurable
def swish(x):
    """Swish activation.

    This is suggested in arXiv:1710.05941

    Args:
        x (Tensor): input
    Returns:
        Tensor
    """
    return x * tf.math.sigmoid(x)


def max_n(inputs):
    """Calculate the maximum of n Tensors

    Args:
        inputs (list[Tensor]): list of Tensors, should have the same shape
    Returns:
        the elementwise maximum of all the tensors in `inputs`
    """
    ret = inputs[0]
    inputs = inputs[1:]
    for x in inputs:
        ret = tf.maximum(ret, x)
    return ret


def shuffle(values, seed=None):
    """Shuffle with gradient defined.

    tf.random.shuffle() has no gradient defined for `value`.  This `shuffle`
    can propagate gradient of `value` by using `gather`.

    Args:
        values (nested Tensor): nested Tensor to be shuffled. All the tensor
            need to have the same batch size (i.e. shape[0]).
        seed (int): Used to create a random seed for the distribution.
    Returns:
        shuffled value along dimension 0.
    """
    batch_size = tf.shape(tf.nest.flatten(values)[0])[0]
    indices = tf.random.shuffle(tf.range(batch_size), seed=seed)
    return tf.nest.map_structure(lambda value: tf.gather(value, indices),
                                 values)
