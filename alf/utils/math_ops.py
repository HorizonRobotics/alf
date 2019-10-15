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

import gin
import tensorflow as tf
"""Various math ops."""


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


def smart_add(x, y):
    """Add two Tensors which may be None or ().

     If x or y is None, the other tensor is returned.
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
