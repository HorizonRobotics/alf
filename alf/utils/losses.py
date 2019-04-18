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

from tf_agents.utils import common as tfa_common


def element_wise_huber_loss(x, y):
    """Elementwise Huber loss
    Args:
        x (tf.Tensor): label
        y (tf.Tensor): prediction
    Returns:
        loss (tf.Tensor)
    """
    return tf.compat.v1.losses.huber_loss(
        x, y, reduction=tf.compat.v1.losses.Reduction.NONE)


def element_wise_squared_loss(x, y):
    """Elementwise squared loss
    Args:
        x (tf.Tensor): label
        y (tf.Tensor): prediction
    Returns:
        loss (tf.Tensor)
    """
    return tf.compat.v1.losses.mean_squared_error(
        x, y, reduction=tf.compat.v1.losses.Reduction.NONE)
