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
"""Classes for doing moving average."""

import gin
import tensorflow as tf


@gin.configurable
class EMAverager(tf.Module):
    """Class for exponential moving average.

    x_t = (1-update_rate)* x_{t-1} + update_Rate * x
    The average is corrected by a mass as x_t / mass_t, and the mass is
    calculated as:
    mass_t = (1-update_rate) * mass_{t-1} + update_rate

    Note that update_rate can be a fixed floating number or a Variable. If it is
    a Variable, the update_rate can be changed by the user.
    """

    def __init__(self, tensor_spec: tf.TensorSpec, update_rate):
        """Create an EMAverager.

        Args:
            tensor_spec (TensorSpec): the TensorSpec for the value to be
                averaged
            update_rate (float|Variable): the update rate
        """
        super().__init__()
        self._tensor_spec = tensor_spec
        self._update_rate = update_rate
        self._average = tf.Variable(
            initial_value=tf.zeros(tensor_spec.shape, tensor_spec.dtype),
            trainable=False,
            dtype=tensor_spec.dtype)
        self._mass = tf.Variable(
            initial_value=0, trainable=False, dtype=tf.float64)

    def update(self, tensor):
        """Update the average.

        Args:
            tensor (Tensor): a value for updating the average
        Returns:
            None
        """
        self._average.assign_add(
            tf.cast(self._update_rate, tensor.dtype) *
            (tensor - self._average))
        self._mass.assign_add(
            tf.cast(self._update_rate, tf.float64) * (1 - self._mass))

    def get(self):
        """Get the current average.

        Returns:
            Tensor: the current average
        """
        return self._average / tf.cast(self._mass, dtype=self._average.dtype)


@gin.configurable
class ScalarEMAverager(EMAverager):
    """EMAverager for scalar value"""

    def __init__(self, update_rate, dtype=tf.float32):
        """Create a ScalarEMAverager.

        Args:
            udpate_rate (float|Variable): update rate
            dtype (tf.dtype): dtype of the scalar
        """
        super().__init__(
            tensor_spec=tf.TensorSpec(shape=(), dtype=dtype),
            update_rate=update_rate)


@gin.configurable
class AdaptiveAverager(EMAverager):
    """Averager with adaptive update_rate.

    This averager gives higher weight to more recent samples for calculating the
    average. Roughly speaking, the weight for each sample at time t is roughly
    proportional to (t/T)^(speed-1), where T is the current time step.
    """

    def __init__(self, tensor_spec: tf.TensorSpec, speed=10.):
        """Create an AdpativeAverager.

        Args:
            tensor_spec (TensorSpec): the TensorSpec for the value to be
                averaged
            speed (float): speed of updating mean and variance.
        """
        update_rate = tf.Variable(1.0, dtype=tf.float64, trainable=False)
        super().__init__(tensor_spec, update_rate)
        self._update_ema_rate = update_rate
        self._total_steps = tf.Variable(
            int(speed), dtype=tf.int64, trainable=False)
        self._speed = speed

    def update(self, tensor):
        """Update the average.

        Args:
            tensor (Tensor): a value for updating the average
        Returns:
            None
        """
        self._update_ema_rate.assign(
            self._speed / tf.cast(self._total_steps, tf.float64))
        self._total_steps.assign_add(1)
        super().update(tensor)


@gin.configurable
class ScalarAdaptiveAverager(AdaptiveAverager):
    """AdaptiveAverager for scalar value."""

    def __init__(self, speed=10, dtype=tf.float32):
        """Create a ScalarAdpativeAverager.

        Args:
            speed (float): speed of updating mean and variance.
            dtype (tf.dtype): dtype of the scalar
        """
        super().__init__(
            tensor_spec=tf.TensorSpec(shape=(), dtype=dtype), speed=speed)
