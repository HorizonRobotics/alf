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

from tf_agents.utils.tensor_normalizer import EMATensorNormalizer
from tf_agents.specs.tensor_spec import TensorSpec


class AdaptiveNormalizer(EMATensorNormalizer):
    def __init__(self,
                 tensor_spec,
                 speed=2.0,
                 auto_update=True,
                 variance_epsilon=1e-3):
        """Create reward normalizer

        This normalizer gives higher weight to more recent samples for
        calculating mean and variance. Roughly speaking, the weight for each
        sample at time t is proportional to (t/T)^(speed-1), where T is the
        current time step.

        Args:
            tensor_spec (TensorSpec): spec of the mean of tensors to be
              normlized.
            speed (float): speed of updating mean and variance.
            auto_update (bool): If True, automatically update mean and variance
              for each call to `normalize()`. Otherwise, the user needs to call
              `update()`
        """
        self._total_env_steps = tf.Variable(
            int(speed), dtype=tf.int64, trainable=False)
        self._update_ema_rate = tf.Variable(
            1.0, dtype=tf.float32, trainable=False)
        self._speed = speed
        self._auto_update = auto_update
        self._variance_epsilon = variance_epsilon
        super(AdaptiveNormalizer, self).__init__(
            tensor_spec, norm_update_rate=self._update_ema_rate)

    def normalize(self, tensors):
        """Normalized the reward

        Args:
            tensors (nested Tensor): tensors to be normalized
        Returns:
            normalized reward. with mean equal to 0 and variance equal to 1
              over the time.
        """
        if self._auto_update:
            self.update(tensors)
        n_tensors = super(AdaptiveNormalizer, self).normalize(
            tensors, center_mean=True, variance_epsilon=self._variance_epsilon)
        return n_tensors

    def update(self, tensors):
        self._update_ema_rate.assign(
            self._speed / tf.cast(self._total_env_steps, tf.float32))
        self._total_env_steps.assign_add(1)
        super(AdaptiveNormalizer, self).update(tensors)


class ScalarAdaptiveNormalizer(AdaptiveNormalizer):
    def __init__(self, speed=2.0, auto_update=True, variance_epsilon=1e-10):
        super(ScalarAdaptiveNormalizer, self).__init__(
            tensor_spec=TensorSpec((), dtype=tf.float32),
            speed=speed,
            auto_update=auto_update,
            variance_epsilon=variance_epsilon)
