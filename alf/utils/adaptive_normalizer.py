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

from tf_agents.specs.tensor_spec import TensorSpec
from tf_agents.utils.nest_utils import get_outer_rank
from tf_agents.networks.utils import BatchSquash

from alf.utils.averager import AdaptiveAverager


class AdaptiveNormalizer(tf.Module):
    def __init__(self,
                 tensor_spec,
                 speed=2.0,
                 auto_update=True,
                 variance_epsilon=1e-8,
                 name="AdaptiveNormalizer"):
        """Create an adaptive normalizer.

        This normalizer gives higher weight to more recent samples for
        calculating mean and variance. Roughly speaking, the weight for each
        sample at time t is proportional to (t/T)^(speed-1), where T is the
        current time step. See docs/streaming_averaging_amd_sampling.py for
        detail.

        Given weights w_i and samples x_i, i = 1...n, let

        m   = \sum_i w_i * x_i
        m2  = \sum_i w_i * x_i^2

        then

        var = \sum_i w_i * (x_i - m)^2
            = \sum_i w_i * (x_i^2 + m^2 - 2*x_i*m)
            = m2 + m^2 - 2m^2
            = m2 - m^2

        which is the same result with the case when w_1=w_2=...=w_n=(1/n)

        Args:
            tensor_spec (nested TensorSpec): specs of the mean of tensors to be
              normalized.
            speed (float): speed of updating mean and variance.
            auto_update (bool): If True, automatically update mean and variance
              for each call to `normalize()`. Otherwise, the user needs to call
              `update()`
            variance_epislon (float): a small value added to std for normalizing
            name (str):
        """
        self._mean_averager = AdaptiveAverager(tensor_spec, speed=speed)
        self._m2_averager = AdaptiveAverager(tensor_spec, speed=speed)
        self._auto_update = auto_update
        self._variance_epsilon = variance_epsilon
        self._tensor_spec = tensor_spec

    def update(self, tensor):
        self._mean_averager.update(tensor)
        sqr_tensor = tf.nest.map_structure(tf.square, tensor)
        self._m2_averager.update(sqr_tensor)

    def normalize(self, tensor, clip_value=-1.0):
        """
        Normalize a tensor with mean and variance

        Args:
            tensor (nested tf.Tensor): each leaf can have arbitrary outer dims
                with shape [B1, B2,...] + tensor_spec.shape.
            clip_value (float): if positive, normalized values will be clipped to
                [-clip_value, clip_value].

        Returns:
            normalized tensor
        """
        if self._auto_update:
            self.update(tensor)

        def _normalize(m, m2, spec, t):
            # in some extreme cases, due to floating errors, var might be a very
            # large negative value (close to 0)
            var = tf.nn.relu(m2 - tf.square(m))
            outer_dims = get_outer_rank(t, spec)
            batch_squash = BatchSquash(outer_dims)
            t = batch_squash.flatten(t)
            t = tf.nn.batch_normalization(
                t,
                m,
                var,
                offset=None,
                scale=None,
                variance_epsilon=self._variance_epsilon)
            if clip_value > 0:
                t = tf.clip_by_value(t, -clip_value, clip_value)
            t = batch_squash.unflatten(t)
            return t

        return tf.nest.map_structure(_normalize, self._mean_averager.get(),
                                     self._m2_averager.get(),
                                     self._tensor_spec, tensor)


class ScalarAdaptiveNormalizer(AdaptiveNormalizer):
    def __init__(self,
                 speed=2.0,
                 auto_update=True,
                 variance_epsilon=1e-8,
                 name="ScalarAdaptiveNormalizer"):
        super(ScalarAdaptiveNormalizer, self).__init__(
            tensor_spec=TensorSpec((), dtype=tf.float32),
            speed=speed,
            auto_update=auto_update,
            variance_epsilon=variance_epsilon,
            name=name)
