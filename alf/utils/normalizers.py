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

from abc import abstractmethod

import torch
import torch.nn as nn

import alf
from alf.nest.utils import get_outer_rank
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops
from alf.utils.averager import WindowAverager, EMAverager, AdaptiveAverager


class Normalizer(nn.Module):
    def __init__(self,
                 tensor_spec,
                 auto_update=True,
                 variance_epsilon=1e-10,
                 name="Normalizer"):
        """Create a base normalizer using a first-moment and a second-moment
        averagers.

        Given weights w_i and samples x_i, i = 1...n, let

        m   = \sum_i w_i * x_i     (first moment)
        m2  = \sum_i w_i * x_i^2   (second moment)

        then

        var = \sum_i w_i * (x_i - m)^2
            = \sum_i w_i * (x_i^2 + m^2 - 2*x_i*m)
            = m2 + m^2 - 2m^2
            = m2 - m^2

        which is the same result with the case when w_1=w_2=...=w_n=(1/n)

        NOTE: tf_agents' normalizer maintains a running average of variance which
            is not correct mathematically, because the estimated variance contains
            early components that don't measure all the current samples.

        Args:
            tensor_spec (nested TensorSpec): specs of the mean of tensors to be
              normalized.
            auto_update (bool): If True, automatically update mean and variance
              for each call to `normalize()`. Otherwise, the user needs to call
              `update()`
            variance_epsilon (float): a small value added to std for normalizing
            name (str):
        """
        super().__init__()
        self._name = name
        self._auto_update = auto_update
        self._variance_epsilon = variance_epsilon
        self._tensor_spec = tensor_spec
        self._mean_averager = self._create_averager()
        self._m2_averager = self._create_averager()

    @abstractmethod
    def _create_averager(self):
        """
        Create an averager. Derived classes must specify what averager to use.
        """
        pass

    def update(self, tensor):
        """Update the statistics given a new tensor.
        """
        self._mean_averager.update(tensor)
        sqr_tensor = alf.nest.map_structure(math_ops.square, tensor)
        self._m2_averager.update(sqr_tensor)

    def normalize(self, tensor, clip_value=-1.0):
        """
        Normalize a tensor with mean and variance

        Args:
            tensor (nested Tensor): each leaf can have arbitrary outer dims
                with shape [B1, B2,...] + tensor_spec.shape.
            clip_value (float): if positive, normalized values will be clipped to
                [-clip_value, clip_value].

        Returns:
            normalized tensor
        """
        if self._auto_update:
            self.update(tensor)

        def _normalize(m, m2, t):
            # in some extreme cases, due to floating errors, var might be a very
            # large negative value (close to 0)
            var = torch.relu(m2 - math_ops.square(m))
            t = alf.layers.normalize_along_batch_dims(
                t, m, var, variance_epsilon=self._variance_epsilon)
            if clip_value > 0:
                t = torch.clamp(t, -clip_value, clip_value)
            return t

        return alf.nest.map_structure(_normalize, self._mean_averager.get(),
                                      self._m2_averager.get(), tensor)


class WindowNormalizer(Normalizer):
    """Normalization according to a recent window of samples.
    """

    def __init__(self,
                 tensor_spec,
                 window_size=1000,
                 auto_update=True,
                 variance_epsilon=1e-10,
                 name="WindowNormalizer"):
        """
        Args:
            tensor_spec (nested TensorSpec): specs of the mean of tensors to be
              normalized.
            window_size (int): the size of the recent window
            auto_update (bool): If True, automatically update mean and variance
              for each call to `normalize()`. Otherwise, the user needs to call
              `update()`
            variance_epislon (float): a small value added to std for normalizing
            name (str):
        """
        self._window_size = window_size
        super(WindowNormalizer, self).__init__(
            tensor_spec=tensor_spec,
            auto_update=auto_update,
            variance_epsilon=variance_epsilon,
            name=name)

    def _create_averager(self):
        """Returns a window averager."""
        return WindowAverager(
            tensor_spec=self._tensor_spec, window_size=self._window_size)


class ScalarWindowNormalizer(WindowNormalizer):
    def __init__(self,
                 window_size=1000,
                 auto_update=True,
                 variance_epsilon=1e-10,
                 name="ScalarWindowNormalizer"):
        super(ScalarWindowNormalizer, self).__init__(
            tensor_spec=TensorSpec((), dtype='float32'),
            window_size=window_size,
            auto_update=auto_update,
            variance_epsilon=variance_epsilon,
            name=name)


class EMNormalizer(Normalizer):
    """Exponential moving normalizer: the normalization assigns exponentially
    decayed weights to history samples.
    """

    def __init__(self,
                 tensor_spec,
                 update_rate=1e-3,
                 auto_update=True,
                 variance_epsilon=1e-10,
                 name="EMNormalizer"):
        """
        Args:
            tensor_spec (nested TensorSpec): specs of the mean of tensors to be
              normalized.
            update_rate (float): the update rate
            auto_update (bool): If True, automatically update mean and variance
              for each call to `normalize()`. Otherwise, the user needs to call
              `update()`
            variance_epislon (float): a small value added to std for normalizing
            name (str):
        """
        self._update_rate = update_rate
        super(EMNormalizer, self).__init__(
            tensor_spec=tensor_spec,
            auto_update=auto_update,
            variance_epsilon=variance_epsilon,
            name=name)

    def _create_averager(self):
        """Returns an exponential moving averager."""
        return EMAverager(self._tensor_spec, self._update_rate)


class ScalarEMNormalizer(EMNormalizer):
    def __init__(self,
                 update_rate=1e-3,
                 auto_update=True,
                 variance_epsilon=1e-10,
                 name="ScalarEMNormalizer"):
        super(ScalarEMNormalizer, self).__init__(
            tensor_spec=TensorSpec((), dtype='float32'),
            update_rate=update_rate,
            auto_update=auto_update,
            variance_epsilon=variance_epsilon,
            name=name)


class AdaptiveNormalizer(Normalizer):
    def __init__(self,
                 tensor_spec,
                 speed=8.0,
                 auto_update=True,
                 variance_epsilon=1e-10,
                 name="AdaptiveNormalizer"):
        """This normalizer gives higher weight to more recent samples for
        calculating mean and variance. Roughly speaking, the weight for each
        sample at time t is proportional to (t/T)^(speed-1), where T is the
        current time step. See docs/streaming_averaging_amd_sampling.py for
        detail.

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
        self._speed = speed
        super(AdaptiveNormalizer, self).__init__(
            tensor_spec=tensor_spec,
            auto_update=auto_update,
            variance_epsilon=variance_epsilon,
            name=name)

    def _create_averager(self):
        """Create an adaptive averager."""
        return AdaptiveAverager(
            tensor_spec=self._tensor_spec, speed=self._speed)


class ScalarAdaptiveNormalizer(AdaptiveNormalizer):
    def __init__(self,
                 speed=8.0,
                 auto_update=True,
                 variance_epsilon=1e-10,
                 name="ScalarAdaptiveNormalizer"):
        super(ScalarAdaptiveNormalizer, self).__init__(
            tensor_spec=TensorSpec((), dtype='float32'),
            speed=speed,
            auto_update=auto_update,
            variance_epsilon=variance_epsilon,
            name=name)
