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

import torch
import torch.nn as nn

import alf
from alf.tensor_specs import TensorSpec
from alf.utils.data_buffer import DataBuffer
from alf.nest.utils import get_outer_rank


def average_outer_dims(tensor, spec):
    """
    Args:
        tensor (Tensor): a single Tensor
        spec (TensorSpec):

    Returns:
        the average tensor across outer dims
    """
    outer_dims = get_outer_rank(tensor, spec)
    return tensor.mean(dim=list(range(outer_dims)))


@alf.configurable
class WindowAverager(nn.Module):
    def __init__(self,
                 tensor_spec: TensorSpec,
                 window_size,
                 name="WindowAverager"):
        """
        WindowAverager calculate the average of the past ``window_size`` samples.
        Args:
            tensor_spec (nested TensorSpec): the ``TensorSpec`` for the value to be
                averaged
            window_size (int): the size of the window
            name (str): name of this averager
        """
        super().__init__()
        self._name = name
        self._buf = alf.nest.map_structure(
            # Should put data on the default device instead of "cpu"
            lambda spec: DataBuffer(spec, window_size, alf.get_default_device(
            )),
            tensor_spec)
        self._tensor_spec = tensor_spec

    def update(self, tensor):
        """Update the average.

        Args:
            tensor (nested Tensor): value for updating the average; outer dims
                will be averaged first before being added.
        Returns:
            None
        """
        alf.nest.map_structure(
            lambda buf, t, spec: buf.add_batch(
                average_outer_dims(t.detach(), spec).unsqueeze(0)), self._buf,
            tensor, self._tensor_spec)

    def get(self):
        """Get the current average.

        Returns:
            Tensor: the current average
        """

        def _get(buf):
            n = torch.max(buf.current_size,
                          torch.ones_like(buf.current_size)).to(torch.float32)
            return torch.sum(buf.get_all(), dim=0) * (1. / n)

        return alf.nest.map_structure(_get, self._buf)

    def average(self, tensor):
        """Combines ``self.update`` and ``self.get`` in one step. Can be handy
        in practice.

        Args:
            tensor (nested Tensor): a value for updating the average;  outer dims
                will be averaged first before being added
        Returns:
            Tensor: the current average
        """
        self.update(tensor)
        return self.get()


@alf.configurable
class ScalarWindowAverager(WindowAverager):
    """WindowAverager for scalar value"""

    def __init__(self,
                 window_size,
                 dtype=torch.float32,
                 name="ScalarWindowAverager"):
        """

        Args:
            window_size (int): the size of the window
            dtype (torch.dtype): dtype of the scalar
            name (str): name of this averager
        """
        super().__init__(
            tensor_spec=TensorSpec(shape=(), dtype=dtype),
            window_size=window_size,
            name=name)


@alf.configurable
class EMAverager(nn.Module):
    r"""Class for exponential moving average. Suppose the update rate is
    :math:`\alpha`, and the quantity to be averaged is denoted as :math:`x`, then

    .. math::

        x_t = (1-\alpha)x_{t-1} + \alpha x

    The average is corrected by a mass :math:`w_t` as :math:`\frac{x_t}{w_t}``,
    and the mass is calculated as:

    .. math::

        w_t = (1-\alpha) * w_{t-1} + \alpha

    Note that update rate can be a fixed floating number or a variable. If it is
    a variable, the update rate can be changed by the user.
    """

    def __init__(self, tensor_spec: TensorSpec, update_rate,
                 name="EMAverager"):
        """

        Args:
            tensor_spec (nested TensorSpec): the ``TensorSpec`` for the value to be
                averaged
            update_rate (float|Variable): the update rate
            name (str): name of this averager
        """
        super().__init__()
        self._name = name
        self._tensor_spec = tensor_spec
        self._update_rate = update_rate

        var_id = [0]

        def _create_variable(tensor_spec):
            var = tensor_spec.zeros()
            self.register_buffer("_var%s" % var_id[0], var)
            var_id[0] += 1
            return var

        self._average = alf.nest.map_structure(_create_variable, tensor_spec)
        # mass can be shared by different structure elements
        self.register_buffer("_mass", torch.zeros((), dtype=torch.float64))

    def update(self, tensor):
        """Update the average.

        Args:
            tensor (nested Tensor): value for updating the average; outer dims
                will be first averaged before being added to the average
        Returns:
            None
        """
        alf.nest.map_structure(
            lambda average, t, spec: average.add_(
                torch.as_tensor(self._update_rate, dtype=t.dtype) * (
                    average_outer_dims(t.detach(), spec) - average)),
            self._average, tensor, self._tensor_spec)
        self._mass.add_(
            torch.as_tensor(self._update_rate, dtype=torch.float64) *
            (1 - self._mass))

    def get(self):
        """Get the current average.

        Returns:
            Tensor: the current average
        """
        return alf.nest.map_structure(
            lambda average: average / self._mass.clamp(min=self._update_rate).
            to(average.dtype), self._average)

    def average(self, tensor):
        """Combines ``self.update`` and ``self.get`` in one step. Can be handy
        in practice.

        Args:
            tensor (nested Tensor): a value for updating the average; outer dims
                will be first averaged before being added to the average
        Returns:
            Tensor: the current average
        """
        self.update(tensor)
        return self.get()


@alf.configurable
class ScalarEMAverager(EMAverager):
    """EMAverager for scalar value"""

    def __init__(self,
                 update_rate,
                 dtype=torch.float32,
                 name="ScalarEMAverager"):
        """

        Args:
            udpate_rate (float|Variable): update rate
            dtype (torch.dtype): dtype of the scalar
            name (str): name of this averager
        """
        super().__init__(
            tensor_spec=TensorSpec(shape=(), dtype=dtype),
            update_rate=update_rate,
            name=name)


@alf.configurable
class AdaptiveAverager(EMAverager):
    """Averager with adaptive update_rate.

    This averager gives higher weight to more recent samples for calculating the
    average. Roughly speaking, the weight for each sample at time :math:`t` is
    roughly proportional to :math:`(t/T)^{speed-1}`, where :math:`T` is the
    current time step. See ``notes/streaming_averaging_amd_sampling.py`` for
    detail.
    """

    def __init__(self,
                 tensor_spec: TensorSpec,
                 speed=10.,
                 name="AdaptiveAverager"):
        """

        Args:
            tensor_spec (nested TensorSpec): the ``TensorSpec`` for the value to be
                averaged
            speed (float): speed of updating mean and variance.
            name (str): name of this averager
        """
        update_rate = torch.ones((), dtype=torch.float64)
        super().__init__(tensor_spec, update_rate)
        self.register_buffer("_update_ema_rate", update_rate)
        self.register_buffer("_total_steps",
                             torch.as_tensor(int(speed), dtype=torch.int64))
        self._speed = speed

    def update(self, tensor):
        """Update the average.

        Args:
            tensor (nested Tensor): a value for updating the average; outer dims
                will be first averaged before being added to the average
        """
        self._update_ema_rate.fill_(
            self._speed / self._total_steps.to(torch.float64))
        self._total_steps.add_(1)
        super().update(tensor)


@alf.configurable
class ScalarAdaptiveAverager(AdaptiveAverager):
    """AdaptiveAverager for scalar value."""

    def __init__(self,
                 speed=10,
                 dtype=torch.float32,
                 name="ScalarAdaptiveAverager"):
        """

        Args:
            speed (float): speed of updating mean and variance.
            dtype (torch.dtype): dtype of the scalar
            name (str): name of this averager
        """
        super().__init__(
            tensor_spec=TensorSpec(shape=(), dtype=dtype),
            speed=speed,
            name=name)
