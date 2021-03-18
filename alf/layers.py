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
"""Some basic layers."""

import copy

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import alf
from alf.initializers import variance_scaling_init
from alf.nest.utils import get_outer_rank, NestConcat, NestMultiply, NestSum
from alf.nest import map_structure, get_field
from alf.tensor_specs import TensorSpec
from alf.utils import common
from alf.utils.math_ops import identity


def normalize_along_batch_dims(x, mean, variance, variance_epsilon):
    """Normalizes a tensor by ``mean`` and ``variance``, which are expected to have
    the same tensor spec with the inner dims of ``x``.

    Args:
        x (Tensor): a tensor of (``[D1, D2, ..] + shape``), where ``D1``, ``D2``, ..
            are arbitrary leading batch dims (can be empty).
        mean (Tensor): a tensor of ``shape``
        variance (Tensor): a tensor of ``shape``
        variance_epsilon (float): A small float number to avoid dividing by 0.
    Returns:
        Normalized tensor.
    """
    spec = TensorSpec.from_tensor(mean)
    assert spec == TensorSpec.from_tensor(variance), \
        "The specs of mean and variance must be equal!"

    bs = BatchSquash(get_outer_rank(x, spec))
    x = bs.flatten(x)

    variance_epsilon = torch.as_tensor(variance_epsilon).to(variance.dtype)
    inv = torch.rsqrt(variance + variance_epsilon)
    x = (x - mean.to(x.dtype)) * inv.to(x.dtype)

    x = bs.unflatten(x)
    return x


class Identity(nn.Module):
    """A layer that simply returns its argument as result."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Cast(nn.Module):
    """A layer that cast the dtype of the elements of the input tensor."""

    def __init__(self, dtype=torch.float32):
        """
        Args:
            dtype (torch.dtype): desired type of the new tensor.
        """
        super().__init__()
        self._dtype = dtype

    def forward(self, x):
        return x.to(self._dtype)


class Transpose(nn.Module):
    """A layer that perform the transpose of channels.

    Note that batch dimention is not considered for transpose. This means that
    dim0=0 means the dimension after batch dimension.
    """

    def __init__(self, dim0=0, dim1=1):
        """
        Args:
            dim0 (int): the first dimension to be transposed.
            dim1 (int): the second dimension to be transposed
        """
        super().__init__()
        if dim0 >= 0:
            dim0 += 1
        self._dim0 = dim0
        if dim1 >= 0:
            dim1 += 1
        self._dim1 = dim1

    def forward(self, x):
        return x.transpose(self._dim0, self._dim1)


class Permute(nn.Module):
    """A layer that perform the permutation of channels."""

    def __init__(self, *dims):
        """
        Args:
            *dims: The desired ordering of dimensions (not including batch dimension)
        """
        super().__init__()
        assert all([d >= 0 for d in dims
                    ]), ("dims should be non-negative. Got %s" % str(dims))
        dims = [1 + d for d in dims]
        self._dims = [0] + dims

    def forward(self, x):
        return x.permute(*self._dims)


class BatchSquash(object):
    """Facilitates flattening and unflattening batch dims of a tensor. Copied
    from `tf_agents`.

    Exposes a pair of matched flatten and unflatten methods. After flattening
    only 1 batch dimension will be left. This facilitates evaluating networks
    that expect inputs to have only 1 batch dimension.
    """

    def __init__(self, batch_dims):
        """Create two tied ops to flatten and unflatten the front dimensions.

        Args:
            batch_dims (int): Number of batch dimensions the flatten/unflatten
                ops should handle.

        Raises:
            ValueError: if batch dims is negative.
        """
        if batch_dims < 0:
            raise ValueError('Batch dims must be non-negative.')
        self._batch_dims = batch_dims
        self._original_tensor_shape = None

    def flatten(self, tensor):
        """Flattens and caches the tensor's batch_dims."""
        if self._batch_dims == 1:
            return tensor
        self._original_tensor_shape = tensor.shape
        return torch.reshape(tensor,
                             (-1, ) + tuple(tensor.shape[self._batch_dims:]))

    def unflatten(self, tensor):
        """Unflattens the tensor's batch_dims using the cached shape."""
        if self._batch_dims == 1:
            return tensor

        if self._original_tensor_shape is None:
            raise ValueError('Please call flatten before unflatten.')

        return torch.reshape(
            tensor, (tuple(self._original_tensor_shape[:self._batch_dims]) +
                     tuple(tensor.shape[1:])))


@alf.configurable
class OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self._num_classes = num_classes

    def forward(self, input):
        return nn.functional.one_hot(
            input, num_classes=self._num_classes).to(torch.float32)


@alf.configurable
class FixedDecodingLayer(nn.Module):
    """A layer that uses a set of fixed basis for decoding the inputs."""

    def __init__(self,
                 input_size,
                 output_size,
                 basis_type="rbf",
                 sigma=1.,
                 tau=0.5):
        """
        Args:
            input_size (int): the size of input to be decoded, representing the
                number of representation coefficients
            output_size (int): the size of the decoded output
            basis_type (str): the type of basis to be used for decoding
                - "poly": polynomial basis using Vandermonde matrix
                - "cheb": polynomial basis using Chebyshev polynomials
                - "rbf": radial basis functions
                - "haar": Haar wavelet basis
            sigma (float): the bandwidth parameter used for RBF basis.
                If None, a default value of 1. will be used.
            tau (float): a factor for weighting the basis exponentially
                according to the order (``n``) of the basis, i.e., ``tau**n```
        """
        # get the argument list with vals
        self._kwargs = copy.deepcopy(locals())
        self._kwargs.pop('self')
        self._kwargs.pop('__class__')

        super(FixedDecodingLayer, self).__init__()

        assert input_size > 0, "input_size should be at least one"
        assert basis_type in {"poly", "cheb", "rbf", "haar"
                              }, ("the specified method "
                                  "{} is not supported".format(basis_type))

        self._B = nn.Linear(input_size, output_size, bias=False)

        def _polyvander_matrix(n, D, tau=tau):
            # non-square matrix [n, D + 1]
            x = torch.linspace(-1, 1, n)
            B = torch.as_tensor(
                np.polynomial.polynomial.polyvander(x.cpu(), D))
            # weight for encoding the preference to low-frequency basis
            exp_factor = torch.arange(D + 1).float()
            basis_weight = tau**exp_factor
            return B * basis_weight

        def _chebvander_matrix(n, D, tau=tau):
            # non-square matrix [n, D + 1]
            x = np.linspace(-1, 1, n)
            B = torch.as_tensor(np.polynomial.chebyshev.chebvander(x, D))
            # weight for encoding the preference to low-frequency basis
            exp_factor = torch.arange(D + 1).float()
            basis_weight = tau**exp_factor
            return B * basis_weight

        def _rbf_matrix(n, sigma=1.0):
            # square matrix [n, n]
            x = torch.linspace(-1, 1, n)
            B = torch.empty(n, n)
            for d in range(n):
                B[:, d] = torch.exp(-(x - x[d])**2 / sigma)
            return B

        def _haar_matrix(n, tau=tau):
            # square matrix [n, n]
            def _is_power_of_two(x):
                return (x & (x - 1)) == 0

            # allow only size n to be the power of 2
            assert _is_power_of_two(n), "n is required to be the power of 2"

            def _get_haar_matrix(n):
                if n > 2:
                    h = _get_haar_matrix(n // 2)
                else:
                    return torch.Tensor([[1, 1], [1, -1]])

                def _kron(A, B):
                    return torch.einsum("ab,cd->acbd", A, B).view(
                        A.size(0) * B.size(0),
                        A.size(1) * B.size(1))

                # calculate upper haar part
                h_n = _kron(h, torch.Tensor([[1], [1]]))
                # calculate lower haar part
                h_i = torch.sqrt(torch.Tensor([n / 2])) * _kron(
                    torch.eye(len(h)), torch.Tensor([[1], [-1]]))
                # combine both parts
                h = torch.cat((h_n, h_i), dim=1)
                return h

            B = _get_haar_matrix(n) / torch.sqrt(torch.Tensor([n]))
            # weight for encoding the preference to low-frequency basis
            exp_factor = torch.ceil(torch.log2(torch.arange(n).float() + 1))
            basis_weight = tau**exp_factor
            return B * basis_weight

        if basis_type == "poly":
            B = _polyvander_matrix(output_size, input_size - 1)
        elif basis_type == "cheb":
            B = _chebvander_matrix(output_size, input_size - 1)
        elif basis_type == "rbf":
            assert input_size == output_size
            B = _rbf_matrix(input_size, sigma=sigma)
        elif basis_type == "haar":
            assert input_size == output_size
            B = _haar_matrix(input_size)

        # assign the constructed transformation matrix and set it to be non-trainable
        self._B.weight.requires_grad = False
        self._B.weight.copy_(B)

    def forward(self, inputs):
        return self._B(inputs)

    @property
    def weight(self):
        return self._B.weight


@alf.configurable
class FC(nn.Module):
    """Fully connected layer."""

    def __init__(self,
                 input_size,
                 output_size,
                 activation=identity,
                 use_bias=True,
                 use_bn=False,
                 use_ln=False,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A fully connected layer that's also responsible for activation and
        customized weights initialization. An auto gain calculation might depend
        on the activation following the linear layer. Suggest using this wrapper
        module instead of ``nn.Linear`` if you really care about weight std after
        init.
        Args:
            input_size (int): input size
            output_size (int): output size
            activation (torch.nn.functional):
            use_bias (bool): whether use bias
            use_bn (bool): whether use batch normalization.
            use_ln (bool): whether use layer normalization
            kernel_initializer (Callable): initializer for the FC layer kernel.
                If none is provided a ``variance_scaling_initializer`` with gain as
                ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to
                the std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_value (float): a constant
        """
        # get the argument list with vals
        self._kwargs = copy.deepcopy(locals())
        self._kwargs.pop('self')
        self._kwargs.pop('__class__')

        super(FC, self).__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._activation = activation
        self._weight = nn.Parameter(torch.Tensor(output_size, input_size))
        if use_bias:
            self._bias = nn.Parameter(torch.Tensor(output_size))
        else:
            self._bias = None

        self._kernel_initializer = kernel_initializer
        self._kernel_init_gain = kernel_init_gain
        self._bias_init_value = bias_init_value
        self._use_bias = use_bias
        self._use_bn = use_bn
        self._use_ln = use_ln
        if use_bn:
            self._bn = nn.BatchNorm1d(output_size)
        else:
            self._bn = None
        if use_ln:
            self._ln = nn.LayerNorm(output_size)
        else:
            self._ln = None
        self.reset_parameters()

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    def reset_parameters(self):
        """Initialize the parameters."""
        if self._kernel_initializer is None:
            variance_scaling_init(
                self._weight.data,
                gain=self._kernel_init_gain,
                nonlinearity=self._activation)
        else:
            self._kernel_initializer(self._weight.data)

        if self._use_bias:
            nn.init.constant_(self._bias.data, self._bias_init_value)

        if self._use_ln:
            self._ln.reset_parameters()
        if self._use_bn:
            self._bn.reset_parameters()

    def forward(self, inputs):
        """Forward computation.

        Args:
            inputs (Tensor): its shape should be ``[batch_size, input_size]`` or
                ``[batch_size, ..., input_size]``
        Returns:
            Tensor: with shape as ``inputs.shape[:-1] + (output_size,)``
        """
        if inputs.dim() == 2 and self._use_bias:
            y = torch.addmm(self._bias, inputs, self._weight.t())
        else:
            y = inputs.matmul(self._weight.t())
            if self._use_bias:
                y += self._bias
        if self._use_ln:
            if not self._use_bias:
                self._ln.bias.data.zero_()
            y = self._ln(y)
        if self._use_bn:
            if not self._use_bias:
                self._bn.bias.data.zero_()
            y = self._bn(y)
        return self._activation(y)

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias

    def make_parallel(self, n):
        """Create a ``ParallelFC`` using ``n`` replicas of ``self``.
        The initialized layer parameters will be different.
        """
        return ParallelFC(n=n, **self._kwargs)


@alf.configurable
class FCBatchEnsemble(FC):
    r"""The BatchEnsemble for FC layer.

    BatchEnsemble is proposed in `Wen et. al. BatchEnsemble: An Alternative Approach
    to Efficient Ensemble and Lifelong Learning <https://arxiv.org/abs/2002.06715>`_

    In a nutshell, a tuple of vector :math:`(r_k, s_k)` is maintained for ensemble
    member k in addition to the original FC weight matrix w. For input x, the
    result for ensemble member k is calculated as :math:`(W \circ (s_k r_k^T)) x`.
    This can be more efficiently calculated as :math:`(W (x \circ r_k)) \circ s_k`.
    Note that for each sample in a batch, a random ensemble member will used for it
    if ``ensemble_ids`` is not provided to ``forward()``.

    """

    def __init__(self,
                 input_size,
                 output_size,
                 ensemble_size,
                 output_ensemble_ids=True,
                 activation=identity,
                 use_bias=True,
                 use_bn=False,
                 use_ln=False,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_range=0.,
                 ensemble_group=0):
        """
        Args:
            input_size (int): input size
            output_size (int): output size
            ensemble_size (int): ensemble size
            output_ensemble_ids (bool): If True, the forward() function will return
                a tuple of (result, ensemble_ids). If False, the forward() function
                will return result only.
            activation (Callable): activation function
            use_bias (bool): whether use bias
            use_bn (bool): whether use batch normalization.
            use_ln (bool): whether use layer normalization
            kernel_initializer (Callable): initializer for the FC layer kernel.
                If none is provided a ``variance_scaling_initializer`` with gain as
                ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to
                the std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_range (float): biases are initialized uniformly in
                [-bias_init_range, bias_init_range]
            ensemble_group (int): the extra attribute ``ensemble_group`` added
                to ``self._r``, ``self._s``, and ``self._ensemble_bias``,
                default value is 0.
                For alf.optimizers whose ``parvi`` is not ``None``, all parameters
                with the same ``ensemble_group`` will be updated by the
                particle-based VI algorithm specified by ``parvi``, options are
                [``svgd``, ``gfsf``],

                * Stein Variational Gradient Descent (SVGD)

                  Liu, Qiang, and Dilin Wang. "Stein Variational Gradient Descent:
                  A General Purpose Bayesian Inference Algorithm." NIPS. 2016.

                * Wasserstein Gradient Flow with Smoothed Functions (GFSF)

                  Liu, Chang, et al. "Understanding and accelerating particle-based
                  variational inference." ICML, 2019.
        """
        nn.Module.__init__(self)
        self._r = nn.Parameter(torch.Tensor(ensemble_size, input_size))
        self._s = nn.Parameter(torch.Tensor(ensemble_size, output_size))
        assert isinstance(ensemble_group,
                          int), ("ensemble_group has to be an integer!")
        self._ensemble_bias = nn.Parameter(
            torch.Tensor(ensemble_size, output_size))
        self._r.ensemble_group = ensemble_group
        self._s.ensemble_group = ensemble_group
        self._ensemble_bias.ensemble_group = ensemble_group
        self._use_ensemble_bias = use_bias
        self._ensemble_size = ensemble_size
        self._output_ensemble_ids = output_ensemble_ids
        self._bias_init_range = bias_init_range
        super().__init__(
            input_size,
            output_size,
            activation=activation,
            use_bias=False,
            use_bn=use_bn,
            use_ln=use_ln,
            kernel_initializer=kernel_initializer,
            kernel_init_gain=kernel_init_gain)

    def reset_parameters(self):
        """Reinitialize parameters."""
        super().reset_parameters()
        # Both r and s are initialized to +1/-1 according to Appendix B
        torch.randint(
            2, size=self._r.shape, dtype=torch.float32, out=self._r.data)
        torch.randint(
            2, size=self._s.shape, dtype=torch.float32, out=self._s.data)
        self._r.data.mul_(2)
        self._r.data.sub_(1)
        self._s.data.mul_(2)
        self._s.data.sub_(1)
        if self._use_ensemble_bias:
            nn.init.uniform_(
                self._ensemble_bias.data,
                a=-self._bias_init_range,
                b=self._bias_init_range)

    def forward(self, inputs):
        """Forward computation.

        Args:
            inputs (Tensor|tuple): if a Tensor, its shape should be ``[batch_size, input_size]`` or
                ``[batch_size, ..., input_size]``. And a random ensemble id will be
                generated for each sample in the batch. If a tuple, it should
                contain two tensors. The first one is the data tensor with shape
                ``[batch_size, input_size]`` or ``[batch_size, ..., input_size]``.
                The second one is ensemble_ids indicating which ensemble member each
                sample should use. Its shape should be [batch_size], and all elements
                should be in [0, ensemble_size).
        Returns:
            tuple if ``output_ensemble_ids`` is True,
            - Tensor: with shape as ``inputs.shape[:-1] + (output_size,)``
            - LongTensor: if enseble_ids is provided, this is same as ``ensemble_ids``,
                otherwise a randomly generated ensemble_ids is returned
            Tensor if ``output_ensemble_ids`` is False. The result of FC.
        """
        if type(inputs) == tuple:
            inputs, ensemble_ids = inputs
        else:
            ensemble_ids = torch.randint(
                self._ensemble_size, size=(inputs.shape[0], ))
        batch_size = inputs.shape[0]
        output_size, input_size = self._weight.shape
        r = self._r[ensemble_ids]  # [batch_size, input_size]
        s = self._s[ensemble_ids]  # [batch_size, output_size]
        if inputs.ndim > 2:
            ones = [1] * (inputs.ndim - 2)
            r = r.reshape(batch_size, *ones, input_size)
            s = s.reshape(batch_size, *ones, output_size)
        y = (inputs * r).matmul(self._weight.t())
        y = y * s
        if self._use_ensemble_bias:
            bias = self._ensemble_bias[ensemble_ids]
            if inputs.ndim > 2:
                bias = bias.reshape(batch_size, *ones, output_size)
            y += bias
        if self._use_ln:
            if not self._use_ensemble_bias:
                self._ln.bias.data.zero_()
            y = self._ln(y)
        if self._use_bn:
            if not self._use_ensemble_bias:
                self._bn.bias.data.zero_()
            y = self._bn(y)

        y = self._activation(y)
        if self._output_ensemble_ids:
            return y, ensemble_ids
        else:
            return y


@alf.configurable
class ParallelFC(nn.Module):
    """Parallel FC layer."""

    def __init__(self,
                 input_size,
                 output_size,
                 n,
                 activation=identity,
                 use_bias=True,
                 use_bn=False,
                 use_ln=False,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """
        It is equivalent to ``n`` separate FC layers with the same
        ``input_size`` and ``output_size``.

        Args:
            input_size (int): input size
            output_size (int): output size
            n (int): n independent ``FC`` layers
            activation (torch.nn.functional):
            use_bn (bool): whether use Batch Normalization.
            use_ln (bool): whether use layer normalization
            use_bias (bool): whether use bias
            kernel_initializer (Callable): initializer for the FC layer kernel.
                If none is provided a ``variance_scaling_initializer`` with gain
                as ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to
                the std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_value (float): a constant
        """
        super().__init__()
        self._activation = activation
        self._weight = nn.Parameter(torch.Tensor(n, output_size, input_size))
        if use_bias:
            self._bias = nn.Parameter(torch.Tensor(n, output_size))
        else:
            self._bias = None

        for i in range(n):
            if kernel_initializer is None:
                variance_scaling_init(
                    self._weight.data[i],
                    gain=kernel_init_gain,
                    nonlinearity=self._activation)
            else:
                kernel_initializer(self._weight.data[i])

        if use_bias:
            nn.init.constant_(self._bias.data, bias_init_value)
        if use_bn:
            self._bn = nn.BatchNorm1d(n * output_size)
        else:
            self._bn = None

        if use_ln:
            self._ln = nn.GroupNorm(n, n * output_size)
        else:
            self._ln = None

    def forward(self, inputs):
        """Forward

        Args:
            inputs (torch.Tensor): with shape ``[B, n, input_size]`` or ``[B, input_size]``
        Returns:
            torch.Tensor with shape ``[B, n, output_size]``
        """
        n, k, l = self._weight.shape
        if inputs.ndim == 2:
            assert inputs.shape[1] == l, (
                "inputs has wrong shape %s. Expecting (B, %d)" % (inputs.shape,
                                                                  l))
            inputs = inputs.unsqueeze(0).expand(n, *inputs.shape)
        elif inputs.ndim == 3:
            assert (inputs.shape[1] == n and inputs.shape[2] == l), (
                "inputs has wrong shape %s. Expecting (B, %d, %d)" %
                (inputs.shape, n, l))
            inputs = inputs.transpose(0, 1)  # [n, B, l]
        else:
            raise ValueError("Wrong inputs.ndim=%d" % inputs.ndim)

        if self.bias is not None:
            y = torch.baddbmm(
                self._bias.unsqueeze(1), inputs,
                self.weight.transpose(1, 2))  # [n, B, k]
        else:
            y = torch.bmm(inputs, self._weight.transpose(1, 2))  # [n, B, k]
        y = y.transpose(0, 1)  # [B, n, k]
        if self._ln is not None:
            if self._bias is None:
                self._ln.bias.data.zero_()
            y1 = y.reshape(-1, n * k)
            y = self._ln(y1)
            y = y1.view(-1, n, k)
        if self._bn is not None:
            if self._bias is None:
                self._bn.bias.data.zero_()
            y1 = y.reshape(-1, n * k)
            y1 = self._bn(y1)
            y = y1.view(-1, n, k)
        return self._activation(y)

    @property
    def weight(self):
        """Get the weight Tensor.

        Returns:
            Tensor: with shape (n, output_size, input_size). ``weight[i]`` is
                the weight for the i-th FC layer. ``weight[i]`` can be used for
                ``FC`` layer with the same ``input_size`` and ``output_size``
        """
        return self._weight

    @property
    def bias(self):
        """Get the bias Tensor.

        Returns:
            Tensor: with shape (n, output_size). ``bias[i]`` is the bias for the
                i-th FC layer. ``bias[i]`` can be used for ``FC`` layer with
                the same ``input_size`` and ``output_size``
        """
        return self._bias


@alf.configurable
class CausalConv1D(nn.Module):
    """1D (Dilated) Causal Convolution layer.
        1D Dilated Causal Convolution is proposed in `Aaron et. al. WaveNet:
        A generative model for raw audio <https://arxiv.org/abs/1609.03499>`_
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 hide_current=False,
                 activation=torch.relu_,
                 use_bias=None,
                 use_bn=False,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A layer implementing the 1D (Dilated) Causal Convolution.
        It is also responsible for activation and customized weights
        initialization. An auto gain calculation might depend on the activation
        following the causal conv1d layer.

        Note that the main difference of causal conv v.s. standard conv is that
        each temporal element in the convolutional output is causal w.r.t.
        the temporal elements from input. For example, for a length ``L``
        sequence ``x`` with the shape of ``[B, C, L]``, and
        ``y = causal_conv(x)``, where the shape of ``y`` is
        ``[B, C', L]``, by causal we mean ``y[..., l]`` only depends on
        ``X[..., :l]`` (i.e. the past), and there is no dependency on
        ``X[..., l:]`` (i.e. future) as in the standard non-causal
        convolution.

        This can implemented by using an asymmetric padding, which in effect
        shift the input to the right (future) according to kernel size.

        Args:
            in_channels (int): channels of the input
            out_channels (int): channels of the output
            kernel_size (int): size of the kernel
            dilation (int): controls the spacing between the kernel points.
                Please refer to here for a visual illustration:
                https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
            hide_current (bool): whether to hide the current by shifting the
                input to the right (future) by one. This is typically needed
                in the first layer of a causal conv net.
            activation (torch.nn.functional): activation to be applied to output
            use_bias (bool|None): whether use bias. If None, will use ``not use_bn``
            use_bn (bool): whether use batch normalization
            kernel_initializer (Callable): initializer for the conv layer kernel.
                If None is provided a variance_scaling_initializer with gain as
                ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_value (float): a constant
        """
        super(CausalConv1D, self).__init__()
        if use_bias is None:
            use_bias = not use_bn
        self._activation = activation

        # use F.pad for asymmetric padding
        if hide_current:
            assert dilation == 1, "the dilation should be 1 for hiding current"
            asymmetric_padding = (kernel_size, -1)
        else:
            asymmetric_padding = ((kernel_size - 1) * dilation, 0)

        self._pad = partial(
            F.pad, pad=asymmetric_padding, mode='constant', value=0)
        self._causal_conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=use_bias)

        self._kernel_initializer = kernel_initializer
        self._kernel_init_gain = kernel_init_gain
        self._bias_init_value = bias_init_value
        self._use_bias = use_bias
        if use_bn:
            self._bn = nn.BatchNorm1d(out_channels)
        else:
            self._bn = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the parameters."""
        if self._kernel_initializer is None:
            variance_scaling_init(
                self._causal_conv1d.weight.data,
                gain=self._kernel_init_gain,
                nonlinearity=self._activation)
        else:
            self._kernel_initializer(self._causal_conv1d.weight.data)
        if self._use_bias:
            nn.init.constant_(self._causal_conv1d.bias.data,
                              self._bias_init_value)
        if self._bn is not None:
            self._bn.reset_parameters()

    def forward(self, x):
        """
        Args:
            x (tensor): input of the shape [B, C, L] where B is the batch size,
                C denotes the number of input channels, and L is the length of
                the signal.

        Returns:
            A tensor of the shape [B, C', L], where C' denotes the number of
                output channels.
        """

        y = self._causal_conv1d(self._pad(x))
        if self._bn is not None:
            y = self._bn(y)
        return self._activation(y)

    @property
    def weight(self):
        return self._causal_conv1d.weight

    @property
    def bias(self):
        return self._causal_conv1d.bias


@alf.configurable
class Conv2D(nn.Module):
    """2D Convolution Layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=torch.relu_,
                 strides=1,
                 padding=0,
                 use_bias=None,
                 use_bn=False,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A 2D Conv layer that's also responsible for activation and customized
        weights initialization. An auto gain calculation might depend on the
        activation following the conv layer. Suggest using this wrapper module
        instead of ``nn.Conv2d`` if you really care about weight std after init.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            activation (torch.nn.functional):
            strides (int or tuple):
            padding (int or tuple):
            use_bias (bool|None): whether use bias. If None, will use ``not use_bn``
            use_bn (bool): whether use batch normalization
            kernel_initializer (Callable): initializer for the conv layer kernel.
                If None is provided a variance_scaling_initializer with gain as
                ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_value (float): a constant
        """
        super(Conv2D, self).__init__()
        if use_bias is None:
            use_bias = not use_bn
        self._activation = activation
        self._conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=strides,
            padding=padding,
            bias=use_bias)

        self._kernel_initializer = kernel_initializer
        self._kernel_init_gain = kernel_init_gain
        self._bias_init_value = bias_init_value
        self._use_bias = use_bias
        if use_bn:
            self._bn = nn.BatchNorm2d(out_channels)
        else:
            self._bn = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the parameters."""
        if self._kernel_initializer is None:
            variance_scaling_init(
                self._conv2d.weight.data,
                gain=self._kernel_init_gain,
                nonlinearity=self._activation)
        else:
            self._kernel_initializer(self._conv2d.weight.data)
        if self._use_bias:
            nn.init.constant_(self._conv2d.bias.data, self._bias_init_value)
        if self._bn is not None:
            self._bn.reset_parameters()

    def forward(self, img):
        y = self._conv2d(img)
        if self._bn is not None:
            y = self._bn(y)
        return self._activation(y)

    @property
    def weight(self):
        return self._conv2d.weight

    @property
    def bias(self):
        return self._conv2d.bias


@alf.configurable
class ParallelConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 n,
                 activation=torch.relu_,
                 strides=1,
                 padding=0,
                 use_bias=None,
                 use_bn=False,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A parallel 2D Conv layer that can be used to perform n independent
        2D convolutions in parallel.

        It is equivalent to ``n`` separate ``Conv2D`` layers with the same
        ``in_channels`` and ``out_channels``.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            n (int): n independent ``Conv2D`` layers
            activation (torch.nn.functional):
            strides (int or tuple):
            padding (int or tuple):
            use_bias (bool|None): whether use bias. If None, will use ``not use_bn``
            use_bn (bool): whether use batch normalization
            kernel_initializer (Callable): initializer for the conv layer kernel.
                If None is provided a ``variance_scaling_initializer`` with gain
                as ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_value (float): a constant
        """
        super(ParallelConv2D, self).__init__()
        if use_bias is None:
            use_bias = not use_bn
        self._activation = activation
        self._n = n
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = common.tuplify2d(kernel_size)
        self._conv2d = nn.Conv2d(
            in_channels * n,
            out_channels * n,
            kernel_size,
            groups=n,
            stride=strides,
            padding=padding,
            bias=use_bias)

        for i in range(n):
            if kernel_initializer is None:
                variance_scaling_init(
                    self._conv2d.weight.data[i * out_channels:(i + 1) *
                                             out_channels],
                    gain=kernel_init_gain,
                    nonlinearity=self._activation)
            else:
                kernel_initializer(
                    self._conv2d.weight.data[i * out_channels:(i + 1) *
                                             out_channels])

        # [n*C', C, kernel_size, kernel_size]->[n, C', C, kernel_size, kernel_size]
        self._weight = self._conv2d.weight.view(
            self._n, self._out_channels, self._in_channels,
            self._kernel_size[0], self._kernel_size[1])

        if use_bias:
            nn.init.constant_(self._conv2d.bias.data, bias_init_value)
            # [n*C']->[n, C']
            self._bias = self._conv2d.bias.view(self._n, self._out_channels)
        else:
            self._bias = None

        if use_bn:
            self._bn = nn.BatchNorm2d(n * out_channels)
        else:
            self._bn = None

    def forward(self, img):
        """Forward

        Args:
            img (torch.Tensor): with shape ``[B, C, H, W]``
                                        or ``[B, n, C, H, W]``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``n``: number of replicas
                - ``C``: number of channels
                - ``H``: image height
                - ``W``: image width.
                When the shape of img is ``[B, C, H, W]``, all the n 2D Conv
                operations will take img as the same shared input.
                When the shape of img is ``[B, n, C, H, W]``, each 2D Conv operator
                will have its own input data by slicing img.

        Returns:
            torch.Tensor with shape ``[B, n, C', H', W']``
                where the meaning of the symbols are:
                - ``B``: batch
                - ``n``: number of replicas
                - ``C'``: number of output channels
                - ``H'``: output height
                - ``W'``: output width
        """

        if img.ndim == 4:
            # the shared input case
            assert img.shape[1] == self._in_channels, (
                "Input img has wrong shape %s. Expecting (B, %d, H, W)" %
                (img.shape, self._in_channels))

            img = img.unsqueeze(1).expand(img.shape[0], self._n,
                                          *img.shape[1:])
        elif img.ndim == 5:
            # the non-shared case
            assert (
                img.shape[1] == self._n
                and img.shape[2] == self._in_channels), (
                    "Input img has wrong shape %s. Expecting (B, %d, %d, H, W)"
                    % (img.shape, self._n, self._in_channels))
        else:
            raise ValueError("Wrong img.ndim=%d" % img.ndim)

        # merge replica and channels
        img = img.reshape(img.shape[0], img.shape[1] * img.shape[2],
                          *img.shape[3:])

        res = self._conv2d(img)

        if self._bn is not None:
            res = self._bn(res)

        # reshape back: [B, n*C', H', W'] -> [B, n, C', H', W']
        res = res.reshape(res.shape[0], self._n, self._out_channels,
                          *res.shape[2:])
        return self._activation(res)

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias


@alf.configurable
class ConvTranspose2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=torch.relu_,
                 strides=1,
                 padding=0,
                 use_bias=None,
                 use_bn=False,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A 2D ConvTranspose layer that's also responsible for activation and
        customized weights initialization. An auto gain calculation might depend
        on the activation following the conv layer. Suggest using this wrapper
        module instead of ``nn.ConvTranspose2d`` if you really care about weight std
        after init.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            activation (torch.nn.functional):
            strides (int or tuple):
            padding (int or tuple):
            use_bias (bool|None): If None, will use ``not use_bn``
            use_bn (bool): whether use batch normalization
            kernel_initializer (Callable): initializer for the conv_trans layer.
                If None is provided a variance_scaling_initializer with gain as
                ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_value (float): a constant
        """
        super(ConvTranspose2D, self).__init__()
        if use_bias is None:
            use_bias = not use_bn
        self._activation = activation
        self._conv_trans2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=strides,
            padding=padding,
            bias=use_bias)
        if kernel_initializer is None:
            variance_scaling_init(
                self._conv_trans2d.weight.data,
                gain=kernel_init_gain,
                nonlinearity=self._activation,
                transposed=True)
        else:
            kernel_initializer(self._conv_trans2d.weight.data)

        if use_bias:
            nn.init.constant_(self._conv_trans2d.bias.data, bias_init_value)

        if use_bn:
            self._bn = nn.BatchNorm2d(out_channels)
        else:
            self._bn = None

    def forward(self, img):
        y = self._conv_trans2d(img)
        if self._bn is not None:
            y = self._bn(y)
        return self._activation(y)

    @property
    def weight(self):
        return self._conv_trans2d.weight

    @property
    def bias(self):
        return self._conv_trans2d.bias


@alf.configurable
class ParallelConvTranspose2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 n,
                 activation=torch.relu_,
                 strides=1,
                 padding=0,
                 use_bias=None,
                 use_bn=False,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A parallel ConvTranspose2D layer that can be used to perform n
        independent 2D transposed convolutions in parallel.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            n (int): n independent ``ConvTranspose2D`` layers
            activation (torch.nn.functional):
            strides (int or tuple):
            padding (int or tuple):
            use_bias (bool|None): If None, will use ``not use_bn``
            use_bn (bool):
            kernel_initializer (Callable): initializer for the conv_trans layer.
                If None is provided a ``variance_scaling_initializer`` with gain
                as ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_value (float): a constant
        """
        super(ParallelConvTranspose2D, self).__init__()
        if use_bias is None:
            use_bias = not use_bn
        self._activation = activation
        self._n = n
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = common.tuplify2d(kernel_size)
        self._conv_trans2d = nn.ConvTranspose2d(
            in_channels * n,
            out_channels * n,
            kernel_size,
            groups=n,
            stride=strides,
            padding=padding,
            bias=use_bias)

        for i in range(n):
            if kernel_initializer is None:
                variance_scaling_init(
                    self._conv_trans2d.weight.data[i * in_channels:(i + 1) *
                                                   in_channels],
                    gain=kernel_init_gain,
                    nonlinearity=self._activation)
            else:
                kernel_initializer(
                    self._conv_trans2d.weight.data[i * in_channels:(i + 1) *
                                                   in_channels])

        # [n*C, C', kernel_size, kernel_size]->[n, C, C', kernel_size, kernel_size]
        self._weight = self._conv_trans2d.weight.view(
            self._n, self._in_channels, self._out_channels,
            self._kernel_size[0], self._kernel_size[1])

        if use_bias:
            nn.init.constant_(self._conv_trans2d.bias.data, bias_init_value)
            # [n*C]->[n, C]
            self._bias = self._conv_trans2d.bias.view(self._n,
                                                      self._out_channels)
        else:
            self._bias = None

        if use_bn:
            self._bn = nn.BatchNorm2d(n * out_channels)
        else:
            self._bn = None

    def forward(self, img):
        """Forward

        Args:
            img (torch.Tensor): with shape ``[B, C, H, W]``
                                        or ``[B, n, C, H, W]``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``n``: number of replicas
                - ``C``: number of channels
                - ``H``: image height
                - ``W``: image width.
                When the shape of img is ``[B, C, H, W]``, all the n transposed 2D
                Conv operations will take img as the same shared input.
                When the shape of img is ``[B, n, C, H, W]``, each transposed 2D
                Conv operator will have its own input data by slicing img.

        Returns:
            torch.Tensor with shape ``[B, n, C', H', W']``
                where the meaning of the symbols are:
                - ``B``: batch
                - ``n``: number of replicas
                - ``C'``: number of output channels
                - ``H'``: output height
                - ``W'``: output width
        """
        if img.ndim == 4:
            # the shared input case
            assert img.shape[1] == self._in_channels, (
                "Input img has wrong shape %s. Expecting (B, %d, H, W)" %
                (img.shape, self._in_channels))

            img = img.unsqueeze(1).expand(img.shape[0], self._n,
                                          *img.shape[1:])
        elif img.ndim == 5:
            # the non-shared case
            assert (
                img.shape[1] == self._n
                and img.shape[2] == self._in_channels), (
                    "Input img has wrong shape %s. Expecting (B, %d, %d, H, W)"
                    % (img.shape, self._n, self._in_channels))
        else:
            raise ValueError("Wrong img.ndim=%d" % img.ndim)

        # merge replica and channels
        img = img.reshape(img.shape[0], img.shape[1] * img.shape[2],
                          *img.shape[3:])

        res = self._conv_trans2d(img)
        if self._bn is not None:
            res = self._bn(res)
        # reshape back: [B, n*C', H', W'] -> [B, n, C', H', W']
        res = res.reshape(res.shape[0], self._n, self._out_channels,
                          res.shape[2], res.shape[3])
        return self._activation(res)

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias


@alf.configurable
class ParamFC(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 activation=torch.relu_,
                 use_bias=True,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A fully connected layer that does not maintain its own weight and bias,
        but accepts both from users. If the given parameter (weight and bias)
        tensor has an extra batch dimension (first dimension), it performs
        parallel FC operation.

        Args:
            input_size (int): input size
            output_size (int): output size
            activation (torch.nn.functional):
            use_bias (bool): whether use bias
        """
        super(ParamFC, self).__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._kernel_init_gain = kernel_init_gain
        self._bias_init_value = bias_init_value

        self._weight_length = output_size * input_size
        self.set_weight(torch.randn(1, self._weight_length))
        if use_bias:
            self._bias_length = output_size
            self.set_bias(torch.randn(1, self._bias_length))
        else:
            self._bias_length = 0
            self._bias = None

    @property
    def weight(self):
        """Get stored weight tensor or batch of weight tensors."""
        return self._weight

    @property
    def bias(self):
        """Get stored bias tensor or batch of bias tensors."""
        return self._bias

    @property
    def weight_length(self):
        """Get the n_element of a single weight tensor. """
        return self._weight_length

    @property
    def bias_length(self):
        """Get the n_element of a single bias tensor. """
        return self._bias_length

    def set_weight(self, weight, reinitialize=False):
        """Store a weight tensor or batch of weight tensors.

        Args:
            weight (torch.Tensor): with shape ``[B, D]``
                where the mining of the symbols are:
                - ``B``: batch size
                - ``D``: length of weight vector, should be self._weight_length
            reinitialize (bool): whether to reinitialize self._weight
        """
        assert (weight.ndim == 2 and weight.shape[1] == self._weight_length), (
            "Input weight has wrong shape %s. Expecting shape (n, %d)" %
            (weight.shape, self._weight_length))
        self._groups = weight.shape[0]
        weight = weight.view(self._groups, self._output_size, self._input_size)
        if reinitialize:
            for i in range(self._groups):
                if self._kernel_initializer is None:
                    variance_scaling_init(
                        weight[i],
                        gain=self._kernel_init_gain,
                        nonlinearity=self._activation)
                else:
                    self._kernel_initializer(weight[i])

        self._weight = weight

    def set_bias(self, bias, reinitialize=False):
        """Store a bias tensor or batch of bias tensors.

        Args:
            bias (torch.Tensor): with shape ``[B, D]``
                where the mining of the symbols are:
                - ``B``: batch size
                - ``D``: length of bias vector, should be self._bias_length
            reinitialize (bool): whether to reinitialize self._bias
        """
        assert (bias.ndim == 2 and bias.shape[1] == self._bias_length), (
            "Input bias has wrong shape %s. Expecting shape (n, %d)" %
            (bias.shape, self._bias_length))
        assert bias.shape[0] == self._groups, (
            "Input bias has wrong shape %s. Expecting shape (%d, %d)" %
            (bias.shape, self._group, self.bias_length))

        if reinitialize:
            if self._use_bias:
                nn.init.constant_(bias, self._bias_init_value)

        self._bias = bias  # [n, bias_length]

    def forward(self, inputs):
        """Forward

        Args:
            inputs (torch.Tensor): with shape ``[B, D] (groups=1)``
                                        or ``[B, n, D] (groups=n)``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``n``: number of replicas
                - ``D``: input dimension
                When the shape of inputs is ``[B, D]``, all the n linear
                operations will take inputs as the same shared inputs.
                When the shape of inputs is ``[B, n, D]``, each linear operator
                will have its own input data by slicing inputs.

        Returns:
            torch.Tensor with shape ``[B, n, D]`` or ``[B, D]``
                where the meaning of the symbols are:
                - ``B``: batch
                - ``n``: number of replicas
                - ``D'``: output dimension
        """

        if inputs.ndim == 2:
            # case 1: non-parallel inputs
            assert inputs.shape[1] == self._input_size, (
                "Input inputs has wrong shape %s. Expecting (B, %d)" %
                (inputs.shape, self._input_size))
            inputs = inputs.unsqueeze(0).expand(self._groups, *inputs.shape)
        elif inputs.ndim == 3:
            # case 2: parallel inputs
            assert (
                inputs.shape[1] == self._groups
                and inputs.shape[2] == self._input_size), (
                    "Input inputs has wrong shape %s. Expecting (B, %d, %d)" %
                    (inputs.shape, self._groups, self._input_size))
            inputs = inputs.transpose(0, 1)  # [n, B, D]
        else:
            raise ValueError("Wrong inputs.ndim=%d" % inputs.ndim)

        if self._bias is not None:
            res = torch.baddbmm(
                self._bias.unsqueeze(1), inputs, self._weight.transpose(1, 2))
        else:
            res = torch.bmm(inputs, self._weight.transpose(1, 2))
        res = res.transpose(0, 1)  # [B, n, D]
        res = res.squeeze(1)  # [B, D] if n=1

        return self._activation(res)


@alf.configurable
class ParamConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=torch.relu_,
                 strides=1,
                 pooling_kernel=None,
                 padding=0,
                 use_bias=False,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A 2D conv layer that does not maintain its own weight and bias,
        but accepts both from users. If the given parameter (weight and bias)
        tensor has an extra batch dimension (first dimension), it performs
        parallel FC operation.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            activation (torch.nn.functional):
            strides (int or tuple):
            pooling_kernel (int or tuple):
            padding (int or tuple):
            use_bias (bool): whether use bias
        """
        super(ParamConv2D, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._activation = activation
        self._kernel_size = common.tuplify2d(kernel_size)
        self._kH, self._kW = self._kernel_size
        self._strides = strides
        self._pooling_kernel = pooling_kernel
        self._padding = padding
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._kernel_init_gain = kernel_init_gain
        self._bias_init_value = bias_init_value

        self._weight_length = out_channels * in_channels * self._kH * self._kW
        self.set_weight(torch.randn(1, self._weight_length))
        if use_bias:
            self._bias_length = out_channels
            self.set_bias(torch.randn(1, self._bias_length))
        else:
            self._bias_length = 0
            self._bias = None

    @property
    def weight(self):
        """Get stored weight tensor or batch of weight tensors."""
        return self._weight

    @property
    def bias(self):
        """Get stored bias tensor or batch of bias tensors."""
        return self._bias

    @property
    def weight_length(self):
        """Get the n_element of a single weight tensor. """
        return self._weight_length

    @property
    def bias_length(self):
        """Get the n_element of a single bias tensor. """
        return self._bias_length

    def set_weight(self, weight, reinitialize=False):
        """Store a weight tensor or batch of weight tensors.

        Args:
            weight (torch.Tensor): with shape ``[B, D]``
                where the mining of the symbols are:
                - ``B``: batch size
                - ``D``: length of weight vector, should be self._weight_length
            reinitialize (bool): whether to reinitialize self._weight
        """
        assert (weight.ndim == 2 and weight.shape[1] == self._weight_length), (
            "Input weight has wrong shape %s. Expecting shape (n, %d)" %
            (weight.shape, self._weight_length))
        if weight.shape[0] == 1:
            # non-parallel weight
            self._groups = 1
            weight = weight.view(self._out_channels, self._in_channels,
                                 self._kH, self._kW)
        else:
            # parallel weight
            self._groups = weight.shape[0]
            weight = weight.view(self._groups, self._out_channels,
                                 self._in_channels, self._kH, self._kW)
            weight = weight.reshape(self._groups * self._out_channels,
                                    self._in_channels, self._kH, self._kW)

        if reinitialize:
            for i in range(self._groups):
                if self._kernel_initializer is None:
                    variance_scaling_init(
                        weight[i * self._out_channels:(i + 1) *
                               self._out_channels],
                        gain=self._kernel_init_gain,
                        nonlinearity=self._activation)
                else:
                    self._kernel_initializer(
                        weight[i * self._out_channels:(i + 1) *
                               self._out_channels])
        self._weight = weight

    def set_bias(self, bias, reinitialize=False):
        """Store a bias tensor or batch of bias tensors.

        Args:
            bias (torch.Tensor): with shape ``[B, D]``
                where the mining of the symbols are:
                - ``B``: batch size
                - ``D``: length of bias vector, should be self._bias_length
            reinitialize (bool): whether to reinitialize self._bias
        """
        assert (bias.ndim == 2 and bias.shape[1] == self._bias_length), (
            "Input bias has wrong shape %s. Expecting shape (n, %d)" %
            (bias.shape, self._bias_length))
        assert bias.shape[0] == self._groups, (
            "Input bias has wrong shape %s. Expecting shape (%d, %d)" %
            (bias.shape, self._group, self.bias_length))

        if reinitialize:
            if self._use_bias:
                nn.init.constant_(bias, self._bias_init_value)

        self._bias = bias.reshape(-1)

    def forward(self, img, keep_group_dim=True):
        """Forward

        Args:
            img (torch.Tensor): with shape ``[B, C, H, W] (groups=1)``
                                        or ``[B, n, C, H, W] (groups=n)``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``n``: number of replicas
                - ``C``: number of channels
                - ``H``: image height
                - ``W``: image width.
                When the shape of img is ``[B, C, H, W]``, all the n 2D Conv
                operations will take img as the same shared input.
                When the shape of img is ``[B, n, C, H, W]``, each 2D Conv operator
                will have its own input data by slicing img.

        Returns:
            torch.Tensor with shape ``[B, n, C', H', W']``
                where the meaning of the symbols are:
                - ``B``: batch
                - ``n``: number of replicas
                - ``C'``: number of output channels
                - ``H'``: output height
                - ``W'``: output width
        """
        if self._groups == 1:
            # non-parallel layer
            assert (img.ndim == 4 and img.shape[1] == self._in_channels), (
                "Input img has wrong shape %s. Expecting (B, %d, H, W)" %
                (img.shape, self._in_channels))
        else:
            # parallel layer
            if img.ndim == 4:
                if img.shape[1] == self._in_channels:
                    # case 1: non-parallel input
                    img = img.repeat(1, self._groups, 1, 1)
                else:
                    # case 2: parallel input
                    assert img.shape[1] == self._groups * self._in_channels, (
                        "Input img has wrong shape %s. Expecting (B, %d, H, W) or (B, %d, H, W)"
                        % (img.shape, self._in_channels,
                           self._groups * self._in_channels))
            elif img.ndim == 5:
                # case 3: parallel input with unmerged group dim
                assert (
                    img.shape[1] == self._groups
                    and img.shape[2] == self._in_channels
                ), ("Input img has wrong shape %s. Expecting (B, %d, %d, H, W)"
                    % (img.shape, self._groups, self._in_channels))
                # merge group and channel dim
                img = img.reshape(img.shape[0], img.shape[1] * img.shape[2],
                                  *img.shape[3:])
            else:
                raise ValueError("Wrong img.ndim=%d" % img.ndim)

        res = self._activation(
            F.conv2d(
                img,
                self._weight,
                bias=self._bias,
                stride=self._strides,
                padding=self._padding,
                groups=self._groups))
        if self._pooling_kernel is not None:
            res = F.max_pool2d(res, self._pooling_kernel)

        if self._groups > 1 and keep_group_dim:
            # reshape back: [B, n*C', H', W'] -> [B, n, C', H', W']
            res = res.reshape(res.shape[0], self._groups, self._out_channels,
                              res.shape[2], res.shape[3])

        return res


@alf.configurable
class Reshape(nn.Module):
    def __init__(self, shape):
        """A layer for reshape the tensor.

        The result of this layer is a tensor reshaped to ``(B, *shape)`` where
        ``B`` is ``x.shape[0]``

        Args:
            shape (tuple): desired shape not including the batch dimension.
        """
        super().__init__()
        self._shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self._shape)


def _tuplify2d(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    return (x, x)


def _conv_transpose_2d(in_channels,
                       out_channels,
                       kernel_size,
                       stride=1,
                       padding=0,
                       bias=True):
    # need output_padding so that output_size is stride * input_size
    # See https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d
    output_padding = stride + 2 * padding - kernel_size
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        bias=bias)


@alf.configurable(
    whitelist=['v1_5', 'with_batch_normalization', 'keep_conv_bias'])
class BottleneckBlock(nn.Module):
    """Bottleneck block for ResNet.

    We allow two slightly different architectures:
    * v1: Placing the stride at the first 1x1 convolution as described in the
      original ResNet paper `Deep residual learning for image recognition
      <https://arxiv.org/abs/1512.03385>`_.
    * v1.5: Placing the stride for downsampling at 3x3 convolution. This variant
      is also known as ResNet V1.5 and improves accuracy according to
      `<https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
    """

    def __init__(self,
                 in_channels,
                 kernel_size,
                 filters,
                 stride,
                 transpose=False,
                 v1_5=True,
                 with_batch_normalization=True,
                 keep_conv_bias=False):
        """
        Args:
            kernel_size (int): the kernel size of middle layer at main path
            filters (int): the filters of 3 layer at main path
            stride (int): stride for this block.
            transpose (bool): a bool indicate using ``Conv2D`` or ``Conv2DTranspose``.
                If two BottleneckBlock layers ``L`` and ``LT`` are constructed
                with the same arguments except ``transpose``, it is gauranteed that
                ``LT(L(x)).shape == x.shape`` if ``x.shape[-2:]`` can be divided
                by ``stride``.
            v1_5 (bool): whether to use the ResNet V1.5 structure
            with_batch_normalization (bool): whether to include batch normalization.
                Note that standard ResNet uses batch normalization.
            keep_conv_bias (bool): by default, if ``with_batch_normalization`` is
                True, the biases of conv layers are not used because they are useless.
                This behavior can be overrided by setting ``keep_conv_bias`` to
                True. The main purpose of this is for loading legacy models.
        Return:
            Output tensor for the block
        """
        super().__init__()
        filters1, filters2, filters3 = filters

        conv_fn = _conv_transpose_2d if transpose else nn.Conv2d

        bias = not with_batch_normalization or keep_conv_bias

        padding = (kernel_size - 1) // 2
        if v1_5:
            a = conv_fn(in_channels, filters1, 1, bias=bias)
            b = conv_fn(
                filters1, filters2, kernel_size, stride, padding, bias=bias)
        else:
            a = conv_fn(in_channels, filters1, 1, stride, bias=bias)
            b = conv_fn(filters1, filters2, kernel_size, 1, padding, bias=bias)

        c = conv_fn(filters2, filters3, 1, bias=bias)

        nn.init.kaiming_normal_(a.weight.data)
        nn.init.kaiming_normal_(b.weight.data)
        nn.init.kaiming_normal_(c.weight.data)

        if bias:
            nn.init.zeros_(a.bias.data)
            nn.init.zeros_(b.bias.data)
            nn.init.zeros_(c.bias.data)

        if stride != 1 or in_channels != filters3:
            s = conv_fn(in_channels, filters3, 1, stride, bias=bias)
            nn.init.kaiming_normal_(s.weight.data)
            if bias:
                nn.init.zeros_(s.bias.data)
            if with_batch_normalization:
                shortcut_layers = nn.Sequential(s, nn.BatchNorm2d(filters3))
            else:
                shortcut_layers = s
        else:
            shortcut_layers = None

        relu = nn.ReLU(inplace=True)

        if with_batch_normalization:
            core_layers = nn.Sequential(a, nn.BatchNorm2d(filters1), relu, b,
                                        nn.BatchNorm2d(filters2), relu, c,
                                        nn.BatchNorm2d(filters3))
        else:
            core_layers = nn.Sequential(a, relu, b, relu, c)

        self._core_layers = core_layers
        self._shortcut_layers = shortcut_layers

    def forward(self, inputs):
        core = self._core_layers(inputs)
        if self._shortcut_layers:
            shortcut = self._shortcut_layers(inputs)
        else:
            shortcut = inputs

        return torch.relu_(core + shortcut)

    def calc_output_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        y = self.forward(x)
        return y.shape[1:]


def _masked_softmax(logits, mask, dim=-1):
    if mask is not None:
        logits.masked_fill_(mask, -float('inf'))
    return nn.functional.softmax(logits, dim=dim)


class TransformerBlock(nn.Module):
    """Transformer residue block.

    The transformer residue block includes two residue blocks with layer normalization (LN):

    1. Multi-head attention (MHA) block
    2. Position-wise MLP

    The overall computation is:

    .. code-block:: python

        y = x + MHA(LN(x))
        z = y + MLP(LN(y))

    The original transformer is described in:
    [1]. Ashish Vaswani et. al. Attention Is All You Need

    This implementation is a variation which places layer norm at a different
    location, which is proposed in:
    [2]. Ruibin XiongOn et. al. Layer Normalization in the Transformer Architecture

    We also support the relative positional encoding proposed in
    [3] Zihang Dai et. al. Transformer-XL: Attentive language models beyond a fixed-length context.

    In this implementation, the positional encodings are learnable parameter instead
    of the sinusoidal matrix proposed in [1]
    """

    def __init__(self,
                 d_model,
                 num_heads,
                 memory_size,
                 d_k=None,
                 d_v=None,
                 d_ff=None,
                 positional_encoding='abs',
                 add_positional_encoding=True,
                 scale_attention_score=True):
        """
        Args:
            d_model (int): dimension of the model, same as d_model in [1]
            num_heads (int): the number of attention heads
            memory_size (int): maximal allowed sequence length
            d_k (int): Dimension of key, same as d_k in [1]. If None, use ``d_model // num_heads``
            d_v (int): Dimension of value, same as d_v in [1]. If None, use ``d_model // num_heads``
            d_ff (int): Diemension of the MLP, same as d_ff in [1]. If None, use ``4 * d_model``
            positional_encoding (str): One of ['none', 'abs', 'rel']. If 'none',
                no position encoding will be used. If 'abs', use absolute positional
                encoding depending on the absolute position in the memory sequence,
                same as that described in [1]. If 'rel', use the relative positional
                encoding proposed in [3].
            add_positional_encoding (bool): If True, in addition to use positional
                encoding for calculating the attention weights, the positional encoding
                is also concatenated to the attention result so that the attention
                result can keep the location information better. Note that using
                this option will increase the number of parameters by about 25%.
                This option cannot be used if positional_encoding is 'none'.
            scale_attention_score (bool): If True, scale the attention score by
                ``d_k ** -0.5`` as suggested in [1]. However, this may not always
                be better since it slows the unittest in layers_test.py
        """
        super().__init__()
        if d_k is None:
            d_k = d_model // num_heads
        if d_v is None:
            d_v = d_model // num_heads
        if d_ff is None:
            d_ff = 4 * d_model
        self._q_proj = nn.Parameter(torch.Tensor(d_model, num_heads * d_k))
        self._k_proj = nn.Parameter(torch.Tensor(d_model, num_heads * d_k))
        self._v_proj = nn.Parameter(torch.Tensor(d_model, num_heads * d_v))
        d_a = d_v
        if add_positional_encoding:
            assert positional_encoding != 'none', (
                "positional_encoding cannot be 'none' for "
                "add_positional_encoding=True")
            d_a = d_v + d_k
        self._o_proj = nn.Parameter(torch.Tensor(num_heads * d_a, d_model))

        self._d_model = d_model
        self._d_k = d_k
        self._d_v = d_v
        self._d_a = d_a
        self._num_heads = num_heads
        self._memory_size = memory_size
        self._relative_positional_encoding = positional_encoding == 'rel'
        self._add_positional_encoding = add_positional_encoding

        self._attention_scale = d_k**-0.5 if scale_attention_score else 1.
        self._mlp = torch.nn.Sequential(
            FC(d_model, d_ff, torch.relu_), FC(d_ff, d_model))
        self._norm1 = torch.nn.LayerNorm(d_model)
        self._norm2 = torch.nn.LayerNorm(d_model)

        l = 2 * memory_size - 1 if positional_encoding == 'rel' else memory_size
        self._positional_encoding = None
        if positional_encoding != 'none':
            self._positional_encoding = nn.Parameter(torch.Tensor(l, d_k))
        # bias over query vectors when calculating score with keys. Introduced in [3].
        self._qk_bias = nn.Parameter(torch.Tensor(num_heads, d_k))
        # bias over query vectors when calculating score with positional encodings.
        # Introduced in [3].
        self._qp_bias = nn.Parameter(torch.Tensor(num_heads, d_k))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the parameters."""
        nn.init.xavier_uniform_(self._q_proj)
        nn.init.xavier_uniform_(self._k_proj)
        nn.init.xavier_uniform_(self._v_proj)
        nn.init.xavier_uniform_(self._o_proj)
        nn.init.zeros_(self._qk_bias)
        nn.init.zeros_(self._qp_bias)
        if self._positional_encoding is not None:
            nn.init.uniform_(self._positional_encoding, -0.1, 0.1)
        for l in self._mlp:
            l.reset_parameters()

    @staticmethod
    def _shift(x, m):
        """
        y[i, j, :] <= x[n - 1 + i - j, :] for 0<=i<m, 0<=j<n
        Args:
            x: [2 * N - 1, d]
        Returns:
            [M, N, d]
        """
        n = (x.shape[0] + 1) // 2
        # [M, N], index[i, j] = n - 1 + i - j
        index = n - 1 + torch.arange(m).unsqueeze(-1) - torch.arange(n)
        return x[index]

    def forward(self, memory, query=None, mask=None):
        """Forward computation.

        Notation: B: batch_size, N: length of ``memory``, M: length of ``query``
        Args:
            memory (Tensor): The shape is [B, N, d_model]
            query (Tensor): The shape [B, d_model] or [B, M, d_model]. If None,
                will use memory as query
            mask (Tensor|None): A tensor for indicating which slot in ``memory``
                will be used. Its shape can be [B, N] or [B, M, N]. If the shape
                is [B, N], mask[b,n] indicates whether to use memory[b, n] for
                calculating the attention result for ``query[b]``. If the shape is
                [B, M, N], maks[b, m, n] indicates whether to use meory[b, n] for
                calculating the attention result for ``query[b, m]``.
        Returns:
            Tensor: the shape is same as query.
        """
        need_squeeze = False
        if query is None:
            original_query = memory
            memory = self._norm1(memory)
            query = memory
        else:
            if query.ndim == 2:
                query = query.unsqueeze(1)
                need_squeeze = True
            original_query = query
            query = self._norm1(query)
            memory = self._norm1(memory)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(1)
            # [B, M, 1, N]
            mask = mask.unsqueeze(2)

        # B: batch_size
        # H: num_heads
        # N: memory_size
        # M: query.shape[1]
        # L: 2N-1 if relative_positional_encoding else N
        batch_size = query.shape[0]
        m = query.shape[1]
        n = memory.shape[1]
        d_k = self._d_k
        d_v = self._d_v
        d_model = self._d_model
        d_a = self._d_a
        num_heads = self._num_heads

        assert query.shape[0] == memory.shape[0]
        assert query.shape[2] == d_model
        assert memory.shape[2] == d_model
        assert n <= self._memory_size
        assert m <= self._memory_size

        # [B, M, H, d_k] <= [B, M, d_model] * [d_model, d_k]
        q = torch.matmul(query, self._q_proj).reshape(batch_size, m, num_heads,
                                                      d_k)

        # We select different versions of calculation based on memory consumption
        if n * d_k <= m * d_model:
            #             computation                  memory
            # k           N * H * d_k * d_model        N * H * d_k
            # a           M * H * N * d_k              M * H * N

            # [B, N, H, d_k] <= [B, N, d_model] * [d_model, H * d_k]
            k = torch.matmul(memory, self._k_proj).reshape(
                batch_size, n, num_heads, d_k)
            # [B, M, H, N] <= [B, M, H, d_k] * [B, N, H, d_k]
            logits = torch.einsum('bmhd,bnhd->bmhn', q + self._qk_bias, k)
        else:
            #             computation                  memory
            # qk          M * H * d_k * d_model        M * H * d_model
            # a           M * H * N * d_model          M * H * N

            # [B, M, H, d_model] <= [B, M, H, d_k] * [d_model, H, d_k]
            qk = torch.einsum('bmhd,ehd->bmhe', q + self._qk_bias,
                              self._k_proj.reshape(d_model, num_heads, d_k))
            # [B, M, H, N] <= [B, M, H, d_model] * [B, N, d_model]
            logits = torch.einsum('bmhd,bnd->bmhn', qk, memory)

        if self._positional_encoding is not None:
            # [N, d_k]
            positional_encoding = self._positional_encoding
            if n < self._memory_size:
                d = self._memory_size - n
                if self._relative_positional_encoding:
                    positional_encoding = positional_encoding[d:-d]
                else:
                    positional_encoding = positional_encoding[:-d]

            if self._relative_positional_encoding:
                # positional_encoding[i, j, :] <= positional_encoding(n - 1 + i - j, d)
                # [M, N, d_k]
                positional_encoding = self._shift(positional_encoding, m)
            # [B, M, H, N] <= [B, M, H, d_k] * ([d_k, N] or [M, d_k, N])
            positional_logits = torch.matmul(
                q + self._qp_bias, positional_encoding.transpose(-2, -1))
            # gradient can still be correctly calculated in this case even though
            # inplace add is used.
            logits.add_(positional_logits)

        if self._attention_scale != 1.0:
            logits.mul_(self._attention_scale)

        # [B, M, H, N]
        a = _masked_softmax(logits, mask)

        if n * d_v <= m * d_model:
            #             computation                  memory
            # v           N * H * d_v * d_model        N * H * d_v
            # att_result  M * H * N * d_v              M * H * d_v

            # [B, N, H, d_v] <= [B, N, d_model] * [d_model, H * d_v]
            v = torch.matmul(memory, self._v_proj).reshape(
                batch_size, n, num_heads, d_v)
            # [B, M, H, d_v] <= [B, M, H, N] * [B, N, H, d_v]
            att_result = torch.einsum('bmhn,bnhd->bmhd', a, v)
        else:
            # computation                  memory
            # att_result  M * H * N * d_model          M * H * d_model
            # att_result  M * H * d_v * d_model        M * H * d_v

            # [B, M, H, d_model] <= [B, M, H, N] * [B, 1, N, d_model]
            att_result = torch.einsum('bmhn,bnd->bmhd', a, memory)
            # [B, M, H, d_v] <= [B, M, H, d_model] * [d_model, H, d_v]
            att_result = torch.einsum(
                'bmhd,dhe->bmhe', att_result,
                self._v_proj.reshape(d_model, self._num_heads, d_v))

        if self._add_positional_encoding:
            # [B, M, H, d_k] <= [B, M, H, N] * ([N, d_k] or [M, N, d_k])
            att_pos = torch.matmul(a, positional_encoding)
            # [B, M, H, d_v + d_k]
            att_result = torch.cat([att_result, att_pos], dim=-1)

        # [B, M, H * d_a]
        att_result = att_result.reshape(batch_size, m, num_heads * d_a)
        # [B, M, d_model]
        x = original_query + torch.matmul(att_result, self._o_proj)
        # [B, M, d_model]
        y = self._mlp(self._norm2(x))
        # [B, M, d_model]
        z = x + y

        if need_squeeze:
            z = z.squeeze(1)

        return z


class Lambda(nn.Module):
    """Wrap a function as an nn.Module."""

    def __init__(self, func):
        """
        Args:
            func (Callable): a function that calculate the output given the input.
                It should be parameterless.
        """
        super().__init__()
        self._func = func

    def forward(self, x):
        return self._func(x)


class GFT(nn.Module):
    """Guided Feature Transformation.

    This class implements the GFT model proposed in the following paper:

    `Yu et al. Guided Feature Transformation (GFT): A Neural language Grounding
    Module for Embodied Agents, CoRL 2018 <https://arxiv.org/pdf/1805.08329.pdf>`_
    """

    def __init__(self, num_transformations, image_channels, language_dim):
        super().__init__()
        self._t_layers = nn.ModuleList([
            FC(language_dim, (1 + image_channels) * image_channels)
            for k in range(num_transformations)
        ])
        self._ones = torch.ones(1, 1, 1)

    def forward(self, input):
        """
        Args:
            input (tuple): the tuple of image features and sentence embedding.
        Returns:
            Tensor: same shape as input[0]
        """
        image, sentence = input
        batch_size, channels = image.shape[:2]
        # [B, C, W*H]
        cnn_out = image.view(batch_size, channels, -1)
        ## compute K transformation matrices
        ts = [
            l(sentence).view(batch_size, channels, channels + 1)
            for l in self._t_layers
        ]

        ones = self._ones.expand(batch_size, 1, cnn_out.shape[-1])
        for t in ts:
            # [B, C+1, W*H]
            cnn_out = torch.cat((cnn_out, ones), dim=1)
            # [B, C, W*H] <= [B, C, C+1] * [B, C+1, W*H]
            cnn_out = torch.relu_(torch.matmul(t, cnn_out))
        return cnn_out.reshape(*image.shape)

    def reset_parameters(self):
        for l in self._t_layers:
            l.reset_parameters()


class GetFields(nn.Module):
    """Get the fields from a nested input."""

    def __init__(self, field_nest=None, **fields):
        """
        Args
            field_nest (nested str): the path of the fields to be retrieved. Each str
                in ``fields`` represents a path to the field with '.' separating
                the field name at different level.
            fields (str): A simpler way of specifying ``field_nest`` when it is
                a dict. ``GetFields(a="field_a", b="field_b")`` is equivalent to
                ``GetFields(dict(a="field_a", b="field_b"))``.
        """
        super().__init__()
        if field_nest is not None:
            assert not fields
            fields = field_nest
        self._fields = fields

    def forward(self, input):
        return alf.nest.map_structure(
            lambda path: alf.nest.get_field(input, path), self._fields)


class Sum(nn.Module):
    """Sum over given dimension(s).

    Note that batch dimention is not counted for dim. This means that
    dim=0 means the dimension after batch dimension.
    """

    def __init__(self, dim):
        """
        Args:
            dim (int|tuple[int]): the dimension(s) to be summed.
        """
        super().__init__()
        dim = alf.nest.map_structure(lambda d: d + 1 if d >= 0 else d, dim)
        self._dim = dim

    def forward(self, input):
        return input.sum(dim=self._dim)


def reset_parameters(module):
    """Reset the parameters for ``module``.

    Args:
        module (nn.Module):
    Returns:
        None
    Raises:
        ValueError: fail to reset the parameters for ``module``
    """
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
    elif isinstance(module, nn.Sequential):
        for l in module:
            reset_parameters(l)
    elif isinstance(module, nn.Module):
        if len(list(module.parameters())) > 0:
            raise ValueError(
                "Cannot reset_parameter for layer type %s." % type(module))


class Detach(nn.Module):
    """Detach nested Tensors."""

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return common.detach(input)


class Branch(nn.Module):
    """Apply multiple modules on the same input.

    Example:

    .. code-block:: python

        net = Branch((module1, module2))
        y = net(x)

    is equivalent to the following:

    .. code-block:: python

        y = module1(x), module2(x)

    """

    def __init__(self, *modules, **named_modules):
        """
        Args:
            modules (nested nn.Module): a nest of ``torch.nn.Module``. Note that
                ``Branch(module_a, module_b)`` is equivalent to
                ``Branch((module_a, module_b))``
            named_modules (nn.Module | Callable): a simpler way of specifying
                a dict of modules. ``Branch(a=model_a, b=module_b)``
                is equivalent to ``Branch(dict(a=module_a, b=module_b))``
        """
        super().__init__()
        if modules:
            assert not named_modules
            if len(modules) == 1:
                modules = modules[0]
        else:
            modules = named_modules
        has_network = any(
            alf.nest.flatten(
                alf.nest.map_structure(
                    lambda m: isinstance(m, alf.networks.Network), modules)))
        assert not has_network, (
            "modules should not contain alf.networks.Network. "
            "Try alf.networks.Branch instead.")

        self._networks = modules
        if alf.nest.is_nested(modules):
            # make it a nn.Module so its parameters can be picked up by the framework
            self._nets = alf.nest.utils.make_nested_module(modules)

    def forward(self, inputs):
        return alf.nest.map_structure(lambda net: net(inputs), self._networks)

    def reset_parameters(self):
        alf.nest.map_structure(reset_parameters, self._networks)


def _get_nested_field(nest, nest_fields):
    return map_structure(lambda f: get_field(nest, f), nest_fields)


class Sequential(nn.Module):
    """A more flexible Sequential than torch.nn.Sequential.

    ``alf.layers.Sequentail`` is similar to ``alf.nn.Sequential``, but does not
    accept ``alf.nn.Network`` as its elements.

    All the modules provided through ``modules`` and ``named_modules`` are calculated
    sequentially in the same order as they appear in the call to ``Sequential``.
    Typically, each module takes the result of the previous module as its input
    (or the input to the Sequential if it is the first module), and the result of
    the last module is the output of the Sequential. But we also allow more
    flexibilities as shown in example 2.

    Example 1:

    .. code-block:: python

        net = Sequential(module1, module2)
        y = net(x)

    is equivalent to the following:

    .. code-block:: python

        z = module1(x)
        y = module2(z)

    Example 2:

    .. code-block:: python

        net = Sequential(
            module1, a=module2, b=(('input', 'a'), module3), output=('a', 'b'))
        output = net(input, state)

    is equivalent to the following:

    .. code-block:: python

        _ = module1(input)
        a = module2(_)
        b = module3((input, a))
        output = (a, b)

    """

    def __init__(self, *modules, output='', **named_modules):
        """
        Args:
            modules (nn.Module | (nested str, nn.Module)): ``torch.nn.Module``
                and optionally their input (which is the
                first element of the tuple). If input is not provided, it is
                assumed to be the result of the previous module (or input to this
                ``Sequential`` for the first module). If input is provided, it
                should be a nested str. It will be used to retrieve results from
                the dictionary of the current ``named_results``. For modules
                specified by ``modules``, because no ``named_modules`` has been
                invoked, ``named_results`` is ``{'input': input}``.
            named_modules (nn.Module | (nested str, nn.Module)): ``torch.nn.Module``
                and optionally their input (which is the
                first element of the tuple). If input is not provided, it is
                assumed to be the result of the previous module (or input to this
                ``Sequential`` for the first module). If input is provided, it
                should be a nested str. It will be used to retrieve results from
                the dictionary of the current ``named_results``. ``named_results``
                is updated once the result of a named module is calculated.
            output (nested str): if not provided, the result from the last module
                will be used as output. Otherwise, it will be used to retrieve
                results from ``named_results`` after the results of all modules
                have been calculated.
        """
        super().__init__()
        named_elements = list(zip([''] * len(modules), modules)) + list(
            named_modules.items())
        modules = []
        inputs = []
        outputs = []
        simple = True
        is_nested_str = lambda s: all(
            map(lambda x: type(x) == str, alf.nest.flatten(s)))
        for i, (out, element) in enumerate(named_elements):
            input = ''
            if isinstance(element, tuple) and len(element) == 2:
                input, module = element
            else:
                module = element
            if not (isinstance(module, nn.Module) and is_nested_str(input)):
                raise ValueError("Argument %s is not in the form of nn.Module "
                                 "or (nested str, nn.Module): %s" %
                                 (out or str(i), element))
            if isinstance(module, alf.networks.Network):
                assert not alf.nest.flatten(module.state_spec), (
                    "Network element of layers.Sequential should be stateless. "
                    "Use networks.Sequential instead")
            inputs.append(input)
            outputs.append(out)
            modules.append(module)
            if out or input:
                simple = False

        self._networks = nn.ModuleList(modules)
        if simple:
            self.forward = self._forward_simple
        else:
            self.forward = self._forward_complex
        self._output = output
        self._inputs = inputs
        self._outputs = outputs

    def _forward_simple(self, input):
        for module in self._networks:
            if isinstance(module, alf.networks.Network):
                input = module(input)[0]
            else:
                input = module(input)
        return input

    def _forward_complex(self, input):
        var_dict = {'input': input}
        for i, net in enumerate(self._networks):
            if self._inputs[i]:
                input = _get_nested_field(var_dict, self._inputs[i])
            if isinstance(net, alf.networks.Network):
                input = net(input)[0]
            else:
                input = net(input)
            if self._outputs[i]:
                var_dict[self._outputs[i]] = input
        if self._output:
            input = _get_nested_field(var_dict, self._output)
        return input

    def reset_parameters(self):
        alf.nest.map_structure(reset_parameters, self._networks)

    def __getitem__(self, i):
        return self._networks[i]
