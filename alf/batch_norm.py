# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
import types

from alf.utils.common import warning_once


class _NormBase(nn.Module):
    """Base of BatchNorm supporting RNN."""

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        self._num_features = num_features
        self._eps = eps
        self._momentum = momentum
        self._affine = affine
        self._track_running_stats = track_running_stats
        if affine:
            self._weight = nn.Parameter(torch.Tensor(num_features))
            self._bias = nn.Parameter(torch.Tensor(num_features))
        self._running_means = []
        self._running_vars = []
        self._num_batches_tracked = []
        self.set_max_steps(1)
        self._current_step = 0
        self.reset_parameters()
        self._clamped = False

    def set_max_steps(self, max_steps: int):
        """Set max steps to keeping running statistics.

        Args:
            max_steps: the maximum steps for which the batch norm running statistics
                are maintained.
        """
        self._max_steps = max_steps
        if not self._track_running_stats:
            return
        for i in range(len(self._running_means), max_steps):
            self._running_means.append(torch.zeros(self._num_features))
            self.register_buffer('_running_means%s' % i,
                                 self._running_means[i])
            self._running_vars.append(torch.ones(self._num_features))
            self.register_buffer('_running_vars%s' % i, self._running_vars[i])
            self._num_batches_tracked.append(torch.zeros((), dtype=torch.long))
            self.register_buffer('_num_batches_tracked%s' % i,
                                 self._num_batches_tracked[i])

    def set_current_step(self, current_step: Union[torch.Tensor, int]):
        """Use and/or update the running statistics at current_step for normalization.

        Args:
            current_step: the current step. If it is a Tensor, it should be a 1D
                int64 Tensor of shape [batch_size,]. And each of its element
                means the current step for the corresponding sample in a batch.
        """
        if not self._track_running_stats:
            return
        self._clamped = False
        if type(current_step) == int:
            if current_step >= self._max_steps:
                warning_once("current_step should be smaller than "
                             "max_steps. Got %s. Will be clamped to %s" %
                             (current_step, self._max_steps - 1))
                current_step = min(current_step, self._max_steps - 1)
                self._clamped = True
        elif isinstance(current_step, torch.Tensor):
            assert 0 <= current_step.ndim <= 1
            if torch.any(current_step >= self._max_steps):
                warning_once("current_step should be smaller than "
                             "max_steps. Got %s. Will be clamped to %s" %
                             (current_step.max(), self._max_steps - 1))
                current_step = current_step.clamp(max=self._max_steps - 1)
                self._clamped = True
        self._current_step = current_step

    def reset_parameters(self):
        """Reset the parameters."""
        if self._track_running_stats:
            for i in range(self._max_steps):
                self._running_means[i].zero_()
                self._running_vars[i].fill_(1)
                self._num_batches_tracked[i].zero_()
        if self._affine:
            nn.init.ones_(self._weight)
            nn.init.zeros_(self._bias)

    def forward(self, input: torch.Tensor):
        self._check_input_dim(input)

        if self.training or not self._track_running_stats:
            if self._track_running_stats:
                current_step = self._current_step
                if isinstance(current_step,
                              torch.Tensor) and current_step.ndim != 0:
                    assert torch.all(current_step == current_step[0]), (
                        "all current_steps must be same for training.")
                    current_step = current_step[0]
                current_step = int(current_step)
                running_mean = self._running_means[current_step]
                running_var = self._running_vars[current_step]
                if not self._clamped:
                    num_batches_tracked = self._num_batches_tracked[
                        current_step]
                    num_batches_tracked.add_(1)
                    if self._momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(
                            num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self._momentum
                else:
                    exponential_average_factor = 0.0
            else:
                running_mean = None
                running_var = None
                exponential_average_factor = 0.0
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                self._weight,
                self._bias,
                # whether the mini-batch stats should be used for normalization
                # rather than the running stats.
                # If current_step is out of limit, we will use the running stats
                # for max_steps - 1 to normalize the batch so that we can keep
                # training and eval consistent.
                not self._clamped,
                exponential_average_factor,
                self._eps)
        else:  # not training and tracking running stats
            running_means = torch.stack(
                self._running_means, dim=0)[self._current_step]
            running_vars = torch.stack(
                self._running_vars, dim=0)[self._current_step]
            if running_means.ndim == 1:
                running_means = running_means[None, :].expand(input.shape[:2])
                running_vars = running_vars[None, :].expand(input.shape[:2])
            running_means = running_means.reshape(-1)
            running_vars = running_vars.reshape(-1)
            weight = self._weight
            bias = self._bias
            if self._affine:
                weight = weight[None, :].expand(input.shape[:2])
                bias = bias[None, :].expand(input.shape[:2])
            batch_size = input.shape[0]
            input = input.reshape(1, -1, *input.shape[2:])
            y = F.batch_norm(
                input,
                running_means,
                running_vars,
                weight,
                bias,
                # whether the mini-batch stats should be used for normalization
                # rather than the running stats.
                False,
                0.0,  # exponential_average_factor
                self._eps)
            y = y.reshape(batch_size, -1, *y.shape[2:])
            return y


class BatchNorm1d(_NormBase):
    r"""Batch Normalization over a 2D or 3D input.

    For detail about Batch Normalization, see
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

    The main difference is that this implementation supports using BN for RNN.
    The reason is that for RNN, the normalization statics can be dramatically different
    for different step of RNN. Hence we need to maintain different running statistics
    for different step of RNN.

    The following example shows how to use it, assuming ``rnn`` is a ``Network``
    which contains some alf.layers.BatchNorm layers.

    .. code-block:: python

        prepare_rnn_batch_norm(rnn)
        rnn.set_batch_norm_max_steps(5)

        for i in range(t):
            rnn.set_batch_norm_current_step(i)
            y, state = rnn(input[i], state)

    Note that ``set_batch_norm_current_step()`` also accepts Tensor as its argument.
    In that case, it means that the current step for each sample in a batch.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(
                input.dim()))


class BatchNorm2d(_NormBase):
    r"""Applies Batch Normalization over a 4D input.

    For detail about Batch Normalization, see
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

    The main difference is that this implementation supports using BN for RNN.
    The reason is that for RNN, the normalization statics can be dramatically different
    for different step of RNN. Hence we need to maintain different running statistics
    for different step of RNN.

    The following example shows how to use it, assuming ``rnn`` is a ``Network``
    which contains some alf.layers.BatchNorm layers.

    .. code-block:: python

        prepare_rnn_batch_norm(rnn)     # Only need to call once in the lifetime of rnn
        rnn.set_batch_norm_max_steps(5) # Only need to call once in the lifetime of rnn

        for i in range(t):
            rnn.set_batch_norm_current_step(i)
            y, state = rnn(input[i], state)

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))


def set_batch_norm_max_steps(module, max_steps: int):
    """Set max_steps for all batch norm layers in ``module``.

    Args:
        max_steps: the maximum steps for which the batch norm running statistics
            are maintained.
    """
    for bn in module._all_bns:
        bn.set_max_steps(max_steps)


def set_batch_norm_current_step(module: nn.Module,
                                current_step: Union[torch.Tensor, int]):
    """Set current_step for all batch norm layers in ``module``.

    Args:
        current_step: the current step for RNN. If it is a Tensor, it means that
            the current step for each sample in a batch.
    """
    for bn in module._all_bns:
        bn.set_current_step(current_step)


def prepare_rnn_batch_norm(module: nn.Module) -> bool:
    """Prepare an RNN network ``module`` to use alf.layers.BatchNorm layers.

    It will report error if any nn.BatchNorm layer is found within ``module``

    Returns:
        True if alf.layers.BatchNorm layers have been found

        False otherwise.
    """
    bns = set()
    todo = [("", module)]
    visited = set()
    while len(todo) > 0:
        path, m = todo.pop()
        if isinstance(m, (BatchNorm1d, BatchNorm2d)):
            bns.add(m)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            warning_once(
                "RNN may not perform well with torch.nn.BatchNorm layer "
                "(at %s). Consider using alf.layers.BatchNorm instead." % path)
        elif isinstance(m, nn.Module):
            for name, submodule in m.named_children():
                if submodule not in visited:
                    todo.append((path + '.' + name, submodule))
                    visited.add(submodule)

    module._all_bns = bns
    module.set_batch_norm_max_steps = types.MethodType(
        set_batch_norm_max_steps, module)
    module.set_batch_norm_current_step = types.MethodType(
        set_batch_norm_current_step, module)

    return len(bns) > 0


class ParamBatchNorm(nn.Module):
    """Base class for ParamBatchNorm, adapted from ``torch.nn.modules._BatchNorm``
    """

    def __init__(self,
                 num_features,
                 n_groups,
                 eps=1e-5,
                 momentum=0.1,
                 track_running_stats=True):
        """A general Batch Normalization layer that does not maintain learnable 
        affine parameters (weight and bias), but accepts both from users. 
        If ``n_groups`` is greater than 1, it performs parallel Batch Normalization 
        operation.

        This layer maintains running estimates of ``mean`` and ``var`` if 
        ``track_running_stats`` is set to ``True``, just like BatchNorm layers of
        ``torch.nn``.

        Args:
            num_features (int): refer to nn.BatchNorm1d and nn.BatchNorm2d 
            n_groups (int): number of parallel groups
            eps (float): refer to nn.BatchNorm1d and nn.BatchNorm2d 
            momentum (float): refer to nn.BatchNorm1d and nn.BatchNorm2d
            track_running_stats (bool): refer to nn.BatchNorm1d and nn.BatchNorm2d
        """
        super().__init__()
        self._num_features = num_features
        self._n_groups = n_groups
        self._eps = eps
        self._momentum = momentum
        self._track_running_stats = track_running_stats
        self._set_weight(torch.ones(n_groups, self.weight_length))
        self._set_bias(torch.zeros(n_groups, self.bias_length))
        if track_running_stats:
            self.register_buffer('running_mean',
                                 torch.zeros(n_groups * num_features))
            self.register_buffer('running_var',
                                 torch.ones(n_groups * num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()
        self._param_length = None

    def reset_parameters(self) -> None:
        self.reset_running_stats()

    def reset_running_stats(self) -> None:
        if self._track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self._track_running_stats is on
            self.running_mean.zero_()  # type: ignore[operator]
            self.running_var.fill_(1)  # type: ignore[operator]
            self.num_batches_tracked.zero_()  # type: ignore[operator]

    @property
    def weight(self):
        """Get stored weight tensor or batch of weight tensors."""
        return self._weight

    @property
    def bias(self):
        """Get stored bias tensor or batch of bias tensors."""
        return self._bias

    @property
    def num_features(self):
        """Get the n_element of a single weight tensor. """
        return self._num_features

    @property
    def weight_length(self):
        """Get the n_element of a single weight tensor. """
        return self._num_features

    @property
    def bias_length(self):
        """Get the n_element of a single bias tensor. """
        return self._num_features

    @property
    def param_length(self):
        """Get total number of parameters for all layers. """
        if self._param_length is None:
            self._param_length = self.weight_length + self.bias_length
        return self._param_length

    def set_parameters(self, theta, reinitialize=False):
        """Distribute parameters to corresponding parameters.

        Args:
            theta (torch.Tensor): with shape ``[D] (groups=1)``
                                        or ``[B, D] (groups=B)``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``D``: length of parameters, should be self.param_length
                When the shape of inputs is ``[D]``, it will be unsqueezed
                to ``[1, D]``.
            reinitialize (bool): whether to reinitialize parameters of
                each layer.
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[0] == self._n_groups
                and (theta.shape[1] == self.param_length)), (
                    "Input theta has wrong shape %s. Expecting shape (%d, %d)"
                    % (theta.shape, self._n_groups, self.param_length))

        weight = theta[:, :self.weight_length]
        self._set_weight(weight, reinitialize=reinitialize)
        bias = theta[:, self.weight_length:]
        self._set_bias(bias, reinitialize=reinitialize)

    def _set_weight(self, weight, reinitialize=False):
        """Store a weight tensor or batch of weight tensors.

        Args:
            weight (torch.Tensor): with shape ``[B, D]``
                where the mining of the symbols are:
                - ``B``: batch size
                - ``D``: length of weight vector, should be self.weight_length
            reinitialize (bool): whether to reinitialize self._weight
        """
        assert (weight.ndim == 2 and weight.shape[0] == self._n_groups
                and (weight.shape[1] == self.weight_length)), (
                    "Input weight has wrong shape %s. Expecting shape (%d, %d)"
                    % (weight.shape, self._n_groups, self.weight_length))
        if reinitialize:
            weight = torch.ones(self._n_groups, self.weight_length)

        self._weight = weight.reshape(-1)  # [n * weight_length]

    def _set_bias(self, bias, reinitialize=False):
        """Store a bias tensor or batch of bias tensors.

        Args:
            bias (torch.Tensor): with shape ``[B, D]``
                where the mining of the symbols are:
                - ``B``: batch size
                - ``D``: length of bias vector, should be self.bias_length
            reinitialize (bool): whether to reinitialize self._bias
        """
        assert (bias.ndim == 2 and bias.shape[0] == self._n_groups
                and (bias.shape[1] == self.bias_length)), (
                    "Input bias has wrong shape %s. Expecting shape (%d, %d)" %
                    (bias.shape, self._n_groups, self.bias_length))
        if reinitialize:
            bias = torch.zeros(self._n_groups, self.bias_length)

        self._bias = bias.reshape(-1)  # [n * bias_length]

    def _preprocess_input(self, inputs):
        raise NotImplementedError

    def forward(self, inputs, keep_group_dim=True):
        """Forward

        Args:
            inputs (torch.Tensor): refer to ``_preprocess_input`` of subclass
                for detailed description.

        Returns:
            torch.Tensor: for BatchNorm1d, with shape ``[B, n, D]`` or ``[B, n*D]``,
                for BatchNorm2d, with shape ``[B, n, C, H, W]`` or ``[B, n*C, H, W]``.
        """
        inputs = self._preprocess_input(inputs)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self._momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self._momentum

        if self.training and self._track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self._momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self._momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is
                                                           None)
        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean,
                                                       torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var,
                                                      torch.Tensor)

        res = F.batch_norm(
            inputs,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self._track_running_stats else None,
            self.running_var
            if not self.training or self._track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self._eps)

        if self._n_groups > 1 and keep_group_dim:
            res = res.reshape(inputs.shape[0], self._n_groups, -1,
                              *inputs.shape[2:])  # [B, n, D]

        return res


class ParamBatchNorm1d(ParamBatchNorm):
    def _preprocess_input(self, inputs):
        """Check inputs shape and preprocess for BatchNorm1d.

        Args:
            inputs (torch.Tensor): with shape ``[B, D] (groups=1)``
                or ``[B, n, D] (groups=n)``; for BatchNorm2d,
                where the meaning of the symbols are:

                - ``B``: batch size
                - ``n``: number of replicas
                - ``D``: input dimension

                When the shape of inputs is ``[B, D]``, all the n linear
                operations will take inputs as the same shared inputs.
                When the shape of inputs is ``[B, n, D]``, each linear operator
                will have its own input data by slicing inputs.

        Returns:
            torch.Tensor: with shape ``[B, n*D]``
        """
        if inputs.ndim == 2:
            # case 1: non-parallel inputs
            assert inputs.shape[1] == self.num_features, (
                "Input inputs has wrong shape %s. Expecting (B, %d)" %
                (inputs.shape, self.num_features))
            inputs = inputs.repeat(1, self._n_groups)  # [B, n*D]
            # inputs = inputs.unsqueeze(0).expand(self._n_groups, *inputs.shape)
        elif inputs.ndim == 3:
            # case 2: parallel inputs
            assert (
                inputs.shape[1] == self._n_groups
                and inputs.shape[2] == self.num_features), (
                    "Input inputs has wrong shape %s. Expecting (B, %d, %d)" %
                    (inputs.shape, self._n_groups, self.num_features))
            # [B, n*D]
            inputs = inputs.reshape(-1, self._n_groups * self.num_features)
        else:
            raise ValueError("Wrong inputs.ndim=%d" % inputs.ndim)

        return inputs


class ParamBatchNorm2d(ParamBatchNorm):
    def _preprocess_input(self, inputs):
        """Check inputs shape and preprocess for BatchNorm2d.

        Args:
            inputs (torch.Tensor): with shape ``[B, C, H, W] (groups=1)``
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
            torch.Tensor with shape ``[B, n*C, H, W]``
        """
        if self._n_groups == 1:
            # non-parallel layer
            assert (inputs.ndim == 4
                    and inputs.shape[1] == self.num_features), (
                        "Input img has wrong shape %s. Expecting (B, %d, H, W)"
                        % (inputs.shape, self.num_features))
        else:
            # parallel layer
            if inputs.ndim == 4:
                if inputs.shape[1] == self.num_features:
                    # case 1: non-parallel input
                    inputs = inputs.repeat(1, self._n_groups, 1, 1)
                else:
                    # case 2: parallel input
                    assert inputs.shape[
                        1] == self._n_groups * self.num_features, (
                            "Input img has wrong shape %s. Expecting (B, %d, H, W) or (B, %d, H, W)"
                            % (inputs.shape, self.num_features,
                               self._n_groups * self.num_features))
            elif inputs.ndim == 5:
                # case 3: parallel input with unmerged group dim
                assert (
                    inputs.shape[1] == self._n_groups
                    and inputs.shape[2] == self.num_features
                ), ("Input img has wrong shape %s. Expecting (B, %d, %d, H, W)"
                    % (inputs.shape, self._n_groups, self.num_features))
                # merge group and channel dim
                inputs = inputs.reshape(inputs.shape[0],
                                        inputs.shape[1] * inputs.shape[2],
                                        *inputs.shape[3:])
            else:
                raise ValueError("Wrong img.ndim=%d" % inputs.ndim)

        return inputs
