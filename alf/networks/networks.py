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
"""Various concrete Networks."""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Callable, Dict, Optional, Tuple

import alf
from alf.initializers import variance_scaling_init
from alf.utils.math_ops import identity
from alf.utils.common import expand_dims_as, is_eval
from .network import Network, wrap_as_network

__all__ = [
    'LSTMCell', 'GRUCell', 'NoisyFC', 'Residue', 'TemporalPool', 'Delay',
    'AMPWrapper'
]


class LSTMCell(Network):
    r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.
    """

    def __init__(self, input_size, hidden_size, name='LSTMCell'):
        """
        Args:
            input_size (int): The number of expected features in the input `x`
            hidden_size (int): The number of features in the hidden state `h`
        """
        state_spec = (alf.TensorSpec((hidden_size, )),
                      alf.TensorSpec((hidden_size, )))
        super().__init__(
            input_tensor_spec=alf.TensorSpec((input_size, )),
            state_spec=state_spec,
            name=name)
        self._cell = nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size)

    def forward(self, input, state):
        h_state, c_state = self._cell(input, state)
        return h_state, (h_state, c_state)


class GRUCell(Network):
    r"""A gated recurrent unit (GRU) cell

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.
    """

    def __init__(self, input_size, hidden_size, name='GRUCell'):
        """
        Args:
            input_size (int): The number of expected features in the input `x`
            hidden_size (int): The number of features in the hidden state `h`
        """
        super().__init__(
            input_tensor_spec=alf.TensorSpec((input_size, )),
            state_spec=alf.TensorSpec((hidden_size, )),
            name=name)
        self._cell = nn.GRUCell(input_size, hidden_size)

    def forward(self, input, state):
        h = self._cell(input, state)
        return h, h


class Residue(Network):
    """Residue block.

    It performs ``y = activation(x + block(x))``.
    """

    def __init__(self,
                 block,
                 input_tensor_spec=None,
                 activation=torch.relu_,
                 name='Residue'):
        """
        Args:
            block (Callable):
            input_tensor_spec (nested TensorSpec): input tensor spec for ``block``
                if it cannot be inferred from ``block``
            activation (Callable): activation function
        """
        block = wrap_as_network(block, input_tensor_spec)
        super().__init__(
            input_tensor_spec=block.input_tensor_spec,
            state_spec=block.state_spec,
            name='Residue')
        self._block = block
        self._activation = activation

    def forward(self, x, state=()):
        y, state = self._block(x, state)
        return self._activation(x + y), state


class TemporalPool(Network):
    """Pool features temporally.

    Suppose input_size=(), stack_size=2, pooling_size=2, the following table
    shows the output of different mode for an input sequence of 1,2,3,4,5 (ignoring
    batch dimension)

           1,        2,        3,        4,          5
    skip:  [0, 1],   [0, 1],   [1, 3],   [1, 3],     [3, 5]
    avg:   [0, 0],   [0, 1.5], [0, 1.5], [1.5, 3.5], [1.5, 3.5]
    max:   [0, 0],   [0, 2],   [0, 2],   [2, 4],     [2, 4]

    Note that for 'avg' and 'max', the result is zero for the first ``pooling_size - 1``
    steps because it needs ``pooling_size`` input to calculate the result. After
    that, the output changes every ``pooling_size`` steps as the new pooling result
    available. On the other hand, for 'skip', the first input is immediately
    reflected in the output because it is a valid way of skipping.

    Example:

    .. code-block:: python

        # A temporal CNN with progressively large temporal receptive field.
        cnn = alf.networks.Sequential([
            alf.networks.TemporalPool(256, 3, 1),
            torch.nn.Flatten(),
            alf.layers.FC(768, 256, activation=torch.relu_),
            alf.networks.TemporalPool(256, 3, 2),
            torch.nn.Flatten(),
            alf.layers.FC(768, 256, activation=torch.relu_),
            alf.networks.TemporalPool(256, 3, 4),
            torch.nn.Flatten(),
            alf.layers.FC(768, 256, activation=torch.relu_)])


    Note that the output of the above network changes every 4 steps, which may make
    the response too slow for many tasks. So a practical way of using ``TemporalPool``
    is to combine it with ``Residue`` so that the output will not lag:

    .. code-block:: python

        block = alf.networks.Residue(
            alf.networks.Sequential([
                alf.networks.TemporalPool(256, 3, 2),
                torch.nn.Flatten(),
                alf.layers.FC(768, 256, activation=torch.relu_)]))

    """

    def __init__(self,
                 input_size,
                 stack_size,
                 pooling_size=1,
                 dtype=torch.float32,
                 mode='skip',
                 name='TemporalPool'):
        """
        Args:
            input_size (int|tuple[int]): shape of the input
            stack_size (int): stack the features from so many steps
            pooling_size (int): if > 1, perform a pooling first. ``pooling_size``
                steps of features will be pooled as single feature vector according
                to ``mode``
            mode (str): one of ('skip', 'avg', 'max'), only effective if pooling_size > 1.
                'skip': only keeping features at step ``t * pooling_size``
                'avg': features are averaged for each window of ``pooling_size`` steps.
                    The pooling results for first ``pooling_size - 1`` steps are 0.
                'max': features are maxed for each window of ``pooling_size`` steps
                    The pooling results for first ``pooling_size - 1`` steps are 0.
        Returns:
            tuple of:
            - tensor of shape (stack_size, input_size)
            - internal states
        """
        if isinstance(input_size, typing.Iterable):
            input_size = tuple(input_size)
        else:
            input_size = (input_size, )
        shape = (stack_size, ) + input_size
        input_tensor_spec = alf.TensorSpec(input_size, dtype=dtype)
        self._pooling_size = pooling_size
        if pooling_size == 1:
            state_spec = alf.TensorSpec((stack_size - 1, ) + input_size, dtype)
        elif mode == 'skip':
            self._pool_func = self._skip_pool
            pool_state_spec = ()
            self._update_step = 1
        elif mode == 'avg':
            self._pool_func = self._avg_pool
            pool_state_spec = input_tensor_spec
            self._update_step = 0
        elif mode == 'max':
            self._pool_func = self._max_pool
            pool_state_spec = input_tensor_spec
            self._update_step = 0
        else:
            raise ValueError("Unknown mode '%s'" % mode)

        if pooling_size > 1:
            state_spec = (alf.TensorSpec(shape, input_tensor_spec.dtype),
                          pool_state_spec, alf.TensorSpec((),
                                                          dtype=torch.int64))
        super().__init__(input_tensor_spec, state_spec=state_spec, name=name)

    def forward(self, x, state):
        if self._pooling_size == 1:
            output = torch.cat([state, x.unsqueeze(1)], dim=1)
            return output, output[:, 1:, ...]
        else:
            output, pool_state, step = state
            step = step + 1
            pool, pool_state = self._pool_func(x, pool_state, step)
            step = step % self._pooling_size
            output = torch.where(
                expand_dims_as(step == self._update_step, output),
                torch.cat(
                    [output[:, 1:, ...], pool.unsqueeze(1)], dim=1), output)
            return output, (output, pool_state, step)

    def _skip_pool(self, x, state, step):
        return x, ()

    def _avg_pool(self, x, state, step):
        w = expand_dims_as(1. / step.to(torch.float32), x)
        state = torch.where(
            expand_dims_as(step == 1, x), x, torch.lerp(state, x, w))
        return state, state

    def _max_pool(self, x, state, step):
        state = torch.where(
            expand_dims_as(step == 1, x), x, torch.max(x, state))
        return state, state


class Delay(Network):
    """The output is the input of the ``delay`` step ago.

    Args:
        input_tensor_spec (nested TensorSpec): representing the input
        delay (int): if 0, there is no delay and the output is same as the input.
    """

    def __init__(self, input_tensor_spec, delay=1, name='Delay'):
        if delay == 0:
            state_spec = ()
            self._forward = lambda i, s: (i, ())
        elif delay == 1:
            state_spec = input_tensor_spec
            self._forward = lambda i, s: (s, i)
        else:
            state_spec = (input_tensor_spec, ) * delay
            self._forward = lambda i, s: (s[0], s[1:] + (i, ))

        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=state_spec,
            name=name)

    def forward(self, input, state):
        return self._forward(input, state)


class AMPWrapper(Network):
    """Wrap a network to run in a given AMP context.

    Args:
        enabled: whether to enable AMP autocast
        net: the wrapped network
    """

    def __init__(self, enabled: bool, net: Network):
        super().__init__(
            net.input_tensor_spec, state_spec=net.state_spec, name=net.name)
        self._net = net
        self._enabled = enabled

    def forward(self, input, state):
        if torch.is_autocast_enabled() and not self._enabled:
            input = alf.layers.to_float32(input)
        with torch.cuda.amp.autocast(self._enabled):
            return self._net(input, state)


@alf.configurable
@alf.repr_wrapper
class NoisyFC(Network):
    r"""The Noisy Linear Layer described in

    Fortunato et. al. `Noisy Networks for Exploration <https://arxiv.org/abs/1706.10295>`_

    In short, the original weight :math:`w` and bias :math:`b` of FC layer are replaced
    with :math:`w + w_\sigma \odot \epislon^w` and :math:`b + b_\sigma \odot \epsion^b` where
    :math:`\epsilon^w` and :math:`\epsilon^b` are noise and :math:`w, w_\sigma, b, b_\sigma`
    are trainable parameters.

    Some details:

    1. The noise for each sample in a batch is different.
    2. The noise is maintained as state. It has a probability of `new_noise_prob`
       to change to new noise.
    3. Since the initial state is always 0, a new noise will always be generated
       for zero state.
    4. If it is running in eval mode (i.e., common.is_eval() is True), noise will
       be disabled (i.e. same as alf.layers.FC).
    5. The noise is factorized Gaussian noise as described in the paper.


    Args:
        input_size: input size.
        output_size: output size.
        activation: activation function.
        std_init: the scaling factor for the initial value of weight_sigma
            and bias_sigma.
        new_noise_prob: the probability of resample the noise.
        use_bn: whether use batch normalization.
        use_ln: whether use layer normalization
        bn_ctor: will be called as ``bn_ctor(num_features)`` to
            create the BN layer.
        kernel_initializer: initializer for the FC layer kernel.
            If none is provided a ``variance_scaling_initializer`` with gain as
            ``kernel_init_gain`` will be used.
        kernel_init_gain: a scaling factor (gain) applied to
            the std of kernel init distribution. It will be ignored if
            ``kernel_initializer`` is not None.
        bias_init_value: a constant for the initial bias value.
            This is ignored if ``bias_initializer`` is provided.
        bias_initializer:  initializer for the bias parameter.
        weight_opt_args: If provided, it will be used as optimizer arguments
            for weight. And it will be combined with zero_mean=False and
            fixed_norm=False as optimizer arguments for weight_sigma.
        bias_opt_args: If provided, it will be used as optimizer arguments
            for bias. And it will be combined with zero_mean=False as
            optimizer arguments for bias_sigma.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 std_init: float = 0.5,
                 new_noise_prob: float = 0.01,
                 activation: Callable = identity,
                 use_bn: bool = False,
                 use_ln: bool = False,
                 bn_ctor: Callable = nn.BatchNorm1d,
                 kernel_initializer: Optional[Callable] = None,
                 kernel_init_gain: float = 1.0,
                 bias_init_value: float = 0.0,
                 bias_initializer: Optional[Callable] = None,
                 weight_opt_args: Optional[Dict] = None,
                 bias_opt_args: Optional[Dict] = None):
        super().__init__(
            input_tensor_spec=alf.TensorSpec((input_size, )),
            state_spec=(alf.TensorSpec((input_size, )),
                        alf.TensorSpec((output_size, )))),
        self._input_size = input_size
        self._output_size = output_size
        self._activation = activation
        self._std_init = std_init
        self._weight = nn.Parameter(torch.empty(output_size, input_size))
        self._weight_sigma = nn.Parameter(torch.empty(output_size, input_size))
        self._bias = nn.Parameter(torch.empty(output_size))
        self._bias_sigma = nn.Parameter(torch.empty(output_size))
        self._use_bn = use_bn
        self._use_ln = use_ln
        if use_bn:
            self._bn = bn_ctor(output_size)
        else:
            self._bn = None
        if use_ln:
            self._ln = nn.LayerNorm(output_size)
        else:
            self._ln = None
        self._new_noise_prob = new_noise_prob
        self._kernel_initializer = kernel_initializer
        self._kernel_init_gain = kernel_init_gain
        self._bias_init_value = bias_init_value
        self._bias_initializer = bias_initializer
        self.reset_parameters()
        if weight_opt_args:
            self._weight.opt_args = weight_opt_args
            weight_opt_args = copy.copy(weight_opt_args)
            weight_opt_args['zero_mean'] = False
            weight_opt_args['fixed_norm'] = False
            self._weight_sigma.opt_args = weight_opt_args
        if bias_opt_args and self._bias is not None:
            self._bias.opt_args = bias_opt_args
            bias_opt_args = copy.copy(bias_opt_args)
            bias_opt_args['zero_mean'] = False
            self._bias_sigma.opt_args = bias_opt_args

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias

    def reset_parameters(self):
        """Initialize the parameters."""
        if self._kernel_initializer is None:
            variance_scaling_init(
                self._weight.data,
                gain=self._kernel_init_gain,
                nonlinearity=self._activation)
        else:
            self._kernel_initializer(self._weight.data)
        self._weight_sigma.data.fill_(
            self._std_init / math.sqrt(self._input_size))
        if self._bias_initializer is not None:
            self._bias_initializer(self._bias.data)
        else:
            nn.init.constant_(self._bias.data, self._bias_init_value)
        self._bias_sigma.data.fill_(
            self._std_init / math.sqrt(self._output_size))
        if self._use_ln:
            self._ln.reset_parameters()
        if self._use_bn:
            self._bn.reset_parameters()

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input: torch.Tensor, state: Tuple[torch.Tensor]):
        """Forward computation.

        Args:
            inputs: its shape should be ``[batch_size, input_size]`
            state: tuple of noise
        Returns:
            Tensor: with shape as ``[batch_size, output_size]``
        """
        epsilon_in, epsilon_out = state
        # y = bias + input @ weight.t()
        y = torch.addmm(self._bias, input, self._weight.t())
        if not is_eval():
            batch_size = input.shape[0]
            new_epsilon_in = self._scale_noise((batch_size, self._input_size))
            new_epsilon_out = self._scale_noise((batch_size,
                                                 self._output_size))
            new_noise = torch.rand(batch_size) < self._new_noise_prob
            # The initial state is always 0. So we need to generate new noise
            # for initial state.
            new_noise = new_noise | ((epsilon_in == 0).all(dim=1) &
                                     (epsilon_out == 0).all(dim=1))
            new_noise = new_noise.unsqueeze(-1)
            epsilon_in = torch.where(new_noise, new_epsilon_in, epsilon_in)
            epsilon_out = torch.where(new_noise, new_epsilon_out, epsilon_out)
            noise_in = input * epsilon_in
            # x = bias_sigma + noise_in @ weight_sigma.t()
            x = torch.addmm(self._bias_sigma, noise_in, self._weight_sigma.t())
            # Although y.addcmul_(x, epsilon_out) is better, it can have problem
            # of dtype mismatch when AMP is enabled.
            y = y + x * epsilon_out
        if self._use_ln:
            y = self._ln(y)
        if self._use_bn:
            y = self._bn(y)
        return self._activation(y), (epsilon_in, epsilon_out)
