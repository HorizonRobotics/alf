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

import torch
import torch.nn as nn
import typing

import alf
from alf.utils.common import expand_dims_as
from .network import Network, wrap_as_network

__all__ = ['LSTMCell', 'GRUCell', 'Residue', 'TemporalPool']


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

    def __init__(self, input_size, hidden_size):
        """
        Args:
            input_size (int): The number of expected features in the input `x`
            hidden_size (int): The number of features in the hidden state `h`
        """
        state_spec = (alf.TensorSpec((hidden_size, )),
                      alf.TensorSpec((hidden_size, )))
        super().__init__(
            input_tensor_spec=alf.TensorSpec((input_size, )),
            state_spec=state_spec)
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

    def __init__(self, input_size, hidden_size):
        """
        Args:
            input_size (int): The number of expected features in the input `x`
            hidden_size (int): The number of features in the hidden state `h`
        """
        super().__init__(
            input_tensor_spec=alf.TensorSpec((input_size, )),
            state_spec=alf.TensorSpec((hidden_size, )))
        self._cell = nn.GRUCell(input_size, hidden_size)

    def forward(self, input, state):
        h = self._cell(input, state)
        return h, h


class Residue(Network):
    """Residue block.

    It performs ``y = activation(x + block(x))``.
    """

    def __init__(self, block, input_tensor_spec=None, activation=torch.relu_):
        """
        Args:
            block (Callable):
            input_tensor_spec (nested TensorSpec): input tensor spec for ``block``
                if it cannot be infered from ``block``
            activation (Callable): activation function
        """
        block = wrap_as_network(block, input_tensor_spec)
        super().__init__(input_tensor_spec=block.input_tensor_spec)
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
                 mode='skip'):
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
        super().__init__(input_tensor_spec, state_spec=state_spec)

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
