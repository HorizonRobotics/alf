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

import alf

from .network import Network, wrap_as_network


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
        super().__init__(input_tensor_spec=alf.TensorSpec((input_size, )))
        self._cell = nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size)
        self._state_spec = (alf.TensorSpec((hidden_size, )),
                            alf.TensorSpec((hidden_size, )))

    @property
    def state_spec(self):
        return self._state_spec

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
        super().__init__(input_tensor_spec=alf.TensorSpec((input_size, )))
        self._cell = nn.GRUCell(input_size, hidden_size)
        self._state_spec = alf.TensorSpec((hidden_size, ))

    @property
    def state_spec(self):
        return self._state_spec

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
