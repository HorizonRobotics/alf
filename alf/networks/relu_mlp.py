# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import gin
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

import alf
from alf.layers import FC
from alf.networks import Network
from alf.tensor_specs import TensorSpec
from alf.utils.math_ops import identity

# @gin.configurable
# class SimpleFC(FC):
#     def __init__(self,
#                  input_size,
#                  output_size,
#                  activation=identity,
#                  kernel_initializer=None):
#         """A fully connected layer that's also responsible for activation and
#         customized weights initialization. An auto gain calculation might depend
#         on the activation following the linear layer. Suggest using this wrapper
#         module instead of ``nn.Linear`` if you really care about weight std after
#         init.
#         Args:
#             input_size (int): input size
#             output_size (int): output size
#             activation (torch.nn.functional):
#             kernel_initializer (Callable): initializer for the FC layer kernel.
#                 If none is provided a ``variance_scaling_initializer`` with gain as
#                 ``kernel_init_gain`` will be used.
#         """
#         super().__init__(input_size,
#                          output_size,
#                          activation=activation,
#                          use_bias=True,
#                          use_bn=False,
#                          kernel_initializer=kernel_initializer)

#         self._hidden_neurons = None

#     @property
#     def hidden_neurons(self):
#         return self._hidden_neurons

#     def forward(self, inputs):
#         self._hidden_neurons = super().forward(inputs)
#         return self._hidden_neurons


@gin.configurable
class SimpleFC(nn.Linear):
    def __init__(self, input_size, output_size, activation=identity):
        super().__init__(input_size, output_size)
        self._activation = activation
        self._hidden_neurons = None

    @property
    def hidden_neurons(self):
        return self._hidden_neurons

    def forward(self, inputs):
        self._hidden_neurons = super().forward(inputs)
        return self._activation(self._hidden_neurons)


@gin.configurable
class ReluMLP(Network):
    def __init__(
            self,
            input_tensor_spec,
            hidden_layers=(64, 64),
            activation=torch.relu_,
            kernel_initializer=None,  # torch.nn.init.normal_,
            name="ReluMLP"):
        """Create a ReluMLP.

        Args:
            input_tensor_spec (TensorSpec):
            hidden_layers (tuple): size of hidden layers.
            activation (nn.functional):
            name (str):
        """
        assert len(input_tensor_spec.shape) == 1, \
            ("The input shape {} should be a 1-d vector!".format(
                input_tensor_spec.shape
            ))

        super().__init__(input_tensor_spec, name=name)

        self._input_size = input_tensor_spec.shape[0]
        self._output_size = self._input_size
        self._hidden_layers = hidden_layers
        self._n_hidden_layers = len(hidden_layers)
        self._kernel_initializer = kernel_initializer

        self._fc_layers = nn.ModuleList()
        input_size = self._input_size
        for size in hidden_layers:
            fc = SimpleFC(input_size, size, activation=activation)
            # kernel_initializer=kernel_initializer)
            self._fc_layers.append(fc)
            input_size = size

        last_fc = SimpleFC(input_size, self._output_size, activation=identity)
        # kernel_initializer=kernel_initializer)
        self._fc_layers.append(last_fc)

    def forward(self, inputs, state=(), requires_jac_diag=False):
        """
        Args:
            inputs (Tensor)
            state: not used
        """
        inputs = inputs.squeeze()
        assert inputs.shape[-1] == self._input_size, \
            ("inputs should has shape {}!".format(self._input_size))

        z = inputs
        for fc in self._fc_layers:
            z = fc(z)
        if requires_jac_diag:
            z = (z, self._compute_jac_diag())

        return z, state

    def compute_jac_diag(self, inputs):
        """Compute diagonals of the input-output jacobian. """

        inputs = inputs.squeeze()
        assert inputs.shape[-1] == self._input_size, \
            ("inputs should has shape {}!".format(self._input_size))

        self.forward(inputs)

        return self._compute_jac_diag()

    def _compute_jac_diag(self):
        """Compute diagonals of the input-output jacobian. """

        mask = (self._fc_layers[-2].hidden_neurons > 0).float()
        if self._n_hidden_layers == 1:
            J = torch.einsum('ia,ba,ai->bi', self._fc_layers[-1].weight, mask,
                             self._fc_layers[0].weight)  # [B, n]
        else:
            J = torch.einsum('ia,ba,aj->bij', self._fc_layers[-1].weight, mask,
                             self._fc_layers[-2].weight)
            for fc in reversed(self._fc_layers[1:-2]):
                mask = (fc.hidden_neurons > 0).float()
                J = torch.einsum('bia,ba,aj->bij', J, mask, fc.weight)

            mask = (self._fc_layers[0].hidden_neurons > 0).float()
            J = torch.einsum('bia,ba,ai->bi', J, mask,
                             self._fc_layers[0].weight)  # [B, n]

        return J
