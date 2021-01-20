# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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


@gin.configurable
class SimpleFC(nn.Linear):
    """
    A simple FC layer that record its output before activation.
    It is for used in the ReluMLP to enable explicit computation
    of diagonals of input-output Jacobian.
    """

    def __init__(self, input_size, output_size, activation=identity):
        """
        Initialize a SimpleFC layer.

        Args:
            input_size (int): input dimension.
            output_size (int): output dimension.
            activation (nn.functional): activation used for this layer.
                Default is math_ops.identity.
        """
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
    """
    A MLP with relu activations. Diagonals of input-output Jacobian
    can be computed directly without calling autograd.
    """

    def __init__(self,
                 input_tensor_spec,
                 output_size=None,
                 hidden_layers=(64, 64),
                 activation=torch.relu_,
                 name="ReluMLP"):
        """Create a ReluMLP.

        Args:
            input_tensor_spec (TensorSpec):
            hidden_layers (tuple): size of hidden layers.
            activation (nn.functional): activation used for each layer.
            name (str):
        """
        assert len(input_tensor_spec.shape) == 1, \
            ("The input shape {} should be a 1-d vector!".format(
                input_tensor_spec.shape
            ))

        super().__init__(input_tensor_spec, name=name)

        self._input_size = input_tensor_spec.shape[0]
        self._output_size = output_size
        if self._output_size is None:
            self._output_size = self._input_size
        self._hidden_layers = hidden_layers
        self._n_hidden_layers = len(hidden_layers)

        self._fc_layers = nn.ModuleList()
        input_size = self._input_size
        for size in hidden_layers:
            fc = SimpleFC(input_size, size, activation=activation)
            self._fc_layers.append(fc)
            input_size = size

        last_fc = SimpleFC(input_size, self._output_size, activation=identity)
        self._fc_layers.append(last_fc)

    def forward(self,
                inputs,
                state=(),
                requires_jac=False,
                requires_jac_diag=False):
        """
        Args:
            inputs (torch.Tensor)
            state: not used
            requires_jac (bool): whether outputs input-output Jacobian.
            requires_jac_diag (bool): whetheer outputs diagonals of Jacobian.
        """
        ndim = inputs.ndim
        if ndim == 1:
            inputs = inputs.unsqueeze(0)
        assert inputs.ndim == 2 and inputs.shape[-1] == self._input_size, \
            ("inputs should has shape (B, {})!".format(self._input_size))

        z = inputs
        for fc in self._fc_layers:
            z = fc(z)
        if ndim == 1:
            z = z.squeeze(0)
        if requires_jac:
            z = (z, self._compute_jac())
        elif requires_jac_diag:
            z = (z, self._compute_jac_diag())

        return z, state

    def compute_jac(self, inputs):
        """Compute the input-output Jacobian. """

        assert inputs.ndim <= 2 and inputs.shape[-1] == self._input_size, \
            ("inputs should has shape {}!".format(self._input_size))

        self.forward(inputs)
        J = self._compute_jac()
        if inputs.ndim == 1:
            J = J.squeeze(0)

        return J

    def _compute_jac(self):
        """Compute the input-output Jacobian. """

        mask = (self._fc_layers[-2].hidden_neurons > 0).float()
        J = torch.einsum('ia,ba,aj->bij', self._fc_layers[-1].weight, mask,
                         self._fc_layers[-2].weight)
        for fc in reversed(self._fc_layers[0:-2]):
            mask = (fc.hidden_neurons > 0).float()
            J = torch.einsum('bia,ba,aj->bij', J, mask, fc.weight)

        return J  # [B, n_out, n_in]

    def compute_jac_diag(self, inputs):
        """Compute diagonals of the input-output Jacobian. """

        assert inputs.ndim <= 2 and inputs.shape[-1] == self._input_size, \
            ("inputs should has shape {}!".format(self._input_size))

        self.forward(inputs)
        J_diag = self._compute_jac_diag()
        if inputs.ndim == 1:
            J_diag = J_diag.squeeze(0)

        return J_diag

    def _compute_jac_diag(self):
        """Compute diagonals of the input-output Jacobian. """

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

    def compute_vjp(self, inputs, vec):
        """Compute vector-Jacobian product. 

        Args:
            inputs (Tensor): size (self._input_size) or (B, self._input_size)
            vec (Tensor): the vector for which the vector-Jacobian product
                is computed. Must be of size (self._output_size) or
                (B, self._output_size). 

        Returns:
            vjp (Tensor): size (self._input_size) or (B, self._input_size).
        """

        ndim = inputs.ndim
        assert vec.ndim == ndim, \
            ("ndim of inputs and vec must be consistent!")
        if ndim > 1:
            assert ndim == 2, \
                ("inputs must be a vector or matrix!")
            assert inputs.shape[0] == vec.shape[0], \
                ("batch size of inputs and vec must agree!")
        assert inputs.shape[-1] == self._input_size, \
            ("inputs should has shape {}!".format(self._input_size))
        assert vec.shape[-1] == self._output_size, \
            ("vec should has shape {}!".format(self._output_size))

        self.forward(inputs)

        return self._compute_vjp(vec)

    def _compute_vjp(self, vec):
        """Compute vector-Jacobian product. """

        ndim = vec.ndim
        if ndim == 1:
            vec = vec.unsqueeze(0)

        J = torch.matmul(vec, self._fc_layers[-1].weight)
        for fc in reversed(self._fc_layers[0:-1]):
            mask = (fc.hidden_neurons > 0).float()
            J = torch.matmul(J * mask, fc.weight)

        if ndim == 1:
            J = J.squeeze(0)

        return J  # [B, n_in] or [n_in]
