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
from torch.autograd.functional import jvp
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

import alf
from alf.algorithms.hypernetwork_layers import ParamFC
from alf.algorithms.hypernetwork_networks import ParamNetwork
from alf.initializers import variance_scaling_init
from alf.layers import FC
from alf.networks import Network
from alf.tensor_specs import TensorSpec
from alf.utils.math_ops import identity


class SimpleFC(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 activation=torch.relu_,
                 use_bias=False):

        super(SimpleFC, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._activation = activation
        self._weight_length = output_size * input_size
        self._bias_length = 0
        self._bias = None
        self._hidden_neurons = None

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

    def set_weight(self, weight):
        """Store a weight tensor or batch of weight tensors."""
        # weight = weight.view(-1)
        assert (weight.ndim == 1 and len(weight) == self._weight_length), (
            "Input weight has wrong shape %s. Expecting shape (%d,)" %
            (weight.shape, self._weight_length))
        self._weight = weight.view(self._output_size, self._input_size)

    @property
    def hidden_neurons(self):
        return self._hidden_neurons

    def forward(self, inputs):
        self._hidden_neurons = self._activation(inputs.matmul(self.weight.t()))
        return self._hidden_neurons


@gin.configurable
class ReluMLP(ParamNetwork):
    """Creates an instance of ``ReluMLP`` with one bottleneck layer.
    """

    def __init__(self,
                 input_tensor_spec,
                 hidden_layers=((64, False), ),
                 activation=torch.relu_,
                 initializer=None,
                 kernel_initializer=torch.nn.init.normal_,
                 kernel_init_gain=1.0,
                 name="ReluMLP"):
        r"""Create a ReluMLP.

        Args:
            input_tensor_spec (TensorSpec):
            hidden_layers (tuple): size of hidden layers.
            activation (nn.functional):
            name (str):
        """
        self._input_size = input_tensor_spec.shape[0]
        self._output_size = self._input_size
        self._n_hidden_layers = len(hidden_layers)
        self._kernel_initializer = kernel_initializer
        self._kernel_init_gain = kernel_init_gain

        super().__init__(
            input_tensor_spec,
            fc_layer_params=hidden_layers,
            fc_layer_ctor=SimpleFC,
            activation=activation,
            last_layer_param=(self._output_size, False),
            last_activation=identity,
            name=name)

        self.set_parameters(torch.randn(self.param_length))

    @property
    def params(self):
        return self._params

    def set_parameters(self, params=None):
        if params is None:
            self._kernel_initializer(self._params)
            self._params = self._params / 8
        else:
            assert (params.ndim == 1 and len(params) == self.param_length)
            params.requires_grad = True
            self._params = params
        pos = 0
        for fc_l in self._fc_layers:
            weight_length = fc_l.weight_length
            fc_l.set_weight(self.params[pos:pos + weight_length])
            pos = pos + weight_length
            if fc_l.bias is not None:
                bias_length = fc_l.bias_length
                fc_l.set_bias(self.params[pos:pos + bias_length])
                pos = pos + bias_length

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

    def ntk_svgd(self, inputs, loss_func, temperature=1.0):
        """Compute the ntk logp and ntk_grad

        """
        assert inputs.ndim == 2 and inputs.shape[-1] == self._input_size, \
            ("inputs should has shape (batch, {})!".format(self._input_size))

        num_particles = inputs.shape[0] // 2
        inputs_i, inputs_j = torch.split(inputs, num_particles, dim=0)

        def _param_forward_i(params):
            params.requires_grad = True
            self.set_parameters(params)
            return self.forward(inputs_i.detach().clone())[0]

        # prepare for the first term: ntk_logp
        loss_inputs = inputs_j
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(),
                                        inputs_j)[0].detach()  # [bj, n]

        # prepare for the second term: grad of ntk
        outputs_j, _ = self.forward(inputs_j)  # [bj, n]
        jac_diag_j = self._compute_jac_diag()  # [bj, n]

        # combine both 'vector' to apply jvp
        combined_loss = (
            outputs_j * loss_grad).sum() - temperature * jac_diag_j.sum()
        combined_grad_j = torch.autograd.grad(combined_loss, self.params)[0]
        combined_vec_j = combined_grad_j.detach() / num_particles
        ntk_grad = jvp(_param_forward_i, self.params,
                       combined_vec_j)[1]  # [bi, n]

        return ntk_grad, inputs_i, loss
