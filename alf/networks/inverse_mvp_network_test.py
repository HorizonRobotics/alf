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

from absl.testing import parameterized
import torch
import torch.nn as nn
import alf
from alf.data_structures import AlgStep, LossInfo
from alf.algorithms.algorithm import Algorithm
from alf.networks.network import Network
from alf.networks.relu_mlp import ReluMLP
from alf.tensor_specs import TensorSpec
from alf.utils.math_ops import identity

import functools
from alf.networks.encoding_networks import EncodingNetwork
from alf.initializers import variance_scaling_init


class InverseMVPAlgorithm(Algorithm):
    r"""InverseMVPNetwork Algorithm
    It is used to predict :math:`x=J^{-1}*vec` given vec for the purpose of 
    optimizing a downstream objective Jx - vec = 0. 
    """

    def __init__(self,
                 net: Network = None,
                 optimizer=None,
                 name="InvMVPAlgorithm"):
        r"""Create a Inverse MVP Algorithm.
        Args:
            net (Network): network for predicting outputs from inputs.
            name (str): name of this Algorithm.
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

        self._net = net

    def predict_step(self, inputs, state=None):
        """Predict for one step of inputs.
        Args:
            inputs (tuple of Tensors): inputs (inputs, vec) for prediction.
            state: not used.
            
        Returns:
            AlgStep:
            - output (torch.Tensor): predictions
                if requires_jac_diag is True.
            - state: not used.
        """
        outputs = self._net(inputs)[0]
        return AlgStep(output=outputs, state=(), info=())


def jacobian(y, x, create_graph=False):
    """It is from Adam Paszke's implementation:
    https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
    """
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(
            flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.

    return torch.stack(jac).reshape(y.shape + x.shape)


def create_mvp_network(input_spec, hidden_size, num_hidden_layers):
    input_dim = input_spec[0].shape[0]
    vec_dim = input_spec[1].shape[0]

    kernel_initializer = functools.partial(
        variance_scaling_init,
        gain=1.0 / 2.0,
        mode='fan_in',
        distribution='truncated_normal',
        nonlinearity=identity)

    mvp_network = EncodingNetwork(
        input_spec,
        input_preprocessors=(torch.nn.Linear(input_dim, hidden_size),
                             torch.nn.Linear(vec_dim, hidden_size)),
        preprocessing_combiner=alf.layers.NestConcat(),
        fc_layer_params=(2 * hidden_size, ) * num_hidden_layers,
        activation=torch.relu_,
        kernel_initializer=kernel_initializer,
        last_layer_size=input_dim,
        last_activation=identity)

    return mvp_network


class InverseMVPTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    def minimize_step(self, inputs, vec):
        y = self.inverse_mvp.predict_step((inputs, vec)).output
        jac_y, _ = self.mlp.compute_vjp(inputs, y)
        jac_y = jac_y.reshape(*vec.shape)
        loss = torch.nn.functional.mse_loss(jac_y, vec)
        return loss

    @parameterized.parameters(
        dict(batch_size=2, input_dim=2, vec_dim=2),
        dict(batch_size=5, input_dim=2, vec_dim=2),
        dict(batch_size=10, input_dim=2, vec_dim=2),
    )
    def test_inverse_mvp(self, batch_size=2, input_dim=2, vec_dim=2):
        r"""
        The InverseMVP network is an encoding network that is trained to
        predict the inverse Jacobian vector product
        :math:`J^{-1}v` with respect to an MLP on a randomly generated input.
        Using relu_mlp we can compute this exactly, and check that the
        trained network is correct. 
        """
        input_spec = TensorSpec(shape=(input_dim, ))
        vec_spec = TensorSpec(shape=(vec_dim, ))
        input_tensor_spec = (input_spec, vec_spec)
        net = create_mvp_network(
            input_tensor_spec, hidden_size=300, num_hidden_layers=1)
        optimizer = alf.optimizers.Adam(lr=1e-4)
        self.inverse_mvp = InverseMVPAlgorithm(net=net, optimizer=optimizer)
        mlp_spec = TensorSpec((input_dim, ))
        self.mlp = ReluMLP(
            mlp_spec,
            output_size=input_dim,
            activation=identity,
            hidden_layers=(1, ))
        # make Jac better behaved
        w1 = torch.tensor([[1., 2.], [2., 1.]])
        w2 = torch.tensor([[2., 1.], [1., 2.]])

        self.mlp._fc_layers[0].weight = nn.Parameter(w1)
        self.mlp._fc_layers[1].weight = nn.Parameter(w2)

        input = torch.rand(batch_size, input_dim, requires_grad=True)
        vec = torch.rand(batch_size, input_dim, requires_grad=True)

        for _ in range(500):
            loss = self.minimize_step(input, vec.detach())
            self.inverse_mvp.update_with_gradient(LossInfo(loss=loss))

        jac = self.mlp.compute_jac(input)
        jac_inv = torch.inverse(jac)
        jac_inv_vec = torch.matmul(jac_inv, vec.unsqueeze(-1)).squeeze(-1)
        y = self.inverse_mvp.predict_step((input, vec)).output
        self.assertArrayEqual(y, jac_inv_vec, 1e-3)


if __name__ == "__main__":
    alf.test.main()
