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

from absl.testing import parameterized
import torch
import torch.nn as nn
import alf
from alf.data_structures import AlgStep, LossInfo
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.generator import InverseMVPAlgorithm
from alf.networks.network import Network
from alf.networks.relu_mlp import ReluMLP
from alf.networks.relu_mlp_test import jacobian
from alf.tensor_specs import TensorSpec
from alf.utils.math_ops import identity

import functools
from alf.networks.encoding_networks import EncodingNetwork
from alf.initializers import variance_scaling_init


class InverseMVPTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    def test_inverse_mvp(self, batch_size=5):
        r"""
        The InverseMVP network is an encoding network that is trained to
        predict the inverse Jacobian vector product :math:`J^{-1}v`,
        where the Jacobian is w.r.t. :math:`f(z)=g(z^{(:k)})+\lambda z`, 
        :math:`z^{(:k)}` denote the vector of the first k components 
        of z. In the following test case, z is a randomly generated input 
        of dimension 3 and :math:`k=2`.
        Using relu_mlp for g we can compute this exactly, and check that the
        trained network is correct. 
        """
        input_dim = 2
        vec_dim = 3
        output_dim = 3
        fullrank_diag_weight = 1.0
        input_spec = TensorSpec(shape=(input_dim, ))
        vec_spec = TensorSpec(shape=(vec_dim, ))
        input_tensor_spec = (input_spec, vec_spec)
        optimizer = alf.optimizers.Adam(lr=5e-4)
        self.inverse_mvp = InverseMVPAlgorithm(
            input_dim,
            output_dim,
            hidden_size=300,
            num_hidden_layers=1,
            optimizer=optimizer)
        mlp_spec = TensorSpec((input_dim, ))
        self.mlp = ReluMLP(
            mlp_spec, output_size=output_dim, hidden_layers=(2, ))
        # make Jac better behaved
        w1 = torch.tensor([[1., 2.], [2., 1.]])
        w2 = torch.tensor([[2., 1.], [1, 1], [1., 2.]])

        self.mlp._fc_layers[0].weight = nn.Parameter(w1)
        self.mlp._fc_layers[1].weight = nn.Parameter(w2)

        def _minimize_step(inputs, vec):
            y, z_inputs = self.inverse_mvp.predict_step((inputs, vec)).output
            jac_y, _ = self.mlp.compute_vjp(z_inputs, y)
            jac_y = torch.cat(
                (jac_y, torch.zeros(jac_y.shape[0], output_dim - input_dim)),
                dim=-1)
            jac_y += fullrank_diag_weight * y
            loss = torch.nn.functional.mse_loss(jac_y, vec)
            return loss

        inputs = torch.rand(batch_size, output_dim, requires_grad=True)
        vec = torch.rand(batch_size, vec_dim, requires_grad=True)
        for _ in range(5000):
            loss = _minimize_step(inputs, vec.detach())
            self.inverse_mvp.update_with_gradient(LossInfo(loss=loss))

        y, z_inputs = self.inverse_mvp.predict_step((inputs, vec)).output
        jac = self.mlp.compute_jac(z_inputs)
        jac = torch.cat(
            (jac, torch.zeros(*jac.shape[:-1] + (output_dim - input_dim, ))),
            dim=-1)
        jac += fullrank_diag_weight * torch.eye(output_dim)
        jac_inv = torch.inverse(jac)
        jac_inv_vec = torch.matmul(vec.unsqueeze(1), jac_inv).squeeze(1)
        self.assertArrayEqual(y, jac_inv_vec, 1e-2)


if __name__ == "__main__":
    alf.test.main()
