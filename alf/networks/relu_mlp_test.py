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
import numpy as np
import torch

import alf
from alf.networks.relu_mlp import ReluMLP
from alf.tensor_specs import TensorSpec


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


class ReluMLPTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    @parameterized.parameters(
        dict(hidden_layers=()),
        dict(hidden_layers=(2, )),
        dict(hidden_layers=(2, 3), batch_size=1),
        dict(hidden_layers=(2, 3, 4)),
    )
    def test_compute_jac(self, hidden_layers=(2, ), batch_size=2,
                         input_size=5):
        """
        Check that the input-output Jacobian computed by the direct(autograd-free)
        approach is consistent with the one computed by calling autograd.
        """
        partial_idx1 = [0, 2]
        partial_idx2 = [1, -1]
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(spec, output_size=4, hidden_layers=hidden_layers)

        # compute jac using direct approach
        x = torch.randn(batch_size, input_size, requires_grad=True)
        x1 = x.detach().clone()
        jac = mlp.compute_jac(x1)
        jac_partial1 = mlp.compute_jac(x1, partial_idx1)
        jac_partial2 = mlp.compute_jac(x1, partial_idx2)

        # compute jac using autograd
        y, _ = mlp(x)
        jac_ad = jacobian(y, x)
        jac2 = []
        for i in range(batch_size):
            jac2.append(jac_ad[i, :, i, :])
        jac2 = torch.stack(jac2, dim=0)
        jac2_partial1 = jac2[:, partial_idx1, :]
        jac2_partial2 = jac2[:, partial_idx2, :]

        self.assertArrayEqual(jac, jac2, 1e-6)
        self.assertArrayEqual(jac2_partial1, jac2_partial1, 1e-6)
        self.assertArrayEqual(jac2_partial2, jac2_partial2, 1e-6)

    @parameterized.parameters(
        dict(hidden_layers=(2, )),
        dict(hidden_layers=(2, 3), batch_size=1),
        dict(hidden_layers=(2, 3, 4)),
    )
    def test_compute_jac_diag(self,
                              hidden_layers=(2, ),
                              batch_size=2,
                              input_size=5):
        """
        Check that the diagonal of input-output Jacobian computed by
        the direct (autograd-free) approach is consistent with the one
        computed by calling autograd.
        """
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(spec, hidden_layers=hidden_layers)

        # compute jac diag using direct approach
        x = torch.randn(batch_size, input_size, requires_grad=True)
        x1 = x.detach().clone()
        jac_diag = mlp.compute_jac_diag(x1)

        # compute jac using autograd
        y, _ = mlp(x)
        jac = jacobian(y, x)
        jac_diag2 = []
        for i in range(batch_size):
            jac_diag2.append(torch.diag(jac[i, :, i, :]))
        jac_diag2 = torch.stack(jac_diag2, dim=0)

        self.assertArrayEqual(jac_diag, jac_diag2, 1e-6)

    @parameterized.parameters(
        dict(hidden_layers=()),
        dict(hidden_layers=(2, )),
        dict(hidden_layers=(2, 3), batch_size=1),
        dict(hidden_layers=(2, 3, 4)),
    )
    def test_compute_vjp(self, hidden_layers=(2, ), batch_size=2,
                         input_size=5):
        """
        Check that the vector-Jacobian product computed by the direct(autograd-free)
        approach is consistent with the one computed by calling autograd.
        """
        output_size = 4
        partial_idx1 = [0, 2]
        partial_idx2 = [1, -1]
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(
            spec, output_size=output_size, hidden_layers=hidden_layers)

        # compute vjp and partial using direct approach
        x = torch.randn(batch_size, input_size, requires_grad=True)
        vec = torch.randn(batch_size, output_size)
        x1 = x.detach().clone()
        vjp, _ = mlp.compute_vjp(x1, vec)
        vjp_partial1, _ = mlp.compute_vjp(
            x1, vec, output_partial_idx=partial_idx1)
        vjp_partial2, _ = mlp.compute_vjp(
            x1, vec, output_partial_idx=partial_idx2)
        vjp_partial2_partial_vec, _ = mlp.compute_vjp(
            x1, vec[:, partial_idx2], output_partial_idx=partial_idx2)

        # # compute vjp using autograd
        y, _ = mlp(x)
        vjp2 = torch.autograd.grad(y, x, grad_outputs=vec)[0]

        # # compute partial vjp using autograd
        x2 = x.detach().clone()
        x2.requires_grad = True
        y2, _ = mlp(x2)
        jac_ad = jacobian(y2, x2)
        jac2 = []
        for i in range(batch_size):
            jac2.append(jac_ad[i, :, i, :])
        jac2 = torch.stack(jac2, dim=0)

        jac2_partial1 = jac2[:, partial_idx1, :]
        vec1 = vec[:, partial_idx1]
        vjp2_partial1 = torch.einsum('bji,bj->bi', jac2_partial1, vec1)

        jac2_partial2 = jac2[:, partial_idx2, :]
        vec2 = vec[:, partial_idx2]
        vjp2_partial2 = torch.einsum('bji,bj->bi', jac2_partial2, vec2)

        self.assertArrayEqual(vjp, vjp2, 1e-6)
        self.assertArrayEqual(vjp_partial1, vjp2_partial1, 1e-6)
        self.assertArrayEqual(vjp_partial2, vjp2_partial2, 1e-6)
        self.assertArrayEqual(vjp_partial2, vjp_partial2_partial_vec, 1e-6)

    @parameterized.parameters(
        dict(hidden_layers=()),
        dict(hidden_layers=(2, )),
        dict(hidden_layers=(2, 3), batch_size=1),
        dict(hidden_layers=(2, 3, 4)),
    )
    def test_compute_jvp(self, hidden_layers=(2, ), batch_size=3,
                         input_size=5):
        """
        Check that the Jacobian-vec product computed by the direct(autograd-free)
        approach is consistent with the one computed by calling autograd.
        """
        output_size = 4
        partial_idx1 = [0, 2]
        partial_idx2 = [1, -1]
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(
            spec, output_size=output_size, hidden_layers=hidden_layers)

        # compute jvp and partial jvp using direct approach
        x = torch.randn(batch_size, input_size, requires_grad=True)
        vec = torch.randn(batch_size, input_size)
        x1 = x.detach().clone()
        jvp, _ = mlp.compute_jvp(x1, vec)
        jvp_partial1, _ = mlp.compute_jvp(
            x1, vec, output_partial_idx=partial_idx1)
        jvp_partial2, _ = mlp.compute_jvp(
            x1, vec, output_partial_idx=partial_idx2)

        # # compute jvp using autograd
        _, jvp2 = torch.autograd.functional.jvp(
            lambda x: mlp(x)[0], inputs=x, v=vec)

        # # compute partial jvp using autograd
        x2 = x.detach().clone()
        x2.requires_grad = True
        y2, _ = mlp(x2)
        jac_ad = jacobian(y2, x2)
        jac2 = []
        for i in range(batch_size):
            jac2.append(jac_ad[i, :, i, :])
        jac2 = torch.stack(jac2, dim=0)
        jac2_partial1 = jac2[:, partial_idx1, :]
        jvp2_partial1 = torch.einsum('bji,bi->bj', jac2_partial1, vec)
        jac2_partial2 = jac2[:, partial_idx2, :]
        jvp2_partial2 = torch.einsum('bji,bi->bj', jac2_partial2, vec)

        self.assertArrayEqual(jvp, jvp2, 1e-6)
        self.assertArrayEqual(jvp_partial1, jvp2_partial1, 1e-6)
        self.assertArrayEqual(jvp_partial2, jvp2_partial2, 1e-6)


if __name__ == "__main__":
    alf.test.main()
