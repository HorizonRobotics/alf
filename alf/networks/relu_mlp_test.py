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

from absl.testing import parameterized
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
        dict(hidden_layers=(2, )),
        dict(hidden_layers=(2, 3)),
        dict(hidden_layers=(2, 3, 4)),
    )
    def test_compute_jac_diag(self, hidden_layers=(2, ), input_size=5):
        """
        Check that the diagonal of input-output Jacobian computed by
        the direct (autograd-free) approach is consistent with the one
        computed by calling autograd.
        """
        batch_size = 2
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(spec, hidden_layers=hidden_layers)

        # compute jac diag using direct approach
        x = torch.randn(batch_size, input_size, requires_grad=True)
        x1 = x.detach().clone()
        x1.requires_grad = True
        jac_diag = mlp.compute_jac_diag(x1)

        # compute jac using autograd
        y, _ = mlp(x)
        jac = jacobian(y, x)
        jac_diag2 = []
        for i in range(batch_size):
            jac_diag2.append(torch.diag(jac[i, :, i, :]))
        jac_diag2 = torch.stack(jac_diag2, dim=0)

        self.assertArrayEqual(jac_diag, jac_diag2, 1e-6)


if __name__ == "__main__":
    alf.test.main()
