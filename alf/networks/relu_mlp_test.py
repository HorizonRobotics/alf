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
        dict(hidden_layers=((2, False), )),
        dict(hidden_layers=((2, False), (3, False))),
        dict(hidden_layers=((2, False), (3, False), (4, False))),
    )
    def test_compute_jac_diag(self, hidden_layers=((2, False), ),
                              input_size=5):
        batch_size = 2
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(spec, hidden_layers=hidden_layers)
        params = torch.randn(mlp.param_length)

        param1 = params.clone()
        param1.requires_grad = True
        mlp.set_parameters(param1)
        x = torch.randn(batch_size, input_size, requires_grad=True)
        x1 = x.detach().clone()
        x1.requires_grad = True
        jac_diag = mlp.compute_jac_diag(x1)

        # compute jac using autograd

        param2 = params.clone()
        param2.requires_grad = True
        mlp.set_parameters(param2)
        y, _ = mlp(x)
        jac = jacobian(y, x)
        jac_diag2 = []
        for i in range(batch_size):
            jac_diag2.append(torch.diag(jac[i, :, i, :]))
        jac_diag2 = torch.stack(jac_diag2, dim=0)

        self.assertArrayEqual(jac_diag, jac_diag2, 1e-6)

    # def test_compute_ntk(self, input_size=5, hidden_layers=(2, )):
    #     batch_size = 2
    #     spec = TensorSpec((input_size, ))
    #     mlp = ReluMLP(spec, hidden_layers=hidden_layers)
    #     x = torch.randn(batch_size, input_size)
    #     y, _ = mlp(x)
    #     ntk = mlp.compute_ntk(x[0], x[1], mlp._hidden_fcs[0].hidden_neurons[0],
    #                           mlp._hidden_fcs[0].hidden_neurons[1])

    #     # compute ntk using autograd
    #     Jd = jacobian(y, mlp._last_fc.weight)
    #     Je = jacobian(y, mlp._hidden_fcs[0].weight)
    #     Jd = Jd.reshape(batch_size, input_size,
    #                     mlp._last_fc.weight.data.nelement())
    #     Je = Je.reshape(batch_size, input_size,
    #                     mlp._hidden_fcs[0].weight.data.nelement())

    #     jac = torch.cat((Jd, Je), dim=-1)
    #     ntk2 = jac[0] @ jac[1].t()

    #     self.assertArrayEqual(ntk, ntk2, 1e-6)

    def test_ntk_svgd(self, input_size=5, hidden_layers=((2, False), )):
        precision = torch.rand(input_size)

        def _neglogprob(x):
            return torch.squeeze(
                0.5 * torch.matmul(x * x,
                                   torch.reshape(precision, (input_size, 1))),
                axis=-1)

        batch_size = 3
        spec = TensorSpec((input_size, ))
        mlp = ReluMLP(spec, hidden_layers=hidden_layers)
        x1 = torch.randn(batch_size, input_size, requires_grad=True)
        x2 = torch.randn(batch_size, input_size, requires_grad=True)
        params = torch.randn(mlp.param_length)

        # ## compute ntk svgd by ReluMLP.ntk_svgd
        x = torch.cat((x1, x2), dim=0).detach().clone()
        x.requires_grad = True
        mlp.set_parameters(params.clone())
        ntk_grad1, vec1 = mlp.ntk_svgd(x, _neglogprob)

        mlp.set_parameters(params.clone())
        y1, _ = mlp(x1)
        y2, _ = mlp(x2)

        # compute the 1st term of the ntk svgd
        Je = jacobian(y1, mlp._fc_layers[0].weight)
        Jd = jacobian(y1, mlp._fc_layers[1].weight)
        Je = Je.reshape(batch_size, input_size,
                        mlp._fc_layers[0].weight.nelement())
        Jd = Jd.reshape(batch_size, input_size,
                        mlp._fc_layers[1].weight.nelement())
        jac1 = torch.cat((Je, Jd), dim=-1)

        # Je = jacobian(y2, mlp._fc_layers[0].weight)
        # Jd = jacobian(y2, mlp._fc_layers[1].weight)
        # Je = Je.reshape(batch_size, input_size,
        #                 mlp._fc_layers[0].weight.nelement())
        # Jd = Jd.reshape(batch_size, input_size,
        #                 mlp._fc_layers[1].weight.nelement())
        # jac2 = torch.cat((Je, Jd), dim=-1)

        # loss2 = _neglogprob(x2)
        # loss_grad2 = torch.autograd.grad(loss2.sum(), x2)[0]
        # vec1 = torch.einsum('ijk,ij->k', jac2, loss_grad2) / batch_size
        # grad1 = torch.matmul(jac1, vec1)

        # compute the 2nd term of the ntk svgd
        vec2 = []
        for i in range(batch_size):
            Tr = []
            for j in range(input_size):
                grad = torch.autograd.grad(y2[i, j], x2, create_graph=True)[0]
                Tr.append(grad[i])
            Tr = torch.stack(Tr, dim=1)
            Tr = torch.trace(Tr)
            Jd = jacobian(Tr, mlp._fc_layers[1].weight)
            Je = jacobian(Tr, mlp._fc_layers[0].weight)
            vec2.append(torch.cat((Je.view(-1), Jd.view(-1)), dim=-1))
        vec2 = torch.stack(vec2, dim=0)
        vec2 = vec2.mean(0)

        ntk_grad2 = torch.matmul(jac1, vec2)

        #     svgd1 = grad1 + grad2

        # # ## compute ntk svgd by ReluMLP.ntk_svgd
        # x = torch.cat((x1, x2), dim=0)
        # grad = mlp.ntk_svgd(x, _neglogprob)

        self.assertArrayEqual(ntk_grad1, ntk_grad2, 1e-6)

        # x = torch.cat((x1, x2), dim=0)
        # ntk_logp, ntk_grad = mlp.ntk_svgd(x, _neglogprob)

    #     svgd2 = ntk_logp + ntk_grad

    #     self.assertArrayEqual(svgd1, svgd2, 1e-6)


if __name__ == "__main__":
    alf.test.main()
