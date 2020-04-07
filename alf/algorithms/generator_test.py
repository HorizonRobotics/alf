# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

import math

from absl import logging
from absl.testing import parameterized
import torch
import torch.nn as nn

import alf
from alf.algorithms.generator import Generator
from alf.networks import Network
from alf.tensor_specs import TensorSpec


class Net(Network):
    def __init__(self, dim=2):
        super().__init__(
            input_tensor_spec=TensorSpec(shape=(dim, )),
            skip_input_preprocessing=True,
            name="Net")

        self.fc = nn.Linear(3, dim, bias=False)
        w = torch.tensor([[1, 2], [-1, 1], [1, 1]], dtype=torch.float32)
        self.fc.weight = nn.Parameter(w.t())

    def forward(self, input, state=()):
        return self.fc(input), ()


class Net2(Network):
    def __init__(self, dim=2):
        super().__init__(
            input_tensor_spec=[
                TensorSpec(shape=(dim, )),
                TensorSpec(shape=(dim, ))
            ],
            skip_input_preprocessing=True,
            name="Net")
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        w = torch.tensor([[1, 2], [1, 1]], dtype=torch.float32)
        u = torch.zeros((dim, dim), dtype=torch.float32)
        self.fc1.weight = nn.Parameter(w.t())
        self.fc2.weight = nn.Parameter(u.t())

    def forward(self, input, state=()):
        return self.fc1(input[0]) + self.fc2(input[1]), ()


class GeneratorTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    @parameterized.parameters(
        dict(entropy_regularization=1.0),
        dict(entropy_regularization=0.0),
        dict(entropy_regularization=0.0, mi_weight=1),
    )
    def test_generator_unconditional(self,
                                     entropy_regularization=0.0,
                                     mi_weight=None):
        """
        The generator is trained to match(STEIN)/maximize(ML) the likelihood
        of a Gaussian distribution with zero mean and diagonal variance (1, 4).
        After training, w^T w is the variance of the distribution implied by the
        generator. So it should be diag(1,4) for STEIN and 0 for 'ML'.
        """
        logging.info("entropy_regularization: %s mi_weight: %s" %
                     (entropy_regularization, mi_weight))
        dim = 2
        batch_size = 512
        net = Net(dim)
        generator = Generator(
            dim,
            noise_dim=3,
            entropy_regularization=entropy_regularization,
            net=net,
            mi_weight=mi_weight,
            optimizer=alf.optimizers.AdamTF(lr=1e-3))

        var = torch.tensor([1, 4], dtype=torch.float32)
        precision = 1. / var

        def _neglogprob(x):
            return torch.squeeze(
                0.5 * torch.matmul(x * x, torch.reshape(precision, (dim, 1))),
                axis=-1)

        def _train():
            alg_step = generator.train_step(
                inputs=None, loss_func=_neglogprob, batch_size=batch_size)
            generator.update_with_gradient(alg_step.info)

        for i in range(5000):
            _train()
            learned_var = torch.matmul(net.fc.weight, net.fc.weight.t())
            if i % 500 == 0:
                print(i, "learned var=", learned_var)

        if entropy_regularization == 1.0:
            self.assertArrayEqual(torch.diag(var), learned_var, 0.1)
        else:
            if mi_weight is None:
                self.assertArrayEqual(torch.zeros(dim, dim), learned_var, 0.1)
            else:
                self.assertGreater(
                    float(torch.sum(torch.abs(learned_var))), 0.5)

    @parameterized.parameters(
        dict(entropy_regularization=1.0),
        dict(entropy_regularization=0.0),
        dict(entropy_regularization=0.0, mi_weight=1),
    )
    def test_generator_conditional(self,
                                   entropy_regularization=0.0,
                                   mi_weight=None):
        """
        The target conditional distribution is :math:`N(\mu; diag(1, 4))`. After training
        net._u should be u for both STEIN and ML. And :math:`w^T w` should be :math:`diag(1, 4)`
        for STEIN and 0 for ML.
        """
        logging.info("entropy_regularization: %s mi_weight: %s" %
                     (entropy_regularization, mi_weight))
        dim = 2
        batch_size = 512
        net = Net2(dim)
        generator = Generator(
            dim,
            noise_dim=dim,
            entropy_regularization=entropy_regularization,
            net=net,
            mi_weight=mi_weight,
            input_tensor_spec=TensorSpec((dim, )),
            optimizer=alf.optimizers.Adam(lr=1e-3))

        var = torch.tensor([1, 4], dtype=torch.float32)
        precision = 1. / var
        u = torch.tensor([[-0.3, 1], [1, 2]], dtype=torch.float32)

        def _neglogprob(xy):
            x, y = xy
            d = x - torch.matmul(y, u)
            return torch.squeeze(
                0.5 * torch.matmul(d * d, torch.reshape(precision, (dim, 1))),
                axis=-1)

        def _train():
            y = torch.randn(batch_size, dim)
            alg_step = generator.train_step(inputs=y, loss_func=_neglogprob)
            generator.update_with_gradient(alg_step.info)

        for i in range(5000):
            _train()
            learned_var = torch.matmul(net.fc1.weight, net.fc1.weight.t())
            if i % 500 == 0:
                print(i, "learned var=", learned_var)
                print("u=", net.fc2.weight.t())

        if mi_weight is not None:
            self.assertGreater(float(torch.sum(torch.abs(learned_var))), 0.5)
        elif entropy_regularization == 1.0:
            self.assertArrayEqual(net.fc2.weight.t(), u, 0.1)
            self.assertArrayEqual(torch.diag(var), learned_var, 0.1)
        else:
            self.assertArrayEqual(net.fc2.weight.t(), u, 0.1)
            self.assertArrayEqual(torch.zeros(dim, dim), learned_var, 0.1)


if __name__ == '__main__':
    alf.test.main()
