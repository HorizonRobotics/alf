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
import numpy as np
import torch

import alf
from alf.algorithms.hypernetwork_algorithm import HyperNetwork
from alf.algorithms.hypernetwork_networks import ParamConvNet, ParamNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops


class HyperNetworkTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayGreater(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertGreater(float(torch.min(x - y)), eps)

    @parameterized.parameters(
        ('svgd', True),
        ('svgd', False),
        ('gfsf', False),
    )
    def test_hypernetwork_regression(self,
                                     par_vi,
                                     use_fc_bn=False,
                                     particles=None):
        """
        The hypernetwork generator is trained to match a random linear function,
        :math:`f(x) = W^T x`, where :math:`W` follows a Gaussian distribution.
        The output variances from the groundtruth random linear function and 
        the sampled functions from hypernetwork are computed during after training.
        The difference between the two variances are expected to decrase with training.
        
        """

        input_spec = TensorSpec((3, 15, 15), torch.float32)
        particles = 64
        train_batch_size = 20
        test_batch_size = 1000
        output_dim = 4
        conv_layer_params = ((16, 3, 1, 2, 2), (10, 2, 1))
        fc_layer_params = (64, )
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            last_layer_size=output_dim,
            last_activation=math_ops.identity,
            use_fc_bn=use_fc_bn,
            loss_type='regression',
            par_vi=par_vi,
            optimizer=alf.optimizers.Adam(lr=1e-4))
        fc = alf.layers.FC(
            np.prod(input_spec.shape),
            output_dim,
            activation=math_ops.identity,
            use_bias=False)
        weight_mean = fc.weight.data

        def fc_predict(inputs):
            weight_noise = torch.randn(fc.weight.shape)
            fc.weight.data.copy_(weight_mean + weight_noise)
            return fc(inputs.view(inputs.shape[0], -1))

        def _train():
            inputs = input_spec.randn(outer_dims=(train_batch_size, ))
            targets = fc_predict(inputs)
            alg_step = algorithm.train_step(
                inputs=(inputs, targets), particles=particles)
            algorithm.update_with_gradient(alg_step.info)

        def _test():
            inputs = input_spec.randn(outer_dims=(test_batch_size, ))
            fc.weight.data.copy_(weight_mean)
            targets_mean = fc(inputs.view(inputs.shape[0], -1))
            targets = []
            for i in range(particles):
                targets.append(fc_predict(inputs))
            targets = torch.stack(targets, dim=0)
            true_var = torch.var(targets, dim=0) / particles

            outputs = algorithm.predict(inputs, particles=particles)
            est_var = torch.var(outputs, dim=1) / particles
            return true_var, est_var

        true_var, est_var = _test()
        init_err = torch.abs(true_var - est_var)
        for i in range(1000):
            _train()
            if i % 100 == 0:
                true_var, est_var = _test()
                err = torch.abs(true_var - est_var)
                print("train_iter {}: mean err {}".format(i, err.mean()))
        true_var, est_var = _test()
        final_err = torch.abs(true_var - est_var)
        self.assertArrayGreater(init_err, final_err, 10.)

    def test_hypernetwork_classification(self):
        # TODO: out of distribution tests
        # If simply use a linear classifier with random weights,
        # the cross_entropy loss does not seem to capture the distribution.
        pass


if __name__ == "__main__":
    alf.test.main()
