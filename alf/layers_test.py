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
import numpy as np

import alf
from alf.utils import math_ops


class LayersTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(
        dict(n=1, act=torch.relu, use_bias=False, parallel_x=False),
        dict(n=1, act=math_ops.identity, use_bias=False, parallel_x=False),
        dict(n=2, act=torch.relu, use_bias=True, parallel_x=False),
        dict(n=2, act=torch.relu, use_bias=True, parallel_x=True),
        dict(n=2, act=torch.relu, use_bias=False, parallel_x=True),
        dict(
            n=2, act=torch.relu, use_bias=False, use_bn=True, parallel_x=True),
    )
    def test_parallel_fc(self,
                         n=2,
                         act=math_ops.identity,
                         use_bias=True,
                         use_bn=False,
                         parallel_x=True):
        batch_size = 3
        x_dim = 4
        pfc = alf.layers.ParallelFC(
            x_dim, 6, n=n, activation=act, use_bias=use_bias, use_bn=use_bn)
        fc = alf.layers.FC(
            x_dim, 6, activation=act, use_bias=use_bias, use_bn=use_bn)

        if parallel_x:
            px = torch.randn((batch_size, n, x_dim))
        else:
            px = torch.randn((batch_size, x_dim))

        py = pfc(px)
        for i in range(n):
            fc.weight.data.copy_(pfc.weight[i])
            if use_bias:
                fc.bias.data.copy_(pfc.bias[i])
            if parallel_x:
                x = px[:, i, :]
            else:
                x = px
            y = fc(x)
            self.assertLess((y - py[:, i, :]).abs().max(), 1e-5)

    @parameterized.parameters(
        dict(n=1, act=torch.relu, use_bias=False, parallel_x=False),
        dict(n=1, act=math_ops.identity, use_bias=False, parallel_x=False),
        dict(n=2, act=torch.relu, use_bias=True, parallel_x=False),
        dict(n=2, act=torch.relu, use_bias=True, parallel_x=True),
        dict(n=2, act=torch.relu, use_bias=False, parallel_x=True),
        dict(
            n=2, act=torch.relu, use_bias=False, use_bn=True, parallel_x=True),
    )
    def test_parallel_conv(self,
                           n=2,
                           act=math_ops.identity,
                           use_bias=True,
                           use_bn=False,
                           parallel_x=True):
        batch_size = 5
        in_channels = 4
        out_channels = 3
        height = 11
        width = 11
        pconv = alf.layers.ParallelConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            n=n,
            activation=act,
            use_bn=use_bn,
            use_bias=use_bias)

        conv = alf.layers.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=act,
            use_bn=use_bn,
            use_bias=use_bias)
        if parallel_x:
            px = torch.randn((batch_size, n, in_channels, height, width))
        else:
            px = torch.randn((batch_size, in_channels, height, width))

        py = pconv(px)
        for i in range(n):
            conv.weight.data.copy_(pconv.weight[i])
            if use_bias:
                conv.bias.data.copy_(pconv.bias[i])
            if parallel_x:
                x = px[:, i, :]
            else:
                x = px
            y = conv(x)
            self.assertLess((y - py[:, i, :]).abs().max(), 1e-5)

    @parameterized.parameters(
        dict(n=1, act=torch.relu, use_bias=False, parallel_x=False),
        dict(n=1, act=math_ops.identity, use_bias=False, parallel_x=False),
        dict(n=2, act=torch.relu, use_bias=True, parallel_x=False),
        dict(n=2, act=torch.relu, use_bias=True, parallel_x=True),
        dict(n=2, act=torch.relu, use_bias=False, parallel_x=True),
        dict(
            n=2, act=torch.relu, use_bias=False, use_bn=True, parallel_x=True),
    )
    def test_parallel_conv_transpose(self,
                                     n=2,
                                     act=math_ops.identity,
                                     use_bias=True,
                                     use_bn=False,
                                     parallel_x=True):
        batch_size = 5
        in_channels = 4
        out_channels = 3
        height = 11
        width = 11
        pconvt = alf.layers.ParallelConvTranspose2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            n=n,
            activation=act,
            use_bn=use_bn,
            use_bias=use_bias)

        convt = alf.layers.ConvTranspose2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=act,
            use_bn=use_bn,
            use_bias=use_bias)
        if parallel_x:
            px = torch.randn((batch_size, n, in_channels, height, width))
        else:
            px = torch.randn((batch_size, in_channels, height, width))

        py = pconvt(px)
        for i in range(n):
            convt.weight.data.copy_(pconvt.weight[i])
            if use_bias:
                convt.bias.data.copy_(pconvt.bias[i])
            if parallel_x:
                x = px[:, i, :]
            else:
                x = px
            y = convt(x)
            self.assertLess((y - py[:, i, :]).abs().max(), 1e-5)

    @parameterized.parameters(
        ("rbf", 8, 8, 0.1),
        ("poly", 4, 8, None),
        ("cheb", 4, 8, None),
        ("haar", 8, 8, None),
        ("rbf", 7, 8, 0.1),
        ("haar", 7, 7, None),
        ("unimplemented", 3, 8, None),
    )
    def test_fixed_decoding_layer(self, basis_type, input_size, output_size,
                                  sigma):
        batch_size = 3

        if (basis_type == "rbf" and input_size != output_size) or \
           (basis_type == "haar" and (input_size & (input_size - 1)) != 0) or \
           basis_type == "unimplemented":
            self.assertRaises(
                AssertionError,
                alf.layers.FixedDecodingLayer,
                input_size,
                output_size,
                basis_type=basis_type,
                sigma=sigma)
        else:
            basis_weight_tau = 0.5
            dec = alf.layers.FixedDecodingLayer(
                input_size,
                output_size,
                basis_type=basis_type,
                sigma=sigma,
                tau=basis_weight_tau)

            self.assertTrue(dec.weight.shape == (output_size, input_size))

            x = torch.randn((batch_size, input_size))
            y = dec(x)
            self.assertTrue(y.shape == (batch_size, output_size))

            # test basis weighting factor
            dec_no_basis_weighting = alf.layers.FixedDecodingLayer(
                input_size,
                output_size,
                basis_type=basis_type,
                sigma=sigma,
                tau=1.0)

            basis_weight = (dec.weight.norm(dim=0) /
                            dec_no_basis_weighting.weight.norm(dim=0))
            if basis_type == "poly" or basis_type == "cheb":
                exp_factor = torch.arange(input_size).float()
                basis_weight_expected = basis_weight_tau**exp_factor
                self.assertTensorClose(basis_weight, basis_weight_expected)
            elif basis_type == "haar":
                exp_factor = torch.ceil(
                    torch.log2(torch.arange(input_size).float() + 1))
                basis_weight_expected = basis_weight_tau**exp_factor
                self.assertTensorEqual(basis_weight, basis_weight_expected)

    @parameterized.parameters((2, 2), (4, 4), (8, 8), (16, 16))
    def test_harr_basis_correctness(self, input_size, output_size):
        basis_weight_tau = 1.0

        dec = alf.layers.FixedDecodingLayer(
            input_size, output_size, basis_type="haar", tau=basis_weight_tau)

        if input_size <= 8:
            # expected Haar matrix are constructed following the reference
            # http://fourier.eng.hmc.edu/e161/lectures/Haar/index.html
            st = np.sqrt(2)
            if input_size == 2:
                # H2^T
                expected_haar_basis = torch.as_tensor(
                    1. / np.sqrt(2) * np.array([[1, 1], [1, -1]]).transpose(
                        1, 0))
            elif input_size == 4:
                # H4^T
                expected_haar_basis = torch.as_tensor(1. / 2 * np.array(
                    [[1, 1, 1, 1], [1, 1, -1, -1], [st, -st, 0, 0],
                     [0, 0, st, -st]]).transpose(1, 0))
            elif input_size == 8:
                # H8^T
                expected_haar_basis = torch.as_tensor(
                    1. / np.sqrt(8) * np.array([
                        [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, -1, -1, -1, -1],
                        [st, st, -st, -st, 0, 0, 0, 0],
                        [0, 0, 0, 0, st, st, -st, -st],
                        [2, -2, 0, 0, 0, 0, 0, 0], [0, 0, 2, -2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 2, -2, 0, 0], [0, 0, 0, 0, 0, 0, 2, -2]
                    ]).transpose(1, 0))

            # test constructed basis against ground-truth reference
            self.assertTensorClose(
                dec.weight, expected_haar_basis, epsilon=1e-6)

        # test constructed basis are orthogonal
        self.assertTensorClose(
            torch.mm(dec.weight, dec.weight.transpose(1, 0)),
            torch.eye(dec.weight.shape[0]),
            epsilon=1e-6)


if __name__ == "__main__":
    alf.test.main()
