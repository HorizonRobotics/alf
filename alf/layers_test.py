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
from alf.utils import math_ops


class LayersTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(
        dict(n=1, act=torch.relu, use_bias=False, parallel_x=False),
        dict(n=1, act=math_ops.identity, use_bias=False, parallel_x=False),
        dict(n=2, act=torch.relu, use_bias=True, parallel_x=False),
        dict(n=2, act=torch.relu, use_bias=True, parallel_x=True),
        dict(n=2, act=torch.relu, use_bias=False, parallel_x=True),
    )
    def test_parallel_fc(self,
                         n=2,
                         act=math_ops.identity,
                         use_bias=True,
                         parallel_x=True):
        batch_size = 3
        x_dim = 4
        pfc = alf.layers.ParallelFC(
            x_dim, 6, n=n, activation=act, use_bias=use_bias)
        fc = alf.layers.FC(x_dim, 6, activation=act, use_bias=use_bias)

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
    )
    def test_parallel_conv(self,
                           n=2,
                           act=math_ops.identity,
                           use_bias=True,
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
            use_bias=use_bias)

        conv = alf.layers.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=act,
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
    )
    def test_parallel_conv_transpose(self,
                                     n=2,
                                     act=math_ops.identity,
                                     use_bias=True,
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
            use_bias=use_bias)

        convt = alf.layers.ConvTranspose2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=act,
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
            dec = alf.layers.FixedDecodingLayer(
                input_size, output_size, basis_type=basis_type, sigma=sigma)

            self.assertTrue(dec.weight.shape == (output_size, input_size))

            x = torch.randn((batch_size, input_size))
            y = dec(x)
            self.assertTrue(y.shape == (batch_size, output_size))


if __name__ == "__main__":
    alf.test.main()
