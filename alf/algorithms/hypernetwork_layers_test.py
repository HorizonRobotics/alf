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
from alf.algorithms.hypernetwork_layers import ParamFC, ParamConv2D
from alf.utils import math_ops


class ParamLayersTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(
        dict(batch_size=1, n=2, act=torch.relu, use_bias=True),
        dict(batch_size=3, n=2, act=torch.relu, use_bias=True),
        dict(batch_size=3, n=2, act=torch.relu, use_bias=False),
    )
    def test_param_fc(self,
                      batch_size=1,
                      n=2,
                      act=math_ops.identity,
                      use_bias=True):
        input_size = 4
        output_size = 5
        pfc = ParamFC(
            input_size, output_size, activation=act, use_bias=use_bias)
        fc = alf.layers.FC(
            input_size, output_size, activation=act, use_bias=use_bias)

        # test param length
        self.assertEqual(pfc.weight_length, fc.weight.nelement())
        if use_bias:
            self.assertEqual(pfc.bias_length, fc.bias.nelement())

        # test non-parallel forward
        fc.weight.data.copy_(pfc.weight[0])
        if use_bias:
            fc.bias.data.copy_(pfc.bias[0])
        inputs = torch.randn(batch_size, input_size)
        p_outs = pfc(inputs)
        outs = fc(inputs)
        self.assertLess((outs - p_outs).abs().max(), 1e-6)

        # test parallel forward
        weight = torch.randn(n, pfc.weight_length)
        pfc.set_weight(weight)
        weight = weight.view(n, output_size, input_size)
        if use_bias:
            bias = torch.randn(n, pfc.bias_length)
            pfc.set_bias(bias)

        n_inputs = inputs.unsqueeze(1).expand(batch_size, n, input_size)
        p_outs = pfc(inputs)
        p_n_outs = pfc(n_inputs)
        self.assertLess((p_outs - p_n_outs).abs().max(), 1e-6)
        for i in range(n):
            fc.weight.data.copy_(weight[i])
            if use_bias:
                fc.bias.data.copy_(bias[i])
            outs = fc(inputs)
            self.assertLess((outs - p_outs[:, i, :]).abs().max(), 1e-6)

    @parameterized.parameters(
        dict(batch_size=1, n=2, act=torch.relu, use_bias=True),
        dict(batch_size=3, n=2, act=torch.relu, use_bias=True),
        dict(batch_size=3, n=2, act=torch.relu, use_bias=False),
    )
    def test_param_conv2d(self,
                          batch_size=1,
                          n=2,
                          act=math_ops.identity,
                          use_bias=True):
        in_channels = 4
        out_channels = 5
        kernel_size = 3
        height = 11
        width = 11
        pconv = ParamConv2D(
            in_channels,
            out_channels,
            kernel_size,
            activation=act,
            use_bias=use_bias)
        conv = alf.layers.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            activation=act,
            use_bias=use_bias)

        # test param length
        self.assertEqual(pconv.weight_length, conv.weight.nelement())
        if use_bias:
            self.assertEqual(pconv.bias_length, conv.bias.nelement())

        # test non-parallel forward
        conv.weight.data.copy_(pconv.weight)
        if use_bias:
            conv.bias.data.copy_(pconv.bias)
        image = torch.randn(batch_size, in_channels, height, width)
        p_outs = pconv(image)
        outs = conv(image)
        self.assertLess((outs - p_outs).abs().max(), 1e-6)

        # test parallel forward
        weight = torch.randn(n, pconv.weight_length)
        pconv.set_weight(weight)
        weight = weight.view(n, out_channels, in_channels, kernel_size,
                             kernel_size)
        if use_bias:
            bias = torch.randn(n, pconv.bias_length)
            pconv.set_bias(bias)
        images = image.repeat(1, n, 1, 1)
        p_outs = pconv(image)
        p_n_outs = pconv(images)
        self.assertLess((p_outs - p_n_outs).abs().max(), 1e-6)
        for i in range(n):
            conv.weight.data.copy_(weight[i])
            if use_bias:
                conv.bias.data.copy_(bias[i])
            outs = conv(image)
            self.assertLess((outs - p_outs[:, i, :, :, :]).abs().max(), 1e-6)


if __name__ == "__main__":
    alf.test.main()
