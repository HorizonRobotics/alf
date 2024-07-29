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

from absl import logging
from absl.testing import parameterized
import torch
import torch.nn as nn
import numpy as np

import alf
from alf.utils import math_ops


class LayersTest(parameterized.TestCase, alf.test.TestCase):
    def _test_make_parallel(
            self,
            net,
            spec,
            tolerance=1e-6,
            get_pnet_parameters=lambda pnet: pnet.parameters()):
        batch_size = 10
        for n in (1, 2, 5):
            pnet = net.make_parallel(n)
            nnet = alf.layers.NaiveParallelLayer(net, n)
            for i in range(n):
                for pp, np in zip(
                        get_pnet_parameters(pnet),
                        nnet._networks[i].parameters()):
                    self.assertEqual(pp.shape, (n, ) + np.shape)
                    np.data.copy_(pp[i])
            pspec = alf.layers.make_parallel_spec(spec, n)
            input = alf.nest.map_structure(lambda s: s.sample([batch_size]),
                                           pspec)
            presult = pnet(input)
            nresult = nnet(input)
            alf.nest.map_structure(
                lambda p, n: self.assertTensorClose(p, n, tolerance), presult,
                nresult)

            # test reset parameter
            nnet.reset_parameters()

    @parameterized.parameters(
        dict(n=1, act=torch.relu, use_bias=False, parallel_x=False),
        dict(n=1, act=math_ops.identity, use_bias=False, parallel_x=False),
        dict(n=2, act=torch.relu, use_bias=True, parallel_x=False),
        dict(n=2, act=torch.relu, use_bias=True, parallel_x=True),
        dict(n=2, act=torch.relu, use_bias=False, parallel_x=True),
        dict(
            n=2, act=torch.relu, use_bias=False, use_bn=True, parallel_x=True),
        dict(
            n=2, act=torch.relu, use_bias=False, use_ln=True, parallel_x=True),
    )
    def test_parallel_fc(self,
                         n=2,
                         act=math_ops.identity,
                         use_bias=True,
                         use_bn=False,
                         use_ln=False,
                         parallel_x=True):
        batch_size = 3
        x_dim = 4
        pfc = alf.layers.ParallelFC(
            x_dim,
            6,
            n=n,
            activation=act,
            use_bias=use_bias,
            use_bn=use_bn,
            use_ln=use_ln)
        fc = alf.layers.FC(
            x_dim,
            6,
            activation=act,
            use_bias=use_bias,
            use_bn=use_bn,
            use_ln=use_ln)

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
        dict(
            n=1,
            act=math_ops.identity,
            use_bias=False,
            specify_comp_weight=True),
        dict(n=1, act=torch.relu, use_bias=False, specify_comp_weight=True),
        dict(
            n=1,
            act=math_ops.identity,
            use_bias=True,
            specify_comp_weight=False),
        dict(n=1, act=torch.relu, use_bias=True, specify_comp_weight=False),
        dict(n=2, act=torch.relu, use_bias=True, specify_comp_weight=True),
        dict(n=5, act=torch.relu, use_bias=True, specify_comp_weight=True),
        dict(n=5, act=torch.relu, use_bias=True, specify_comp_weight=False),
        dict(
            n=5,
            act=torch.relu,
            use_bias=True,
            use_bn=True,
            specify_comp_weight=False),
        dict(
            n=5,
            act=torch.relu,
            use_bias=True,
            use_ln=True,
            specify_comp_weight=False))
    def test_compositional_fc(self,
                              n=2,
                              act=math_ops.identity,
                              use_bias=True,
                              use_bn=False,
                              use_ln=False,
                              specify_comp_weight=True):
        batch_size = 3
        x_dim = 4
        cfc = alf.layers.CompositionalFC(
            x_dim,
            6,
            n=n,
            activation=act,
            use_bias=use_bias,
            use_bn=use_bn,
            use_ln=use_ln)

        fc = alf.layers.FC(
            x_dim,
            6,
            activation=math_ops.identity,
            use_bias=use_bias,
            use_bn=False,
            use_ln=False)

        # only used for constructing proper bn/ln
        fc_bn_ln = alf.layers.FC(
            x_dim,
            6,
            activation=math_ops.identity,
            use_bias=use_bias,
            use_bn=True,
            use_ln=True)

        x = torch.randn((batch_size, x_dim))

        if specify_comp_weight:
            comp_weight = torch.randn(batch_size, n)
        else:
            comp_weight = None

        cy, _ = cfc(inputs=(x, comp_weight))

        comp_y = 0
        for i in range(n):
            fc.weight.data.copy_(cfc.weight[i])
            if use_bias:
                fc.bias.data.copy_(cfc.bias[i])
            y = fc(x)
            if specify_comp_weight:
                comp_y += comp_weight[..., i:i + 1] * y
            else:
                comp_y += y
        if use_bn:
            comp_y = fc_bn_ln._bn(comp_y)
        if use_ln:
            comp_y = fc_bn_ln._ln(comp_y)

        comp_y = act(comp_y)

        self.assertLess((comp_y - cy).abs().max(), 1e-5)

    @parameterized.parameters(
        dict(n=1, act=torch.relu, use_bias=True, specify_comp_weight=True),
        dict(n=1, act=torch.relu, use_bias=True, specify_comp_weight=False),
        dict(n=5, act=torch.relu, use_bias=True, specify_comp_weight=True),
        dict(n=5, act=torch.relu, use_bias=True, specify_comp_weight=False),
        dict(n=5, act=torch.relu, use_bn=True, specify_comp_weight=False),
        dict(n=5, act=torch.relu, use_bn=True, specify_comp_weight=True))
    def test_compositional_fc_chaining(self,
                                       n=2,
                                       act=math_ops.identity,
                                       use_bias=True,
                                       use_bn=False,
                                       use_ln=False,
                                       specify_comp_weight=True):
        batch_size = 3
        x_dim = 4
        cfc1 = alf.layers.CompositionalFC(
            x_dim,
            8,
            n=n,
            activation=act,
            use_bias=use_bias,
            use_bn=use_bn,
            use_ln=use_ln)

        cfc2 = alf.layers.CompositionalFC(
            8,
            6,
            n=n,
            activation=act,
            use_bias=use_bias,
            use_bn=use_bn,
            use_ln=use_ln)

        x = torch.randn((batch_size, x_dim))

        if specify_comp_weight:
            comp_weight = torch.randn(batch_size, n)
        else:
            comp_weight = None

        cy, _ = cfc2(cfc1(inputs=(x, comp_weight)))

        fc1 = alf.layers.FC(
            x_dim,
            8,
            activation=math_ops.identity,
            use_bias=use_bias,
            use_bn=False,
            use_ln=False)
        fc2 = alf.layers.FC(
            8,
            6,
            activation=math_ops.identity,
            use_bias=use_bias,
            use_bn=False,
            use_ln=False)

        # only used for constructing proper bn/ln
        fc1_bn_ln = alf.layers.FC(
            x_dim,
            8,
            activation=math_ops.identity,
            use_bias=use_bias,
            use_bn=True,
            use_ln=True)
        fc2_bn_ln = alf.layers.FC(
            8,
            6,
            activation=math_ops.identity,
            use_bias=use_bias,
            use_bn=True,
            use_ln=True)

        cfcs = [cfc1, cfc2]
        fcs = [fc1, fc2]
        fcs_bn_ln = [fc1_bn_ln, fc2_bn_ln]

        for fc, fc_bn_ln, cfc in zip(fcs, fcs_bn_ln, cfcs):
            comp_y = 0
            for i in range(n):
                fc.weight.data.copy_(cfc.weight[i])
                if use_bias:
                    fc.bias.data.copy_(cfc.bias[i])
                y = fc(x)
                if specify_comp_weight:
                    comp_y += comp_weight[..., i:i + 1] * y
                else:
                    comp_y += y
            if use_bn:
                comp_y = fc_bn_ln._bn(comp_y)
            if use_ln:
                comp_y = fc_bn_ln._ln(comp_y)

            comp_y = act(comp_y)
            x = comp_y

        self.assertLess((comp_y - cy).abs().max(), 1e-5)

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
        dict(batch_size=1, n=2, act=torch.relu, use_bias=True),
        dict(batch_size=3, n=2, act=torch.relu, use_bias=False),
        dict(batch_size=3, n=2, act=torch.relu, use_bias=True, use_ln=True),
    )
    def test_param_fc(self,
                      batch_size=1,
                      n=2,
                      act=math_ops.identity,
                      use_bias=True,
                      use_ln=False):
        input_size = 4
        output_size = 5
        pfc = alf.layers.ParamFC(
            input_size,
            output_size,
            activation=act,
            use_bias=use_bias,
            use_ln=use_ln,
            n_groups=n)
        fc = alf.layers.FC(
            input_size,
            output_size,
            activation=act,
            use_ln=use_ln,
            use_bias=use_bias)

        # test param length
        self.assertEqual(pfc.weight_length, fc.weight.nelement())
        if use_bias:
            self.assertEqual(pfc.bias_length, fc.bias.nelement())
        if use_ln:
            self.assertEqual(pfc._ln.param_length,
                             fc._ln.weight.nelement() + fc._ln.bias.nelement())

        # test parallel forward
        params = torch.randn(n, pfc.param_length)
        pfc.set_parameters(params)
        inputs = torch.randn(batch_size, input_size)
        n_inputs = inputs.unsqueeze(1).expand(batch_size, n, input_size)
        p_outs = pfc(inputs)
        p_n_outs = pfc(n_inputs)
        self.assertLess((p_outs - p_n_outs).abs().max(), 1e-6)
        weight = pfc.weight
        if use_bias:
            bias = pfc.bias
        if use_ln:
            norm_weight = pfc._ln.weight.reshape(n, -1)
            norm_bias = pfc._ln.bias.reshape(n, -1)
        for i in range(n):
            fc.weight.data.copy_(weight[i])
            if use_bias:
                fc.bias.data.copy_(bias[i])
            if use_ln:
                fc._ln.weight.data.copy_(norm_weight[i])
                fc._ln.bias.data.copy_(norm_bias[i])
            outs = fc(inputs)
            self.assertLess((outs - p_outs[:, i, :]).abs().max(), 1e-6)

    @parameterized.parameters(
        dict(batch_size=1, n=2, act=torch.relu, use_bias=True, use_ln=True),
        dict(batch_size=3, n=2, act=torch.relu, use_bias=False),
        dict(batch_size=3, n=2, act=torch.relu, use_bias=True, use_ln=True),
        dict(batch_size=3, n=2, act=torch.relu, use_bias=False, use_ln=True),
    )
    def test_param_conv2d(self,
                          batch_size=1,
                          n=2,
                          act=math_ops.identity,
                          use_bias=True,
                          use_ln=False):
        in_channels = 4
        out_channels = 5
        kernel_size = 3
        height = 11
        width = 11
        pconv = alf.layers.ParamConv2D(
            in_channels,
            out_channels,
            kernel_size,
            activation=act,
            use_bias=use_bias,
            use_ln=use_ln,
            n_groups=n)
        conv = alf.layers.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            activation=act,
            use_bias=use_bias,
            use_ln=use_ln)

        # test param length
        self.assertEqual(pconv.weight_length, conv.weight.nelement())
        if use_bias:
            self.assertEqual(pconv.bias_length, conv.bias.nelement())
        if use_ln:
            self.assertEqual(
                pconv._ln.param_length,
                conv._ln.weight.nelement() + conv._ln.bias.nelement())

        # test parallel forward
        params = torch.randn(n, pconv.param_length)
        pconv.set_parameters(params)
        image = torch.randn(batch_size, in_channels, height, width)
        images = image.repeat(1, n, 1, 1)
        p_outs = pconv(image)
        p_n_outs = pconv(images)
        self.assertLess((p_outs - p_n_outs).abs().max(), 1e-6)
        weight = pconv.weight.reshape(n, -1, *pconv.weight.shape[1:])
        if use_bias:
            bias = pconv.bias.reshape(n, -1)
        if use_ln:
            norm_weight = pconv._ln.weight.reshape(n, -1)
            norm_bias = pconv._ln.bias.reshape(n, -1)
        for i in range(n):
            conv.weight.data.copy_(weight[i])
            if use_bias:
                conv.bias.data.copy_(bias[i])
            if use_ln:
                conv._ln.weight.data.copy_(norm_weight[i])
                conv._ln.bias.data.copy_(norm_bias[i])
            outs = conv(image)
            self.assertLess((outs - p_outs[:, i, :, :, :]).abs().max(), 1e-5)

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

    def test_transformer_block_shift(self):
        n = 32
        m = 7
        l = 2 * n - 1
        d = 24
        x = torch.rand(l, d)
        y = alf.layers.TransformerBlock._shift(x, m)
        for i in range(m):
            for j in range(n):
                self.assertEqual(y[i, j], x[n - 1 + i - j, :])

    def test_transformer_block_with_mask(self):
        batch_size = 100
        max_len = 32
        actual_len = 20
        d_model = 64

        x = torch.randn((batch_size, max_len, d_model))

        # We are going to create ``batch_size`` of batches, where each batch has
        # a sequence of ``max_len``. However, for the purpose of this unit test,
        # we are going to treat only a random ``actual_len`` elements in the
        # sequence of each batch as actual elements, where the rest ``max_len -
        # actual_len`` elements are masked out.
        #
        # The index tuple B is created such that x[B] is a tensor of shape
        # [batch_size, actual_len, d_model], which picks out the actual elements
        # and ignores the masked elements.
        B = (torch.arange(batch_size).unsqueeze(1),
             torch.argsort(torch.rand(batch_size, max_len),
                           dim=1)[:, :actual_len])

        # mask[b, i] == True means the i-th element in the sequence of batch b
        # is MASKED OUT. This is a bit counterintuitive and thus worth noting.
        mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        mask[B] = False

        tf = alf.layers.TransformerBlock(
            d_model=d_model,
            d_k=d_model,
            d_v=d_model,
            d_ff=d_model,
            num_heads=3,
            memory_size=max_len,
            scale_attention_score=False,
            positional_encoding='none')

        # The following tests that feeding the full x with the mask is
        # equivalent to feeding the masked x to the transformer block up to
        # certain numerical error.
        y_lean = tf(x[B])
        y = tf(x, mask=mask)
        self.assertTrue((torch.abs(y_lean - y[B]) < 2e-3).all())

    @parameterized.parameters(0, 1, 2, 3)
    def test_transformer_block(self, task_type=2):
        batch_size = 100
        max_len = 32
        d_model = 64
        layers = [alf.layers.FC(4, d_model, torch.relu_)]
        for i in range(2):
            layers.append(
                alf.layers.TransformerBlock(
                    d_model=d_model,
                    d_k=d_model,
                    d_v=d_model,
                    d_ff=d_model,
                    num_heads=3,
                    memory_size=max_len,
                    scale_attention_score=False,
                    positional_encoding='rel' if task_type >= 3 else 'abs'))
        layers.append(alf.layers.FC(d_model, 1))
        model = nn.Sequential(*layers)

        def _get_batch_content_based(batch_size):
            """
            A simple memory task:
            In the first half of the sequence x[n] (i, j < max_len // 2):
                one of x[n, i, 0] (over i) is set to 1, which means type 0 value is x[n, i, 3]
                one of x[n, j, 1] (over j) is set to 1, which means type 1 value is x[n, j, 3]
            In the second half of the sequence (k >= max_len // 2)
                x[n, k, type] = 1 means the desired result for y[n, k] is the value of the type
            x[n, i, 2] is used for indicating whether i is at the second half or not.
            This feature is not necessary for successfully train the task, but it
            make the training much faster, which is required for a unittest.
            """
            x = torch.zeros(batch_size, max_len, 4)
            x[:, :, 3] = torch.rand(batch_size, max_len)
            # indicating second half
            x[:, max_len // 2:, 2] = 1
            index = torch.randint(max_len // 2, size=(batch_size, 2))
            bindex = torch.arange(batch_size)
            x[bindex, index[:, 0], 0] = 1
            x[bindex, index[:, 1], 1] = 1
            read_type = torch.randint(2, size=(batch_size, max_len // 2))
            read_index = max_len // 2 + torch.arange(max_len // 2).unsqueeze(0)

            bindex = bindex.unsqueeze(-1)
            x[bindex, read_index, read_type] = 1
            y = torch.zeros(batch_size, max_len)
            y[bindex, read_index] = x[bindex, index[bindex, read_type], 3]
            return x, y

        def _get_batch_position_based(batch_size):
            """
            A simple memory task:
            In the second half of the sequence (k >= max_len // 2)
                x[n, k, type] = 1 means the desired result for y[n, k] is x[n, 4 + type, 3]
            x[n, i, 2] is used for indicating whether i is at the second half or not.
            """
            x = torch.zeros(batch_size, max_len, 4)
            x[:, :, 3] = torch.rand(batch_size, max_len)
            # indicating second half
            x[:, max_len // 2:, 2] = 1
            read_type = torch.randint(2, size=(batch_size, max_len // 2))
            read_index = max_len // 2 + torch.arange(max_len // 2).unsqueeze(0)
            bindex = torch.arange(batch_size).unsqueeze(-1)
            x[bindex, read_index, read_type] = 1
            y = torch.zeros(batch_size, max_len)
            y[bindex, read_index] = x[bindex, 4 + read_type, 3]
            return x, y

        def _get_batch_position_target(batch_size):
            """
            A simple memory task:
            In the first half of the sequence x[n] (i, j < max_len // 2):
                one of x[n, i, 0] (over i) is set to 1, which means type 0 is at location i
                one of x[n, j, 1] (over j) is set to 1, which means type 1 is at location j
            In the second half of the sequence (k >= max_len // 2)
                x[n, k, type] = 1 means the desired result for y[n, k] is the location of the type
            x[n, i, 2] is used for indicating whether i is at the second half or not.
            """
            x = torch.zeros(batch_size, max_len, 4)
            #x[:, :, 3] = torch.rand(batch_size, max_len)
            # indicating second half
            x[:, max_len // 2:, 2] = 1
            index = torch.randint(max_len // 2, size=(batch_size, 2))
            bindex = torch.arange(batch_size)
            x[bindex, index[:, 0], 0] = 1
            x[bindex, index[:, 1], 1] = 1
            read_type = torch.randint(2, size=(batch_size, max_len // 2))
            read_index = max_len // 2 + torch.arange(max_len // 2).unsqueeze(0)

            bindex = bindex.unsqueeze(-1)
            x[bindex, read_index, read_type] = 1
            y = torch.zeros(batch_size, max_len)
            y[bindex, read_index] = index[bindex, read_type].to(torch.float32)
            return x, y

        def _get_batch_relative_position_based(batch_size):
            """
            A simple memory task:
            In the second half of the sequence (k >= max_len // 2)
                x[n, k, type] = 1 means the desired result for y[n, k] is x[n, k - 8 - type, 3]
            x[n, i, 2] is used for indicating whether i is at the second half or not.
            """
            x = torch.zeros(batch_size, max_len, 4)
            x[:, :, 3] = torch.rand(batch_size, max_len)
            # indicating second half
            x[:, max_len // 2:, 2] = 1
            read_type = torch.randint(2, size=(batch_size, max_len // 2))
            read_index = max_len // 2 + torch.arange(max_len // 2).unsqueeze(0)
            bindex = torch.arange(batch_size).unsqueeze(-1)
            x[bindex, read_index, read_type] = 1
            y = torch.zeros(batch_size, max_len)
            y[bindex, read_index] = x[bindex, read_index - 8 - read_type, 3]
            return x, y

        get_batch = [
            _get_batch_content_based, _get_batch_position_based,
            _get_batch_position_target, _get_batch_relative_position_based
        ][task_type]

        iters = [200, 500, 900, 500][task_type]
        optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
        for i in range(iters):
            optimizer.zero_grad()
            x, y = get_batch(batch_size)
            pred = model(x).squeeze(-1)
            loss = torch.mean((pred - y)**2)
            logging.log_every_n(
                logging.INFO,
                "%s loss=%s" % (i, loss.detach().cpu().numpy()),
                n=100)
            loss.backward()
            optimizer.step()

        logging.info("loss=%s" % loss.detach().cpu().numpy())
        self.assertLess(loss, 0.01)

    def test_fc_batch_ensemble(self):
        num_threads = torch.get_num_threads()
        # Use single thread to make the equality tests more robust. Different
        # addition order from multi-thread may lead to slightly different result.
        torch.set_num_threads(1)

        batch_size = 256
        x = torch.randn((batch_size, 16))
        layer1 = alf.layers.FCBatchEnsemble(
            16, 24, ensemble_size=8, use_bias=True)
        layer2 = alf.layers.FCBatchEnsemble(
            24, 1, ensemble_size=8, use_bias=False, output_ensemble_ids=False)
        y = layer1(x)
        # Test correct output type and shape
        self.assertEqual(type(y), tuple)
        self.assertEqual(y[0].shape, (batch_size, 24))
        self.assertEqual(y[1].shape, (batch_size, ))
        z = layer2(y)
        self.assertEqual(z.shape, (256, 1))

        x = torch.randn((8, 16))
        x = torch.cat([x] * 32, dim=0)
        ensemble_ids = torch.arange(8).unsqueeze(-1).expand(-1, 8).reshape(-1)
        ensemble_ids = torch.cat([ensemble_ids] * 4, dim=0)
        y = layer1((x, ensemble_ids))
        z = layer2(y)
        # test same ensemble id leads to same result
        # different ensemble id leads to different result
        self.assertTensorClose(z[0:64], z[64:128], epsilon=1e-6)
        self.assertTensorClose(z[0:64], z[128:192], epsilon=1e-6)
        self.assertTensorClose(z[0:64], z[192:256], epsilon=1e-6)

        self.assertTrue((z[0:8] != z[8:16]).all())
        self.assertTrue((z[0:8] != z[16:24]).all())
        self.assertTrue((z[0:8] != z[24:32]).all())
        self.assertTrue((z[0:8] != z[32:40]).all())
        self.assertTrue((z[0:8] != z[40:48]).all())
        self.assertTrue((z[0:8] != z[48:56]).all())
        self.assertTrue((z[0:8] != z[56:64]).all())

        torch.set_num_threads(num_threads)

    def test_conv2d_batch_ensemble(self):
        num_threads = torch.get_num_threads()
        # Use single thread to make the equality tests more robust. Different
        # addition order from multi-thread may lead to slightly different result.
        torch.set_num_threads(1)

        batch_size = 256
        x = torch.randn((batch_size, 3, 10, 10))
        layer1 = alf.layers.Conv2DBatchEnsemble(
            3, 16, 5, ensemble_size=8, use_bias=True)
        layer2 = alf.layers.Conv2DBatchEnsemble(
            16,
            1,
            3,
            ensemble_size=8,
            use_bias=False,
            output_ensemble_ids=False)
        y = layer1(x)
        # Test correct output type and shape
        self.assertEqual(type(y), tuple)
        self.assertEqual(y[0].shape, (batch_size, 16, 6, 6))
        self.assertEqual(y[1].shape, (batch_size, ))
        z = layer2(y)
        self.assertEqual(z.shape, (batch_size, 1, 4, 4))

        x = torch.randn((8, 3, 10, 10))
        x = torch.cat([x] * 32, dim=0)
        ensemble_ids = torch.arange(8).unsqueeze(-1).expand(-1, 8).reshape(-1)
        ensemble_ids = torch.cat([ensemble_ids] * 4, dim=0)
        y = layer1((x, ensemble_ids))
        z = layer2(y)
        # test same ensemble id leads to same result
        # different ensemble id leads to different result
        self.assertTensorClose(z[0:64], z[64:128], epsilon=1e-6)
        self.assertTensorClose(z[0:64], z[128:192], epsilon=1e-6)
        self.assertTensorClose(z[0:64], z[192:256], epsilon=1e-6)

        z = z.view(batch_size, -1)
        self.assertTrue(((z[0:8] != z[8:16]).sum(-1) > 0).all())
        self.assertTrue(((z[0:8] != z[16:24]).sum(-1) > 0).all())
        self.assertTrue(((z[0:8] != z[24:32]).sum(-1) > 0).all())
        self.assertTrue(((z[0:8] != z[32:40]).sum(-1) > 0).all())
        self.assertTrue(((z[0:8] != z[40:48]).sum(-1) > 0).all())
        self.assertTrue(((z[0:8] != z[48:56]).sum(-1) > 0).all())
        self.assertTrue(((z[0:8] != z[56:64]).sum(-1) > 0).all())

        torch.set_num_threads(num_threads)

    @parameterized.parameters((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3),
                              (3, 1), (3, 2), (3, 3))
    def test_causal_conv1d_shape(self,
                                 kernel_size,
                                 dilation,
                                 batch_size=5,
                                 act=math_ops.identity,
                                 use_bias=True):
        in_channels = 4
        out_channels = 5
        signal_length = 40

        # create a batched multi-channel 1d signal
        signal = torch.randn(batch_size, in_channels, signal_length)

        causal_conv = alf.layers.CausalConv1D(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            activation=act,
            use_bias=use_bias)

        out = causal_conv(signal)

        # the output signal should have the same length as the original one
        self.assertTrue(out.shape == (batch_size, out_channels, signal_length))

    def test_causal_conv1d_output_causality(self, batch_size=5):

        in_channels = 1
        out_channels = 1
        signal_length = 40

        # create a batched multi-channel 1d signal
        signal = torch.randn(batch_size, in_channels, signal_length)

        causal_conv = alf.layers.CausalConv1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            dilation=1,
            hide_current=True,
            activation=math_ops.identity,
            use_bias=False)

        # here we create a 1x1 identity filter
        causal_conv.weight.data = torch.full_like(causal_conv.weight.data, 1.0)

        out = causal_conv(signal)

        # the output signal should be equal to input signal shifted to the
        # right by one step
        self.assertTensorClose(out[..., 1:], signal[..., :-1], epsilon=1e-6)

    def test_sequential1(self):
        net = alf.layers.Sequential(
            alf.layers.FC(4, 6),
            a=alf.layers.FC(6, 8),
            b=alf.layers.FC(8, 12),
            c=(('a', 'b', 'input'), alf.layers.NestConcat()))

        batch_size = 24
        x = torch.randn((batch_size, 4))
        y = net(x)

        x1 = net[0](x)
        x2 = net[1](x1)
        x3 = net[2](x2)
        x4 = net[3]((x2, x3, x))
        self.assertEqual(x4, y)

        input_spec = alf.BoundedTensorSpec((4, ))
        self._test_make_parallel(net, input_spec)

    def test_sequential2(self):
        # test wrong field name
        net = alf.layers.Sequential(
            alf.layers.FC(4, 6),
            a=alf.layers.FC(6, 8),
            c=(('a', 'b', 'input'), alf.layers.NestConcat()),
            b=alf.layers.FC(8, 12))

        batch_size = 24
        x = torch.randn((batch_size, 4))
        self.assertRaises(LookupError, net, x)

    def test_sequential3(self):
        # test output
        net = alf.layers.Sequential(
            alf.layers.FC(4, 6),
            a=alf.layers.FC(6, 8),
            b=alf.layers.FC(8, 12),
            c=(('a', 'b'), alf.layers.NestConcat()),
            output=dict(a='a', b='b', c='c'))

        batch_size = 24
        x = torch.randn((batch_size, 4))
        y = net(x)

        x1 = net[0](x)
        a = net[1](x1)
        b = net[2](a)
        c = net[3]((a, b))
        self.assertEqual(y['a'], a)
        self.assertEqual(y['b'], b)
        self.assertEqual(y['c'], c)

        input_spec = alf.BoundedTensorSpec((4, ))
        self._test_make_parallel(net, input_spec)

    def test_sequential4(self):
        # test output
        net = alf.layers.Sequential(
            alf.layers.FC(4, 6),
            a=alf.layers.FC(6, 8),
            b=alf.layers.FC(8, 8),
            c=(('a', 'b'), lambda x: x[0] + x[1]))

        batch_size = 24
        x = torch.randn((batch_size, 4))
        y = net(x)

        x1 = net[0](x)
        a = net[1](x1)
        b = net[2](a)
        c = a + b
        self.assertEqual(c, y)

        input_spec = alf.BoundedTensorSpec((4, ))
        self._test_make_parallel(net, input_spec)

    def test_branch(self):
        net = alf.layers.Branch(alf.layers.FC(4, 6), alf.layers.FC(4, 8))
        input_spec = alf.BoundedTensorSpec((4, ))
        self._test_make_parallel(net, input_spec)

    def test_fc(self):
        input_spec = alf.BoundedTensorSpec((5, ))
        layer = alf.layers.FC(5, 7)
        self._test_make_parallel(layer, input_spec)

    def test_conv2d(self):
        input_spec = alf.BoundedTensorSpec((3, 10, 10))
        layer = alf.layers.Conv2D(3, 5, 3)
        self._test_make_parallel(
            layer,
            input_spec,
            get_pnet_parameters=lambda pnet: (pnet.weight, pnet.bias))

    def test_conv_transpose_2d(self):
        input_spec = alf.BoundedTensorSpec((3, 10, 10))
        layer = alf.layers.ConvTranspose2D(3, 5, 3)
        self._test_make_parallel(
            layer,
            input_spec,
            get_pnet_parameters=lambda pnet: (pnet.weight, pnet.bias))

    def test_cast(self):
        input_spec = alf.BoundedTensorSpec((8, ), dtype=torch.uint8)
        layer = alf.layers.Cast()
        self._test_make_parallel(layer, input_spec)

    def test_transpose(self):
        input_spec = alf.BoundedTensorSpec((8, 4, 10))
        layer = alf.layers.Transpose()
        self._test_make_parallel(layer, input_spec)
        layer = alf.layers.Transpose(0, 2)
        self._test_make_parallel(layer, input_spec)
        layer = alf.layers.Transpose(-2, -1)
        self._test_make_parallel(layer, input_spec)

    def test_permute(self):
        input_spec = alf.BoundedTensorSpec((8, 4, 10))
        layer = alf.layers.Permute(2, 1, 0)
        self._test_make_parallel(layer, input_spec)

    def test_onehot(self):
        input_spec = alf.BoundedTensorSpec((10, ),
                                           dtype=torch.int64,
                                           minimum=0,
                                           maximum=11)
        layer = alf.layers.OneHot(12)
        self._test_make_parallel(layer, input_spec)

    def test_reshape(self):
        input_spec = alf.BoundedTensorSpec((8, 4, 10))
        layer = alf.layers.Reshape((32, 10))
        self._test_make_parallel(layer, input_spec)

    def test_get_fields(self):
        input_spec = dict(
            a=alf.BoundedTensorSpec((8, 4, 10)),
            b=alf.BoundedTensorSpec((4, )),
            c=alf.BoundedTensorSpec((3, )))
        layer = alf.layers.GetFields(('a', 'c'))
        self._test_make_parallel(layer, input_spec)

    def test_sum(self):
        input_spec = alf.BoundedTensorSpec((8, 4, 10))
        layer = alf.layers.Sum(dim=1)
        self._test_make_parallel(layer, input_spec)
        layer = alf.layers.Sum(dim=-1)
        self._test_make_parallel(layer, input_spec)

    def test_replication_pad_2d(self):
        layer = alf.layers.ReplicationPad2d((1, 2, 3, 4))
        x = torch.arange(120).reshape(2, 3, 4, 5)
        y = layer(x)
        self.assertEqual(y.shape, (2, 3, 11, 8))
        self.assertTensorEqual(y[:, :, 3:7, 1:6], x)

        self.assertTensorEqual(
            y[:, :, :3, 1:6], torch.repeat_interleave(
                x[:, :, :1, :], 3, dim=2))
        self.assertTensorEqual(
            y[:, :, 7:, 1:6], torch.repeat_interleave(
                x[:, :, 3:, :], 4, dim=2))

        self.assertTensorEqual(y[:, :, 3:7, :1], x[:, :, :, :1])
        self.assertTensorEqual(
            y[:, :, 3:7, 6:], torch.repeat_interleave(
                x[:, :, :, 4:], 2, dim=-1))

        self.assertTensorEqual(
            y[:, :, :3, :1], torch.repeat_interleave(
                x[:, :, :1, :1], 3, dim=2))
        self.assertTensorEqual(
            y[:, :, :3, 6:],
            torch.einsum('ijkl, kl->ijkl', x[:, :, :1, 4:], torch.ones(3, 2)))
        self.assertTensorEqual(
            y[:, :, 7:, :1], torch.repeat_interleave(
                x[:, :, 3:, :1], 4, dim=2))
        self.assertTensorEqual(
            y[:, :, 7:, 6:],
            torch.einsum('ijkl, kl->ijkl', x[:, :, 3:, 4:], torch.ones(4, 2)))

    def test_random_crop(self):
        # It's hard to test the randomness. Here we just make the crop
        # size same as the padded size so that there is no randomness and
        # it is same as ReplicationPad2d((1,2,3,4))
        layer = alf.layers.RandomCrop((11, 8), (1, 2, 3, 4))
        x = torch.arange(120).reshape(2, 3, 4, 5)
        y = layer(x)
        self.assertEqual(y.shape, (2, 3, 11, 8))

        self.assertTensorEqual(y[:, :, 3:7, 1:6], x)

        self.assertTensorEqual(
            y[:, :, :3, 1:6], torch.repeat_interleave(
                x[:, :, :1, :], 3, dim=2))
        self.assertTensorEqual(
            y[:, :, 7:, 1:6], torch.repeat_interleave(
                x[:, :, 3:, :], 4, dim=2))

        self.assertTensorEqual(y[:, :, 3:7, :1], x[:, :, :, :1])
        self.assertTensorEqual(
            y[:, :, 3:7, 6:], torch.repeat_interleave(
                x[:, :, :, 4:], 2, dim=-1))

        self.assertTensorEqual(
            y[:, :, :3, :1], torch.repeat_interleave(
                x[:, :, :1, :1], 3, dim=2))
        self.assertTensorEqual(
            y[:, :, :3, 6:],
            torch.einsum('ijkl, kl->ijkl', x[:, :, :1, 4:], torch.ones(3, 2)))
        self.assertTensorEqual(
            y[:, :, 7:, :1], torch.repeat_interleave(
                x[:, :, 3:, :1], 4, dim=2))
        self.assertTensorEqual(
            y[:, :, 7:, 6:],
            torch.einsum('ijkl, kl->ijkl', x[:, :, 3:, 4:], torch.ones(4, 2)))


if __name__ == "__main__":
    alf.test.main()
