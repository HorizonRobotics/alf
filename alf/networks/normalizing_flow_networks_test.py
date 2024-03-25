# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from absl import logging

import numpy as np

import torch

import alf
from alf.networks.normalizing_flow_networks import _RealNVPTransform
from alf.networks import RealNVPNetwork, NetworkWrapper
from alf.tensor_specs import TensorSpec


class RealNVPTransformTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((4, 1.), (10, 0.5))
    def test_RealNVP_zero(self, D, prob):
        spec = TensorSpec((D, ))
        mask = torch.rand((D, )) > prob
        # scale=1, translation=0
        scale_trans_net = NetworkWrapper(lambda x: x * 0.,
                                         spec).make_parallel(2)
        transform = _RealNVPTransform(
            spec,
            scale_trans_net,
            mask,
            cache_size=0,
            scale_nonlinear=torch.exp)
        x = spec.rand((1, ))
        y = transform(x)
        self.assertTensorClose(x, y)
        y_inv = transform.inv(y)
        self.assertTensorClose(y_inv, x)

    @parameterized.parameters((alf.math.identity, ), (torch.relu_),
                              (alf.math.square))
    def test_RealNVP_elementwise_zero(self, elementwise_func):
        """If the scale and translation networks are both elementwise functions that
        map 0 to 0, then RealNVPTransform with torch.exp is always an identical
        transformation.
        """
        spec = TensorSpec((100, ))
        mask = torch.rand((100, )) > 0.5
        scale_trans_net = NetworkWrapper(elementwise_func,
                                         spec).make_parallel(2)
        transform = _RealNVPTransform(
            spec,
            scale_trans_net,
            mask,
            cache_size=0,
            scale_nonlinear=torch.exp)

        x = spec.rand((1, ))
        y = transform(x)
        self.assertTensorClose(y, x)
        y_inv = transform.inv(y)
        self.assertTensorClose(y_inv, x)

    def test_RealNVP_transform(self):
        spec = TensorSpec((2, 2))
        mask = torch.tensor([[0, 0], [1, 1]]).to(torch.bool)
        matrix = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 0, 1]]).to(torch.float32)
        scale_trans_net = NetworkWrapper(
            lambda x: torch.matmul(x.reshape(-1, 4), matrix),
            spec).make_parallel(2)
        transform = _RealNVPTransform(
            spec,
            scale_trans_net,
            mask,
            cache_size=0,
            scale_nonlinear=torch.exp)

        x = torch.tensor([[[1, 2], [3, 4]]]).to(torch.float32)
        y = transform(x)
        # x * mask -> [0,0,3,4]
        # scale/trans -> [0,3,0,4]
        # (1-mask) * (exp(scale)*x+trans) -> [1*exp(0)+0,2*exp(3)+3,0,0]
        expected_y = torch.tensor([[[1, 3 + 2 * np.exp(3)], [3, 4]]])
        self.assertTensorClose(y, expected_y, epsilon=1e-4)
        y_inv = transform.inv(y)
        self.assertTensorClose(y_inv, x)

        # test Jacobian
        j = transform.log_abs_det_jacobian(x, y)
        expected_j = torch.tensor([[[0, 3], [0, 0]]]).reshape(1, -1).sum(-1)
        self.assertTensorClose(j, expected_j)

    @parameterized.parameters((10, 0, torch.exp),
                              (100, 1, alf.math.clipped_exp),
                              (500, 0, torch.nn.functional.softplus))
    def test_RealNVP_transform_jacobian_diagonal(self, D, cache_size,
                                                 scale_nonlinear):
        spec = TensorSpec((D, ))
        mask = torch.rand((D, )) > 0.5

        scale_trans_net = alf.networks.EncodingNetwork(
            input_tensor_spec=(spec, spec),
            preprocessing_combiner=alf.nest.utils.NestConcat(),
            fc_layer_params=(256, ) * 3,
            last_layer_size=D,
            last_activation=alf.math.identity)
        scale_trans_net = scale_trans_net.make_parallel(2)

        z = spec.rand((1, ))
        transform = _RealNVPTransform(
            spec,
            scale_trans_net,
            mask,
            z=z,
            conditional_input_tensor_spec=spec,
            cache_size=cache_size,
            scale_nonlinear=scale_nonlinear)

        x = spec.rand((1, ))
        y = transform(x)
        y_inv = transform.inv(y)
        self.assertTensorClose(x, y_inv, epsilon=1e-4)

        # compare PyTorch's Jacobian diagonal with our manual result
        jacob = torch.autograd.functional.jacobian(transform, x)  # [1,D,1,D]
        jacob = jacob.squeeze(0).squeeze(1)
        jacob_diag = torch.diagonal(jacob, 0)
        self.assertTrue(torch.all(jacob_diag > 0))
        j = transform.log_abs_det_jacobian(x, y)
        self.assertTensorClose(
            j, jacob_diag.log().sum(-1, keepdim=True), epsilon=1e-4)

    @parameterized.parameters(((1, ), (10, )), ((1, 2), (10, 10)),
                              ((3, 4, 5, 6), (5, 5, 5)))
    def test_RealNVP_transform_outer_dims(self, outer_dims, input_shape):
        x_spec = TensorSpec(input_shape)
        z_spec = TensorSpec((10, 10))

        scale_trans_net = alf.networks.EncodingNetwork(
            input_tensor_spec=(x_spec, z_spec),
            input_preprocessors=(alf.layers.Reshape(-1),
                                 alf.layers.Reshape(-1)),
            preprocessing_combiner=alf.nest.utils.NestConcat(),
            last_layer_size=x_spec.numel,
            last_activation=alf.math.identity).make_parallel(2)

        x = x_spec.rand(outer_dims)
        if len(outer_dims) > 2:
            z = z_spec.rand(outer_dims[-2:])
        else:
            z = z_spec.rand(outer_dims[-1:])
        mask = x_spec.zeros().to(torch.bool)

        transform = _RealNVPTransform(
            x_spec,
            scale_trans_net,
            mask,
            conditional_input_tensor_spec=z_spec,
            z=z)
        y = transform(x)

        self.assertEqual(y.shape, x.shape)


class RealNVPNetworkTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((
        10,
        None,
        3,
        'contiguous',
        4,
        (10, 10),
        False,
    ), (30, 30, None, 'random', 5, (200, 200), True), (
        10,
        20,
        4,
        'distributed',
        1,
        (10, 10),
        True,
    ), (100, 200, 40, 'distributed', 3, (50, 50), False))
    def test_RealNVPNetwork(self, input_size, conditional_input_size, sub_dim,
                            mask_mode, num_layers, fc_layers,
                            use_transform_cache):

        spec = TensorSpec((input_size, ))
        if conditional_input_size is not None:
            conditional_spec = TensorSpec((conditional_input_size, ))
        else:
            conditional_spec = None
        network = RealNVPNetwork(
            input_tensor_spec=spec,
            conditional_input_tensor_spec=conditional_spec,
            preprocessing_combiner=alf.nest.utils.NestConcat(),
            fc_layer_params=fc_layers,
            sub_dim=sub_dim,
            mask_mode=mask_mode,
            use_transform_cache=use_transform_cache,
            num_layers=num_layers)

        B = 3

        for i in range(3):
            x = spec.randn(outer_dims=(B, ))
            if conditional_spec is not None:
                z = conditional_spec.randn(outer_dims=(B, ))
                y = network((x, z))[0]
                y_inv = network.inverse((y, z))[0]
            else:
                z = None
                y = network(x)[0]
                y_inv = network.inverse(y)[0]

            # check inverse result
            self.assertTensorClose(y_inv, x, epsilon=1e-3)

        # check mask
        if num_layers == 1 and mask_mode == "distributed":
            # check the unchanged portion of 'x'
            mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])
            self.assertTensorClose(x * mask, y * mask)

        # check Jacobian
        transform = network.make_invertible_transform(z)
        # compare PyTorch's Jacobian diagonal with our manual result
        jacob = torch.autograd.functional.jacobian(transform, x)  # [B,D,B,D]
        # Note that 'jacob' might not be triangular for composed transform
        jacob = jacob.reshape(B * input_size, -1)
        jacob_det = torch.det(jacob)
        j = transform.log_abs_det_jacobian(x, y)  # [B,D]
        j = j.reshape(-1)
        self.assertTensorClose(
            j.sum(), jacob_det.abs().log().sum(-1), epsilon=1e-3)

    @parameterized.parameters((True, 10), (True, None), (False, None),
                              (False, 10))
    def test_cached_transform(self, use_transform_cache,
                              conditional_input_size):
        input_size = 10
        spec = TensorSpec((input_size, ))
        if conditional_input_size is not None:
            conditional_spec = TensorSpec((conditional_input_size, ))
        else:
            conditional_spec = None
        network = RealNVPNetwork(
            input_tensor_spec=spec,
            conditional_input_tensor_spec=conditional_spec,
            preprocessing_combiner=alf.nest.utils.NestConcat(),
            use_transform_cache=use_transform_cache)

        x = spec.randn(outer_dims=(1, ))
        if conditional_spec is not None:
            z = conditional_spec.randn(outer_dims=(1, ))
            t1 = network.make_invertible_transform(z)
            t2 = network.make_invertible_transform(z)
            if use_transform_cache:
                self.assertTrue(t1 is t2)
            else:
                self.assertFalse(t1 is t2)
            z = conditional_spec.randn(outer_dims=(1, ))
            t3 = network.make_invertible_transform(z)
            self.assertFalse(t2 is t3)
        else:
            t1 = network.make_invertible_transform()
            t2 = network.make_invertible_transform()
            if use_transform_cache:
                self.assertTrue(t1 is t2)
            else:
                self.assertFalse(t1 is t2)


if __name__ == "__main__":
    alf.test.main()
