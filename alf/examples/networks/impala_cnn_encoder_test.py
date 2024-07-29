# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import torch

import alf
from alf.tensor_specs import TensorSpec
from alf.examples.networks import impala_cnn_encoder


class TestImpalaCnnEncoder(alf.test.TestCase):
    def test_residual_cnn_block_shape(self):
        # Such a residual CNN block does not change the shape of the input
        input_tensor_spec = TensorSpec((3, 20, 20), torch.float32)
        block = impala_cnn_encoder._create_residual_cnn_block(
            input_tensor_spec)
        self.assertEqual(input_tensor_spec, block.output_spec)

    def test_create_downsampling_cnn_stack_shape(self):
        input_tensor_spec = TensorSpec((3, 64, 64), torch.float32)
        single_stack = impala_cnn_encoder._create_downsampling_cnn_stack(
            input_tensor_spec=input_tensor_spec,
            output_channels=9,
            num_residual_blocks=2)
        self.assertEqual((9, 32, 32), single_stack.output_spec.shape)

    def test_impala_cnn_encoder_shape(self):
        observation_spec = TensorSpec((3, 64, 64), torch.float32)
        encoder = impala_cnn_encoder.create(
            input_tensor_spec=observation_spec,
            cnn_channel_list=(16, 32, 32),
            num_blocks_per_stack=2,
            flatten_output_size=256)
        self.assertEqual((256, ), encoder.output_spec.shape)
