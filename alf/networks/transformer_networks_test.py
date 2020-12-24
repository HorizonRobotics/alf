# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import torch.nn as nn
from alf.networks import TransformerNetwork

import alf


class TransformerNetworkTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(True, False)
    def test_transformer_network(self, centralized_memory=True):
        d_model = 32
        core_size = 2
        memory_size = 128
        num_memory_layers = 8
        input_tensor_spec = [
            alf.TensorSpec((), dtype=torch.int64),
            alf.TensorSpec((3, 7, 7), dtype=torch.float32)
        ]
        input_preprocessors = [
            nn.Sequential(
                nn.Embedding(100, d_model), alf.layers.Reshape((1, d_model))),
            nn.Sequential(
                alf.layers.Conv2D(3, d_model, kernel_size=1),
                alf.layers.Reshape((d_model, 49)), alf.layers.Transpose())
        ]
        transformer = TransformerNetwork(
            input_tensor_spec,
            memory_size=memory_size,
            core_size=core_size,
            num_prememory_layers=2,
            num_memory_layers=num_memory_layers,
            num_attention_heads=8,
            d_ff=d_model,
            centralized_memory=centralized_memory,
            input_preprocessors=input_preprocessors)

        state_spec = transformer.state_spec
        if centralized_memory:
            self.assertEqual(len(state_spec), 1)
            self.assertEqual(state_spec[0][0].shape, (memory_size, d_model))
        else:
            self.assertEqual(len(state_spec), 8)
            for i in range(num_memory_layers):
                self.assertEqual(state_spec[i][0].shape,
                                 (memory_size, d_model))
        batch_size = 64
        x = [
            torch.randint(100, size=(batch_size, )),
            torch.rand((batch_size, 3, 7, 7))
        ]
        state = alf.utils.spec_utils.zeros_from_spec(transformer.state_spec,
                                                     batch_size)
        y, state = transformer(x, state)

        self.assertEqual(y.shape, (batch_size, core_size * d_model))


if __name__ == '__main__':
    alf.test.main()
