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
import os
import tempfile

import torch

import alf
from alf.tensor_specs import TensorSpec
from alf.utils.data_buffer import DataBuffer
from alf.utils.checkpoint_utils import Checkpointer


class DataBufferTest(alf.test.TestCase):
    def test_data_buffer(self):
        dim = 20
        capacity = 256
        data_spec = (TensorSpec(shape=()), TensorSpec(shape=(dim // 3 - 1, )),
                     TensorSpec(shape=(dim - dim // 3, )))

        data_buffer = DataBuffer(data_spec=data_spec, capacity=capacity)

        def _get_batch(batch_size):
            x = torch.randn(batch_size, dim)
            x = (x[:, 0], x[:, 1:dim // 3], x[..., dim // 3:])
            return x

        data_buffer.add_batch(_get_batch(100))
        self.assertEqual(int(data_buffer.current_size), 100)
        batch = _get_batch(1000)
        data_buffer.add_batch(batch)
        self.assertEqual(int(data_buffer.current_size), capacity)
        ret = data_buffer.get_batch_by_indices(torch.arange(capacity))
        self.assertEqual(ret[0], batch[0][-capacity:])
        self.assertEqual(ret[1], batch[1][-capacity:])
        self.assertEqual(ret[2], batch[2][-capacity:])
        batch = _get_batch(100)
        data_buffer.add_batch(batch)
        ret = data_buffer.get_batch_by_indices(
            torch.arange(data_buffer.current_size - 100,
                         data_buffer.current_size))
        self.assertEqual(ret[0], batch[0])
        self.assertEqual(ret[1], batch[1])
        self.assertEqual(ret[2], batch[2][-capacity:])

        # Test checkpoint working
        with tempfile.TemporaryDirectory() as checkpoint_directory:
            checkpoint = Checkpointer(
                checkpoint_directory, data_buffer=data_buffer)
            checkpoint.save(10)
            data_buffer = DataBuffer(data_spec=data_spec, capacity=capacity)
            checkpoint = Checkpointer(
                checkpoint_directory, data_buffer=data_buffer)
            global_step = checkpoint.load()
            self.assertEqual(global_step, 10)

        ret = data_buffer.get_batch_by_indices(
            torch.arange(data_buffer.current_size - 100,
                         data_buffer.current_size))
        self.assertEqual(ret[0], batch[0])
        self.assertEqual(ret[1], batch[1])
        self.assertEqual(ret[2], batch[2][-capacity:])


if __name__ == '__main__':
    alf.test.main()
