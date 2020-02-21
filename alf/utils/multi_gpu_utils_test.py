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

import unittest
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from alf.utils.multi_gpu_utils import MultiGPU


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        # the standard interface
        output = self.fc(input)
        global in_model_batch_size
        in_model_batch_size = input.size(0)
        return output

    def train_step(self, input):
        # self-defined non-standard interface
        output = self.fc(input)
        return output


class DummyDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


# some shared parameters for testing
input_size = 4
output_size = 3

batch_size = 30
data_size = 50

in_model_batch_size = 0

gpu_available = torch.cuda.is_available()
if gpu_available:
    device = torch.device("cuda:0")


@unittest.skipUnless(gpu_available, "requires GPU")
class TestSingleGPU(unittest.TestCase):
    def test_single_gpu(self):
        model = Model(input_size, output_size).to(device)

        gpu_ids = [0]
        model = MultiGPU(model, gpu_ids)

        data_loader = DataLoader(
            dataset=DummyDataset(input_size, data_size),
            batch_size=batch_size,
            shuffle=True)

        for data in data_loader:
            input = data.to(device)
            output = model(input)
            outside_batch_size = input.size(0)

            self.assertEqual(in_model_batch_size * len(gpu_ids),
                             outside_batch_size)


@unittest.skipUnless(gpu_available, "requires GPU")
class TestMultiGPU(unittest.TestCase):
    def test_multi_gpu(self):
        model = Model(input_size, output_size).to(device)

        gpu_ids = list(range(torch.cuda.device_count()))
        model = MultiGPU(model, gpu_ids)

        data_loader = DataLoader(
            dataset=DummyDataset(input_size, data_size),
            batch_size=batch_size,
            shuffle=True)

        for data in data_loader:
            input = data.to(device)
            output = model(input)
            outside_batch_size = input.size(0)

            self.assertEqual(in_model_batch_size * len(gpu_ids),
                             outside_batch_size)


@unittest.skipUnless(gpu_available, "requires GPU")
class TestMultiGPUCustomizedFunction(unittest.TestCase):
    def test_multi_gpu_customized_function(self):
        model = Model(input_size, output_size).to(device)

        gpu_ids = list(range(torch.cuda.device_count()))
        model = MultiGPU(model, gpu_ids)

        data_loader = DataLoader(
            dataset=DummyDataset(input_size, data_size),
            batch_size=batch_size,
            shuffle=True)

        for data in data_loader:
            input = data.to(device)
            output = model.train_step(input)
            outside_batch_size = input.size(0)

            self.assertEqual(in_model_batch_size * len(gpu_ids),
                             outside_batch_size)


class TestRunOnCPU(unittest.TestCase):
    def test_run_on_cpu(self):
        model = Model(input_size, output_size)  # model is on cpu

        model = MultiGPU(model)
        data_loader = DataLoader(
            dataset=DummyDataset(input_size, data_size),
            batch_size=batch_size,
            shuffle=True)

        # check model is till on cpu
        self.assertEqual(next(model.parameters()).is_cuda, False)
        for data in data_loader:
            input = data
            output = model(input)
            outside_batch_size = input.size(0)

            self.assertEqual(in_model_batch_size, outside_batch_size)


if __name__ == '__main__':
    unittest.main()
