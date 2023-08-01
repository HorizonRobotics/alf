# Copyright (c) 2023 Horizon Robotics and Hobot Contributors. All Rights Reserved.
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

import torch
import copy

import alf
from alf.optimizers import AdamTF

from alf.pretrained_models.pretrained_model import PretraindModel
from alf.pretrained_models.model_adapters import LinearAdapter, Conv2dAdapter


class ModelAdaptersTest(unittest.TestCase):
    def _test(self, x, model, pretrained):
        opt = AdamTF(lr=0.1)
        opt.add_param_group({'params': pretrained.parameters()})

        paras = [copy.deepcopy(p) for p in model.parameters()]
        adapter_paras = [
            copy.deepcopy(p) for p in pretrained._adapters.parameters()
        ]

        for i in range(2):
            y = pretrained(x).sum()
            opt.zero_grad()
            y.backward()
            opt.step()

        paras1 = list(model.parameters())
        adapter_paras1 = list(pretrained._adapters.parameters())

        for p, p1 in zip(paras, paras1):
            self.assertTrue(torch.all(p == p1))
        for ap, ap1 in zip(adapter_paras, adapter_paras1):
            self.assertTrue(torch.all(ap != ap1))

    def test_linear_adapter(self):
        x = torch.tensor([0.1, 0.2])

        fc = torch.nn.Sequential(
            torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 4))
        pretrained = PretraindModel(fc)

        y0 = pretrained(x)

        alf.config('LinearAdapter', rank=1)
        pretrained.add_adapter(LinearAdapter)

        y1 = pretrained(x)
        # The initial adapter weights are all zeros
        self.assertTrue(torch.all(y1 == y0))

        self._test(x, fc, pretrained)

        y2 = pretrained(x)
        # after training
        self.assertTrue(torch.all(y2 != y0))

        pretrained.merge_adapter()
        y2_ = pretrained(x)
        self.assertTrue(torch.allclose(y2_, y2, atol=1e-7))

        # Check if removing adapter works or not
        pretrained.remove_adapter()
        y3 = pretrained(x)
        self.assertTrue(torch.allclose(y3, y0, atol=1e-7))

    def test_conv_adapter(self):
        x = torch.rand([1, 10, 10])
        conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3, padding=1), torch.nn.Tanh(),
            torch.nn.Conv2d(3, 2, kernel_size=1))

        pretrained = PretraindModel(conv)

        y0 = pretrained(x)

        alf.config('Conv2dAdapter', rank=4)
        pretrained.add_adapter(Conv2dAdapter)

        y1 = pretrained(x)
        # The initial adapter weights are all zeros
        self.assertTrue(torch.all(y1 == y0))

        self._test(x, conv, pretrained)

        y2 = pretrained(x)
        # after training
        self.assertTrue(torch.all(y2 != y0))

        pretrained.merge_adapter()
        y2_ = pretrained(x)
        self.assertTrue(torch.allclose(y2_, y2, atol=1e-7))

        # Check if removing adapter works or not
        pretrained.remove_adapter()
        y3 = pretrained(x)
        self.assertTrue(torch.allclose(y3, y0, atol=1e-7))

    def test_double_adaptation(self):
        x = torch.rand([1, 1, 10, 10])
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3, padding=1), torch.nn.Tanh(),
            torch.nn.Conv2d(3, 2, kernel_size=1), alf.layers.Reshape(-1),
            torch.nn.Linear(2 * 10 * 10, 10))
        pretrained = PretraindModel(model)

        y0 = pretrained(x)

        pretrained.add_adapter(LinearAdapter)
        pretrained.add_adapter(Conv2dAdapter)

        self.assertEqual(len(pretrained._adapters), 3)  # 2 conv + 1 linear

        y1 = pretrained(x)
        # The initial adapter weights are all zeros
        self.assertTrue(torch.all(y1 == y0))

        self._test(x, model, pretrained)

        y2 = pretrained(x)
        # after training
        self.assertTrue(torch.all(y2 != y0))


if __name__ == "__main__":
    unittest.main()
