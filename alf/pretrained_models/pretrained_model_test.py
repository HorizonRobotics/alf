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
import tempfile
from absl.testing import parameterized
import os
import json

import torch
import torch.nn as nn

import alf
from alf.pretrained_models.pretrained_model import PretrainedModel
import alf.utils.checkpoint_utils as ckpt_utils
from alf.pretrained_models.model_adapters.lora import (LinearAdapter,
                                                       Conv2dAdapter)


class Net(nn.Module):
    def __init__(self, size=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 1, 3, padding=1)
        self.fc1 = nn.Linear(size**2, size**2)
        self.fc2 = nn.Linear(size**2, 10)
        self._size = size

    def forward(self, input):
        return self.fc2(
            self.fc1(self.conv2(self.conv1(input)).reshape(-1, self._size**2)))


class PretrainedModelTest(alf.test.TestCase):
    def test_pretrained_model_ckpt(self):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            net = Net()
            pretrained_net = PretrainedModel(
                net, adapter_cls=[LinearAdapter, Conv2dAdapter])

            # check the base model will be ignored for params
            named_paras = pretrained_net.named_parameters()
            for name, para in named_paras:
                self.assertFalse('conv' in name or 'fc' in name)

            ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, net=pretrained_net)
            # This merge doesn't affect ckpt
            pretrained_net.merge_adapter()
            ckpt_mngr.save(0)

            model_structure_file = os.path.join(ckpt_dir,
                                                'ckpt-structure.json')
            with open(model_structure_file, 'r') as f:
                model_structure = json.load(f)

            expected_model_structure = {
                'global_step': -1,
                'net': {
                    '_adapters.0._wA': -1,
                    '_adapters.1._wA': -1,
                    '_adapters.2._wA': -1,
                    '_adapters.3._wA': -1
                }
            }
            self.assertEqual(expected_model_structure, model_structure)

            ckpt_mngr.load(0)

    def test_finetuning_grad(self):
        alf.reset_configs()
        alf.config('Conv2dAdapter', rank=32)
        net = Net(10)
        net.half()
        pretrained_net = PretrainedModel(
            net, adapter_cls=[LinearAdapter, Conv2dAdapter])
        x = torch.zeros([1, 3, 10, 10]).to(torch.float16)
        y = pretrained_net(x).sum()
        y.float().backward()
        for name, para in pretrained_net.named_parameters():
            self.assertTrue(para.grad is not None)

    def test_module_blacklist(self):
        alf.reset_configs()
        alf.config('Conv2dAdapter', rank=32)
        alf.config('LinearAdapter', rank=32)
        net = Net(10)
        # This regex will exclude 'conv1 and 'fc1'
        blacklist = ['.*1']
        pretrained_net = PretrainedModel(
            net,
            adapter_cls=[LinearAdapter, Conv2dAdapter],
            module_blacklist=blacklist)
        for name in pretrained_net.adapted_module_names:
            for b in blacklist:
                assert b not in name
        # because of blacklist, adapters only have two weights
        self.assertEqual(len(list(pretrained_net.parameters())), 2)

        pretrained_net.remove_adapter()

        whitelist = ['.*conv.*']
        pretrained_net = PretrainedModel(
            net,
            adapter_cls=[LinearAdapter, Conv2dAdapter],
            module_whitelist=whitelist)
        # because of whitelist, adapters only have two weights
        self.assertEqual(len(list(pretrained_net.parameters())), 2)


if __name__ == '__main__':
    alf.test.main()
