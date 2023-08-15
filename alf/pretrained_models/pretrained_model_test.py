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

import alf
from alf.pretrained_models.pretrained_model import PretrainedModel
import alf.utils.checkpoint_utils as ckpt_utils
from alf.utils.checkpoint_utils_test import Net
from alf.pretrained_models.model_adapters.lora import (LinearAdapter,
                                                       Conv2dAdapter)


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


if __name__ == '__main__':
    alf.test.main()
