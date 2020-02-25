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
"""Simple wrapper over unittest.TestCase to provide extra functionality."""

import torch
import unittest


class TestCase(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.addTypeEqualityFunc(torch.Tensor, 'assertTensorEqual')

    def assertTensorEqual(self, t1, t2, msg=None):
        self.assertIsInstance(t1, torch.Tensor,
                              'First argument is not a Tensor')
        self.assertIsInstance(t2, torch.Tensor,
                              'Second argument is not a Tensor')

        if not torch.all(t1 == t2):
            standardMsg = '%s != %s' % (t1, t2)
            self.fail(self._formatMessage(msg, standardMsg))
