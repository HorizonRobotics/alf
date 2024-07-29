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
"""Simple wrapper over unittest.TestCase to provide extra functionality."""

import numpy as np
import torch
import unittest
import alf
from alf.utils import common


class TestCase(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.addTypeEqualityFunc(torch.Tensor, 'assertTensorEqual')

    def setUp(self):
        # create_environment() might have been used in other tests.
        # We need to reset_configs() so we can avoid the error of configuring
        # a function after it is used.
        alf.reset_configs()
        # Some test may create a global env. We need to close it.
        alf.close_env()
        common.set_random_seed(1)
        alf.summary.reset_global_counter()

    def assertTensorEqual(self, t1, t2, msg=None):
        self.assertIsInstance(t1, torch.Tensor,
                              'First argument is not a Tensor')
        self.assertIsInstance(t2, torch.Tensor,
                              'Second argument is not a Tensor')
        self.assertEqual(t1.shape, t2.shape, msg=msg)
        if not torch.all(t1.cpu() == t2.cpu()):
            standardMsg = '%s != %s' % (t1, t2)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertTensorClose(self, t1, t2, epsilon=1e-6, msg=None):
        self.assertIsInstance(t1, torch.Tensor,
                              'First argument is not a Tensor')
        self.assertIsInstance(t2, torch.Tensor,
                              'Second argument is not a Tensor')
        self.assertEqual(t1.shape, t2.shape, msg=msg)
        diff = torch.max(torch.abs(t1 - t2))
        if not (diff <= epsilon):
            standardMsg = '%s is not close to %s. diff=%s' % (t1, t2, diff)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertTensorNotClose(self, t1, t2, epsilon=1e-6, msg=None):
        self.assertIsInstance(t1, torch.Tensor,
                              'First argument is not a Tensor')
        self.assertIsInstance(t2, torch.Tensor,
                              'Second argument is not a Tensor')
        self.assertEqual(t1.shape, t2.shape, msg=msg)
        if torch.max(torch.abs(t1 - t2)) < epsilon:
            standardMsg = '%s is actually close to %s' % (t1, t2)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertArrayEqual(self, t1, t2, msg=None):
        t1 = np.array(t1)
        t2 = np.array(t2)
        self.assertEqual(t1.shape, t2.shape, msg=msg)
        if not np.all(t1 == t2):
            standardMsg = '%s != %s' % (t1, t2)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertArrayClose(self, t1, t2, epsilon=1e-6, msg=None):
        t1 = np.array(t1)
        t2 = np.array(t2)
        self.assertEqual(t1.shape, t2.shape, msg=msg)
        diff = np.max(np.abs(t1 - t2))
        if not (diff <= epsilon):
            standardMsg = '%s is not close to %s. diff=%s' % (t1, t2, diff)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertArrayNotClose(self, t1, t2, epsilon=1e-6, msg=None):
        t1 = np.array(t1)
        t2 = np.array(t2)
        self.assertEqual(t1.shape, t2.shape, msg=msg)
        if np.max(np.abs(t1 - t2)) < epsilon:
            standardMsg = '%s is actually close to %s' % (t1, t2)
            self.fail(self._formatMessage(msg, standardMsg))
