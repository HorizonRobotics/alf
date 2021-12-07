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

from functools import partial
import torch
import torch.nn.functional as F
import alf

from alf.initializers import _is_elementwise_op


class InitializerTest(alf.test.TestCase):
    def test_is_elementwise_op(self):
        self.assertFalse(_is_elementwise_op(torch.nn.Softmax(dim=-1)))
        self.assertTrue(_is_elementwise_op(torch.relu))
        self.assertTrue(_is_elementwise_op(torch.relu_))
        self.assertTrue(_is_elementwise_op(torch.sigmoid))
        self.assertFalse(
            _is_elementwise_op(lambda x: x * (x**2).sum().rsqrt()))


if __name__ == "__main__":
    alf.test.main()
