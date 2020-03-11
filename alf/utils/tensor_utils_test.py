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

import torch

import alf
from alf.utils import tensor_utils


class NormClippingTest(alf.test.TestCase):
    def test_clip_by_norm(self):
        tensor = torch.ones([5])
        tensor_utils.clip_by_norm(tensor, clip_norm=1.0)
        self.assertTensorClose(torch.norm(tensor), torch.as_tensor(1.0))

    def test_clip_by_norms(self):
        tensors = [torch.randn([3, 4, 5]) for _ in range(10)]
        tensor_utils.clip_by_norms(tensors, clip_norm=1.0)
        for t in tensors:
            self.assertTensorClose(
                torch.norm(torch.reshape(t, [-1])), torch.as_tensor(1.0))

    def test_clip_by_global_norm(self):
        tensors = [torch.randn([3, 4, 5]) for _ in range(10)]
        tensor_utils.clip_by_global_norm(tensors, clip_norm=1.0)
        self.assertTensorClose(
            sum([torch.norm(torch.reshape(t, [-1]))**2 for t in tensors]),
            torch.as_tensor(1.0))


if __name__ == "__main__":
    alf.test.main()
