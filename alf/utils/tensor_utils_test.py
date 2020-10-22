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

import torch

import alf
from alf.utils import tensor_utils


class NormClippingTest(alf.test.TestCase):
    def _sqr_norm(self, tensors):
        tensors = alf.nest.flatten(tensors)
        return sum([torch.norm(torch.reshape(t, [-1]))**2 for t in tensors])

    def test_clip_by_norms(self):
        tensor = torch.ones([5])
        clipped_tensor = tensor_utils.clip_by_norms(
            tensor, clip_norm=1.0, in_place=True)
        self.assertTensorClose(
            self._sqr_norm(clipped_tensor), torch.as_tensor(1.0))
        self.assertTensorClose(self._sqr_norm(tensor), torch.as_tensor(1.0))

        tensors = [torch.randn([3, 4, 5]) for _ in range(10)]
        tensor_utils.clip_by_norms(tensors, clip_norm=1.0, in_place=True)
        for t in tensors:
            self.assertTensorClose(self._sqr_norm(t), torch.as_tensor(1.0))

    def test_no_clip_by_norms(self):
        tensor = torch.ones([5])
        tensor_utils.clip_by_norms(tensor, clip_norm=100.0, in_place=True)
        self.assertTensorNotClose(
            self._sqr_norm(tensor), torch.as_tensor(100.0))

    def test_clip_by_global_norm(self):
        tensors = [torch.randn([3, 4, 5]) for _ in range(10)]
        sqr_norm = self._sqr_norm(tensors)
        clipped_tensors, _ = tensor_utils.clip_by_global_norm(
            tensors, clip_norm=1.0, in_place=False)
        self.assertTensorNotClose(
            self._sqr_norm(tensors), torch.as_tensor(1.0))
        self.assertTensorClose(self._sqr_norm(tensors), sqr_norm)
        self.assertTensorClose(
            self._sqr_norm(clipped_tensors), torch.as_tensor(1.0))

    def test_clip_by_global_norm_in_place(self):
        tensors = [torch.randn([3, 4, 5]) for _ in range(10)]
        tensor_utils.clip_by_global_norm(tensors, clip_norm=1.0, in_place=True)
        self.assertTensorClose(
            sum([torch.norm(torch.reshape(t, [-1]))**2 for t in tensors]),
            torch.as_tensor(1.0))


if __name__ == "__main__":
    alf.test.main()
