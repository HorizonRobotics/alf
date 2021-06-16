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

import torch

import alf
from alf.utils.lean_function import lean_function


def func(x, w, b, scale=1.0):
    return torch.sigmoid(scale * (x @ w) + b)


class TestLeanFunction(alf.test.TestCase):
    def test_lean_function(self):
        x = torch.randn((3, 4), requires_grad=True)
        w = torch.randn((4, 5), requires_grad=True)
        b = torch.randn(5, requires_grad=True)
        lean_func = lean_function(func)
        y1 = func(x, w, b)
        y2 = lean_func(x, w, b)
        self.assertTensorEqual(y1, y2)
        grad1 = torch.autograd.grad(y1.sum(), x)[0]
        grad2 = torch.autograd.grad(y2.sum(), x)[0]
        self.assertTensorEqual(grad1, grad2)

        y3 = lean_func(x, b=b, w=w)
        self.assertTensorEqual(y1, y3)
        grad3 = torch.autograd.grad(y3.sum(), x)[0]
        self.assertTensorEqual(grad1, grad3)

        y1 = func(x, w, b, scale=2.0)
        y2 = lean_func(x, w=w, b=b, scale=2.0)
        self.assertTensorEqual(y1, y2)
        grad1 = torch.autograd.grad(y1.sum(), x)[0]
        grad2 = torch.autograd.grad(y2.sum(), x)[0]
        self.assertTensorEqual(grad1, grad2)


if __name__ == '__main__':
    alf.test.main()
