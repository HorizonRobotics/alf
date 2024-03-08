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

import copy
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

    def test_lean_function_module(self):
        func1 = alf.layers.FC(3, 5, activation=torch.relu_)
        func2 = copy.deepcopy(func1)
        x = torch.randn((4, 3), requires_grad=True)
        func2 = lean_function(func2)
        y1 = func1(x)
        y2 = func2(x)
        self.assertTensorEqual(y1, y2)

        grad1 = torch.autograd.grad(y1.sum(), x)[0]
        grad2 = torch.autograd.grad(y2.sum(), x)[0]
        self.assertTensorEqual(grad1, grad2)

        y1 = func1(x)
        y2 = func2(x)
        y1.sum().backward()
        y2.sum().backward()
        for p1, p2 in zip(func1.parameters(), func2.parameters()):
            self.assertTensorEqual(p1.grad, p2.grad)

    def test_lean_function_network(self):
        func1 = alf.nn.Sequential(
            alf.layers.FC(3, 5, activation=torch.relu_),
            alf.layers.FC(5, 1, activation=torch.sigmoid))
        func2 = func1.copy()
        for p1, p2 in zip(func1.parameters(), func2.parameters()):
            p2.data.copy_(p1)
        x = torch.randn((4, 3), requires_grad=True)
        func2 = lean_function(func2)
        y1 = func1(x)[0]
        y2 = func2(x)[0]
        self.assertTensorEqual(y1, y2)

        grad1 = torch.autograd.grad(y1.sum(), x)[0]
        grad2 = torch.autograd.grad(y2.sum(), x)[0]
        self.assertTensorEqual(grad1, grad2)

        y1 = func1(x)[0]
        y2 = func2(x)[0]
        y1.sum().backward()
        y2.sum().backward()
        for p1, p2 in zip(func1.parameters(), func2.parameters()):
            self.assertTensorEqual(p1.grad, p2.grad)

    def test_lean_fucntion_autocast(self):
        func1 = alf.nn.Sequential(
            alf.layers.FC(3, 5, activation=torch.relu_),
            alf.layers.FC(5, 1, activation=torch.sigmoid))
        func2 = func1.copy()
        for p1, p2 in zip(func1.parameters(), func2.parameters()):
            p2.data.copy_(p1)
        x = torch.randn((4, 3), requires_grad=True)
        func2 = lean_function(func2)
        with torch.cuda.amp.autocast(enabled=True):
            y1 = func1(x)[0]
            y2 = func2(x)[0]
        self.assertTensorEqual(y1, y2)

        grad1 = torch.autograd.grad(y1.sum(), x)[0]
        grad2 = torch.autograd.grad(y2.sum(), x)[0]
        self.assertTensorEqual(grad1, grad2)

        y1 = func1(x)[0]
        y2 = func2(x)[0]
        y1.sum().backward()
        y2.sum().backward()
        for p1, p2 in zip(func1.parameters(), func2.parameters()):
            self.assertTensorEqual(p1.grad, p2.grad)

    def test_lean_fucntion_deepcopy(self):
        func1 = alf.layers.FC(3, 5, activation=torch.relu_)
        func2 = alf.layers.FC(3, 5, activation=torch.relu_)
        lean_f1 = lean_function(func1)
        lean_f2 = copy.deepcopy(lean_f1)
        lean_f2.reset_parameters()
        func2.weight.data.copy_(lean_f2.weight)
        func2.bias.data.copy_(lean_f2.bias)
        x = torch.randn((4, 3), requires_grad=True)
        lean_y1 = lean_f1(x)
        lean_y2 = lean_f2(x)
        y1 = func1(x)
        y2 = func2(x)

        # Test that the copied one is different from the original one
        self.assertTensorNotClose(lean_y1, lean_y2)

        self.assertTensorEqual(y1, lean_y1)
        self.assertTensorEqual(y2, lean_y2)
        lean_y2.sum().backward()
        y2.sum().backward()
        for p1, p2 in zip(lean_f2.parameters(), func2.parameters()):
            self.assertTensorEqual(p1.grad, p2.grad)


if __name__ == '__main__':
    alf.test.main()
