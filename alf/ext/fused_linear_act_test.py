# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from absl.testing import parameterized
import torch
import torch.nn.functional as F
import alf

from alf.ext import fused_linear_act, relu_backward


class FusedLinearActTest(alf.test.TestCase, parameterized.TestCase):

    def _do_one_test_fused_linear_act(self, m, n, k, transa, transb, act,
                                      dtype):
        # Test the fused linear activation function
        if transa:
            A = torch.randn(k, m, device='cuda', dtype=dtype)
            A1 = A.clone().T
            A2 = A.clone().T
            A1.requires_grad = True
            A2.requires_grad = True
        else:
            A = torch.randn(10, m, k, device='cuda', dtype=dtype)
            A1 = A.clone()
            A2 = A.clone()
            A1.requires_grad = True
            A2.requires_grad = True
        if transb:
            B = torch.randn(k, n, device='cuda', dtype=dtype)
            B1 = B.clone().T
            B2 = B.clone().T
            B1.requires_grad = True
            B2.requires_grad = True
        else:
            B = torch.randn(n, k, device='cuda', dtype=dtype)
            B1 = B.clone()
            B2 = B.clone()
            B1.requires_grad = True
            B2.requires_grad = True
        bias = torch.randn(n, device='cuda', dtype=dtype)
        bias1 = bias.clone()
        bias2 = bias1.clone()
        bias1.requires_grad = True
        bias2.requires_grad = True

        C1 = fused_linear_act(A1, B1, bias1, act)

        C2 = F.linear(A2, B2, bias2)
        if act == "RELU":
            C2 = F.relu(C2)

        # Check output shape
        self.assertEqual(C1.shape, C2.shape)
        # Check output values
        print(
            f"m={m} n={n} k={k} transa={transa} transb={transb} act={act} dtype={dtype}"
        )
        print("C max_diff", torch.max(torch.abs(C1 - C2)).item())
        atol = 1e-5 if dtype == torch.float32 else 2e-2
        self.assertTrue(torch.allclose(C1, C2, atol=atol))

        C1.sum().backward()
        C2.sum().backward()
        # Check gradient input shape
        self.assertEqual(A1.grad.shape, A2.shape)
        self.assertEqual(B1.grad.shape, B2.shape)
        self.assertEqual(bias1.grad.shape, bias2.shape)
        # Check gradient input values
        print("A grad max_diff",
              torch.max(torch.abs(A1.grad - A2.grad)).item())
        print("B grad max_diff",
              torch.max(torch.abs(B1.grad - B2.grad)).item())
        print("bias grad max_diff",
              torch.max(torch.abs(bias1.grad - bias2.grad)).item())
        self.assertTrue(torch.allclose(A1.grad, A2.grad, atol=atol))
        self.assertTrue(torch.allclose(B1.grad, B2.grad, atol=atol))
        self.assertTrue(torch.allclose(bias1.grad, bias2.grad, atol=atol))

    def test_fused_linear_act(self):
        for dtype in [torch.float16, torch.float32]:
            for m, n, k in [(4, 8, 16), (8, 4, 16)]:
                for transa in [False, True]:
                    for transb in [False, True]:
                        for act in ["RELU", "NONE"]:
                            self._do_one_test_fused_linear_act(
                                m, n, k, transa, transb, act, dtype)

    @parameterized.parameters(
        (torch.float16, ),
        (torch.float32, ),
    )
    def test_relu_backward(self, dtype):
        # Test the ReLU backward function
        x = torch.randn(4, 8, device='cuda', dtype=dtype)
        grad_output = torch.randn_like(x)
        grad_input_torch = grad_output * (x > 0).float()
        grad_input = relu_backward(x, grad_output)
        # Check gradient input shape
        self.assertEqual(grad_input.shape, (4, 8))
        # Check gradient input values
        self.assertTrue((grad_input == grad_input_torch).all())

    def benchmark_fused_linear_act(self):
        B, m, n, k = 64000, 8192, 8192, 8192
        act = "RELU"
        dtype = torch.float16

        A = torch.randn(m, k, device='cuda', dtype=dtype)
        B = torch.randn(n,
                        k,
                        device='cuda',
                        dtype=torch.float32,
                        requires_grad=True)
        bias = torch.randn(n,
                           device='cuda',
                           dtype=torch.float32,
                           requires_grad=True)

        def fused_linear_act_func():
            with torch.autocast('cuda', dtype=torch.float16):
                C = fused_linear_act(A, B, bias, act)
                C.sum().backward()
            return C

        def linear_act_func():
            with torch.autocast('cuda', dtype=torch.float16):
                C = F.linear(A, B, bias)
                if act == "RELU":
                    C = F.relu_(C)
                C.sum().backward()
            return C

        self.benchmark("fused_linear_act", fused_linear_act_func)
        self.benchmark("linear_act", linear_act_func)

    def benchmark(self, name, f):
        # Warm up
        for _ in range(10):
            f()
        torch.cuda.synchronize()
        # Benchmark
        num_iterations = 100
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        for _ in range(num_iterations):
            f()
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        print(f"{name} Elapsed time: {elapsed_time / num_iterations:.2f} ms")


if __name__ == '__main__':
    alf.test.main()
