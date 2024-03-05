# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import time
import torch

import alf
from alf.networks.s5 import s5
from alf.networks.s5.utils import diag_ssm_forward_slow, diag_ssm_forward_triton


class S5SSMTest(parameterized.TestCase, alf.test.TestCase):
    def test_as_real_to_complex_matrix(self):
        complex = torch.view_as_complex(torch.randn(3, 4, 2))
        real = s5._as_real_to_complex_matrix(complex)
        x = torch.randn(4)
        y1 = real @ x
        y2 = s5._view_as_real(complex @ x.to(complex.dtype))
        self.assertTensorClose(y1, y2)

    def test_as_complex_to_real_matrix(self):
        complex = torch.view_as_complex(torch.randn(3, 4, 2))
        real = s5._as_complex_to_real_matrix(complex)
        x = torch.view_as_complex(torch.randn(4, 2))
        y1 = real @ s5._view_as_real(x)
        y2 = (complex @ x).real
        self.assertTensorClose(y1, y2)

    def test_diag_ssm_forward(self):
        batch_size = 50
        state_dim = 64
        length = 100
        x = torch.view_as_complex(
            torch.randn(length, batch_size, state_dim, 2))
        s = torch.zeros((batch_size, state_dim), dtype=x.dtype)
        Lambda = torch.view_as_complex(torch.randn(state_dim, 2))
        Lambda = Lambda / Lambda.abs()
        Lambda = Lambda * torch.rand(state_dim)

        Lambda.requires_grad_(True)
        x.requires_grad_(True)
        s.requires_grad_(True)

        mask = torch.view_as_complex(
            torch.randn((length, batch_size, state_dim, 2)))

        Lambda.grad = None
        x.grad = None
        s.grad = None
        y1 = diag_ssm_forward_slow(s, x, Lambda)
        Lambda_grad1, x_grad1, s_grad1 = torch.autograd.grad((y1 * mask).sum(),
                                                             [Lambda, x, s])

        t0 = time.time()
        for i in range(10):
            Lambda.grad = None
            x.grad = None
            s.grad = None
            y1 = diag_ssm_forward_slow(s, x, Lambda)
            Lambda_grad1, x_grad1, s_grad1 = torch.autograd.grad(
                (y1 * mask).sum(), [Lambda, x, s])

        t1 = time.time()
        print("time1:", t1 - t0)

        Lambda.grad = None
        x.grad = None
        s.grad = None
        y2 = diag_ssm_forward_triton(s, x, Lambda)
        Lambda_grad2, x_grad2, s_grad2 = torch.autograd.grad((y2 * mask).sum(),
                                                             [Lambda, x, s])

        t0 = time.time()
        for i in range(10):
            Lambda.grad = None
            x.grad = None
            s.grad = None
            y2 = diag_ssm_forward_triton(s, x, Lambda)
            Lambda_grad2, x_grad2, s_grad2 = torch.autograd.grad(
                (y2 * mask).sum(), [Lambda, x, s])

        t1 = time.time()
        print("time2:", t1 - t0)
        self.assertTensorClose(y1, y2, 2e-5)
        self.assertLess((Lambda_grad1 - Lambda_grad2).abs().max() /
                        (Lambda_grad1.abs().max() + 1e-8), 1e-5)
        self.assertTensorClose(x_grad1, x_grad2)
        self.assertTensorClose(s_grad1, s_grad2)

    def test_complex_grad(self):
        # Test complex grad is conjugated
        batch_size = 50
        state_dim = 64
        length = 1000
        x = torch.view_as_complex(
            torch.randn(length, batch_size, state_dim, 2))
        s = torch.zeros((batch_size, state_dim), dtype=x.dtype)
        Lambda = torch.view_as_complex(torch.randn(state_dim, 2))
        Lambda = Lambda / Lambda.abs()
        Lambda = Lambda * torch.rand(state_dim)

        Lambda.requires_grad_(True)
        x.requires_grad_(True)
        s.requires_grad_(True)

        mask = torch.view_as_complex(
            torch.randn((length, batch_size, state_dim, 2)))

        x_grad = torch.autograd.grad((x * mask).sum(), [x])[0]
        self.assertTensorEqual(x_grad, mask.conj())

    def test_s5ssm(self):
        data_dim = 4
        state_dim = 18
        num_blocks = 3
        ssm = s5.S5SSM(data_dim, state_dim, num_blocks)

        # Test that the parameters are initialized correctly
        self.assertEqual(ssm._B.shape, (9, 4, 2))
        self.assertEqual(ssm._Lambda.shape, (9, 2))
        self.assertEqual(ssm._C.shape, (4, 18))
        self.assertEqual(ssm._D.shape, (4, ))
        self.assertEqual(ssm._log_step.shape, (9, ))
        self.assertEqual(ssm._step_rescale, 1.0)
        self.assertEqual(ssm._dt_min, 0.001)
        self.assertEqual(ssm._dt_max, 0.1)
        self.assertEqual(ssm._num_blocks, 3)

        # Test step by step forward is same forward_sequence
        batch_size = 5
        seq_len = 10
        input = torch.randn(seq_len, batch_size, data_dim)
        state = torch.randn(batch_size, state_dim)
        output1, state1 = ssm.forward_sequence(input, state)
        state2 = state
        output2 = []
        for i in range(seq_len):
            output, state2 = ssm(input[i], state2)
            output2.append(output)
        output2 = torch.stack(output2)
        self.assertTensorClose(output1, output2, epsilon=1e-5)
        self.assertTensorClose(state1, state2, epsilon=1e-5)


if __name__ == '__main__':
    alf.test.main()
