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
import torch.nn as nn
import alf


class BatchNormTest(alf.test.TestCase):
    def test_torch_batch_norm(self):
        # Verify that BN also correctly calculate the gradient for its input
        # when using running stats for normalization
        bn = nn.BatchNorm1d(100)
        for i in range(100):
            x = 0.5 * torch.randn((40, 100))
            y = bn(x)

        x1 = torch.randn((40, 100), requires_grad=True)
        bn.eval()
        y1 = bn(x1)
        grad1 = torch.autograd.grad(y1.sum(), x1, retain_graph=True)[0]

        y2 = (x1 - bn.running_mean) * (bn.running_var + bn.eps).rsqrt()
        grad2 = torch.autograd.grad(y2.sum(), x1, retain_graph=True)[0]

        self.assertTensorClose(y1, y2)
        self.assertTensorClose(grad1, grad2)

    def test_batch_norm(self):
        n = 1000000
        dim = 4
        bn = alf.layers.Sequential(alf.layers.BatchNorm1d(dim))
        alf.layers.prepare_rnn_batch_norm(bn)
        bn.set_batch_norm_max_steps(2)
        for i in range(50):
            bn.set_batch_norm_current_step(0)
            x = torch.randn((n, dim))
            y = bn(0.5 * x)
            bn.set_batch_norm_current_step(1)
            y = bn(1 + x)
            # step out of limit should not affect running stats
            bn.set_batch_norm_current_step(2)
            y = bn(10 + 10 * x)

        bn.eval()
        bn.set_batch_norm_current_step(
            torch.tensor([0] * n + [1] * n + [2] * n))
        for i in range(2):
            # multiple evals should not change the statistics
            x = torch.randn((n, dim))
            x = torch.cat([0.5 * x, 1 + 2 * x, 10 + 10 * x], dim=0)
            y = bn(x)
            y0 = y[:n]
            y1 = y[n:2 * n]
            y2 = y[2 * n:]
            self.assertLess(y0.mean(dim=0).abs().max(), 0.02)
            self.assertLess((y0.var(dim=0) - 1).max(), 0.02)
            self.assertLess(y1.mean(dim=0).abs().max(), 0.02)
            self.assertLess((y1.var(dim=0) - 4).max(), 0.04)
            # step out of limit will use the running stats for max_steps - 1
            self.assertLess((y2.mean(dim=0) - 9).abs().max(), 0.1)
            self.assertLess((y2.var(dim=0) - 100).max(), 1.0)


if __name__ == "__main__":
    alf.test.main()
