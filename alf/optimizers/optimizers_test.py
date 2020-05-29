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

import torch

import alf
from alf.optimizers import Adam, AdamTF
from alf.utils import tensor_utils


class OptimizersTest(alf.test.TestCase):
    def test_optimizer_name(self):
        i = Adam.counter
        j = AdamTF.counter
        opt1 = Adam(lr=0.1)
        opt2 = AdamTF(lr=0.1)
        opt3 = Adam(lr=0.1)
        opt4 = AdamTF(lr=0.1, name="AdamTF")
        self.assertEqual(opt1.name, "Adam_%s" % i)
        self.assertEqual(opt2.name, "AdamTF_%s" % j)
        self.assertEqual(opt3.name, "Adam_%s" % (i + 1))
        self.assertEqual(opt4.name, "AdamTF")

    def test_gradient_clipping(self):
        layer = torch.nn.Linear(5, 3)
        x = torch.randn(2, 5)
        y = layer(x)
        loss = torch.sum(y**2)
        clip_norm = 1e-4
        opt = AdamTF(
            lr=0.1, gradient_clipping=clip_norm, clip_by_global_norm=True)
        opt.add_param_group({'params': layer.parameters()})
        opt.zero_grad()
        loss.backward()

        def _grad_norm(params):
            grads = [p.grad for p in params]
            return tensor_utils.global_norm(grads)

        params = []
        for param_group in opt.param_groups:
            params.extend(param_group["params"])
        self.assertGreater(_grad_norm(params), clip_norm)
        opt.step()
        self.assertTensorClose(_grad_norm(params), torch.as_tensor(clip_norm))


if __name__ == "__main__":
    alf.test.main()
