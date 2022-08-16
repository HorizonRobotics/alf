# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.utils.common import zero_tensor_from_nested_spec
from alf.utils.action_samplers import _CategoricalSeedSamplerBase


class ActionSamplersTest(alf.test.TestCase):
    def test_categorical_seed_sampler(self):
        n_classes = 3
        n_probs = 2
        repeat = 10000

        # Test E(\tilde{\pi}) = \pi
        l = _CategoricalSeedSamplerBase(n_classes, new_noise_prob=1)
        probs = torch.rand((n_probs, n_classes))
        probs = probs / probs.sum(dim=-1, keepdim=True)
        x = probs.unsqueeze(0).expand(repeat, n_probs, n_classes).reshape(
            -1, n_classes)
        state = zero_tensor_from_nested_spec(l.state_spec, x.shape[0])
        new_probs, state = l(x, state)
        new_probs = new_probs.reshape(repeat, n_probs, n_classes)
        mean_probs = new_probs.mean(dim=0)
        print('probs', probs)
        print('mean_probs', mean_probs)
        self.assertTensorClose(mean_probs, probs.cpu(), epsilon=0.01)
        self.assertTrue((state != 0).all())

        # Test that epsilon are regenerated with probability 0.1
        l = _CategoricalSeedSamplerBase(n_classes, new_noise_prob=0.1)
        y1, state = l(x, state)
        y2, state = l(x, state)
        y3, state = l(x, state)
        batch_size = repeat * n_probs
        diff1 = batch_size - (y1 == y2).all(dim=1).sum()
        diff2 = batch_size - (y2 == y3).all(dim=1).sum()
        print("diff1=", diff1, "diff2=", diff2)
        self.assertAlmostEqual(diff1 / batch_size, 0.1, delta=0.01)
        self.assertAlmostEqual(diff2 / batch_size, 0.1, delta=0.01)


if __name__ == '__main__':
    alf.test.main()
