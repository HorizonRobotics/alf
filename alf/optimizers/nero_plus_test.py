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

from absl.testing import parameterized
from absl import logging
import torch
import torch.nn.functional as F

import alf

from alf.optimizers import NeroPlus
from alf.utils.datagen import load_mnist


class NeroPlusTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(
        dict(betas=(0, 0.999), eps=1e-30, normalizing_grad_by_norm=True),
        dict(betas=(0.9, 0.999), eps=1e-7, normalizing_grad_by_norm=False))
    def test_nero_plus(self, betas, eps, normalizing_grad_by_norm):
        train_set, test_set = load_mnist(train_bs=256, test_bs=256)
        num_classes = len(train_set.dataset.classes)
        model = alf.layers.Sequential(
            alf.layers.Conv2D(1, 32, 3, strides=2, padding=1),
            alf.layers.Conv2D(32, 32, 3, strides=2, padding=1),
            alf.layers.Conv2D(32, 32, 3, strides=2, padding=1),
            alf.layers.Reshape(-1),
            alf.layers.FC(
                4 * 4 * 32,
                num_classes,
                weight_opt_args=dict(
                    fixed_norm=False,
                    l2_regularization=1e-3,
                    zero_mean=True,
                    max_norm=float('inf'))))
        NeroPlus.initialize(model)
        opt = NeroPlus(
            lr=0.01,
            betas=betas,
            eps=eps,
            normalizing_grad_by_norm=normalizing_grad_by_norm)
        opt.add_param_group(dict(params=list(model.parameters())))

        for epoch in range(5):
            for data, target in train_set:
                logits = model(data)
                loss = F.cross_entropy(logits, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            correct = 0
            total = 0
            for data, target in test_set:
                logits = model(data)
                correct += (logits.argmax(dim=1) == target).sum()
                total += target.numel()
            logging.info("epoch=%s loss=%s acc=%s" % (epoch, loss.item(),
                                                      correct.item()))
        self.assertGreater(correct / total, 0.97)


if __name__ == '__main__':
    alf.test.main()
