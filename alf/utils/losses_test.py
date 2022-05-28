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

from absl import logging
import torch

import alf
from alf.utils import losses


class LossTest(alf.test.TestCase):
    def test_discrete_regression_loss(self):
        n = 10
        batch_size = 256
        for loss_cls in [
                losses.DiscreteRegressionLoss,
                losses.OrderedDiscreteRegressionLoss,
                losses.QuantileRegressionLoss
        ]:

            for transform in [None, alf.math.Sqrt1pTransform()]:
                param = torch.nn.Parameter(torch.zeros(2 * n))
                opt = torch.optim.Adam([param], lr=0.01)
                lossf = loss_cls(transform)
                logging.info("lossf=%s" % lossf)
                lossf.initialize_bias(param.data)
                logging.info("initial bias=%s" % param)
                ex = lossf.calc_expectation(param.unsqueeze(0))
                logging.info("initial expectation=%s" % ex.item())

                if loss_cls == losses.DiscreteRegressionLoss:
                    probs = param.softmax(dim=-1)
                elif loss_cls == losses.OrderedDiscreteRegressionLoss:
                    probs = param.sigmoid()
                    probs = torch.cat(
                        [probs[..., :-1] - probs[..., 1:], probs[..., -1:]],
                        dim=-1)
                else:
                    probs = param

                logging.info("initial probs=%s" % probs)

                for _ in range(2000):
                    target = torch.rand(batch_size) * 10
                    logits = param.unsqueeze(dim=0).expand(target.shape[0], -1)
                    loss = lossf(logits, target)
                    self.assertEqual(loss.shape, target.shape)
                    loss = loss.mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                if loss_cls == losses.DiscreteRegressionLoss:
                    probs = param.softmax(dim=-1)
                elif loss_cls == losses.OrderedDiscreteRegressionLoss:
                    probs = param.sigmoid()
                    probs = torch.cat(
                        [probs[..., :-1] - probs[..., 1:], probs[..., -1:]],
                        dim=-1)
                else:
                    probs = param
                logging.info("probs=%s" % probs)
                if transform is None and loss_cls != losses.QuantileRegressionLoss:
                    self.assertAlmostEqual(probs[9], 0.05, delta=0.005)
                    self.assertAlmostEqual(probs[19], 0.05, delta=0.005)
                    self.assertTrue(((probs[10:19] - 0.1).abs() < 0.01).all())

                ex = lossf.calc_expectation(param.unsqueeze(0))
                logging.info("expectation=%s" % ex.item())
                self.assertAlmostEqual(ex.item(), 5.0, delta=0.1)

    def test_calc_bin(self):
        lossf = losses.SymmetricOrderedDiscreteRegressionLoss(
            alf.math.Sqrt1pTransform(), inverse_after_mean=False)
        target = torch.full((256, 6), 574.99996)
        bin1, bin2, w2 = lossf._calc_bin(target, 257)
        print("target", target, "bin1", bin1, "bin2", bin2, "w2", w2)
        self.assertTrue((w2 >= 0).all())
        self.assertTrue((w2 <= 1.0).all())


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    alf.test.main()
