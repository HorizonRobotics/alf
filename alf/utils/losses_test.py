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

from functools import partial
from absl import logging
from absl.testing import parameterized
import torch
import time

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


class BipartiteMatchingLossTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(('mean', ), ('sum', ), ('none', ))
    def test_loss_shape(self, reduction):
        prediction = torch.rand([2, 5, 4])
        target = torch.rand([2, 5, 4])

        matcher = losses.BipartiteMatchingLoss(reduction=reduction)
        cost_mat = torch.cdist(prediction, target, p=1)
        loss, _ = matcher(cost_mat)
        if reduction == 'none':
            self.assertEqual(loss.shape, (2, 5))
        else:
            self.assertEqual(loss.shape, (2, ))

    def test_forward_loss(self):
        prediction = torch.tensor(
            [[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]],
            dtype=torch.float32,
            requires_grad=True)
        target = torch.tensor([[[0.9, 0.9, 0.9], [0, 0, 0.1]],
                               [[0.1, 0.1, 0.1], [0.5, 0.6, 0.5]]])
        matcher = losses.BipartiteMatchingLoss(reduction='none')
        cost_mat = torch.cdist(prediction, target, p=1)
        loss, _ = matcher(cost_mat)
        self.assertTrue(loss.requires_grad)
        self.assertTensorClose(loss, torch.tensor([[0.1, 0.3], [0.3, 1.4]]))

    def test_loss_training(self):
        """A simple toy training task where each target is a set of unordered
        1d vectors whose means are ``torch.arange(N) + input``, where ``input``
        follows a standard Gaussian. The task is for the model to predict this
        unordered set given the ``input``.
        """
        samples_n = 20200
        N = 5
        mean = torch.arange(
            N, dtype=torch.float32).unsqueeze(0).expand(samples_n,
                                                        -1)  # [samples_n, N]
        std = torch.ones_like(mean) * 0.01
        target = torch.normal(mean, std).unsqueeze(-1)  # [samples_n, N, 1]

        idx = torch.argsort(torch.randn(samples_n, N), dim=1)
        # randomly shuffle the objects in the target set
        target = torch.gather(target, dim=1, index=idx.unsqueeze(-1))

        inputs = torch.randn(samples_n, 1).unsqueeze(-1)  # [samples_n, 1, 1]
        # offset the target objects by the inputs, to make the target input-dependent
        target = target + inputs

        d_model = 64
        transform_layers = []
        for i in range(3):
            transform_layers.append(
                alf.layers.TransformerBlock(
                    d_model=d_model,
                    num_heads=3,
                    memory_size=N + 1,
                    positional_encoding='abs'))
        model = torch.nn.Sequential(
            alf.layers.FC(1, d_model, torch.relu_),
            alf.layers.FC(d_model, d_model, torch.relu_), *transform_layers,
            alf.layers.FC(d_model, 1))

        # We prepend the input to some random noise vectors for the transformer
        # We expect the transformer converts the noise vectors to correct predictions
        # Note: noise is important. Constant vectors are hard to train.
        # [samples_n, N, 1]
        inputs = torch.cat([inputs, torch.randn((samples_n, N, 1))], dim=1)
        val_n = 200
        tr_inputs = inputs[:-val_n, ...]
        val_inputs = inputs[-val_n:, ...]
        tr_target = target[:-val_n, ...]
        val_target = target[-val_n:, ...]

        t0 = time.time()

        optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
        epochs = 10
        batch_size = 100
        matcher = losses.BipartiteMatchingLoss(reduction='mean')
        for _ in range(epochs):
            idx = torch.randperm(tr_inputs.shape[0])
            tr_inputs = tr_inputs[idx]
            tr_target = tr_target[idx]
            l = []
            for i in range(0, idx.shape[0], batch_size):
                optimizer.zero_grad()
                b_inputs = tr_inputs[i:i + batch_size]
                b_target = tr_target[i:i + batch_size]
                b_pred = model(b_inputs)  # [b,N+1,1]
                b_pred = b_pred[:, 1:, :]
                cost_mat = torch.cdist(b_pred, b_target, p=2.)
                loss = matcher(cost_mat)[0].mean()
                loss.backward()
                optimizer.step()
                l.append(loss)
            print("Training loss: ", sum(l) / len(l))

        print("Training time: ", time.time() - t0)

        val_pred = model(val_inputs)
        val_pred = val_pred[:, 1:, :]
        cost_mat = torch.cdist(val_pred, val_target, p=2.)
        val_loss = matcher(cost_mat)[0]

        print("Validation prediction - inputs")
        print(torch.round(val_pred[:3] - val_inputs[:3, :1, :], decimals=2))

        print("Validation loss: ", val_loss.mean())
        self.assertLess(float(val_loss.mean()), 0.15)

    def test_loss_training_discrete_target(self):
        """A simple toy task for testing bipartite matching loss on discrete
        target variables.

        The inputs are sampled from some fixed random Gaussians, each Gaussian
        representing a different object class. The target values are Gaussian ids,
        but shuffled in a random order for each sample.
        """
        samples_n = 10200
        M, N, D = 3, 5, 10
        mean = torch.randn((N, D)).unsqueeze(0)  # [1,N,D]
        std = torch.ones_like(mean) * 0.5  # [1,N,D]

        inputs = torch.normal(
            mean.repeat(samples_n, 1, 1), std.repeat(samples_n, 1,
                                                     1))  # [samples_n,N,D]
        target = torch.arange(N).unsqueeze(0).repeat(samples_n,
                                                     1)  # [samples_n,N]

        # randomly shuffle
        idx = torch.argsort(torch.randn(samples_n, N), dim=1)
        inputs = torch.gather(
            inputs, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, D))
        target = torch.gather(target, dim=1, index=idx)
        # Only take the first M objects for each sample
        inputs = inputs[:, :M, :]
        target = target[:, :M]
        # shuffle the target again ...
        idx = torch.argsort(torch.randn(samples_n, M), dim=1)
        target = torch.gather(target, dim=1, index=idx)

        d_model = 64
        transform_layers = []
        for i in range(3):
            transform_layers.append(
                alf.layers.TransformerBlock(
                    d_model=d_model,
                    num_heads=3,
                    memory_size=M * 2,  # input + queries
                    positional_encoding='abs' if i == 0 else 'none'))
        model = torch.nn.Sequential(*transform_layers, alf.layers.FC(
            d_model, N))
        input_fc = alf.layers.FC(D, d_model)

        queries = torch.nn.Parameter(torch.Tensor(M, d_model))
        torch.nn.init.normal_(queries)

        val_n = 200
        tr_inputs = inputs[:-val_n, ...]
        val_inputs = inputs[-val_n:, ...]
        tr_target = target[:-val_n, ...]
        val_target = target[-val_n:, ...]

        def _compute_cost_mat(p, t):
            p = torch.nn.functional.log_softmax(p, dim=-1)
            oh_t = torch.nn.functional.one_hot(
                t, num_classes=N).to(torch.float32)
            return -torch.einsum('bnk,bmk->bnm', p, oh_t)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(input_fc.parameters()) + [queries],
            lr=1e-3)
        epochs = 5
        batch_size = 100
        matcher = losses.BipartiteMatchingLoss(reduction='mean')
        for _ in range(epochs):
            idx = torch.randperm(tr_inputs.shape[0])
            tr_inputs = tr_inputs[idx]
            tr_target = tr_target[idx]
            l = []
            for i in range(0, idx.shape[0], batch_size):
                optimizer.zero_grad()
                b_inputs = tr_inputs[i:i + batch_size]
                b_target = tr_target[i:i + batch_size]
                b_inputs = input_fc(b_inputs)
                b_queries = queries.unsqueeze(0).expand(
                    b_inputs.shape[0], -1, -1)
                b_inputs = torch.cat([b_inputs, b_queries], dim=1)  # [b,2M,..]
                b_pred = model(b_inputs)  # [b,2M,N]
                b_pred = b_pred[:, M:, :]
                cost_mat = _compute_cost_mat(b_pred, b_target)
                loss = matcher(cost_mat)[0].mean()
                loss.backward()
                optimizer.step()
                l.append(loss)
            print("Training loss: ", sum(l) / len(l))

        val_fc_out = input_fc(val_inputs)
        val_queries = queries.unsqueeze(0).expand(val_fc_out.shape[0], -1, -1)
        val_fc_out = torch.cat([val_fc_out, val_queries], dim=1)  # [b,2M,..]
        val_pred = model(val_fc_out)
        val_pred = val_pred[:, M:, :]
        cost_mat = _compute_cost_mat(val_pred, val_target)
        val_loss = matcher(cost_mat)[0]

        print("Cluster mean")
        print(mean)

        print("Validation inputs")
        print(val_inputs[:5])

        print("Validation prediction")
        print(torch.argmax(val_pred, dim=-1)[:5])

        print("Validation target")
        print(val_target[:5])

        print("Validation loss: ", val_loss.mean())
        self.assertLess(float(val_loss.mean()), 0.01)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    alf.test.main()
