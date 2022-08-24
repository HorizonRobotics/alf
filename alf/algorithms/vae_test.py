# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
from absl.testing import parameterized
import os
import numpy as np
import tempfile
from functools import partial

import torch

import alf
import alf.algorithms.vae as vae
from alf.layers import FC
from alf.nest.utils import NestConcat
from alf.networks import EncodingNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import math_ops


def _make_cond_vae_dataset(train_size, input_spec, prior_input_spec):
    x_train = input_spec.randn(outer_dims=(train_size, ))
    y_train = x_train.clone()
    y_train[:train_size // 2] = y_train[:train_size // 2] + 1.0
    pr_train = torch.cat([
        prior_input_spec.zeros(outer_dims=(train_size // 2, )),
        prior_input_spec.ones(outer_dims=(train_size // 2, ))
    ],
                         dim=0)

    x_test = input_spec.randn(outer_dims=(100, ))
    y_test = x_test.clone()
    y_test[:50] = y_test[:50] + 1.0
    pr_test = torch.cat([
        prior_input_spec.zeros(outer_dims=(50, )),
        prior_input_spec.ones(outer_dims=(50, ))
    ],
                        dim=0)
    pr_test = torch.nn.functional.one_hot(pr_test, 2).to(torch.float32)
    return x_train, y_train, pr_train, x_test, y_test, pr_test


class VaeTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        self._input_spec = TensorSpec((1, ))
        self._epochs = 10
        self._batch_size = 100
        self._latent_dim = 2
        self._loss_f = math_ops.square

    def test_vae(self):
        """Test for one dimensional Gaussion."""
        encoder = vae.VariationalAutoEncoder(
            self._latent_dim, input_tensor_spec=self._input_spec)
        decoding_layers = FC(self._latent_dim, 1)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoding_layers.parameters()),
            lr=0.1)

        x_train = self._input_spec.randn(outer_dims=(10000, ))
        x_test = self._input_spec.randn(outer_dims=(10, ))

        for _ in range(self._epochs):
            x_train = x_train[torch.randperm(x_train.shape[0])]
            for i in range(0, x_train.shape[0], self._batch_size):
                optimizer.zero_grad()
                batch = x_train[i:i + self._batch_size]
                alg_step = encoder.train_step(batch)
                outputs = decoding_layers(alg_step.output.z)
                loss = torch.mean(100 * self._loss_f(batch - outputs) +
                                  alg_step.info.loss)
                loss.backward()
                optimizer.step()

        y_test = decoding_layers(encoder.train_step(x_test).output.z)
        reconstruction_loss = float(torch.mean(self._loss_f(x_test - y_test)))
        print("reconstruction_loss:", reconstruction_loss)
        self.assertLess(reconstruction_loss, 0.05)

    def test_conditional_vae(self):
        """Test for one dimensional Gaussion, conditioned on a Bernoulli variable.
        """
        prior_input_spec = BoundedTensorSpec((), 'int64')

        z_prior_network = EncodingNetwork(
            TensorSpec(
                (prior_input_spec.maximum - prior_input_spec.minimum + 1, )),
            fc_layer_params=(10, ) * 2,
            last_layer_size=2 * self._latent_dim,
            last_activation=math_ops.identity)
        preprocess_network = EncodingNetwork(
            input_tensor_spec=(
                z_prior_network.input_tensor_spec,
                self._input_spec,
                z_prior_network.output_spec,
            ),
            preprocessing_combiner=NestConcat(),
            fc_layer_params=(10, ) * 2,
            last_layer_size=self._latent_dim,
            last_activation=math_ops.identity)

        encoder = vae.VariationalAutoEncoder(
            self._latent_dim,
            preprocess_network=preprocess_network,
            z_prior_network=z_prior_network)
        decoding_layers = FC(self._latent_dim, 1)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoding_layers.parameters()),
            lr=0.1)

        (x_train, y_train, pr_train,
         x_test, y_test, pr_test) = _make_cond_vae_dataset(
             10000, self._input_spec, prior_input_spec)

        for _ in range(self._epochs):
            idx = torch.randperm(x_train.shape[0])
            x_train = x_train[idx]
            y_train = y_train[idx]
            pr_train = pr_train[idx]
            for i in range(0, x_train.shape[0], self._batch_size):
                optimizer.zero_grad()
                batch = x_train[i:i + self._batch_size]
                y_batch = y_train[i:i + self._batch_size]
                pr_batch = torch.nn.functional.one_hot(
                    pr_train[i:i + self._batch_size],
                    int(z_prior_network.input_tensor_spec.shape[0])).to(
                        torch.float32)
                alg_step = encoder.train_step([pr_batch, batch])
                outputs = decoding_layers(alg_step.output.z)
                loss = torch.mean(100 * self._loss_f(y_batch - outputs) +
                                  alg_step.info.loss)
                loss.backward()
                optimizer.step()

        y_hat_test = decoding_layers(
            encoder.train_step([pr_test, x_test]).output.z)
        reconstruction_loss = float(
            torch.mean(self._loss_f(y_test - y_hat_test)))
        print("reconstruction_loss:", reconstruction_loss)
        self.assertLess(reconstruction_loss, 0.05)


class DiscreteVAETest(parameterized.TestCase, alf.test.TestCase):
    def setUp(self):
        super().setUp()
        self._input_spec = TensorSpec((1, ))
        self._epochs = 10
        self._batch_size = 200
        self._loss_f = math_ops.square
        self._encoder_cls = partial(
            alf.networks.EncodingNetwork,
            preprocessing_combiner=NestConcat(),
            activation=torch.tanh,
            fc_layer_params=(256, ) * 3)
        self._decoder_cls = partial(
            alf.networks.EncodingNetwork,
            fc_layer_params=(256, ) * 3,
            activation=torch.tanh,
            last_layer_size=1,
            last_activation=alf.math.identity)

    @parameterized.parameters(
        dict(z_shape=(20, ), n_categories=2, mode='st'),
        dict(z_shape=(10, ), n_categories=4, mode='st'),
        dict(z_shape=(8, ), n_categories=20, mode='st'),
        dict(z_shape=(10, ), n_categories=3, mode='st'),
        dict(z_shape=(10, ), n_categories=3, mode='st-gumbel'),
    )
    def test_discrete_vae(self, z_shape, n_categories, mode):
        """Test for multiple categoricals."""
        z_spec = BoundedTensorSpec(
            shape=z_shape,
            minimum=0,
            maximum=n_categories - 1,
            dtype=torch.int64)
        encoder = vae.DiscreteVAE(
            z_spec=z_spec,
            beta=0.001,
            mode=mode,
            input_tensor_spec=self._input_spec,
            z_network_cls=self._encoder_cls)

        self.assertEqual(encoder.output_spec.shape,
                         (z_spec.numel, ) + (n_categories, ))

        decoder = self._decoder_cls(
            input_tensor_spec=TensorSpec((encoder.output_spec.numel, )))

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

        x_train = self._input_spec.randn(outer_dims=(40000, ))
        x_test = self._input_spec.randn(outer_dims=(100, ))

        for _ in range(self._epochs):
            x_train = x_train[torch.randperm(x_train.shape[0])]
            rec_loss = []
            for i in range(0, x_train.shape[0], self._batch_size):
                optimizer.zero_grad()
                batch = x_train[i:i + self._batch_size]
                alg_step = encoder.train_step(batch)
                z = alg_step.output.z
                z = z.reshape(z.shape[0], -1)
                outputs = decoder(z)[0]
                l = torch.mean(self._loss_f(batch - outputs))
                loss = l + torch.mean(alg_step.info.loss)
                loss.backward()
                optimizer.step()
                rec_loss.append(l)
            print("training rec loss: ", sum(rec_loss) / len(rec_loss))

        z = encoder.train_step(x_test).output.z
        z = z.reshape(z.shape[0], -1)
        y_test = decoder(z)[0]
        reconstruction_loss = float(torch.mean(self._loss_f(x_test - y_test)))
        print("reconstruction_loss:", reconstruction_loss)
        self.assertLess(reconstruction_loss, 0.05)

    @parameterized.parameters(dict(mode='st'), dict(mode='st-gumbel'))
    def test_cond_discrete_vae(self, mode):
        """The input has a shift of 1. depending on the Bernoulli variable.
        """
        prior_input_spec = BoundedTensorSpec((), 'int64')

        z_spec = BoundedTensorSpec(
            shape=(20, ), minimum=0, maximum=1, dtype=torch.int64)
        encoder = vae.DiscreteVAE(
            z_spec=z_spec,
            beta=0.0001,
            input_tensor_spec=self._input_spec,
            mode=mode,
            prior_z_network_cls=self._encoder_cls,
            prior_input_tensor_spec=TensorSpec((2, )),
            z_network_cls=self._encoder_cls)
        decoder = self._decoder_cls(
            input_tensor_spec=TensorSpec((encoder.output_spec.numel, )))

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

        (x_train, y_train, pr_train,
         x_test, y_test, pr_test) = _make_cond_vae_dataset(
             40000, self._input_spec, prior_input_spec)

        for _ in range(self._epochs * 2):
            idx = torch.randperm(x_train.shape[0])
            x_train = x_train[idx]
            y_train = y_train[idx]
            pr_train = pr_train[idx]
            rec_loss = []
            for i in range(0, x_train.shape[0], self._batch_size):
                optimizer.zero_grad()
                batch = x_train[i:i + self._batch_size]
                y_batch = y_train[i:i + self._batch_size]
                pr_batch = torch.nn.functional.one_hot(
                    pr_train[i:i + self._batch_size], 2).to(torch.float32)
                alg_step = encoder.train_step([pr_batch, batch])
                z = alg_step.output.z
                z = z.reshape(z.shape[0], -1)
                outputs = decoder(z)[0]
                l = torch.mean(self._loss_f(y_batch - outputs))
                loss = l + torch.mean(alg_step.info.loss)
                loss.backward()
                optimizer.step()
                rec_loss.append(l)
            print("training rec loss: ", sum(rec_loss) / len(rec_loss))

        z = encoder.train_step([pr_test, x_test]).output.z
        z = z.reshape(z.shape[0], -1)
        y_hat_test = decoder(z)[0]
        reconstruction_loss = float(
            torch.mean(self._loss_f(y_test - y_hat_test)))
        print("reconstruction_loss:", reconstruction_loss)
        self.assertLess(reconstruction_loss, 0.05)


if __name__ == '__main__':
    alf.test.main()
