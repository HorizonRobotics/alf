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
import os
import numpy as np
import tempfile

import torch

import alf
import alf.algorithms.vae as vae
from alf.layers import FC
from alf.nest.utils import NestConcat
from alf.networks import EncodingNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import math_ops


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
        encoder = vae.VariationalAutoEncoder(self._latent_dim,
                                             self._input_spec)
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
                outputs = decoding_layers(alg_step.output)
                loss = torch.mean(100 * self._loss_f(batch - outputs) +
                                  alg_step.info.loss)
                loss.backward()
                optimizer.step()

        y_test = decoding_layers(encoder.train_step(x_test).output)
        reconstruction_loss = float(torch.mean(self._loss_f(x_test - y_test)))
        print("reconstruction_loss:", reconstruction_loss)
        self.assertLess(reconstruction_loss, 0.05)

    def test_conditional_vae(self):
        """Test for one dimensional Gaussion, conditioned on a Bernoulli variable.
        """
        prior_input_spec = BoundedTensorSpec((), 'int64')

        z_mean_prior_network = EncodingNetwork(
            TensorSpec(
                (prior_input_spec.maximum - prior_input_spec.minimum + 1, )),
            fc_layer_params=(10, ) * 2,
            last_layer_size=self._latent_dim,
            last_activation=math_ops.identity)
        z_log_var_prior_network = z_mean_prior_network.copy()
        preprocess_network = EncodingNetwork(
            input_tensor_spec=[
                z_mean_prior_network.input_tensor_spec, self._input_spec,
                z_mean_prior_network.output_spec,
                z_log_var_prior_network.output_spec
            ],
            preprocessing_combiner=NestConcat(),
            fc_layer_params=(10, ) * 2,
            last_layer_size=self._latent_dim,
            last_activation=math_ops.identity)

        encoder = vae.VariationalAutoEncoder(
            self._latent_dim, None, preprocess_network, z_mean_prior_network,
            z_log_var_prior_network)
        decoding_layers = FC(self._latent_dim, 1)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoding_layers.parameters()),
            lr=0.1)

        x_train = self._input_spec.randn(outer_dims=(10000, ))
        y_train = x_train.clone()
        y_train[:5000] = y_train[:5000] + 1.0
        pr_train = torch.cat([
            prior_input_spec.zeros(outer_dims=(5000, )),
            prior_input_spec.ones(outer_dims=(5000, ))
        ],
                             dim=0)

        x_test = self._input_spec.randn(outer_dims=(100, ))
        y_test = x_test.clone()
        y_test[:50] = y_test[:50] + 1.0
        pr_test = torch.cat([
            prior_input_spec.zeros(outer_dims=(50, )),
            prior_input_spec.ones(outer_dims=(50, ))
        ],
                            dim=0)
        pr_test = torch.nn.functional.one_hot(
            pr_test, int(z_mean_prior_network.input_tensor_spec.shape[0])).to(
                torch.float32)

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
                    int(z_mean_prior_network.input_tensor_spec.shape[0])).to(
                        torch.float32)
                alg_step = encoder.train_step([pr_batch, batch])
                outputs = decoding_layers(alg_step.output)
                loss = torch.mean(100 * self._loss_f(y_batch - outputs) +
                                  alg_step.info.loss)
                loss.backward()
                optimizer.step()

        y_hat_test = decoding_layers(
            encoder.train_step([pr_test, x_test]).output)
        reconstruction_loss = float(
            torch.mean(self._loss_f(y_test - y_hat_test)))
        print("reconstruction_loss:", reconstruction_loss)
        self.assertLess(reconstruction_loss, 0.05)


if __name__ == '__main__':
    alf.test.main()
