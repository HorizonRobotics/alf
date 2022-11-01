# Copyright (c) 2022 Horizon Robotics. All Rights Reserved.
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
from functools import partial
import torch

import alf
from alf.algorithms.vq_vae import Vqvae
from alf.layers import FC
from alf.nest.utils import NestConcat
from alf.networks import EncodingNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import math_ops


class VQVaeTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        self._input_spec = TensorSpec((1, ))
        self._epochs = 10
        self._batch_size = 100
        self._num_embeddings = 32
        self._embedding_dim = 10
        self._loss_f = math_ops.square
        self._learning_rate = 1e-3
        self._commitment_loss_weight = 0.1

    def test_vq_vae(self):
        """Test for one dimensional signal."""

        fc_layers_params = (256, ) * 2
        encoder_cls = partial(
            alf.networks.EncodingNetwork,
            fc_layer_params=fc_layers_params,
            last_layer_size=self._embedding_dim,
            last_activation=math_ops.identity,
            last_kernel_initializer=partial(torch.nn.init.uniform_, \
                                    a=-0.03, b=0.03)
        )
        decoder_cls = partial(
            alf.networks.EncodingNetwork,
            fc_layer_params=fc_layers_params,
            last_layer_size=1,
            last_activation=math_ops.identity,
            last_kernel_initializer=partial(torch.nn.init.uniform_, \
                                    a=-0.03, b=0.03)
        )

        optimizer = alf.optimizers.Adam(lr=self._learning_rate)

        vq_vae = Vqvae(
            input_tensor_spec=self._input_spec,
            num_embeddings=self._num_embeddings,
            embedding_dim=self._embedding_dim,
            encoder_ctor=encoder_cls,
            decoder_ctor=decoder_cls,
            optimizer=optimizer,
            commitment_loss_weight=self._commitment_loss_weight)

        # construct 1d samples around two centers with additive noise
        num_centers = 2
        x_train = 1e-1 * self._input_spec.randn(outer_dims=(10000, ))
        x_test = 1e-1 * self._input_spec.randn(outer_dims=(10, ))

        x_train = x_train.view(-1, num_centers) + torch.arange(0, num_centers)
        x_test = x_test.view(-1, num_centers) + torch.arange(0, num_centers)

        x_train = x_train.view(-1, 1)
        x_test = x_test.view(-1, 1)

        for _ in range(self._epochs):
            x_train = x_train[torch.randperm(x_train.shape[0])]
            for i in range(0, x_train.shape[0], self._batch_size):
                batch = x_train[i:i + self._batch_size]
                alg_step = vq_vae.train_step(batch)
                vq_vae.update_with_gradient(alg_step.info)

        alg_step = vq_vae.predict_step(x_test)

        reconstruction_loss = float(
            torch.mean(self._loss_f(x_test - alg_step.output)))
        print("reconstruction_loss:", reconstruction_loss)
        self.assertLess(reconstruction_loss, 0.05)


if __name__ == '__main__':
    alf.test.main()
