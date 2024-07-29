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
"""Vector Quantized Variational AutoEncoder Algorithm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.networks import EncodingNetwork

VqvaeLossInfo = namedtuple(
    "VqvaeLossInfo", ["quantization", "commitment", "reconstruction"],
    default_value=())


class Vqvae(Algorithm):
    r"""Vector Quantized Variational AutoEncoder (VQVAE) algorithm, described in:

    ::
        A van den Oord et al. "Neural Discrete Representation Learning", NeurIPS 2017.

    VQVAE is different from standard VAE mainly in the follows aspects:

    1. Discrete latent is used, instead of continuous latent as in standard VAE.
    2. Standard VAE uses Gaussian prior and posterior. VQVAE can be viewed as
       using a deterministic form of posterior, which is a categorical
       distribution with onehot samples computed by nearest neighbor matching
       (Eq.1 of the paper). By using a uniform prior, the KL divergence is constant.


    """

    def __init__(self,
                 input_tensor_spec: alf.NestedTensorSpec,
                 num_embeddings: int,
                 embedding_dim: int,
                 encoder_ctor: Callable = EncodingNetwork,
                 decoder_ctor: Callable = EncodingNetwork,
                 optimizer: torch.optim.Optimizer = None,
                 commitment_loss_weight: float = 1.0,
                 checkpoint=None,
                 debug_summaries: bool = False,
                 name: str = "Vqvae"):
        """
        Args:
            input_tensor_spec (TensorSpec): the tensor spec of
                the input.
            num_embeddings (int): the number of embeddings (size of codebook)
            embedding_dim (int): the dimensionality of embedding vectors
            encoder_ctor (Callable): called as ``encoder_ctor(observation_spec)``
                to construct the encoding ``Network``. The network takes raw observation
                as input and output the latent representation.
            decoder_ctor (Callable): called as ``decoder_ctor(latent_spec)`` to
                construct the decoder.
            optimizer (Optimizer|None): if provided, it will be used to optimize
                the parameter of encoder_net, decoder_net and embedding vectors.
            commitment_loss_weight (float): the weight for commitment loss.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
        """
        super().__init__(
            checkpoint=checkpoint, debug_summaries=debug_summaries, name=name)

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # [n, d]
        self._embedding = torch.nn.Parameter(
            torch.FloatTensor(self._num_embeddings, self._embedding_dim))

        torch.nn.init.uniform_(
            self._embedding,
            a=-1 / self._num_embeddings,
            b=1 / self._num_embeddings)

        self._encoding_net = encoder_ctor(input_tensor_spec)

        self._decoding_net = decoder_ctor(self._encoding_net.output_spec)

        if optimizer is not None:
            self.add_optimizer(
                optimizer,
                [self._encoding_net, self._decoding_net, self._embedding])
        self._optimizer = optimizer

        self._commitment_loss_weight = commitment_loss_weight

    def _predict_step(self, inputs, state=()):
        """
        Args:
            inputs (tensor): with the shape the same as input_tensor_spec
        """
        # [B, d]
        input_embedding, _ = self._encoding_net(inputs)

        # calculate distances
        # [B, 1] + [n] + [B, n]
        distances = (torch.sum(input_embedding**2, dim=1, keepdim=True) +
                     torch.sum(self._embedding**2, dim=1) -
                     2 * torch.matmul(input_embedding, self._embedding.t()))

        encoding_indices = torch.argmin(distances, dim=1)

        quantized = self._embedding[encoding_indices]

        # straight through
        quantized_st = input_embedding + (quantized - input_embedding).detach()
        return input_embedding, quantized, quantized_st

    def predict_step(self, inputs, state=()):
        _, _, quantized_st = self._predict_step(inputs)
        rec = self._decoding_net(quantized_st)[0]
        return AlgStep(output=rec, state=state, info=quantized_st)

    def train_step(self, inputs, state=()):
        """
        Args:
            inputs (tensor): with the shape the same as input_tensor_spec
        """

        input_embedding, quantized, quantized_st = self._predict_step(inputs)

        e_latent_loss = F.mse_loss(
            quantized.detach(), input_embedding, reduction="none")
        q_latent_loss = F.mse_loss(
            quantized, input_embedding.detach(), reduction="none")

        # encoding loss
        enc_loss = (q_latent_loss +
                    self._commitment_loss_weight * e_latent_loss).mean(dim=1)

        # decoding loss
        rec = self._decoding_net(quantized_st)[0]

        recon_loss = F.mse_loss(rec, inputs, reduction="none").mean(dim=1)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.embedding("vq_embedding", self._embedding.detach())

        loss = (enc_loss + recon_loss)
        info = VqvaeLossInfo(
            quantization=q_latent_loss.mean(1),
            commitment=e_latent_loss.mean(1),
            reconstruction=recon_loss)
        loss_info = LossInfo(loss=loss, extra=info)
        return AlgStep(output=rec, state=state, info=loss_info)
