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
"""Variational auto encoder."""

import torch
import torch.nn as nn

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.layers import FC
from alf.networks import EncodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops
from alf.utils.tensor_utils import tensor_extend_new_dim

VAEInfo = namedtuple(
    "VAEInfo", ["kld", "z_std", "loss", "beta_loss", 'beta'], default_value=())
VAEOutput = namedtuple("VAEOutput", ["z", "z_mean", "z_std"])


@alf.configurable
class VariationalAutoEncoder(Algorithm):
    r"""VariationalAutoEncoder encodes data into diagonal multivariate gaussian,
    performs sampling with reparametrization trick, and returns KL divergence
    between posterior and prior.

    Mathematically:

    :math:`\log p(x) >= E_z \log P(x|z) - \beta KL(q(z|x) || prior(z))`

    ``train_step()`` method returns sampled z and KLD, it is up to the user of
    this class to use the returned z to decode and compute reconstructive loss to
    combine with kl loss returned here to optimize the whole network.

    See vae_test.py for example usages to train vanilla vae, conditional vae and
    vae with prior network on mnist dataset.
    """

    def __init__(self,
                 z_dim: int,
                 input_tensor_spec: alf.nest.NestedTensorSpec = None,
                 preprocess_network: EncodingNetwork = None,
                 z_prior_network: EncodingNetwork = None,
                 beta: float = 1.0,
                 target_kld_per_dim: float = None,
                 beta_optimizer: torch.optim.Optimizer = None,
                 name: str = "VariationalAutoEncoder"):
        """

        Args:
            z_dim: dimension of latent vector ``z``, namely, the dimension
                for generating ``z_mean`` and ``z_log_var``.
            input_tensor_spec: the input spec which can be
                a nest. If `preprocess_network` is None, then it must be provided.
            preprocess_network: an encoding network to
                preprocess input data before projecting it into (mean, log_var).
                If ``z_prior_network`` is None, this network must be handle input
                with spec ``input_tensor_spec``. If ``z_prior_network`` is not
                None, this network must be handle input with spec
                ``(z_prior_network.input_tensor_spec, input_tensor_spec, z_prior_network.output_spec)``.
                If this is None, an MLP of hidden sizes ``(z_dim*2, z_dim*2)``
                will be used.
            z_prior_network: an encoding network that
                outputs concatenation of a prior mean and prior log var given
                the prior input. The network shouldn't activate its output.
            beta: the weight for KL-divergence
            target_kld_per_dim: if not None, then this will be used as the
                target KLD per dim to automatically tune beta.
            beta_optimizer: if not None, will be used to train beta.
            name (str):
        """
        super(VariationalAutoEncoder, self).__init__(name=name)

        self._preprocess_network = preprocess_network
        if preprocess_network is None:
            # according to appendix 2.4-2.5 in paper: https://arxiv.org/pdf/1803.10760.pdf
            if z_prior_network is None:
                preproc_input_spec = input_tensor_spec
            else:
                preproc_input_spec = (z_prior_network.input_tensor_spec,
                                      input_tensor_spec,
                                      z_prior_network.output_spec)
            self._preprocess_network = EncodingNetwork(
                input_tensor_spec=preproc_input_spec,
                preprocessing_combiner=alf.nest.utils.NestConcat(),
                fc_layer_params=(2 * z_dim, 2 * z_dim),
                activation=torch.tanh,
            )
        self._z_prior_network = z_prior_network

        size = self._preprocess_network.output_spec.shape[0]
        self._z_mean = FC(input_size=size, output_size=z_dim)
        self._z_log_var = FC(input_size=size, output_size=z_dim)
        self._log_beta = nn.Parameter(torch.tensor(beta).log())
        self._target_kld = None
        if target_kld_per_dim is not None:
            self._target_kld = target_kld_per_dim * z_dim
        self._z_dim = z_dim

        if beta_optimizer is not None:
            self.add_optimizer(beta_optimizer, [self._log_beta])

    def _sampling_forward(self, inputs):
        """Encode the data into latent space then do sampling.

        Args:
            inputs (nested Tensor): if a prior network is provided, this is a
                tuple of ``(prior_input, new_observation)``.

        Returns:
            tuple:
            - z (Tensor): ``z`` is a tensor of shape (``B``, ``z_dim``).
            - kl_loss (Tensor): ``kl_loss`` is a tensor of shape (``B``,).
        """
        if self._z_prior_network:
            prior_input, new_obs = inputs
            prior_z_mean_and_log_var, _ = self._z_prior_network(prior_input)
            prior_z_mean = prior_z_mean_and_log_var[..., :self._z_dim]
            prior_z_log_var = prior_z_mean_and_log_var[..., self._z_dim:]
            inputs = (prior_input, new_obs, prior_z_mean_and_log_var)

        latents, _ = self._preprocess_network(inputs)
        z_mean = self._z_mean(latents)
        z_log_var = self._z_log_var(latents)

        if self._z_prior_network:
            kl_div_loss = math_ops.square(z_mean) / torch.exp(prior_z_log_var) + \
                          torch.exp(z_log_var) - z_log_var - 1.0
            z_mean = z_mean + prior_z_mean
            z_log_var = z_log_var + prior_z_log_var
        else:
            kl_div_loss = math_ops.square(z_mean) + torch.exp(
                z_log_var) - 1.0 - z_log_var

        kl_div_loss = 0.5 * torch.sum(kl_div_loss, dim=-1)
        # reparameterization sampling: z = u + var ** 0.5 * eps
        eps = torch.randn(z_mean.shape)
        z_std = torch.exp(z_log_var * 0.5)
        z = z_mean + z_std * eps
        output = VAEOutput(z=z, z_std=z_std, z_mean=z_mean)
        return output, kl_div_loss

    def train_step(self, inputs, state=()):
        """
        Args:
            inputs (nested Tensor): data to be encoded. If there is a prior
                network, then ``inputs`` is a tuple of ``(prior_input, new_obs)``.
            state (Tensor): empty tuple ()

        Returns:
            AlgStep:
            - output (VAEOutput):
            - state: empty tuple ()
            - info (VAEInfo):
        """
        output, kld_loss = self._sampling_forward(inputs)
        beta = self._log_beta.exp().detach()
        info = VAEInfo(loss=beta * kld_loss, kld=kld_loss, z_std=output.z_std)
        if self._target_kld is not None:
            beta_loss = self._beta_train_step(kld_loss)
            info = info._replace(
                beta_loss=beta_loss,
                loss=info.loss + beta_loss,
                beta=tensor_extend_new_dim(beta, 0, beta_loss.shape[0]))
        return AlgStep(output=output, state=state, info=info)

    def _beta_train_step(self, kld_loss):
        beta_loss = self._log_beta * (self._target_kld - kld_loss).detach()
        return beta_loss
