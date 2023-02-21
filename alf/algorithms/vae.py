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

from typing import Callable

import numpy as np

import torch
import torch.distributions as td
import torch.nn as nn

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.layers import FC
from alf.networks import EncodingNetwork
from alf.tensor_specs import BoundedTensorSpec
from alf.utils import math_ops, dist_utils
from alf.utils.tensor_utils import tensor_extend_new_dim
from alf.utils.schedulers import ConstantScheduler, Scheduler

VAEInfo = namedtuple(
    "VAEInfo", ["kld", "z_std", "loss", "beta_loss", 'beta'], default_value=())
VAEOutput = namedtuple("VAEOutput", ["z", "z_mode", "z_std"], default_value=())


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
                 input_tensor_spec: alf.NestedTensorSpec = None,
                 preprocess_network: EncodingNetwork = None,
                 z_prior_network: EncodingNetwork = None,
                 beta: float = 1.0,
                 target_kld_per_dim: float = None,
                 beta_optimizer: torch.optim.Optimizer = None,
                 checkpoint=None,
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
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            name (str):
        """
        super(VariationalAutoEncoder, self).__init__(
            checkpoint=checkpoint, name=name)

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
        output = VAEOutput(z=z, z_std=z_std, z_mode=z_mean)
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


@alf.configurable
class DiscreteVAE(VariationalAutoEncoder):
    r"""VAE with a discrete posterior distribution. The latent ``z`` might be
    a single categorical variable or a vector of categorials. Because the
    re-parameterization trick can no longer be applied to the discrete distribution,
    we instead use the straight-through (ST) gradient estimator to train the encoder.

    ::

        Bengio et al., "Estimating or Propagating Gradients Through Stochastic
        Neurons for Conditional Computation", 2013.

    In short, we can re-parameterize the one-hot latent embedding :math:`z` as

    .. math::

        \hat{z} = z + z_{prob} - SG(z_{prob})

    Because :math:`z` is a sampled discrete variable, it has no gradient. So
    the parameter gradient is

    .. math::

        \frac{\partial L}{\partial \hat{z}}\frac{\partial \hat{z}}{\partial \theta}
        = \frac{\partial L}{\partial \hat{z}}\frac{\partial z_{prob}}{\partial \theta}

    Alternatively, we provide the option of ST Gumbel Softmax gradient estimator.

    ::

        Jang et al., "CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX", 2017.

    Which applies the above ST trick to the Gumbel-softmax distribution that uses
    the Gumbel trick to reparameterize the categorical sampling process. The paper
    claims that ST Gumbel-softmax gradient estimator has a lower variance than the
    plain ST estimator.
    """

    def __init__(self,
                 z_spec: BoundedTensorSpec,
                 input_tensor_spec: alf.NestedTensorSpec = None,
                 z_network_cls: Callable = EncodingNetwork,
                 prior_input_tensor_spec: alf.NestedTensorSpec = None,
                 prior_z_network_cls: Callable = None,
                 mode: str = "st",
                 gumbel_temp_scheduler: Scheduler = ConstantScheduler(1.),
                 beta: float = 1.,
                 target_kld_per_categorical: float = None,
                 beta_optimizer: torch.optim.Optimizer = None,
                 name: str = "DiscreteVAE"):
        """
        Args:
            z_spec: a tensor spec for the discrete posterior. It has to be
                rank-one, representing a vector of discrete variables.
                The value bould of each variable must be identical and the lower
                bound has to be 0.
            input_tensor_spec: the input spec.
            z_network_cls: an encoding network to encode input data into a vector
                of logits. If ``prior_z_network_cls`` is None, this network must
                handle input with spec ``input_tensor_spec``. If ``prior_z_network_cls``
                is not None, this network must be handle input with spec
                ``(prior_input_tensor_spec, input_tensor_spec, prior_z_network.output_spec)``.
            prior_input_tensor_spec: the input spec for ``prior_z_network``.
            prior_z_network_cls: an encoding network that outputs a vector of logits
                representing the a prior ``z`` distribution given the prior input.
            mode: either 'st' or 'st-gumbel'.
            gumbel_temp_scheduler: the temperature scheduler for gumbel-softmax.
                Only used when ``mode=='st-gumbel'``.
            beta: the weight for KL-divergence
            target_kld_per_categorical: if not None, then this will be used as the
                target KLD *per Categorical* to automatically tune beta.
            beta_optimizer: if not None, will be used to train beta.
            name (str):
        """
        Algorithm.__init__(self, name=name)

        assert (z_spec.is_discrete and z_spec.ndim == 1
                and z_spec.minimum == 0)
        self._n_categories = int(z_spec.maximum + 1)

        prior_z_network = None
        if prior_z_network_cls is not None:
            prior_z_network = prior_z_network_cls(
                input_tensor_spec=prior_input_tensor_spec,
                last_layer_size=z_spec.numel * self._n_categories,
                last_activation=alf.math.identity)
            input_tensor_spec = (prior_input_tensor_spec, input_tensor_spec,
                                 prior_z_network.output_spec)
        self._prior_z_network = prior_z_network

        self._z_network = z_network_cls(
            input_tensor_spec=input_tensor_spec,
            last_layer_size=z_spec.numel * self._n_categories,
            last_activation=alf.math.identity)

        self._z_spec = z_spec
        assert mode in ['st', 'st-gumbel'], f"Wrong mode {mode}"
        self._mode = mode
        self._gumbel_temp_scheduler = gumbel_temp_scheduler
        self._log_beta = nn.Parameter(torch.tensor(beta).log())
        self._target_kld = None
        if target_kld_per_categorical is not None:
            self._target_kld = target_kld_per_categorical * z_spec.numel

        if beta_optimizer is not None:
            self.add_optimizer(beta_optimizer, [self._log_beta])

    @property
    def output_spec(self):
        """Because the output is a floating one-hot vector, the shape is rank-two.
        """
        return BoundedTensorSpec(
            shape=self._z_spec.shape + (self._n_categories, ),
            minimum=0.,
            maximum=1.,
            dtype=torch.float32)

    def _kl_divergence(self, logits1, logits2=None):
        if logits2 is None:
            logits2 = torch.zeros_like(logits1)  # assume uniform
        logits1 = torch.nn.functional.log_softmax(logits1, dim=-1)
        logits2 = torch.nn.functional.log_softmax(logits2, dim=-1)
        # The expectation is over the target distribution
        kld = torch.nn.functional.kl_div(
            input=logits2, target=logits1, reduction='none', log_target=True)
        return kld.sum(dim=(1, 2))  # [B,L,K] -> [B]

    def _sampling_forward(self, inputs):
        """Encode the data into latent space then do sampling.

        Args:
            inputs: if a prior network is provided, this is a tuple of
                ``(prior_input, new_observation)``.
        """
        logits_shape = (-1, ) + self._z_spec.shape + (self._n_categories, )

        if self._prior_z_network is not None:
            prior_input, new_obs = inputs
            prior_z_logits, _ = self._prior_z_network(prior_input)
            inputs = (prior_input, new_obs, prior_z_logits)
            prior_z_logits = prior_z_logits.reshape(logits_shape)

        z_logits, _ = self._z_network(inputs)
        z_logits = z_logits.reshape(logits_shape)

        if self._prior_z_network is not None:
            z_logits += prior_z_logits
            kl_div_loss = self._kl_divergence(z_logits, prior_z_logits)
        else:
            kl_div_loss = self._kl_divergence(z_logits)

        if self._mode == 'st':
            z_dist = dist_utils.OneHotCategoricalStraightThrough(
                logits=z_logits)
        else:
            z_dist = dist_utils.OneHotCategoricalGumbelSoftmax(
                hard_sample=True,
                tau=self._gumbel_temp_scheduler(),
                logits=z_logits)

        output = VAEOutput(z=z_dist.rsample(), z_mode=z_dist.mode)
        return output, kl_div_loss
