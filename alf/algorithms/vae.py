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
from alf.networks import EncodingNetwork, OnehotCategoricalProjectionNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import math_ops, dist_utils
from alf.utils.tensor_utils import tensor_extend_new_dim

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
    we instead use the straight-through gradient estimator to train the encoder.

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

    For implementation, we directly use ``OnehotCategoricalProjectionNetwork`` to
    model the posterior distribution.
    """

    def __init__(self,
                 z_spec: BoundedTensorSpec,
                 input_tensor_spec: alf.NestedTensorSpec = None,
                 encoder_cls: Callable = EncodingNetwork,
                 prior_input_tensor_spec: alf.NestedTensorSpec = None,
                 prior_encoder_cls: Callable = None,
                 beta: float = 1.,
                 target_kld_per_categorical: float = None,
                 beta_optimizer: torch.optim.Optimizer = None,
                 name: str = "DiscreteVAE"):
        """
        Args:
            z_spec: a tensor spec for the discrete posterior. Regardless of its
                rank, it will be converted to rank-1, representing a vector of
                discrete variables. The value bould of each variable must be
                identical.
            input_tensor_spec: the input spec.
            encoder_cls: an encoding network to
                preprocess input data before projecting it into a discrete
                distribution. If ``prior_encoder_cls`` is None, this network must
                handle input with spec ``input_tensor_spec``. If ``prior_encoder_cls``
                is not None, this network must be handle input with spec
                ``(prior_input_tensor_spec, input_tensor_spec, prior_encoder.output_spec)``.
            prior_input_tensor_spec: the input spec for the prior encoder.
            prior_encoder_cls: an encoding network that outputs an embedding to
                be projected into a prior ``z`` distribution given the prior input.
            beta: the weight for KL-divergence
            target_kld_per_categorical: if not None, then this will be used as the
                target KLD per Categorical to automatically tune beta.
            beta_optimizer: if not None, will be used to train beta.
            name (str):
        """
        Algorithm.__init__(self, name=name)

        assert z_spec.is_discrete
        self._n_categories = int(z_spec.maximum - z_spec.minimum + 1)

        # convert z_spec to rank-1
        z_spec = BoundedTensorSpec(
            shape=(z_spec.numel, ),
            minimum=0,
            maximum=self._n_categories - 1,
            dtype=z_spec.dtype)

        prior_z_network = None
        if prior_encoder_cls is not None:
            prior_encoder = prior_encoder_cls(
                input_tensor_spec=prior_input_tensor_spec)
            prior_proj_net = OnehotCategoricalProjectionNetwork(
                input_size=prior_encoder.output_spec.numel, action_spec=z_spec)
            prior_z_network = alf.nn.Sequential(prior_encoder, prior_proj_net)
            logits_spec = TensorSpec((np.prod(prior_proj_net._output_shape), ))
            input_tensor_spec = (prior_input_tensor_spec, input_tensor_spec,
                                 logits_spec)

        self._prior_z_network = prior_z_network

        encoder = encoder_cls(input_tensor_spec=input_tensor_spec)
        proj_net = OnehotCategoricalProjectionNetwork(
            input_size=encoder.output_spec.numel, action_spec=z_spec)
        self._z_network = alf.nn.Sequential(encoder, proj_net)

        self._z_spec = z_spec

        self._log_beta = nn.Parameter(torch.tensor(beta).log())
        self._target_kld = None
        if target_kld_per_categorical is not None:
            self._target_kld = target_kld_per_categorical * z_spec.numel

        if beta_optimizer is not None:
            self.add_optimizer(beta_optimizer, [self._log_beta])

    @property
    def output_spec(self):
        """Because the output is a floating one-hot vector, the shape is rank-2.
        """
        return BoundedTensorSpec(
            shape=self._z_spec.shape + (self._n_categories, ),
            minimum=0.,
            maximum=1.,
            dtype=torch.float32)

    def _sampling_forward(self, inputs):
        """Encode the data into latent space then do sampling.

        Args:
            inputs (nested Tensor): if a prior network is provided, this is a
                tuple of ``(prior_input, new_observation)``.
        """
        if self._prior_z_network is not None:
            prior_input, new_obs = inputs
            prior_z_dist, _ = self._prior_z_network(prior_input)
            # probably should detach ``prior_z_logits``??
            prior_z_logits = dist_utils.distributions_to_params(
                prior_z_dist)['logits']
            inputs = (prior_input, new_obs,
                      prior_z_logits.reshape(prior_z_logits.shape[0], -1))

        z_dist, _ = self._z_network(inputs)
        if self._prior_z_network is not None:
            # This will sum over kld of each Categorical pair
            kl_div_loss = td.kl.kl_divergence(z_dist, prior_z_dist)
            z_logits = prior_z_logits + dist_utils.distributions_to_params(
                z_dist)['logits']
            z_dist_spec = dist_utils.extract_spec(z_dist)
            z_dist = dist_utils.params_to_distributions({
                'logits': z_logits
            }, z_dist_spec)
        else:
            uniform_prob = 1. / self._n_categories
            entropy = dist_utils.compute_entropy(z_dist)
            kl_div_loss = -np.log(uniform_prob) * self._z_spec.numel - entropy

        # sample z with straight-through enabled
        z = dist_utils.rsample_action_distribution(z_dist)
        z_mode = dist_utils.get_mode(z_dist).to(z)  # to float onehot
        output = VAEOutput(z=z, z_mode=z_mode)
        return output, kl_div_loss
