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

import scipy.special
import torch
import torch.nn as nn
import torch.distributions as td

import alf
from alf.nest.utils import convert_device


def _gammaincinv(a, y):
    # Inverse to the regularized lower incomplete gamma function.
    # pytorch does not have a native implementation of gammaincinv, so we
    # have to use scipy.
    return convert_device(
        torch.as_tensor(
            scipy.special.gammaincinv(a.cpu().numpy(),
                                      y.cpu().numpy()),
            device='cpu'))


class _CategoricalSeedSamplerBase(alf.nn.Network):
    # The reason of separate _CategoricalSeedSamplerBase from CategoricalSeedSampler
    # is for easier unittest.
    def __init__(self, num_classes, new_noise_prob=0.01, concentration=1):
        input_tensor_spec = alf.TensorSpec((num_classes, ))
        super().__init__(
            input_tensor_spec=input_tensor_spec, state_spec=input_tensor_spec)
        self._concentration = concentration
        self._new_noise_prob = new_noise_prob

    def forward(self, input, state):
        """
        Args:
            input: categorical probabilities
        """
        epsilon = state
        batch_size = input.shape[0]
        new_epsilon = torch.rand_like(input)
        new_noise = torch.rand(batch_size) < self._new_noise_prob
        # The initial state is always 0. So we need to generate new noise
        # for initial state.
        new_noise = new_noise | (epsilon == 0).all(dim=1)
        new_noise = new_noise.unsqueeze(-1)
        # epsilon follows Uniform(0,1)
        epsilon = torch.where(new_noise, new_epsilon, epsilon)
        alpha = self._concentration * input
        # Use inverse transform sampling to obtain gamma samples.
        # gamma follows Gamma distribution Gamma(alpha, 1)
        gamma = _gammaincinv(alpha.clamp(min=1e-30), epsilon).clamp(min=1e-30)
        # prob follows Dirichlet distribution Dirichlet(alpha)
        # see https://en.wikipedia.org/wiki/Dirichlet_distribution#Related_distributions
        prob = gamma / gamma.sum(dim=-1, keepdim=True)
        return prob, epsilon


@alf.repr_wrapper
@alf.configurable
class CategoricalSeedSampler(_CategoricalSeedSamplerBase):
    r"""Sample actions with temporal consistency.

    In order to do so, we maintain an internal stateful noise vector :math:`\epsilon`
    and use it to modify the original categorical distribution :math:`\pi` to a new
    distribution :math:`\tilde{\pi}=f(\pi, \epsilon)`. The evolution of :math:`\epsilon`
    and :math:`f` are chosen so that :math:`E(\tilde{\pi})=\pi`. More specifically,
    :math:`f` is chosen so that :math:`\tilde{\pi}` follows Dirichlet distribution
    :math:`Dir(c \pi)`.

    Args:
        num_classes: number of classes for the categorical distribution
        new_noise_prob: the probability of generating a new :math:`\epsilon`
        concentration: the concentration scaling factor c. Larger ``concentration``
            tends to generate :math:`\tilde{\pi}` closer to :math:`\pi`.
    """

    def __init__(self,
                 num_classes: int,
                 new_noise_prob: float = 0.01,
                 concentration: float = 1):
        super().__init__(num_classes, new_noise_prob, concentration)

    def forward(self, input: torch.Tensor, state: torch.Tensor):
        """
        Args:
            input: the parameter of the categorical distribution with the shape of
                ``[batch_size, num_classes]``
            state: noise state (i.e. :math:`\epsilon`)
        """
        prob, state = super().forward(input, state)
        action_id = torch.multinomial(prob, num_samples=1).squeeze(1)
        return action_id, state


@alf.repr_wrapper
class EpsilonGreedySampler(nn.Module):
    """Epsilon greedy sampler.

    With probability ``1 - epsilon_greedy``, sample actions with the largest
    probability. With probability ``epsilon_greedy``, sample actions according
    to the given categorical distribution.

    Args:
        epsilon_greedy: see above.
    """

    def __init__(self, epsilon_greedy=0.1):
        super().__init__()
        self._epsilon_greedy = epsilon_greedy

    def forward(self, input):
        """
        Args:
            input: categorical probabilities with the shape of  ``[batch_size, num_classes]``
        """
        action_id = torch.multinomial(input, num_samples=1).squeeze(1)
        if self._epsilon_greedy < 1:
            greedy_action_id = input.argmax(dim=1)
            if self._epsilon_greedy > 0:
                r = torch.rand(action_id.shape) >= self._epsilon_greedy
                action_id[r] = greedy_action_id[r]
            else:
                action_id = greedy_action_id
        return action_id


@alf.repr_wrapper
class MultinomialSampler(nn.Module):
    """Sample actions according to the given multinomial distribution.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        """
        Args:
            input: categorical probabilities with the shape of  ``[batch_size, num_classes]``
        """
        action_id = torch.multinomial(input, num_samples=1).squeeze(1)
        return action_id
