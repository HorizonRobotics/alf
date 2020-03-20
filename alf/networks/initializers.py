# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import gin
import math
from scipy.stats import truncnorm

import torch
import torch.nn as nn

import alf.utils.math_ops as math_ops


@gin.configurable
def _numerical_calculate_gain(nonlinearity, dz=0.01, r=5.0):
    """Compute the gain in a numerical way by integration. Assume y is the output,
    w is the weight (mean=0, std=1), and x is the input, then

    Var(y) = Var(w) * E(x^2)

    So we need to approximate E(x^2) numerically.

    Args:
        nonlinearity (Callable): any callable activation function
        dz (float): `dz` in the integration
        r (float): `z` range will be `[-r, r]`

    Returns:
        gain (float): a gain factor that will be applied to the init weights
    """
    dist = torch.distributions.normal.Normal(0, 1)
    z = torch.arange(-r, r, dz)
    x = nonlinearity(z)
    Ex2 = (torch.exp(dist.log_prob(z)) * x**2).sum() * dz
    return torch.sqrt(1.0 / Ex2).cpu().numpy()


def _calculate_gain(nonlinearity, nonlinearity_param=0.01):
    """Deprecated: now use _numerical_calculate_gain instead.

    Args:
        nonlinearity (str): the name of the activation function
        nonlinearity_param (float): additional parameter of the nonlinearity;
            currently only used by 'leaky_relu' as the negative slope (pytorch
            default 0.01)
    """
    if nonlinearity == "elu":
        # ELU paper: "The weights have been initialized according to (He et al.,
        # 2015)". Also there is another suggestion for math.sqrt(1.55) in:
        # https://stats.stackexchange.com/questions/229885/whats-the-recommended-weight-initialization-strategy-when-using-the-elu-activat
        return math.sqrt(1.55)
    elif nonlinearity == "sigmoid":
        # pytorch's init.calculate_gain has 1.0 for sigmoid, which is obviously
        # wrong!
        return math.sqrt(3.41)
    else:
        return nn.init.calculate_gain(nonlinearity, nonlinearity_param)


@gin.configurable
def variance_scaling_init(tensor,
                          gain=1.0,
                          mode="fan_in",
                          distribution="truncated_normal",
                          calc_gain_after_activation=True,
                          nonlinearity=math_ops.identity,
                          transposed=False):
    """Implements TensorFlow's `VarianceScaling` initializer.
    https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/ops/init_ops.py#L437

    A potential benefit of this intializer is that we can sample from a truncated
    normal distribution: `scipy.stats.truncnorm(a=-2, b=2, loc=0., scale=1.)`

    Also incorporates PyTorch's calculation of the recommended gains that taking
    nonlinear activations into account, so that after N layers, the final output
    std (in linear space) will be a constant regardless of N's value (when N is
    large). This auto gain probably won't make much of a difference if the
    network is shallow, as in most RL cases.

    Example usage:
        from alf.networks.initializers import variance_scaling_init
        layer = nn.Linear(2, 2)
        variance_scaling_init(layer.weight.data,
                              nonlinearity=nn.functional.leaky_relu)
        nn.init.zeros_(layer.bias.data)

    Args:
        tensor (torch.Tensor): the weights to be initialized
        gain (float): a positive scaling factor for weight std. Different from
            tf's implementation, this number is applied outside of `math.sqrt`.
            Note that if `calc_gain_after_activation=True`, this number will be
            an additional gain factor on top of that.
        mode (str): one of "fan_in", "fan_out", and "fan_avg"
        distribution (str): one of "uniform", "untruncated_normal" and
            "truncated_normal". If the latter, the weights will be sampled
            from a normal distribution truncated at (-2, 2).
        calc_gain_after_activation (bool): whether automatically calculate the
            std gain of applying nonlinearity after this layer. A nonlinear
            activation (e.g., relu) might change std after the transformation,
            so we need to compensate for that. Only used when mode=="fan_in".
        nonlinearity (Callable): any callable activation function
        transposed (bool): a flag indicating if the weight tensor has been
            tranposed (e.g., nn.ConvTranspose2d). In that case, `fan_in` and
            `fan_out` should be swapped.

    Returns:
        tensor (torch.Tensor): a randomly initialized weight tensor
    """

    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if transposed:
        fan_in, fan_out = fan_out, fan_in

    assert mode in ["fan_in", "fan_out", "fan_avg"], \
        "Unrecognized mode %s!" % mode
    if mode == "fan_in":
        size = max(1.0, fan_in)
    elif mode == "fan_out":
        size = max(1.0, fan_out)
    else:
        size = max(1.0, (fan_in + fan_out) / 2.0)

    if (calc_gain_after_activation and mode == "fan_in"):
        gain *= _numerical_calculate_gain(nonlinearity)

    std = gain / math.sqrt(size)
    if distribution == "truncated_normal":
        threshold = 2.0  # truncate within 2 std
        std /= truncnorm.std(-threshold, threshold)
        with torch.no_grad():
            return tensor.copy_(
                # use as_tensor to avoid a copy (memory shared)
                torch.as_tensor(
                    truncnorm.rvs(-threshold, threshold, size=tensor.size()) *
                    std))
    elif distribution == "uniform":
        limit = math.sqrt(3.0) * std
        with torch.no_grad():
            return tensor.uniform_(-limit, limit)
    elif distribution == "untruncated_normal":
        with torch.no_grad():
            return tensor.normal_(0, std)
    else:
        raise ValueError("Invalid `distribution` argument:", distribution)
