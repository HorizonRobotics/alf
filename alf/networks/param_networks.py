# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Networks with input parameters."""

import functools
import math
import torch
import torch.nn as nn
import torch.distributions as td

import alf
from alf.initializers import variance_scaling_init
from alf.layers import ParamFC, ParamConv2D
from alf.networks.network import Network
from alf.tensor_specs import TensorSpec
from alf.utils import common, dist_utils
import alf.utils.math_ops as math_ops
from alf.utils.summary_utils import safe_mean_hist_summary, summarize_tensor_gradients


@alf.configurable
class ParamConvNet(Network):
    def __init__(self,
                 input_channels,
                 input_size,
                 conv_layer_params,
                 same_padding=False,
                 activation=torch.relu_,
                 use_bias=False,
                 use_ln=False,
                 n_groups=None,
                 kernel_initializer=None,
                 flatten_output=False,
                 name="ParamConvNet"):
        """A fully 2D conv network that does not maintain its own network parameters,
        but accepts them from users. If the given parameter tensor has an extra batch
        dimension (first dimension), it performs parallel operations.

        Args:
            input_channels (int): number of channels in the input image
            input_size (int or tuple): the input image size (height, width)
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            same_padding (bool): similar to TF's conv2d ``same`` padding mode. If
                True, the user provided paddings in `conv_layer_params` will be
                replaced by automatically calculated ones; if False, it
                corresponds to TF's ``valid`` padding mode (the user can still
                provide custom paddings though)
            activation (torch.nn.functional): activation for all the layers
            use_bias (bool): whether use bias.
            use_ln (bool): whether use layer normalization
            n_groups (int): number of parallel groups, must be specified if 
                ``use_ln``
            kernel_initializer (Callable): initializer for all the layers.
            flatten_output (bool): If False, the output will be an image
                structure of shape ``(B, n, C, H, W)``; otherwise the output
                will be flattened into a feature of shape ``(B, n, C*H*W)``.
            name (str):
        """

        input_size = common.tuplify2d(input_size)
        super().__init__(
            input_tensor_spec=TensorSpec((input_channels, ) + input_size),
            name=name)

        assert isinstance(conv_layer_params, tuple)
        assert len(conv_layer_params) > 0

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

        self._flatten_output = flatten_output
        self._conv_layer_params = conv_layer_params
        self._conv_layers = nn.ModuleList()
        self._param_length = None
        for paras in conv_layer_params:
            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            pooling_kernel = paras[4] if len(paras) > 4 else None
            if same_padding:  # overwrite paddings
                kernel_size = common.tuplify2d(kernel_size)
                padding = ((kernel_size[0] - 1) // 2,
                           (kernel_size[1] - 1) // 2)
            self._conv_layers.append(
                ParamConv2D(
                    input_channels,
                    filters,
                    kernel_size,
                    activation=activation,
                    strides=strides,
                    pooling_kernel=pooling_kernel,
                    padding=padding,
                    use_bias=use_bias,
                    use_ln=use_ln,
                    n_groups=n_groups,
                    kernel_initializer=kernel_initializer))
            input_channels = filters

    @property
    def param_length(self):
        """Get total number of parameters for all layers. """
        if self._param_length is None:
            length = 0
            for conv_l in self._conv_layers:
                length = length + conv_l.param_length
            self._param_length = length
        return self._param_length

    def set_parameters(self, theta, reinitialize=False):
        """Distribute parameters to corresponding layers.

        Args:
            theta (torch.Tensor): with shape ``[D] (groups=1)``
                                        or ``[B, D] (groups=B)``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``D``: length of parameters, should be self.param_length
                When the shape of inputs is ``[D]``, it will be unsqueezed
                to ``[1, D]``.
            reinitialize (bool): whether to reinitialize parameters of
                each layer.
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[1] == self.param_length), (
            "Input theta has wrong shape %s. Expecting shape (, %d)" %
            self.param_length)
        pos = 0
        for conv_l in self._conv_layers:
            param_length = conv_l.param_length
            conv_l.set_parameters(
                theta[:, pos:pos + param_length], reinitialize=reinitialize)
            pos = pos + param_length
        self._output_spec = None

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (Tensor):
            state: not used, just keeps the interface same with other networks.
        """
        x = inputs
        for conv_l in self._conv_layers[:-1]:
            x = conv_l(x, keep_group_dim=False)
        x = self._conv_layers[-1](x)
        if self._flatten_output:
            x = x.reshape(*x.shape[:-3], -1)
        return x, state


@alf.configurable
class ParamNetwork(Network):
    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 use_conv_bias=False,
                 use_conv_ln=False,
                 use_fc_bias=True,
                 use_fc_ln=False,
                 n_groups=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 last_layer_size=None,
                 last_activation=None,
                 last_use_bias=True,
                 last_use_ln=False,
                 name="ParamNetwork"):
        """A network with Fc and conv2D layers that does not maintain its own
        network parameters, but accepts them from users. If the given parameter
        tensor has an extra batch dimension (first dimension), it performs
        parallel operations.

        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[int]): a tuple of integers
                representing FC layer sizes.
            use_conv_bias (bool): whether use bias for conv layers. 
            use_conv_ln (bool): whether use layer normalization for conv layers.
            use_fc_bias (bool): whether use bias for fc layers.
            use_fc_ln (bool): whether use layer normalization for fc layers.
            n_groups (int): number of parallel groups, must be specified if ``use_bn``
            activation (torch.nn.functional): activation for all the layers
            kernel_initializer (Callable): initializer for all the layers.
            last_layer_size (int): an optional size of an additional layer
                appended at the very end. Note that if ``last_activation`` is
                specified, ``last_layer_size`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.
            last_use_bias (bool): whether use bias for the additional layer.
            last_use_fn (bool): whether use layer normalization for the additional layer.
            name (str):
        """

        super().__init__(input_tensor_spec=input_tensor_spec, name=name)

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

        self._param_length = None
        self._conv_net = None
        if conv_layer_params:
            assert isinstance(conv_layer_params, tuple), \
                "The input params {} should be tuple".format(conv_layer_params)
            assert input_tensor_spec.ndim == 3, \
                "The input shape {} should be like (C,H,W)!".format(
                    input_tensor_spec.shape)
            input_channels, height, width = input_tensor_spec.shape
            self._conv_net = ParamConvNet(
                input_channels, (height, width),
                conv_layer_params,
                activation=activation,
                use_bias=use_conv_bias,
                use_ln=use_conv_ln,
                n_groups=n_groups,
                kernel_initializer=kernel_initializer,
                flatten_output=True)
            input_size = self._conv_net.output_spec.shape[-1]
        else:
            assert input_tensor_spec.ndim == 1, \
                "The input shape {} should be like (N,)!".format(
                    input_tensor_spec.shape)
            input_size = input_tensor_spec.shape[0]

        self._fc_layers = nn.ModuleList()
        if fc_layer_params is None:
            fc_layer_params = []
        else:
            assert isinstance(fc_layer_params, tuple)
            fc_layer_params = list(fc_layer_params)

        for size in fc_layer_params:
            self._fc_layers.append(
                ParamFC(
                    input_size,
                    size,
                    activation=activation,
                    use_bias=use_fc_bias,
                    use_ln=use_fc_ln,
                    n_groups=n_groups,
                    kernel_initializer=kernel_initializer))
            input_size = size

        if last_layer_size is not None or last_activation is not None:
            assert last_layer_size is not None and last_activation is not None, \
            "Both last_layer_param and last_activation need to be specified!"
            self._fc_layers.append(
                ParamFC(
                    input_size,
                    last_layer_size,
                    activation=last_activation,
                    use_bias=last_use_bias,
                    use_ln=last_use_ln,
                    n_groups=n_groups,
                    kernel_initializer=kernel_initializer))
            input_size = last_layer_size

        self._output_spec = TensorSpec((input_size, ),
                                       dtype=self._input_tensor_spec.dtype)

    @property
    def param_length(self):
        """Get total number of parameters for all layers. """
        if self._param_length is None:
            length = 0
            if self._conv_net is not None:
                length += self._conv_net.param_length
            for fc_l in self._fc_layers:
                length = length + fc_l.param_length
            self._param_length = length
        return self._param_length

    def set_parameters(self, theta, reinitialize=False):
        """Distribute parameters to corresponding layers.

        Args:
            theta (torch.Tensor): with shape ``[D] (groups=1)``
                                        or ``[B, D] (groups=B)``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``D``: length of parameters, should be self.param_length
                When the shape of inputs is ``[D]``, it will be unsqueezed
                to ``[1, D]``.
            reinitialize (bool): whether to reinitialize parameters of
                each layer.
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[1] == self.param_length), (
            "Input theta has wrong shape %s. Expecting shape (, %d)" %
            self.param_length)
        if self._conv_net is not None:
            split = self._conv_net.param_length
            conv_theta = theta[:, :split]
            self._conv_net.set_parameters(
                conv_theta, reinitialize=reinitialize)
            fc_theta = theta[:, self._conv_net.param_length:]
        else:
            fc_theta = theta

        pos = 0
        for fc_l in self._fc_layers:
            param_length = fc_l.param_length
            fc_l.set_parameters(
                fc_theta[:, pos:pos + param_length], reinitialize=reinitialize)
            pos = pos + param_length

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (Tensor):
            state: not used, just keeps the interface same with other networks.
        """
        x = inputs
        if self._conv_net is not None:
            x, state = self._conv_net(x, state=state)
        for fc_l in self._fc_layers:
            x = fc_l(x)
        return x, state


@alf.configurable
class NormalProjectionParamNetwork(Network):
    def __init__(self,
                 input_size,
                 output_tensor_spec,
                 activation=math_ops.identity,
                 projection_output_init_gain=0.3,
                 std_bias_initializer_value=0.0,
                 squash_mean=False,
                 state_dependent_std=False,
                 std_transform=nn.functional.softplus,
                 scale_distribution=False,
                 dist_squashing_transform=dist_utils.StableTanh(),
                 name="NormalProjectionParamNetwork"):
        """Creates an instance of NormalProjectionParamNetwork.

        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        Args:
            input_size (int): input vector dimension
            output_tensor_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            activation (Callable): activation function to use in
                dense layers.
            projection_output_init_gain (float): Output gain for initializing
                output means and std weights.
            std_bias_initializer_value (float): Initial value for the bias of the
                ``std_projection_layer``.
            squash_mean (bool): If True, ``output_tensor_spec`` is ``action_spec``,
                squash the output mean to fit the action spec. If 
                ``scale_distribution`` is also True, this value will be ignored.
            state_dependent_std (bool): If True, std will be generated depending
                on the current state; otherwise a global std will be generated
                regardless of the current state.
            std_transform (Callable): Transform to apply to the std, on top of
                `activation`.
            scale_distribution (bool): Whether or not to scale the output
                distribution to ensure that the output tensor fits within the
                `output_tensor_spec`, used when output space is the action space.
                Note that this is different from `mean_transform` which merely 
                squashes the mean to fit within the spec.
            dist_squashing_transform (td.Transform):  A distribution Transform
                which transforms values into :math:`(-1, 1)`. 
                Default to ``dist_utils.StableTanh()``
            name (str): name of this network.
        """
        super().__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        assert isinstance(output_tensor_spec, TensorSpec)

        self._param_length = None
        self._output_size = output_tensor_spec.numel
        self._mean_transform = math_ops.identity
        self._scale_distribution = scale_distribution

        if squash_mean or scale_distribution:
            action_high = torch.as_tensor(output_tensor_spec.maximum)
            action_low = torch.as_tensor(output_tensor_spec.minimum)
            self._action_means = (action_high + action_low) / 2
            self._action_magnitudes = (action_high - action_low) / 2
            # Do not transform mean if scaling distribution
            if not scale_distribution:
                self._mean_transform = (
                    lambda inputs: self._action_means + self._action_magnitudes
                    * inputs.tanh())
            else:
                self._transforms = [
                    dist_squashing_transform,
                    dist_utils.AffineTransform(
                        loc=self._action_means, scale=self._action_magnitudes)
                ]

        self._std_transform = math_ops.identity
        if std_transform is not None:
            self._std_transform = std_transform

        self._means_projection_layer = ParamFC(
            input_size,
            self._output_size,
            activation=activation,
            kernel_init_gain=projection_output_init_gain)

        self._state_dependent_std = state_dependent_std
        if state_dependent_std:
            self._std_projection_layer = ParamFC(
                input_size,
                self._output_size,
                activation=activation,
                kernel_init_gain=projection_output_init_gain,
                bias_init_value=std_bias_initializer_value)
        else:
            self._std = output_tensor_spec.constant(std_bias_initializer_value)
            self._std_projection_layer = lambda _: self._std

    @property
    def param_length(self):
        """Get total number of parameters for all modules. """
        if self._param_length is None:
            length = self._means_projection_layer.weight_length \
                     + self._means_projection_layer.bias_length
            if self._state_dependent_std:
                length += self._std_projection_layer.weight_length \
                          + self._std_projection_layer.bias_length
            else:
                length += self._output_size
            self._param_length = length
        return self._param_length

    def set_parameters(self, theta, reinitialize=False):
        """Distribute parameters to corresponding layers.

        Args:
            theta (torch.Tensor): with shape ``[D] (groups=1)``
                                        or ``[n, D] (groups=n)``
                where the meaning of the symbols are:
                - ``n``: number of replica (groups)
                - ``D``: length of parameters, should be self.param_length
                When the shape of inputs is ``[D]``, it will be unsqueezed
                to ``[1, D]``.
            reinitialize (bool): whether to reinitialize parameters of
                each layer.
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[1] == self.param_length), (
            "Input theta has wrong shape %s. Expecting shape (, %d)" %
            self.param_length)
        split = self._means_projection_layer.param_length
        self._means_projection_layer.set_parameters(
            theta[:, :split], reinitialize=reinitialize)
        if self._state_dependent_std:
            self._std_projection_layer.set_parameters(
                theta[:, split:], reinitialize=reinitialize)
        else:
            self._std = theta[:, split:]

    def _normal_dist(self, means, stds):
        normal_dist = dist_utils.DiagMultivariateNormal(loc=means, scale=stds)
        if self._scale_distribution:
            # The transformed distribution can also do reparameterized sampling
            # i.e., `.has_rsample=True`
            # Note that in some cases kl_divergence might no longer work for this
            # distribution! Assuming the same `transforms`, below will work:
            # ````
            # kl_divergence(Independent, Independent)
            #
            # kl_divergence(TransformedDistribution(Independent, transforms),
            #               TransformedDistribution(Independent, transforms))
            # ````
            squashed_dist = td.TransformedDistribution(
                base_distribution=normal_dist, transforms=self._transforms)
            return squashed_dist
        else:
            return normal_dist

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (Tensor):
            state: not used, just keeps the interface same with other networks.
        """
        # [B, n, D] or [B, D] (n=1)
        means = self._mean_transform(self._means_projection_layer(inputs))
        stds = self._std_transform(self._std_projection_layer(inputs))
        safe_mean_hist_summary("CriticProjNet/batch_means", means.mean(1))
        safe_mean_hist_summary("CriticProjNet/batch_stds", stds.mean(1))

        # means = means.squeeze(-1)
        # stds = stds.squeeze(-1)
        def _summarize_grad(x, name):
            if not x.requires_grad:
                return x
            if alf.summary.should_record_summaries():
                return summarize_tensor_gradients(
                    "CriticProjNet/" + name, x, clone=True)
            else:
                return x

        means = _summarize_grad(means, name='means_grad')
        stds = _summarize_grad(stds, name='stds_grad')
        return self._normal_dist(means, stds), state


@alf.configurable
class StableNormalProjectionParamNetwork(NormalProjectionParamNetwork):
    def __init__(self,
                 input_size,
                 output_tensor_spec,
                 activation=math_ops.identity,
                 projection_output_init_gain=1e-5,
                 squash_mean=False,
                 state_dependent_std=False,
                 inverse_std_transform='softplus',
                 scale_distribution=False,
                 init_std=1.0,
                 min_std=0.0,
                 max_std=None,
                 dist_squashing_transform=dist_utils.StableTanh(),
                 name="StableNormalProjectionParamNetwork"):
        """Creates an instance of StableNormalProjectionParamNetwork.

        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        Args:
            input_size (int): input vector dimension
            output_tensor_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            activation (Callable): activation function to use in
                dense layers.
            projection_output_init_gain (float): Output gain for initializing
                output means and std weights.
            squash_mean (bool): If True, ``output_tensor_spec`` is ``action_spec``,
                squash the output mean to fit the action spec. If 
                ``scale_distribution`` is also True, this value will be ignored.
            state_dependent_std (bool): If True, std will be generated depending
                on the current state; otherwise a global std will be generated
                regardless of the current state.
            inverse_std_transform (str): Currently supports
                "exp" and "softplus". Transformation to obtain inverse std. The
                transformed values are further transformed according to min_std
                and max_std.
            scale_distribution (bool): Whether or not to scale the output
                distribution to ensure that the output tensor fits within the
                `output_tensor_spec`, used when output space is the action space.
                Note that this is different from `mean_transform` which merely 
                squashes the mean to fit within the spec.
            dist_squashing_transform (td.Transform):  A distribution Transform
                which transforms values into :math:`(-1, 1)`. 
                Default to ``dist_utils.StableTanh()``
            name (str): name of this network.
        """
        self._min_std = min_std
        self._max_std = max_std
        assert init_std > min_std

        c = 1 / (init_std - min_std)
        if max_std is not None:
            assert init_std < max_std
            c -= 1 / (max_std - min_std)

        if inverse_std_transform == 'exp':
            std_transform = torch.exp
            std_bias_initializer_value = math.log(c)
        elif inverse_std_transform == 'softplus':
            std_transform = nn.functional.softplus
            std_bias_initializer_value = math.log(math.exp(c) - 1)
        else:
            raise ValueError(
                "Unsupported inverse_std_transform %s" % inverse_std_transform)

        super().__init__(
            input_size=input_size,
            output_tensor_spec=output_tensor_spec,
            activation=activation,
            projection_output_init_gain=projection_output_init_gain,
            std_bias_initializer_value=std_bias_initializer_value,
            squash_mean=squash_mean,
            state_dependent_std=state_dependent_std,
            std_transform=std_transform,
            scale_distribution=scale_distribution,
            dist_squashing_transform=dist_squashing_transform,
            name=name)

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (Tensor):
            state: not used, just keeps the interface same with other networks.
        """
        # [B, n, D] or [B, D] (n=1)
        inv_stds = self._std_transform(self._std_projection_layer(inputs))
        if self._max_std is not None:
            inv_stds = inv_stds + 1 / (self._max_std - self._min_std)
        stds = 1. / inv_stds
        if self._min_std > 0:
            stds = stds + self._min_std

        means = self._mean_transform(
            self._means_projection_layer(inputs) * stds)
        # means = means.squeeze(-1)
        # stds = stds.squeeze(-1)
        safe_mean_hist_summary("CriticProjNet/batch_means", means.mean(1))
        safe_mean_hist_summary("CriticProjNet/batch_stds", stds.mean(1))

        def _summarize_grad(x, name):
            if not x.requires_grad:
                return x
            if alf.summary.should_record_summaries():
                return summarize_tensor_gradients(
                    "CriticProjNet/" + name, x, clone=True)
            else:
                return x

        means = _summarize_grad(means, name='means_grad')
        stds = _summarize_grad(stds, name='stds_grad')
        return self._normal_dist(means, stds), state


@alf.configurable
class CriticDistributionParamNetwork(Network):
    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec=TensorSpec(()),
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=(32, ),
                 action_fc_layer_params=(32, ),
                 joint_fc_layer_params=(64, ),
                 use_conv_bias=False,
                 use_conv_ln=False,
                 use_fc_bias=True,
                 use_fc_ln=False,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 deterministic=False,
                 projection_net_ctor=StableNormalProjectionParamNetwork,
                 state_dependent_std=False,
                 name="CriticDistributionParamNetwork"):
        """A network with Fc and conv2D layers that does not maintain its own
        network parameters, but accepts them from users. If the given parameter
        tensor has an extra batch dimension (first dimension), it performs
        parallel operations.

        Args:
            input_tensor_spec: A tuple of ``TensorSpec``s ``(observation_spec, action_spec)``
                representing the inputs.
            output_tensor_spec (TensorSpec): spec for the output
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where
                ``use_bias`` is optional.
            activation (torch.nn.functional): activation for all the layers
            kernel_initializer (Callable): initializer for all the layers.
            last_layer_size (int): an optional size of an additional layer
                appended at the very end. Note that if ``last_activation`` is
                specified, ``last_layer_size`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_size``. Note that if
                ``last_layer_size`` is not None, ``last_activation`` has to be
                specified explicitly.
            deterministic (bool): whether to make this network deterministic.
            projection_net_ctor (Callable): constructor that generates a 
                projection network.
            state_dependent_std (bool): If True, std will be generated depending
                on the current state; otherwise a global std will be generated
                regardless of the current state.
            name (str):
        """
        super().__init__(input_tensor_spec=input_tensor_spec, name=name)

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                gain=math.sqrt(1.0 / 3),
                mode='fan_in',
                distribution='uniform')

        observation_spec, action_spec = input_tensor_spec

        self._obs_encoder = ParamNetwork(
            observation_spec,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            use_conv_bias=use_conv_bias,
            use_conv_ln=use_conv_ln,
            use_fc_bias=use_fc_bias,
            use_fc_ln=use_fc_ln,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=self.name + ".obs+encoder")

        self._action_encoder = ParamNetwork(
            action_spec,
            fc_layer_params=action_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=self.name + ".action_encoder")

        last_kernel_initializer = functools.partial(
            torch.nn.init.uniform_, a=-0.003, b=0.003)

        if deterministic:
            self._joint_encoder = ParamNetwork(
                TensorSpec((self._obs_encoder.output_spec.shape[0] + \
                            self._action_encoder.output_spec.shape[0], )),
                fc_layer_params=joint_fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer,
                last_layer_size=output_tensor_spec.numel,
                last_activation=math_ops.identity,
                name=self.name + ".joint_encoder")
            self._projection_net = None
        else:
            self._joint_encoder = ParamNetwork(
                TensorSpec((self._obs_encoder.output_spec.shape[0] + \
                            self._action_encoder.output_spec.shape[0], )),
                fc_layer_params=joint_fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer,
                name=self.name + ".joint_encoder")

            self._projection_net = projection_net_ctor(
                input_size=self._joint_encoder.output_spec.shape[0],
                output_tensor_spec=output_tensor_spec,
                state_dependent_std=state_dependent_std)

        self._param_length = None
        # self._output_spec = output_tensor_spec

    @property
    def param_length(self):
        """Get total number of parameters for all modules. """
        if self._param_length is None:
            length = self._obs_encoder.param_length
            length += self._action_encoder.param_length
            length += self._joint_encoder.param_length
            if self._projection_net is not None:
                length += self._projection_net.param_length
            self._param_length = length
        return self._param_length

    def set_parameters(self, theta, reinitialize=False):
        """Distribute parameters to corresponding layers.

        Args:
            theta (torch.Tensor): with shape ``[D] (groups=1)``
                                        or ``[n, D] (groups=n)``
                where the meaning of the symbols are:
                - ``n``: number of replica (groups)
                - ``D``: length of parameters, should be self.param_length
                When the shape of inputs is ``[D]``, it will be unsqueezed
                to ``[1, D]``.
            reinitialize (bool): whether to reinitialize parameters of
                each layer.
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[1] == self.param_length), (
            "Input theta has wrong shape %s. Expecting shape (, %d)" %
            self.param_length)
        split = self._obs_encoder.param_length
        obs_theta = theta[:, :split]
        self._obs_encoder.set_parameters(obs_theta, reinitialize=reinitialize)
        action_theta = theta[:, split:split +
                             self._action_encoder.param_length]
        self._action_encoder.set_parameters(
            action_theta, reinitialize=reinitialize)
        split = split + self._action_encoder.param_length
        joint_theta = theta[:, split:split + self._joint_encoder.param_length]
        self._joint_encoder.set_parameters(
            joint_theta, reinitialize=reinitialize)
        if self._projection_net is not None:
            split = split + self._joint_encoder.param_length
            projection_theta = theta[:, split:]
            self._projection_net.set_parameters(
                projection_theta, reinitialize=reinitialize)

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (Tensor):
            state: not used, just keeps the interface same with other networks.
        """
        observations, actions = inputs

        encoded_obs, _ = self._obs_encoder(observations)  # [B, n, D] or [B, D]
        encoded_action, _ = self._action_encoder(
            actions)  # [B, n, D] or [B, D]
        joint = torch.cat([encoded_obs, encoded_action], -1)
        encoded_joint, _ = self._joint_encoder(joint)  # [B, n, D] or [B, D]
        if self._projection_net is None:
            return encoded_joint, state
        else:
            critics_dist, _ = self._projection_net(encoded_joint)
            return critics_dist, state
