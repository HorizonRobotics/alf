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
r"""Different normalizing flow networks.

A normalizing flow network :math:`f: \mathbb{R}^N \rightarrow \mathbb{R}^N`

1. is invertible, namely given any output :math:`y=f(x)`, we can easily compute the
   corresponding input :math:`x=f^{-1}(y)`, and
2. whose Jacobian determinant is easy to compute, for example, the product of
   diagonal elements.
"""

from typing import Union, Callable, Tuple

from absl import logging

import torch
import torch.nn as nn
import torch.distributions as td

import alf
from alf.utils.math_ops import clipped_exp
from .network import Network
from .encoding_networks import EncodingNetwork


class NormalizingFlowNetwork(Network):
    """The base class for normalizing flow networks.

    Compared to traditional ``Network`` classes, its subclass needs to implement
    the interface ``make_invertible_transform()``.
    """

    def __init__(
            self,
            input_tensor_spec: alf.nest.NestedTensorSpec,
            conditional_input_tensor_spec: alf.nest.NestedTensorSpec = None,
            use_transform_cache: bool = True,
            name: str = "NormalizingFlowNetwork"):
        """
        Args:
            input_tensor_spec: a rank-1 tensor spec
            conditional_input_tensor_spec: a nested tensor spec
            use_transform_cache: whether use cached transform. When there
                is a conditional input, different transforms might be created
                depending on the conditonal inputs. When there is no conditional
                input, the same transform will always be used.
                Note that this only caches the transform itself; to correctly
                cache the inverse result, you also have to set ``cache_size=1``
                when creating the transform.
            name: name of the network
        """
        if conditional_input_tensor_spec is None:
            super().__init__(input_tensor_spec, name=name)
            self._conditional_inputs = False
        else:
            super().__init__(
                (input_tensor_spec, conditional_input_tensor_spec), name=name)
            self._conditional_inputs = True
        self._use_transform_cache = use_transform_cache
        self._cached_transform = (None, None)

    @property
    def use_conditional_inputs(self):
        """
        Returns:
            bool: Whether this normalizing flow uses inputs to condition the
                transforms.
        """
        return self._conditional_inputs

    def make_invertible_transform(
            self, conditional_inputs: alf.nest.NestedTensor = None):
        r"""Express the network forward computation as an invertible PyTorch
        ``Transform``. This overall transformation can be a composed one chaining
        many transformation layers.

        Args:
            conditional_inputs: an optional nested conditional inputs that
                condition the mapping :math:`x \rightarrow y`.

        Returns:
            torch.td.Transform: an invertible transform
        """
        if not self._use_transform_cache:
            return self._make_invertible_transform(conditional_inputs)
        old_t, old_z = self._cached_transform
        if conditional_inputs is old_z and old_t is not None:
            return old_t
        t = self._make_invertible_transform(conditional_inputs)
        self._cached_transform = (t, conditional_inputs)
        return t

    def _make_invertible_transform(
            self, conditional_inputs: alf.nest.NestedTensor = None):
        raise NotImplementedError()

    def forward(self,
                xz: Union[torch.Tensor, Tuple[torch.Tensor, alf.nest.
                                              NestedTensor]],
                state: alf.nest.NestedTensor = ()):
        """When we have no conditional input for forward: ``y=self.forward(x)``.
        Otherwise ``y=self.forward((x,z))`` where ``z`` is the conditional input.

        Args:
            xz: the input can be either an unnested tensor ``x`` or a tuple of
                an unnested tensor and a nested tensor ``(x, z)``. In both cases,
                currently we only support ``x`` to be a rank-1 tensor. ``z`` is
                an optional conditional input that conditions the normalizing
                flow mapping from ``x`` to ``y``.
            state: should be an empty tuple
        """
        if self._conditional_inputs:
            x, z = xz
        else:
            x, z = xz, None
        transform = self.make_invertible_transform(z)
        return transform(x), ()

    def inverse(self,
                yz: Union[torch.Tensor, Tuple[torch.Tensor, alf.nest.
                                              NestedTensor]],
                state: alf.nest.NestedTensor = ()):
        """When we have no conditional input for forward: ``x=self.inverse(y)``.
        Otherwise ``x=self.inverse((y,z))`` where ``z`` is the conditional input.

        Args:
            yz: the input can be either an unnested tensor ``y`` or a tuple of
                an unnested tensor and a nested tensor ``(y, z)``. In both cases,
                currently we only support ``y`` to be a rank-1 tensor. ``z`` is
                an optional conditional input that conditions the normalizing
                flow inverse mapping from ``y`` to ``x``.
            state: should be an empty tuple
        """
        if self._conditional_inputs:
            y, z = yz
        else:
            y, z = yz, None
        transform = self.make_invertible_transform(z)
        return transform.inv(y), ()


@alf.configurable
class realNVPNetwork(NormalizingFlowNetwork):
    r"""Real-valued non-volume preserving transformations.

    "DENSITY ESTIMATION USING REAL NVP", Dinh et al., ICLR 2017.

    In short, each transformation layer does

    .. math::

        \begin{array}{rcl}
            y_{1:d} &=& x_{1:d}\\
            y_{d+1:D} &=& x_{d+1:D}\bigodot \exp(s(x_{1:d};z)) + t(x_{1:d};z)\\
        \end{array}

    where :math:`d` is a hyperparameter that determines the two-way split of the
    input vector :math:`x`, :math:`D` the total length of :math:`x`, :math:`s`
    a (learned) scale function, and :math:`t` a (learned) translation function.
    The scale and translation functions can depend on other input :math:`z`.
    It can be verified that the Jacobian is a lower-triangular matrix and its
    diagonal elements are :math:`\mathbb{I}_d` and :math:`\text{diag}(\exp(s(x_{1:d};z)))`,
    regardless of how complex :math:`s` and :math:`t` are.

    The original paper suggests to alternate the computations of :math:`y_{1:d}`
    and :math:`y_{d+1:D}` to avoid some part of :math:`x` always getting copied.

    Our implementation also allows specifying other binary masks. We additionally
    support a random binary mask and an evenly distributed mask. The reason is that
    we can always re-arrange the 0s and 1s and swap the rows of the Jacobian to
    make it triangular. Because we always take the absolute of Jacobian determinant,
    row swapping will not change the result of ``log_abs_det_jacobian()``.

    Note that whichever binary mask is used, an alternating computation is always
    used. For example, let :math:`b` be the mask, then

    .. math::
        \begin{array}{rcl}
            y &=& b\bigodot x + (1-b)\bigodot(x\bigodot \exp(s(x\bigodot b;z))
            + t(x\bigodot b;z))\\
        \end{array}

    At even layers, we flip the values of :math:`b`.

    For inverse computation,

    .. math::

        \begin{array}{rcl}
            x &=& b\bigodot y + (1-b)\bigodot((y - t(y\bigodot b;z)) \div \exp(s(y\bigodot b;z)))\\
        \end{array}

    .. note::

        The scale and translation network's initial output should be in a good
        range, so their hidden activations default to ``torch.tanh``.
    """

    def __init__(
            self,
            input_tensor_spec: alf.nest.NestedTensorSpec,
            conditional_input_tensor_spec: alf.nest.NestedTensorSpec = None,
            input_preprocessors: alf.nest.Nest = None,
            preprocessing_combiner: alf.nest.utils.NestCombiner = None,
            conv_layer_params: Tuple[Tuple[int]] = None,
            fc_layer_params: Tuple[int] = None,
            activation: Callable = torch.tanh,
            transform_scale_nonlinear: Callable = torch.exp,
            sub_dim: int = None,
            mask_mode: str = "contiguous",
            num_layers: int = 2,
            use_transform_cache: bool = True,
            name: str = "realNVPNetwork"):
        r"""
        Args:
            input_tensor_spec: a rank-1 tensor spec
            conditional_input_tensor_spec: a nested tensor spec
            input_preprocessors: a nest of ``InputPreprocessor``, each of
                which will be applied to the corresponding input. If not None,
                then it must have the same structure with ``input_tensor_spec``
                (after reshaping). If any element is None, then it will be treated
                as math_ops.identity. Only used when conditional inputs are present,
                where its structure should be ``(x_processor, z_processor)``.
            preprocessing_combiner: preprocessing called on complex inputs.
                Note that this combiner must also accept ``input_tensor_spec``
                as the input to compute the processed tensor spec. For example,
                see `alf.nest.utils.NestConcat`. Only used when conditional inputs
                are present.
            conv_layer_params: a tuple of tuples where each tuple takes a format
                ``(filters, kernel_size, strides, padding)``, where ``padding``
                is optional. Used by the scale and translation networks.
            fc_layer_params: a tuple of integers representing FC layer sizes of
                the scale and translation networks.
            activation: hidden activation of the scale and translation networks
            transform_scale_nonlinear: nonlinear function applied to the
                scale network output. Its codomain should be :math:`[0,+\infty)`. Make
                sure that the value of this function won't explode after several
                realNVP transform layers.
            sub_dim: the dimensionality to keep unchanged at odd layers. If None,
                then half of the input is unchanged at a time. When it's 0, all
                input dims will be changed by an affine transform independent of
                the input. This case can still be interesting because the affine
                transform could depend on other variables (i.e., conditional
                ``AffineTransform``).
            mask_mode: three options are supported: "contiguous" (default),
                "distributed", and "random". "contiguous" means at odd layers,
                the first ``sub_dim`` elements are kept unchanged; "distributed"
                means that the ``sub_dim`` elements evenly distributed on the vector
                (good for vector with local similarity); "random" means that the
                mask is randomized.
            num_layers: number of transformation layers. Note that for mask
                mode of "random", every two layers will have a different randomized
                mask.
            use_transform_cache: whether use cached transform. Note that
                this only stores the transform itself; you also have to set
                ``cache_size=1`` for the created transform to correctly cache
                the inverse result.
            name: name of the network
        """
        super(realNVPNetwork, self).__init__(
            input_tensor_spec,
            conditional_input_tensor_spec,
            use_transform_cache=use_transform_cache,
            name=name)
        assert not alf.nest.is_nested(input_tensor_spec), (
            f"Only unnested input spec is supported! Got {input_tensor_spec}")
        assert input_tensor_spec.ndim == 1, (
            f"Only rank-1 tensor spec is supported! Got {input_tensor_spec.ndim}"
        )

        self._transform_scale_nonlinear = transform_scale_nonlinear

        D = input_tensor_spec.numel
        if sub_dim is None:
            sub_dim = D // 2
        assert 0 <= sub_dim <= D, f"Invalid sub dim {sub_dim}!"
        assert num_layers >= 1

        if sub_dim == 0 or sub_dim == D:
            logging.warning("For certain layers, the transform is identity!!")

        self._masks = self._generate_masks(D, sub_dim, mask_mode, num_layers)

        if activation in (torch.relu, torch.relu_):
            logging.warning(
                "Using relu activation for scaling might be unstable!")

        if self.use_conditional_inputs and preprocessing_combiner is None:
            preprocessing_combiner = alf.nest.utils.NestConcat()

        networks = []
        for i in range(num_layers):
            scale_trans_net = EncodingNetwork(
                input_tensor_spec=self._input_tensor_spec,
                input_preprocessors=input_preprocessors,
                preprocessing_combiner=preprocessing_combiner,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                last_layer_size=D,
                last_activation=alf.math.identity,
                activation=activation)
            networks.append(scale_trans_net.make_parallel(2))
        self._networks = nn.ModuleList(networks)

    def _generate_masks(self, D, sub_dim, mask_mode, num_layers):
        masks = []
        for i in range(num_layers):
            if i % 2 == 0:
                new_mask = torch.zeros((D, ), dtype=torch.bool)
                if mask_mode == "contiguous":
                    new_mask[:sub_dim] = 1
                elif mask_mode == "distributed":
                    if sub_dim > 0:
                        delta = D // sub_dim
                        idx = torch.arange(0, delta * sub_dim,
                                           delta).to(torch.int64)
                        new_mask[idx] = 1
                else:
                    assert mask_mode == "random", (
                        f"Invalid mask mode {mask_mode}")
                    idx = torch.randperm(D)[:sub_dim].to(torch.int64)
                    new_mask[idx] = 1
                masks.append(new_mask)
            else:  # flip
                masks.append(~masks[i - 1])
        return masks

    def _make_invertible_transform(self, conditional_inputs=None):
        transforms = []
        for net, mask in zip(self._networks, self._masks):
            transforms.append(
                _realNVPTransform(
                    net,
                    mask,
                    conditional_inputs,
                    scale_nonlinear=self._transform_scale_nonlinear))
        return td.transforms.ComposeTransform(transforms)


class _realNVPTransform(td.Transform):
    """This class implements each transformation layer of ``realNVPNetwork``. For
    details, refer to the docstring of ``realNVPNetwork``.
    """
    domain = td.constraints.real
    codomain = td.constraints.real
    bijective = True
    sign = +1

    def __init__(self,
                 scale_trans_net: Network,
                 mask,
                 z=None,
                 cache_size=1,
                 scale_nonlinear=torch.exp):

        super().__init__(cache_size=cache_size)
        self._scale_trans_net = scale_trans_net
        self._b = mask
        self._scale_nonlinear = scale_nonlinear
        self._z = z

    def __eq__(self, other):
        return (isinstance(other, _realVNPTransform)
                and self._scale_trans_net is other._scale_trans_net
                and self._z is other._z
                and self._scale_nonlinear is other._scale_nonlinear
                and torch.equal(self._b, other._b))

    def _get_scale_trans(self, inputs):
        inputs = inputs * self._b
        if self._z is not None:
            inputs = (inputs, self._z)
        inputs = alf.layers.make_parallel_input(inputs, 2)
        return self._scale_trans_net(inputs)[0]  # [B,2,D]

    def _call(self, x):
        scale_trans = self._get_scale_trans(x)
        new_x = x * self._scale_nonlinear(
            scale_trans[:, 0, ...]) + scale_trans[:, 1, ...]
        y = x * self._b + new_x * (~self._b)
        return y

    def _inverse(self, y):
        scale_trans = self._get_scale_trans(y)
        new_y = (y - scale_trans[:, 1, ...]) / self._scale_nonlinear(
            scale_trans[:, 0, ...])
        x = y * self._b + new_y * (~self._b)
        return x

    def log_abs_det_jacobian(self, x, y):
        scale_trans = self._get_scale_trans(x)
        if self._scale_nonlinear is torch.exp:
            return scale_trans[:, 0, ...] * (~self._b)
        else:
            return self._scale_nonlinear(
                scale_trans[:, 0, ...]).log() * (~self._b)
