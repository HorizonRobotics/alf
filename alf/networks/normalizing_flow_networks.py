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

from functools import partial

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

    def __init__(self,
                 input_tensor_spec: alf.TensorSpec,
                 conditional_input_tensor_spec: alf.NestedTensorSpec = None,
                 use_transform_cache: bool = True,
                 name: str = "NormalizingFlowNetwork"):
        """
        Args:
            input_tensor_spec: input tensor spec
            conditional_input_tensor_spec: a nested tensor spec
            use_transform_cache: whether to cache transforms. When there
                is a conditional input, different transforms might be created
                depending on the conditional inputs. When there is no conditional
                input, the same transform will always be used.
                Note that this only caches the transform itself; to correctly
                cache the inverse result, you also have to set ``cache_size=1``
                when creating the transform.
            name: name of the network
        """
        assert not alf.nest.is_nested(input_tensor_spec), (
            f"Only unnested input spec is supported! Got {input_tensor_spec}")

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
    def use_conditional_inputs(self) -> bool:
        """
        Returns:
            Whether this normalizing flow uses inputs to condition the
                transforms.
        """
        return self._conditional_inputs

    def make_invertible_transform(
            self,
            conditional_inputs: alf.nest.NestedTensor = None) -> td.Transform:
        r"""Express the network forward computation as an invertible PyTorch
        ``Transform``. This overall transformation can be a composed one chaining
        many transformation layers.

        Args:
            conditional_inputs: an optional nested conditional inputs that
                condition the mapping :math:`x \rightarrow y`.

        Returns:
            an invertible transform
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
                an unnested tensor and a nested tensor ``(x, z)``. ``z`` is
                an optional conditional input that conditions the normalizing
                flow mapping from ``x`` to ``y``.
            state: should be an empty tuple
        """
        if self.use_conditional_inputs:
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
                an unnested tensor and a nested tensor ``(y, z)``. ``z`` is
                an optional conditional input that conditions the normalizing
                flow inverse mapping from ``y`` to ``x``.
            state: should be an empty tuple
        """
        if self.use_conditional_inputs:
            y, z = yz
        else:
            y, z = yz, None
        transform = self.make_invertible_transform(z)
        return transform.inv(y), ()


@alf.configurable
class RealNVPNetwork(NormalizingFlowNetwork):
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

    def __init__(self,
                 input_tensor_spec: alf.TensorSpec,
                 conditional_input_tensor_spec: alf.NestedTensorSpec = None,
                 input_preprocessors: alf.nest.Nest = None,
                 preprocessing_combiner: alf.nest.utils.NestCombiner = None,
                 conv_layer_params: Tuple[Tuple[int]] = None,
                 fc_layer_params: Tuple[int] = None,
                 activation: Callable = torch.tanh,
                 transform_scale_nonlinear: Callable = partial(
                     clipped_exp, clip_value_min=-10, clip_value_max=2),
                 sub_dim: int = None,
                 mask_mode: str = "contiguous",
                 num_layers: int = 2,
                 use_transform_cache: bool = True,
                 name: str = "RealNVPNetwork"):
        r"""
        Args:
            input_tensor_spec: input tensor spec
            conditional_input_tensor_spec: a nested tensor spec
            input_preprocessors: a nest of input preprocessors, each of
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
                RealNVP transform layers.
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
                this only stores the transform itself; you also have to use
                ``cache_size=1`` for the created transform to correctly cache
                the inverse result.
            name: name of the network
        """
        super(RealNVPNetwork, self).__init__(
            input_tensor_spec,
            conditional_input_tensor_spec,
            use_transform_cache=use_transform_cache,
            name=name)

        self._transform_scale_nonlinear = transform_scale_nonlinear

        D = input_tensor_spec.numel
        if sub_dim is None:
            sub_dim = D // 2
        assert 0 <= sub_dim <= D, f"Invalid sub dim {sub_dim}!"
        assert num_layers >= 1

        if sub_dim == 0 or sub_dim == D:
            logging.warning("For certain layers, the transform is identity!!")

        self._masks = self._generate_masks(input_tensor_spec, sub_dim,
                                           mask_mode, num_layers)

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

    def _generate_masks(self, spec, sub_dim, mask_mode, num_layers):
        masks = []
        for i in range(num_layers):
            if i % 2 == 0:
                new_mask = spec.zeros().to(torch.bool).reshape(-1)
                if mask_mode == "contiguous":
                    new_mask[:sub_dim] = 1
                elif mask_mode == "distributed":
                    if sub_dim > 0:
                        delta = spec.numel // sub_dim
                        idx = torch.arange(0, delta * sub_dim,
                                           delta).to(torch.int64)
                        new_mask[idx] = 1
                else:
                    assert mask_mode == "random", (
                        f"Invalid mask mode {mask_mode}")
                    idx = torch.randperm(spec.numel)[:sub_dim].to(torch.int64)
                    new_mask[idx] = 1
                new_mask = new_mask.reshape(spec.shape)
                masks.append(new_mask)
            else:  # flip
                masks.append(~masks[i - 1])
        return masks

    def _make_invertible_transform(self, conditional_inputs=None):
        transforms = []
        if self.use_conditional_inputs:
            i_spec, ci_spec = self._input_tensor_spec
        else:
            i_spec, ci_spec = self._input_tensor_spec, None
        for net, mask in zip(self._networks, self._masks):
            transforms.append(
                _RealNVPTransform(
                    input_tensor_spec=i_spec,
                    conditional_input_tensor_spec=ci_spec,
                    scale_trans_net=net,
                    mask=mask,
                    z=conditional_inputs,
                    scale_nonlinear=self._transform_scale_nonlinear))
        return td.ComposeTransform(transforms)


def _prepare_conditional_flow_inputs(
        xy_spec: alf.TensorSpec,
        xy: torch.Tensor,
        z_spec: alf.NestedTensorSpec = None,
        z: alf.nest.NestedTensor = None
) -> Tuple[alf.nest.NestedTensor, alf.utils.tensor_utils.BatchSquash]:
    """A general function for adjusting the shapes of inputs and conditional inputs
    of a conditional flow, prepared for a forward of a network next. Some networks
    assume only one batch dim, for example, when using ``alf.layers.Reshape()``.

    The reason why we need to do this is because the flow transform can be called
    with an arbitrary batch shape of ``x`` or ``y``, for example, when computing
    a loss with time dimension, or sampling a particular shape from a distribution.

    Args:
        xy_spec: tensor spec of ``x`` (forward) or ``y`` (inverse)
        xy:
        z_spec: tensor spec of ``z`` (conditional input)
        z:

    Returns:
        the prepared flow inputs and a ``BatchSquash`` object for unflattening
            the obtained network output if needed (None if not).
    """
    xy_outer_rank = alf.nest.utils.get_outer_rank(xy, xy_spec)
    xy_batch_shape = xy.shape[:xy_outer_rank]
    ret, bs = xy, None

    if xy_outer_rank > 1:
        # If there are extra outer dims of inputs, first squash them into one.
        bs = alf.utils.tensor_utils.BatchSquash(xy_outer_rank)
        ret = bs.flatten(xy)

    if z is not None:
        z_outer_rank = alf.nest.utils.get_outer_rank(z, z_spec)
        z_batch_shape = alf.nest.get_nest_shape(z)[:z_outer_rank]
        assert z_batch_shape == xy_batch_shape[-z_outer_rank:], (
            "xy batch shape is incompatible with z batch shape. "
            f"{xy_batch_shape} vs. {z_batch_shape}")

        if z_outer_rank > 1:
            bs_ = alf.utils.tensor_utils.BatchSquash(z_outer_rank)
            z = alf.nest.map_structure(bs_.flatten, z)

        B = alf.nest.get_nest_batch_size(z)
        if B < ret.shape[0]:
            # When the total outer dim of ``z`` is smaller than that of ``xy``,
            # it means that multiple samples of ``xy`` correspond to one ``z``,
            # so we need to repeat ``z``'s batch dim.
            z = alf.nest.map_structure(
                lambda e: e.repeat(ret.shape[0] // B, *((e.ndim - 1) * [1])),
                z)
        ret = (ret, z)

    return ret, bs


class _RealNVPTransform(td.Transform):
    """This class implements each transformation layer of ``RealNVPNetwork``. For
    details, refer to the docstring of ``RealNVPNetwork``.
    """
    domain: td.constraints.Constraint
    codomain: td.constraints.Constraint
    bijective = True
    sign = +1

    def __init__(self,
                 input_tensor_spec: alf.TensorSpec,
                 scale_trans_net: EncodingNetwork,
                 mask: torch.Tensor,
                 conditional_input_tensor_spec: alf.NestedTensorSpec = None,
                 z: alf.nest.NestedTensor = None,
                 cache_size: int = 1,
                 scale_nonlinear: Callable = torch.exp):
        """
        Args:
            input_tensor_spec: the tensor spec of ``x`` or ``y``
            scale_trans_net: an encoding network that computes the scale and
                translation given ``x`` or ``y``, optionally conditioned on ``z``.
            mask: a bool tensor indicates which part of ``x`` or ``y`` is preserved
                after the transformation.
            conditional_input_tensor_spec: tensor spec of ``z``
            z: a nest of conditional inputs to ``scale_trans_net``
            cache_size: the cache size of the transform
            scale_nonlinear: the nonlinear function applied to the scale; should
                be non-negative.
        """

        super().__init__(cache_size=cache_size)
        self._tensor_specs = (input_tensor_spec, conditional_input_tensor_spec)
        self._scale_trans_net = scale_trans_net
        self._b = mask
        self._scale_nonlinear = scale_nonlinear
        self._z = z
        self.domain = td.constraints.independent(td.constraints.real,
                                                 input_tensor_spec.ndim)
        self.codomain = td.constraints.independent(td.constraints.real,
                                                   input_tensor_spec.ndim)

    @property
    def params(self):
        """Let ALF know what parameters to store when extracting params from
        a transformed distribution."""
        return {'z': self._z}

    def get_builder(self):
        """If a transform has its ``get_builder`` implemented, then when building
        a transformed distribution from the extracted params, this builder will
        be called; otherwise its class will be used.

        This builder needs ``z`` provided as the input, which is also defined as
        the conditional variable. By assumption, this builder can create multiple
        transform instances that have different ``z``s but share other properties
        including scale&translation encoding networks.
        """
        return partial(
            _RealNVPTransform,
            input_tensor_spec=self._tensor_specs[0],
            scale_trans_net=self._scale_trans_net,
            mask=self._b,
            conditional_input_tensor_spec=self._tensor_specs[1],
            cache_size=self._cache_size,
            scale_nonlinear=self._scale_nonlinear)

    def __eq__(self, other):
        return (isinstance(other, _realVNPTransform)
                and self._tensor_specs == other._tensor_specs
                and self._scale_trans_net is other._scale_trans_net
                and self._z is other._z
                and self._scale_nonlinear is other._scale_nonlinear
                and torch.equal(self._b, other._b))

    def _get_scale_trans(self, x_or_y):
        """Compute the scale and translation for the transformation, where both
        of them depend on a part of the inputs and optionally on the conditional
        inputs (if not None).

        For efficiency, we compute scale and translation with the same network
        structure but different weights. This can be achieved by using a parallel
        network.

        One thing to note is that the inputs might have arbitrary outer dims in
        a scenario where a sampled batch with some shape from a distribution is
        being transformed. So we need to take special care of this.
        """
        xy_spec, z_spec = self._tensor_specs
        inputs = x_or_y * self._b

        inputs, bs = _prepare_conditional_flow_inputs(xy_spec, inputs, z_spec,
                                                      self._z)

        inputs = alf.layers.make_parallel_input(inputs, 2)  # [B,2,...]
        scale_trans = self._scale_trans_net(inputs)[0]  # [B,2,D]
        # reshape back to input tensor spec
        scale_trans = scale_trans.reshape(-1, 2, *xy_spec.shape)  # [B,2,...]
        scale, trans = scale_trans[:, 0, ...], scale_trans[:, 1, ...]

        if bs is not None:
            scale = bs.unflatten(scale)
            trans = bs.unflatten(trans)

        return scale, trans

    def _call(self, x):
        """Only use elements of ``x`` selected by ``1-self._b`` for computing
        the scale and translation. Those selected by ``self._b`` are unchanged.
        """
        scale, trans = self._get_scale_trans(x)
        new_x = x * self._scale_nonlinear(scale) + trans
        y = x * self._b + new_x * (~self._b)
        return y

    def _inverse(self, y):
        """Only use elements of ``y`` selected by ``1-self._b`` for computing
        the scale and translation. Those selected by ``self._b`` are unchanged.
        """
        scale, trans = self._get_scale_trans(y)
        new_y = (y - trans) / self._scale_nonlinear(scale)
        x = y * self._b + new_y * (~self._b)
        return x

    def log_abs_det_jacobian(self, x, y):
        r"""The Jacobian is always a triangular matrix (or can be converted into by
        row swapping). The diagonal elements are :math:`\mathbb{I}_d` and
        :math:`\text{diag}(\exp(scale(x_{1:d};z)))`, where the first :math:`d` dims
        are assumed to be selected by the mask ``self._b``.
        """
        scale, trans = self._get_scale_trans(x)
        if self._scale_nonlinear is torch.exp:
            jacob_diag = scale * (~self._b)
        else:
            jacob_diag = self._scale_nonlinear(scale).log() * (~self._b)
        dim = self.domain.event_dim
        shape = jacob_diag.shape[:-dim] + (-1, )
        return jacob_diag.reshape(shape).sum(-1)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        builder = self.get_builder()
        return builder(cache_size=cache_size)
