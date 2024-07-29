# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""ActorDistributionNetwork and ActorRNNDistributionNetwork."""
from typing import Callable
from functools import partial

import torch
import torch.distributions as td
import torch.nn as nn

import alf
import alf.nest as nest
from .encoding_networks import EncodingNetwork, LSTMEncodingNetwork
from .normalizing_flow_networks import RealNVPNetwork
from .projection_networks import NormalProjectionNetwork, CategoricalProjectionNetwork
from .preprocessor_networks import PreprocessorNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.networks.network import Network


@alf.configurable
class ActorDistributionNetworkBase(Network):
    """A base class for ``ActorDistributionNetwork`` and ``ActorDistributionRNNNetwork``.

    Can also be used to create customized actor networks by providing
    different encoding network creators.
    """

    def __init__(self,
                 input_tensor_spec: alf.NestedTensorSpec,
                 action_spec: alf.NestedTensorSpec,
                 encoding_network_ctor: Callable,
                 discrete_projection_net_ctor: Callable,
                 continuous_projection_net_ctor: Callable,
                 name: str = 'ActorDistributionNetworkBase',
                 **encoder_kwargs):
        """
        Args:
            input_tensor_spec: the tensor spec of the input.
            action_spec: the tensor spec of the action.
            encoding_network_ctor: the creator of the encoding network that does
                the heavy lifting of the actor.
            discrete_projection_net_ctor (ProjectionNetwork): constructor that
                generates a discrete projection network that outputs discrete
                actions.
            continuous_projection_net_ctor (ProjectionNetwork): constructor that
                generates a continuous projection network that outputs
                continuous actions.
            name: name of the network
            encoder_kwargs: the extra keyword arguments to the encoding network
        """

        super().__init__(input_tensor_spec, name=name)

        if encoder_kwargs.get('kernel_initializer', None) is None:
            encoder_kwargs[
                'kernel_initializer'] = torch.nn.init.xavier_uniform_

        self._action_spec = action_spec
        self._encoding_net = encoding_network_ctor(input_tensor_spec,
                                                   **encoder_kwargs)
        self._create_projection_net(discrete_projection_net_ctor,
                                    continuous_projection_net_ctor)

    def _create_projection_net(self, discrete_projection_net_ctor,
                               continuous_projection_net_ctor):
        """If there are :math:`N` action specs, then create :math:`N` projection
        networks which can be a mixture of categoricals and normals.
        """

        def _create(spec):
            if spec.is_discrete:
                net = discrete_projection_net_ctor(
                    input_size=self._encoding_net.output_spec.shape[0],
                    action_spec=spec)
            else:
                net = continuous_projection_net_ctor(
                    input_size=self._encoding_net.output_spec.shape[0],
                    action_spec=spec)
            return net

        self._projection_net = nest.map_structure(_create, self._action_spec)
        if nest.is_nested(self._projection_net):
            # need this for torch to pickup the parameters of all the modules
            self._projection_net_module_list = nn.ModuleList(
                nest.flatten(self._projection_net))

    def forward(self, observation, state=()):
        """Computes an action distribution given an observation.

        Args:
            observation (torch.Tensor): consistent with ``input_tensor_spec``
            state: empty for API consistent with ``ActorRNNDistributionNetwork``

        Returns:
            act_dist (torch.distributions): action distribution
            state: empty
        """
        encoding, state = self._encoding_net(observation, state)
        act_dist = nest.map_structure(lambda proj: proj(encoding)[0],
                                      self._projection_net)
        return act_dist, state

    def make_parallel(self, n):
        """Create a ``ParallelActorDistributionNetwork`` using ``n`` replicas of ``self``.
        The initialized network parameters will be different.
        """
        return ParallelActorDistributionNetwork(self, n,
                                                "parallel_" + self._name)

    @property
    def state_spec(self):
        """Return the state spec of the actor network. It is simply the state spec
        of the encoding network."""
        return self._encoding_net.state_spec


@alf.configurable
class ActorDistributionNetwork(ActorDistributionNetworkBase):
    """Network which outputs temporally uncorrelated action distributions."""

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 input_preprocessors=None,
                 input_preprocessors_ctor=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_fc_bn=False,
                 use_fc_ln=False,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=NormalProjectionNetwork,
                 name="ActorDistributionNetwork"):
        """

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the action spec
            input_preprocessors (nested InputPreprocessor): a nest of
                `InputPreprocessor`, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with ``input_tensor_spec`` (after reshaping).
                If any element is None, then it will be treated as math_ops.identity.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
            input_preprocessors_ctor (Callable): if ``input_preprocessors`` is None
                and ``input_preprocessors_ctor`` is provided, then ``input_preprocessors``
                will be constructed by calling ``input_preprocessors_ctor(input_tensor_spec)``.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                `input_tensor_spec` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layer sizes.
            activation (nn.functional): activation used for hidden layers.
            kernel_initializer (Callable): initializer for all the layers
                excluding the projection net. If none is provided a default
                xavier_uniform will be used.
            use_fc_bn (bool): whether use Batch Normalization for the internal
                FC layers (i.e. FC layers except the last one).
            use_fc_ln (bool): whether use Layer Normalization for the internal
                fc layers (i.e. FC layers except the last one).
            discrete_projection_net_ctor (ProjectionNetwork): constructor that
                generates a discrete projection network that outputs discrete
                actions.
            continuous_projection_net_ctor (ProjectionNetwork): constructor that
                generates a continuous projection network that outputs
                continuous actions.
            name (str):
        """
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            encoding_network_ctor=EncodingNetwork,
            discrete_projection_net_ctor=discrete_projection_net_ctor,
            continuous_projection_net_ctor=continuous_projection_net_ctor,
            name=name,
            input_preprocessors=input_preprocessors,
            input_preprocessors_ctor=input_preprocessors_ctor,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_fc_bn=use_fc_bn,
            use_fc_ln=use_fc_ln)


class ParallelActorDistributionNetwork(Network):
    """Perform ``n`` actor distribution computations in parallel."""

    def __init__(self,
                 actor_network: ActorDistributionNetwork,
                 n: int,
                 name="ParallelActorDistributionNetwork"):
        """
        It creates a parallelized version of ``actor_network``.
        Args:
            actor_network (ActorDistributionNetwork): non-parallelized actor network
            n (int): make ``n`` replicas from ``actor_network`` with different
                initialization.
            name (str):
        """

        super().__init__(
            input_tensor_spec=actor_network.input_tensor_spec, name=name)
        self._encoding_net = actor_network._encoding_net.make_parallel(n)
        self._projection_net = actor_network._projection_net.make_parallel(n)
        self._output_spec = self._projection_net.output_spec

    def forward(self, observation, state=()):
        """Computes action distribution given a batch of observations.
        Args:
            inputs (tuple):  A tuple of Tensors consistent with `input_tensor_spec``.
            state (tuple): Empty for API consistent with ``ActorDistributionRNNNetwork``.
        """
        encoding, state = self._encoding_net(observation, state)
        act_dist = nest.map_structure(lambda proj: proj(encoding)[0],
                                      self._projection_net)
        return act_dist, state

    @property
    def state_spec(self):
        """Return the state spec of the actor network. It is simply the state spec
        of the encoding network."""
        return self._encoding_net.state_spec


@alf.configurable
class ActorDistributionRNNNetwork(ActorDistributionNetworkBase):
    """Network which outputs temporally correlated action distributions."""

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 actor_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=NormalProjectionNetwork,
                 name="ActorRNNDistributionNetwork"):
        """

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the action spec
            input_preprocessors (nested InputPreprocessor): a nest of
                ``InputPreprocessor``, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with ``input_tensor_spec`` (after reshaping).
                If any element is None, then it will be treated as math_ops.identity.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                ``input_tensor_spec`` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layers for encoding the observation.
            lstm_hidden_size (int or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
            actor_fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers.
            kernel_initializer (Callable): initializer for all the layers
                excluding the projection net. If none is provided a default
                xavier_uniform will be used.
            discrete_projection_net_ctor (ProjectionNetwork): constructor that
                generates a discrete projection network that outputs discrete
                actions.
            continuous_projection_net_ctor (ProjectionNetwork): constructor that
                generates a continuous projection network that outputs
                continuous actions.
            name (str):
        """
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            encoding_network_ctor=LSTMEncodingNetwork,
            discrete_projection_net_ctor=discrete_projection_net_ctor,
            continuous_projection_net_ctor=continuous_projection_net_ctor,
            name=name,
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            pre_fc_layer_params=fc_layer_params,
            hidden_size=lstm_hidden_size,
            post_fc_layer_params=actor_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)


class UnitNormalActorDistributionNetwork(Network):
    """Outputs a constant unit normal regardless of the inputs.
    """

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 name="UnitNormalActorDistributionNetwork"):
        super().__init__(input_tensor_spec, name=name)
        self._action_spec = action_spec

    def forward(self, inputs, state=()):
        outer_rank = alf.nest.utils.get_outer_rank(inputs,
                                                   self._input_tensor_spec)
        outer_dims = alf.nest.get_nest_shape(inputs)[:outer_rank]
        means = self._action_spec.zeros(outer_dims)
        stds = self._action_spec.ones(outer_dims)
        normal_dist = alf.utils.dist_utils.DiagMultivariateNormal(
            loc=means, scale=stds)
        return normal_dist, state


@alf.configurable
class LatentActorDistributionNetwork(Network):
    r"""Generating an actor distribution by transforming a prior action distribution
    (e.g., standard Normal noise :math:`\mathcal{N}(0,1)`) with a normalizing
    flow network. The resulting distribution might have an arbitrary shape.

    .. warning::

        Like some invertible transform such as ``StableTanh``, the inverse computation
        of a normalizing flow transform might cause numerical issues.
        For policy gradient methods like AC and PPO, transform caches are usually
        invalidated because of detaching actions for PG loss. So
        ``LatentActorDistributionNetwork`` is best suitable for non PG algorithms
        like DDPG and SAC. See ``alf/docs/notes/compute_probs_of_transformed_dist.rst``
        for details.
    """

    def __init__(self,
                 input_tensor_spec: alf.NestedTensorSpec,
                 action_spec: alf.NestedTensorSpec,
                 prior_actor_distribution_network_ctor:
                 Callable = UnitNormalActorDistributionNetwork,
                 normalizing_flow_network_ctor: Callable = RealNVPNetwork,
                 conditional_flow: bool = True,
                 scale_distribution: bool = False,
                 dist_squashing_transform: td.Transform = alf.utils.dist_utils.
                 StableTanh(),
                 name: str = "LatentActorDistributionNetwork"):
        """
        Args:
            input_tensor_spec: the tensor spec of the input
            action_spec: the action spec
            prior_actor_distribution_network_ctor: a constructor that creates
                any actor distribution network. The only requirement is that
                this class returns an action distribution (could be transformed)
                for ``forward()``.
            normalizing_flow_network_ctor: a constructor that creates a normalizing
                flow network which is used to transform the prior action
                distribution.
            conditional_flow: whether to make the normalizing flow network use
                inputs to condition its transformations. Only valid for normalizing
                flow nets that support this option.
            scale_distribution: Whether or not to scale the output
                distribution to ensure that the output action fits within the
                ``action_spec``.
            dist_squashing_transform:  A distribution Transform
                which transforms values into :math:`(-1, 1)`. Default to
                ``dist_utils.StableTanh()``
            name: name of the network
        """
        super().__init__(input_tensor_spec, name=name)
        self._prior_actor_network = prior_actor_distribution_network_ctor(
            input_tensor_spec=input_tensor_spec, action_spec=action_spec)
        self._nf_network = normalizing_flow_network_ctor(
            input_tensor_spec=action_spec,
            conditional_input_tensor_spec=(input_tensor_spec
                                           if conditional_flow else None))
        self._conditional_flow = conditional_flow
        self._scale_distribution = scale_distribution

        if scale_distribution:
            assert isinstance(action_spec, BoundedTensorSpec), \
                ("When squashing the mean or scaling the distribution, bounds "
                 + "are required for the action spec!")
            means, magnitudes = alf.utils.spec_utils.spec_means_and_magnitudes(
                action_spec)
            self._squash_transforms = [
                dist_squashing_transform,
                alf.utils.dist_utils.AffineTransform(
                    loc=means, scale=magnitudes)
            ]

    def forward(self, inputs, state=()):
        distribution, state = self._prior_actor_network(inputs, state)
        if not self._conditional_flow:
            inputs = None
        nf_transform = self._nf_network.make_invertible_transform(inputs)
        transforms = [nf_transform]
        if self._scale_distribution:
            transforms = transforms + self._squash_transforms
        transformed_dist = td.TransformedDistribution(distribution, transforms)
        return transformed_dist, state
