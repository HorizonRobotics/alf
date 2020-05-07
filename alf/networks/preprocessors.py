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
"""Some network input preprocessors."""

import abc
import gin

import torch
import torch.nn as nn

import alf
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.nest.utils import get_outer_rank
from alf.networks.network import Network, SingletonInstanceNetwork
import alf.utils.math_ops as math_ops


class InputPreprocessor(Network):
    """A preprocessor applied to a Network's input.

    It's meant to be applied to either individual inputs or individual
    TensorSpecs. Mainly for the purpose of input preprocessing and making gin
    files more convenient to configure.

    Example:
    In your gin file, below will be possible to configure:
    input1 (img) -> InputPreprocessor1 -> embed1    ----> EncodingNetwork
    input2 (action) -> InputPreprocessor2 -> embed2   /   (with `NestCombiner`)
    """

    def __init__(self, input_tensor_spec, name="InputPreprocessor"):
        assert isinstance(input_tensor_spec, TensorSpec)
        super().__init__(input_tensor_spec, name)

    @abc.abstractmethod
    def _preprocess(self, tensor):
        """Preprocess a tensor input.

        Args:
            tensor (Tensor):

        Returns:
            a preprocessed tensor
        """
        pass

    def forward(self, inputs):
        """Preprocess either a tensor input or a TensorSpec.

        Args:
            inputs (TensorSpec or Tensor):

        Returns:
            Tensor or TensorSpec: if ``Tensor``, the returned is the preprocessed
                result; otherwise it's the tensor spec of the result.
        """
        if isinstance(inputs, TensorSpec):
            tensor = inputs.zeros(outer_dims=(1, ))
        else:
            tensor = inputs
        ret = self._preprocess(tensor)
        if isinstance(inputs, TensorSpec):
            return TensorSpec.from_tensor(ret, from_dim=1)
        return ret


@gin.configurable
class EmbeddingPreprocessor(InputPreprocessor):
    """A preprocessor that converts the input to an embedding vector. This can
    be used when the input is a discrete scalar, or a continuous vector to be
    projected to a different dimension (to have the same length with other
    vectors). Different from an ``EncodingNetwork``, the input can be in the
    original format from the environment.
    """

    def __init__(self,
                 input_tensor_spec,
                 embedding_dim,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 last_activation=math_ops.identity,
                 name="EmbeddingPreproc"):
        """
        Args:
            input_tensor_spec (TensorSpec): the input spec
            embedding_dim (int): output embedding size
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing FC
                layer sizes.
            activation (torch.nn.functional): activation applied to the embedding
            last_activation (nn.functional): activation function of the
                last layer specified by embedding_dim. ``math_ops.identity`` is
                used by default.
            name (str):
        """
        super().__init__(input_tensor_spec, name)
        if input_tensor_spec.is_discrete:
            assert isinstance(input_tensor_spec, BoundedTensorSpec)
            N = input_tensor_spec.maximum - input_tensor_spec.minimum + 1
            # use nn.Embedding to support a large dictionary
            self._embedding_net = nn.Embedding(N, embedding_dim)
        else:
            # Only use an MLP for embedding a continuous input
            self._embedding_net = alf.networks.EncodingNetwork(
                input_tensor_spec=input_tensor_spec,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                activation=activation,
                last_layer_size=embedding_dim,
                last_activation=last_activation)

    def _preprocess(self, tensor):
        assert get_outer_rank(tensor, self._input_tensor_spec) == 1, \
            "Only supports one outer rank (batch dim)!"
        ret = self._embedding_net(tensor)
        # EncodingNetwork returns a pair
        return (ret if self._input_tensor_spec.is_discrete else ret[0])


@gin.configurable
class SharedEmbeddingPreprocessor(InputPreprocessor, SingletonInstanceNetwork):
    """An embedding preprocessor that can be shared by multiple networks.
    """

    def __init__(self,
                 input_tensor_spec,
                 embedding_dim,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 last_activation=math_ops.identity,
                 name="EmbeddingPreproc"):
        """
        Args:
            input_tensor_spec (TensorSpec): the input spec
            embedding_dim (int): output embedding size
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing FC
                layer sizes.
            activation (torch.nn.functional): activation applied to the embedding
            last_activation (nn.functional): activation function of the
                last layer specified by embedding_dim. ``math_ops.identity`` is
                used by default.
            name (str):
        """
        super().__init__(input_tensor_spec, name)
        if input_tensor_spec.is_discrete:
            assert isinstance(input_tensor_spec, BoundedTensorSpec)
            N = input_tensor_spec.maximum - input_tensor_spec.minimum + 1
            # use nn.Embedding to support a large dictionary
            self._embedding_net = nn.Embedding(N, embedding_dim)
        else:
            # Only use an MLP for embedding a continuous input
            self._embedding_net = alf.networks.EncodingNetwork(
                input_tensor_spec=input_tensor_spec,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                activation=activation,
                last_layer_size=embedding_dim,
                last_activation=last_activation)

    def _preprocess(self, tensor):
        assert get_outer_rank(tensor, self._input_tensor_spec) == 1, \
            "Only supports one outer rank (batch dim)!"
        ret = self._embedding_net(tensor)
        # EncodingNetwork returns a pair
        return (ret if self._input_tensor_spec.is_discrete else ret[0])


class PreprocessorNetwork(Network):
    """A base class for networks with input processing need."""

    def __init__(self,
                 input_tensor_spec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 name="PreprocessorNetwork"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then `preprocessing_combiner` must not be
                None.
            input_preprocessors (nested InputPreprocessor): a nest of
                `InputPreprocessor`, each of which will be applied to the
                corresponding input. If not None, then it must have the same
                structure with `input_tensor_spec`. If any element is None, then
                it will be treated as math_ops.identity. This arg is helpful if
                you want to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                `input_tensor_spec` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            name (str): name of the network
        """
        super().__init__(input_tensor_spec, name)

        # make sure the network holds the parameters of any trainable input
        # preprocessor
        self._input_preprocessor_modules = nn.ModuleList()

        def _get_preprocessed_spec(preproc, spec):
            if not isinstance(preproc, InputPreprocessor):
                # In this case we just assume the spec won't change after the
                # preprocessing. If it does change, then you should consider
                # defining an `InputPreprocessor` instead.
                return spec
            return preproc(spec)

        self._input_preprocessors = None
        if input_preprocessors is not None:
            input_tensor_spec = alf.nest.map_structure(
                _get_preprocessed_spec, input_preprocessors, input_tensor_spec)

            def _return_or_copy_preprocessor(preproc):
                if preproc is None:
                    # allow None as a placeholder in the nest
                    preproc = math_ops.identity
                elif isinstance(preproc, InputPreprocessor):
                    preproc = preproc.copy()
                    self._input_preprocessor_modules.append(preproc)
                return preproc

            self._input_preprocessors = alf.nest.map_structure(
                _return_or_copy_preprocessor, input_preprocessors)

        self._preprocessing_combiner = preprocessing_combiner
        if alf.nest.is_nested(input_tensor_spec):
            assert preprocessing_combiner is not None, \
                ("When a nested input tensor spec is provided, a input " +
                "preprocessing combiner must also be provided!")
            input_tensor_spec = preprocessing_combiner(input_tensor_spec)
        else:
            assert isinstance(input_tensor_spec, TensorSpec), \
                "The spec must be an instance of TensorSpec!"
            self._preprocessing_combiner = math_ops.identity

        # This input spec is the final resulting spec after input preprocessors
        # and the nest combiner.
        self._processed_input_tensor_spec = input_tensor_spec

    def forward(self, inputs, state=(), min_outer_rank=1, max_outer_rank=1):
        """Preprocessing nested inputs.

        Args:
            inputs (nested Tensor): inputs to the network
            state (nested Tensor): RNN state of the network
            min_outer_rank (int): the minimal outer rank allowed
            max_outer_rank (int): the maximal outer rank allowed
        Returns:
            Tensor: tensor after preprocessing.
        """
        if self._input_preprocessors:
            inputs = alf.nest.map_structure(
                lambda preproc, tensor: preproc(tensor),
                self._input_preprocessors, inputs)
        proc_inputs = self._preprocessing_combiner(inputs)
        outer_rank = get_outer_rank(proc_inputs,
                                    self._processed_input_tensor_spec)
        assert min_outer_rank <= outer_rank <= max_outer_rank, \
            ("Only supports {}<=outer_rank<={}! ".format(min_outer_rank, max_outer_rank)
            + "After preprocessing: inputs size {} vs. input tensor spec {}".format(
                proc_inputs.size(), self._processed_input_tensor_spec)
            + "\n Make sure that you have provided the right input preprocessors"
            + " and nest combiner!\n"
            + "Before preprocessing: inputs size {} vs. input tensor spec {}".format(
                alf.nest.map_structure(lambda tensor: tensor.size(), inputs),
                self._input_tensor_spec))
        return proc_inputs, state
