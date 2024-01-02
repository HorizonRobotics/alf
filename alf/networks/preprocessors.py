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
"""This file contains input preprocessors as stateless Networks, used for the
purpose of preprocessing input and making gin files more convenient to configure.

Example:
In your gin file, below will be possible to configure:
input1 (img) -> preprocessor1 -> embed1    ----> EncodingNetwork
input2 (action) -> preprocessor2 -> embed2   /   (with `NestCombiner`)

"""
import abc
import math

import torch
import torch.nn as nn

import alf
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.nest.utils import get_outer_rank
from alf.networks.network import Network
import alf.utils.math_ops as math_ops


@alf.configurable
class EmbeddingPreprocessor(Network):
    """A preprocessor that converts the input to an embedding vector. This can
    be used when the input is a discrete scalar, or a continuous vector to be
    projected to a different dimension (to have the same length with other
    vectors). In the former case, ``torch.nn.Embedding`` is used without any
    activation. In the latter case, an ``EncodingNetwork`` is used with the
    specified network hyperparameters.
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
            activation (torch.nn.functional): activation of hidden layers if the
                input is a continuous vector.
            last_activation (nn.functional): activation function of the
                last layer specified by embedding_dim. ``math_ops.identity`` is
                used by default. Only used when the input is continuous.
            name (str):
        """
        super().__init__(input_tensor_spec, name=name)
        if input_tensor_spec.is_discrete:
            assert isinstance(input_tensor_spec, BoundedTensorSpec)
            N = input_tensor_spec.maximum - input_tensor_spec.minimum + 1
            # use nn.Embedding to support a large dictionary
            self._embedding_net = nn.Embedding(N, embedding_dim)
        else:
            # Only use an MLP for embedding a continuous input
            # Manually specify all arguments to avoid being overwritten by gin
            # configuration accidentally
            self._embedding_net = alf.networks.EncodingNetwork(
                input_tensor_spec=input_tensor_spec,
                input_preprocessors=None,
                preprocessing_combiner=None,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                activation=activation,
                last_layer_size=embedding_dim,
                last_activation=last_activation,
                name="preprocessor_embedding_net")

    def _preprocess(self, tensor):
        assert get_outer_rank(tensor, self._input_tensor_spec) == 1, \
            "Only supports one outer rank (batch dim)!"
        ret = self._embedding_net(tensor)
        # EncodingNetwork returns a pair
        return (ret if self._input_tensor_spec.is_discrete else ret[0])

    def forward(self, inputs, state=()):
        """Preprocess either a tensor input or a TensorSpec.

        Args:
            inputs (TensorSpec or Tensor):

        Returns:
            Tensor or TensorSpec: if ``Tensor``, the returned is the preprocessed
                result; otherwise it's the tensor spec of the result.
        """
        assert state is (), \
            "The preprocessor is assumed to be stateless currently."

        ret = self._preprocess(inputs)
        return ret, state


@alf.configurable
class CosineEmbeddingPreprocessor(Network):
    """A preprocessor that converts the input to an embedding vector of 
    cosine functions. It is suggested by the following IQN paper:

    ::
        
        Dabney et al "Implicit Quantile Networks for Distributional Reinforcement Learning",
        arXiv:1806.06923

    """

    def __init__(self,
                 input_tensor_spec,
                 embedding_dim,
                 name="CosineEmbeddingNetwork"):
        """
        Args:
            input_tensor_spec (TensorSpec): the input spec
            embedding_dim (int): output embedding size
        """
        super().__init__(input_tensor_spec, name=name)
        self.const_vec = torch.arange(1, 1 + embedding_dim)

    def forward(self, tau, state=()):
        return torch.cos(tau.unsqueeze(-1) * self.const_vec * math.pi), state

    def make_parallel(self, n):
        return self
