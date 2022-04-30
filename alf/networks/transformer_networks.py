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

import torch
import torch.nn as nn

import alf
from alf.networks import PreprocessorNetwork
from alf.networks.memory import FIFOMemory
from alf.nest.utils import NestConcat


@alf.configurable
class TransformerNetwork(PreprocessorNetwork):
    """A Network composed of Memory and TransformerBlock.

    The following is the pseudocode for the computation:

    .. code-block:: python

        for i in range(num_prememory_layers):
            core, inputs = T_i([core, inputs], [core, inputs])
        for j in range(num_memory_layers):
            new_core, inputs = TM_j([memory_j, core, inputs], [core, inputs])
            memory_j.write(core)
            core = new_core
        return core, new_memory_state

    where T_i denotes the ``TransformerBlock``  for the i-th prememory layers
    and TM_j denotes the ``TransformerBlock`` for the j-th memory layers. memory_j
    is an ``FIFOMemory`` object (not to be confused with the ``memory`` argument
    of ``TransformerBlock.forward() function``)

    The core embedding serves the same purpose of [CLS] in the BERT model in [1],
    which is to generate a fixed dimensional representation for downstream tasks.
    Different from BERT, which only has one [CLS] embedding, we allow the option
    of having multiple core embeddings. In addition to generating a fixed dimensional
    representation, the core embedding is also used to update the memory.

    [1]. Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for
         Language Understanding
    """

    def __init__(self,
                 input_tensor_spec,
                 num_prememory_layers,
                 num_attention_heads,
                 d_ff=None,
                 core_size=1,
                 use_core_embedding=True,
                 memory_size=0,
                 num_memory_layers=0,
                 return_core_only=True,
                 centralized_memory=True,
                 input_preprocessors=None,
                 name="TransformerNetwork"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If ``input_tensor_spec`` is not nested, it should
                represent a rank-2 tensor of shape ``[input_size, d_model]``, where
                ``input_size`` is the length of the input sequence, and ``d_model``
                is the dimension of embedding.
            num_prememory_layers (int): number of TransformerBlock calculation
                without using memory
            num_attention_heads (int): number of attention heads for each
                ``TransformerBlock``
            d_ff (int): the size of the hidden layer of the feedforward network
                in each ``TransformerBlock``. If None, ``TransformerBlock`` will
                calculate it as ``4*d_model``.
            memory_size (int): size of memory.
            num_memory_layers (int): number of TransformerBlock calculation
                using memory
            return_core_only (bool): If True, only return the core embedding.
                Otherwise, return all embeddings
            core_size (int): size of core (i.e. number of embeddings of core)
            use_core_embedding (bool): whether to use learnable core embedding.
                If True, will use additional learnable core embedding to augment
                the input. If False, the first ``core_size`` embeddings of the
                input are treated as core.
            centralized_memory (bool): if False, there will be a separate memory
                for each memory layers. if True, there will be a single memory
                for all the memroy layers and it is updated using the last core
                embeddings.
            input_preprocessors (nested Network|nn.Module): a nest of
                stateless preprocessor networks, each of which will be applied to the
                corresponding input. If not None, then it must have the same
                structure with ``input_tensor_spec``. If any element is None, then
                it will be treated as math_ops.identity. This arg is helpful if
                you want to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector. The output_spec of each input preprocessor i
                should be [input_size_i, d_model]. The result of all the preprocessors
                will be concatenated as a Tensor of shape ``[batch_size, input_size, d_model]``,
                where ``input_size = sum_i input_size_i``.
        """
        preprocessing_combiner = None
        if input_preprocessors is not None:
            preprocessing_combiner = NestConcat(dim=-2)
        super().__init__(
            input_tensor_spec,
            input_preprocessors,
            preprocessing_combiner=preprocessing_combiner,
            name=name)

        assert self._processed_input_tensor_spec.ndim == 2

        input_size, d_model = self._processed_input_tensor_spec.shape
        if num_memory_layers > 0:
            assert memory_size > 0, ("memory_size needs to be set if "
                                     "num_memory_layers > 0")
            if centralized_memory:
                self._memories = [FIFOMemory(d_model, memory_size)]
            else:
                self._memories = [
                    FIFOMemory(d_model, memory_size)
                    for _ in range(num_memory_layers)
                ]
        else:
            self._memories = []
        self._centralized_memory = centralized_memory

        self._core_size = core_size
        if use_core_embedding:
            self._core_embedding = nn.Parameter(
                torch.Tensor(1, core_size, d_model))
            nn.init.uniform_(self._core_embedding, -0.1, 0.1)
        else:
            self._core_embedding = None

        self._state_spec = [mem.state_spec for mem in self._memories]
        self._num_memory_layers = num_memory_layers
        self._num_prememory_layers = num_prememory_layers

        self._transformers = nn.ModuleList()

        for i in range(num_prememory_layers):
            self._transformers.append(
                alf.layers.TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_attention_heads,
                    memory_size=input_size + core_size,
                    positional_encoding='abs' if i == 0 else 'none'))

        for i in range(num_memory_layers):
            self._transformers.append(
                alf.layers.TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_attention_heads,
                    memory_size=memory_size + input_size + core_size,
                    positional_encoding='abs' if i == 0 else 'none'))

        self._return_core_only = return_core_only

    @property
    def state_spec(self):
        return self._state_spec

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (nested Tensor): consistent with ``input_tensor_spec`` provided
                at ``__init__()``
            state (nested Tensor): states
        Returns:
            - Tensor: shape is [B, core_size * d_model] if ``return_core_only``,
                    and [B, core_size + input_size, d_model] if not ``return_core_only``,
                    where ``input_size`` is the number of embeddings from the
                    (processed) input.
            - nested Tensor: network states.
        """
        z, _ = super().forward(inputs, state)
        batch_size = z.shape[0]
        if self._core_embedding is not None:
            core_embedding = self._core_embedding.expand(batch_size, -1, -1)
            query = torch.cat([core_embedding, z], dim=-2)
        else:
            query = z
        for i in range(self._num_prememory_layers):
            query = self._transformers[i].forward(query)

        if self._num_memory_layers > 0 and self._centralized_memory:
            memory = self._memories[0]
            memory.from_states(state[0])
            mem = memory.memory()
            for i in range(self._num_memory_layers):
                transformer = self._transformers[self._num_prememory_layers +
                                                 i]
                query = transformer.forward(
                    memory=torch.cat([mem, query], dim=-2), query=query)
            memory.write(query[:, :self._core_size, :])
        else:
            for i in range(self._num_memory_layers):
                memory = self._memories[i]
                memory.from_states(state[i])
                transformer = self._transformers[self._num_prememory_layers +
                                                 i]
                new_query = transformer.forward(
                    memory=torch.cat([memory.memory(), query], dim=-2),
                    query=query)
                memory.write(query[:, :self._core_size, :])
                query = new_query

        new_state = [mem.states for mem in self._memories]

        if self._return_core_only:
            return query[:, :self._core_size, :].reshape(batch_size,
                                                         -1), new_state
        else:
            return query, new_state
