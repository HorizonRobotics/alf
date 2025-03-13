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
"""PreprocessorNetworks."""

import functools
import math
import typing

import torch
import torch.nn as nn

import alf
import alf.utils.math_ops as math_ops
import alf.nest as nest
from alf.initializers import variance_scaling_init
from alf.nest.utils import get_outer_rank
from alf.networks.network import Network
from alf.tensor_specs import TensorSpec

from .network import Network, NetworkWrapper


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
                the input.
            input_preprocessors (nested Network|nn.Module|None): a nest of
                preprocessors, each of which will be applied to the
                corresponding input. If None, it is treated as ``math_ops.identity``.
                If not None, ``input_tensor_spec`` must have the same structure
                with ``input_preprocessors`` upto the structure defined by
                ``input_preprocessors`` (see ``alf.nest.map_structure_upto``),
                and each element of ``input_preproccessors`` will be applied to the
                corresponding subnest in ``input_tensor_spec``. If any element is None,
                it will be treated as ``math_ops.identity``. This arg is helpful if
                you want to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector. Note that only stateless networks are supported
                as input preprocessors by ``PreprocessorNetwork``.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. It must be provided if the result from
                ``input_preprocessors`` is nested. This combiner must accept the result
                from ``input_preprocessors`` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            name (str): name of the network
        """
        super().__init__(input_tensor_spec, name=name)

        # make sure the network holds the parameters of any trainable input
        # preprocessor
        self._input_preprocessor_modules = nn.ModuleList()
        self._input_preprocessors = None
        if input_preprocessors is not None:

            def _return_or_copy_preprocessor(preproc, input_spec):
                if preproc is None:
                    # allow None as a placeholder in the nest
                    return NetworkWrapper(lambda x: x, input_spec)
                elif isinstance(preproc, Network):
                    assert not nest.flatten(preproc.state_spec), (
                        "stateful preprocessor is not supported: %s" %
                        type(preproc))
                    preproc = preproc.copy()
                    self._input_preprocessor_modules.append(preproc)
                    return preproc
                elif isinstance(preproc, typing.Callable):
                    preproc = NetworkWrapper(preproc, input_spec).copy()
                    self._input_preprocessor_modules.append(preproc)
                else:
                    raise ValueError(
                        "Unsupported type in input_preprocessors: %s" %
                        type(preproc))
                return preproc

            self._input_preprocessors = alf.nest.map_structure_up_to(
                input_preprocessors, _return_or_copy_preprocessor,
                input_preprocessors, input_tensor_spec)

            input_tensor_spec = alf.nest.map_structure(
                lambda net: net.output_spec, self._input_preprocessors)

        self._preprocessing_combiner = preprocessing_combiner
        if alf.nest.is_nested(input_tensor_spec):
            assert preprocessing_combiner is not None, \
                ("When a nested input tensor spec is provided, an input " +
                "preprocessing combiner must also be provided!")
            input_tensor_spec = preprocessing_combiner(input_tensor_spec)
        else:
            assert isinstance(input_tensor_spec, TensorSpec), \
                "The spec must be an instance of TensorSpec!"
            self._preprocessing_combiner = alf.layers.Identity()

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
            inputs = alf.nest.map_structure_up_to(
                self._input_preprocessors, lambda preproc, tensor: preproc(
                    tensor)[0], self._input_preprocessors, inputs)

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
