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
"""Base extension to torch.nn.Module.
   Adapted from tf_agents/tf_agents/networks/network.py
"""

import abc
import functools
import inspect
import six
import torch

import torch.nn as nn

import alf
from alf.tensor_specs import TensorSpec
from alf.networks.preprocessors import InputPreprocessor
from alf.nest.utils import get_outer_rank
from alf.utils.dist_utils import DistributionSpec, extract_spec
import alf.utils.math_ops as math_ops
from alf.utils import common


class _NetworkMeta(abc.ABCMeta):
    """Meta class for Network object.

    We mainly use this class to capture all args to `__init__` of all `Network`
    instances, and store them in `instance._saved_kwargs`.  This in turn is
    used by the `instance.copy` method.

    """

    def __new__(mcs, classname, baseclasses, attrs):
        """Control the creation of subclasses of the Network class.

        Args:
            classname: The name of the subclass being created.
            baseclasses: A tuple of parent classes.
            attrs: A dict mapping new attributes to their values.

        Returns:
            The class object.

        Raises:
            RuntimeError: if the class __init__ has *args in its signature.
        """
        if baseclasses[0] == nn.Module:
            # This is just Network below.  Return early.
            return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)

        init = attrs.get("__init__", None)

        if not init:
            # This wrapper class does not define an __init__.  When someone creates
            # the object, the __init__ of its parent class will be called.  We will
            # call that __init__ instead separately since the parent class is also a
            # subclass of Network. Here just create the class and return.
            return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)

        arg_spec = inspect.getfullargspec(init)
        if arg_spec.varargs is not None:
            raise RuntimeError(
                "%s.__init__ function accepts *args.  This is not allowed." %
                classname)

        def _capture_init(self, *args, **kwargs):
            """Captures init args and kwargs and stores them into `_saved_kwargs`."""
            if len(args) > len(arg_spec.args) + 1:
                # Error case: more inputs than args.  Call init so that the appropriate
                # error can be raised to the user.
                init(self, *args, **kwargs)
            for i, arg in enumerate(args):
                # Add +1 to skip `self` in arg_spec.args.
                kwargs[arg_spec.args[1 + i]] = arg
            init(self, **kwargs)
            setattr(self, "_saved_kwargs", kwargs)

        attrs["__init__"] = functools.update_wrapper(_capture_init, init)
        return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)


@six.add_metaclass(_NetworkMeta)
class Network(nn.Module):
    """A base class for various networks.

    Base extension to nn.Module to simplify copy operations.
    """

    def __init__(self, input_tensor_spec, name="Network"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then `preprocessing_combiner` must not be
                None.
            name (str):
        """
        super(Network, self).__init__()
        self._name = name
        self._input_tensor_spec = input_tensor_spec
        self._output_spec = None
        self._is_distribution = None

    def _test_forward(self):
        """Generate a dummy input according to `nested_input_tensor_spec` and
        forward. Can be used to calculate output spec or testing the network.
        """
        inputs = common.zero_tensor_from_nested_spec(
            self._input_tensor_spec, batch_size=1)
        states = common.zero_tensor_from_nested_spec(
            self.state_spec, batch_size=1)
        return self.forward(inputs, states)

    def copy(self, **kwargs):
        """Create a shallow copy of this network.

        **NOTE** Network layer weights are *never* copied.  This method recreates
        the `Network` instance with the same arguments it was initialized with
        (excepting any new kwargs).

        Args:
            **kwargs: Args to override when recreating this network.  Commonly
                overridden args include 'name'.

        Returns:
            A shallow copy of this network.
        """
        return type(self)(**dict(self._saved_kwargs, **kwargs))

    @property
    def saved_args(self):
        """Return the dictionary of the arguments used to construct the network."""
        return self._saved_kwargs

    @property
    def input_tensor_spec(self):
        """Return the input tensor spec BEFORE preprocessings have been applied.
        """
        return self._input_tensor_spec

    @property
    def name(self):
        """Name of this ``Network``."""
        return self._name

    @property
    def output_spec(self):
        """Return the spec of the network's encoding output. By default, we use
        `_test_forward` to automatically compute the output and get its spec.
        For efficiency, subclasses can overwrite this function if the output spec
        can be obtained easily in other ways.
        """
        if self._output_spec is None:
            self._output_spec = extract_spec(
                self._test_forward()[0], from_dim=1)
        return self._output_spec

    @property
    def state_spec(self):
        """Return the state spec to be used by an ``Algorithm``.

        Subclass should override this to return the correct ``state_spec``.
        """
        return ()

    @property
    def is_distribution_output(self):
        """Whether the output is Distribution."""
        if self._is_distribution is None:
            self._is_distribution = all(
                map(lambda spec: isinstance(spec, DistributionSpec),
                    alf.nest.flatten(self.output_spec)))
        return self._is_distribution

    def make_parallel(self, n):
        """Make a parllelized version of this network.

        A parallel network has ``n`` copies of network with the same structure but
        different indepently initialized parameters.

        By default, it creates ``NaiveParallelNetwork``, which simply making
        ``n`` copies of this network and use a loop to call them in ``forward()``.
        If possible, the subclass should override this to generate an optimized
        parallel implementation.

        Returns:
            Network: A parallel network
        """
        return NaiveParallelNetwork(self, n)


class NaiveParallelNetwork(Network):
    """Naive implementation of parallel network."""

    def __init__(self, network, n, name_prefix="naive_parallel_"):
        """
        A parallel network has ``n`` copies of network with the same structure but
        different indepently initialized parameters.

        ``NaiveParallelNetwork`` created ``n`` independent networks with the same
        structure as ``network`` and evaluate them separately in loop during
        ``forward()``.

        Args:
            network (Network): the parallel network will have ``n`` copies of
                ``network``.
            n (int): ``n`` copies of ``network``
            name_prefix (str): a string that will be added as a prefix to the
                name of the ``network`` as the name of the NaiveParallelNetwork
        """
        super().__init__(network.input_tensor_spec, name_prefix + network.name)
        self._networks = nn.ModuleList(
            [network.copy(name=self.name + '_%d' % i) for i in range(n)])
        self._n = n
        self._state_spec = alf.nest.map_structure(
            lambda spec: alf.TensorSpec((n, ) + spec.shape, spec.dtype),
            network.state_spec)

    def forward(self, inputs, state=()):
        """Compute the output and the next state.

        Args:
            inputs (nested torch.Tensor): its shape can be ``[B, n, ...]``, or
                ``[B, ...]``
            state (nested torch.Tensor): its shape must be ``[B, n, ...]``
        Returns:
            output (nested torch.Tensor): its shape is ``[B, n, ...]``
            next_state (nested torch.Tensor): its shape is ``[B, n, ...]``
        """
        outer_rank = alf.nest.utils.get_outer_rank(inputs,
                                                   self._input_tensor_spec)
        assert 1 <= outer_rank <= 2, ("inputs should have shape [B, %d, ...] "
                                      " or [B, ...]" % self._n)

        if state != ():
            state_outer_rank = alf.nest.utils.get_outer_rank(
                state, self.state_spec)
            assert state_outer_rank == 1, (
                "state should have shape [B, %d, ...] " % self._n)

        output_states = []
        for i in range(self._n):
            if outer_rank == 1:
                inp = inputs
            else:
                inp = alf.nest.map_structure(lambda x: x[:, i, ...], inputs)
            s = alf.nest.map_structure(lambda x: x[:, i, ...], state)
            ret = self._networks[i](inp, s)
            ret = alf.nest.map_structure(lambda x: x.unsqueeze(1), ret)
            output_states.append(ret)
        output, new_state = alf.nest.map_structure(
            lambda *tensors: torch.cat(tensors, dim=1), *output_states)
        return output, new_state

    @property
    def state_spec(self):
        return self._state_spec


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
                corresponding input. If not None, then it must
                have the same structure with `input_tensor_spec` (after reshaping).
                If any element is None, then it will be treated as math_ops.identity.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
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
            self._input_preprocessor_modules.append(preproc)
            return preproc(spec)

        self._input_preprocessors = None
        if input_preprocessors is not None:
            input_preprocessors = alf.nest.pack_sequence_as(
                input_tensor_spec, alf.nest.flatten(input_preprocessors))
            input_tensor_spec = alf.nest.map_structure(
                _get_preprocessed_spec, input_preprocessors, input_tensor_spec)
            # allow None as a placeholder in the nest
            self._input_preprocessors = alf.nest.map_structure(
                lambda preproc: math_ops.identity
                if preproc is None else preproc, input_preprocessors)

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
