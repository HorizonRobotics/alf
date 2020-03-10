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

import torch.nn as nn

import alf
import alf.layers as layers
from alf.tensor_specs import TensorSpec
from alf.networks.preprocessors import InputPreprocessor
from alf.nest.utils import get_outer_rank
from alf.utils.dist_utils import DistributionSpec, extract_spec
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

    It also handles complex nested inputs.
    """

    def __init__(self,
                 input_tensor_spec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 skip_input_preprocessing=False,
                 name="Network"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then `preprocessing_combiner` must not be
                None.
            input_preprocessors (nested InputPreprocessor): a nest of
                `InputPreprocessor`, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with `input_tensor_spec` (after reshaping).
                If any element is None, then it will be treated as alf.layers.identity.
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
            skip_input_preprocessing (bool): If True, skip preprocessing and
                checking the inputs, and the subclass will be responsible for
                the input handling. This can be set for a subclass that has its
                own fixed and specialized way of handling nested inputs.
            name (str):
        """
        super(Network, self).__init__()
        self._name = name
        self._skip_input_preprocessing = skip_input_preprocessing
        self._input_tensor_spec = input_tensor_spec

        if not skip_input_preprocessing:
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

            self._input_preprocessors = alf.nest.map_structure(
                lambda _: layers.identity, input_tensor_spec)
            if input_preprocessors is not None:
                input_preprocessors = alf.nest.pack_sequence_as(
                    input_tensor_spec, alf.nest.flatten(input_preprocessors))
                input_tensor_spec = alf.nest.map_structure(
                    _get_preprocessed_spec, input_preprocessors,
                    input_tensor_spec)
                # allow None as a placeholder in the nest
                self._input_preprocessors = alf.nest.map_structure(
                    lambda preproc: layers.identity
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
                self._preprocessing_combiner = layers.identity

        # This input spec is the final resulting spec after input preprocessors
        # and the nest combiner.
        self._processed_input_tensor_spec = input_tensor_spec
        self._output_spec = None

    def forward(self, inputs, state=()):
        """Preprocessing nested inputs."""
        if self._skip_input_preprocessing:
            return inputs, state
        else:
            proc_inputs = self._preprocessing_combiner(
                alf.nest.map_structure(lambda preproc, tensor: preproc(tensor),
                                       self._input_preprocessors, inputs))
            assert get_outer_rank(proc_inputs, self._processed_input_tensor_spec) == 1, \
                ("Only supports one outer rank (batch dim)! "
                + "After preprocessing: inputs size {} vs. input tensor spec {}".format(
                    proc_inputs.size(), self._processed_input_tensor_spec)
                + "\n Make sure that you have provided the right input preprocessors"
                + " and nest combiner!\n"
                + "Before preprocessing: inputs size {} vs. input tensor spec {}".format(
                    alf.nest.map_structure(lambda tensor: tensor.size(), inputs),
                    self._input_tensor_spec))
            return proc_inputs, state

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
    def input_tensor_spec(self):
        """Return the input tensor spec BEFORE preprocessings have been applied.
        """
        return self._input_tensor_spec

    @property
    def name(self):
        return self._name

    @property
    def output_spec(self):
        """Return the spec of the network's encoding output. By default, we use
        `_test_forward` to automatically compute the output and get its spec.
        For efficiency, subclasses can overwrite this function if the output spec
        can be obtained easily in other ways.
        """
        if self._output_spec is None:
            self._output_spec = TensorSpec.from_tensor(
                self._test_forward()[0], from_dim=1)
        return self._output_spec

    @property
    def state_spec(self):
        """Return the state spec to be used by an `Algorithm`."""
        return ()


class DistributionNetwork(Network):
    """A network that outputs distribution."""

    def __init__(self,
                 input_tensor_spec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 skip_input_preprocessing=False,
                 name="DistributionNetwork"):
        super(DistributionNetwork, self).__init__(
            input_tensor_spec, input_preprocessors, preprocessing_combiner,
            skip_input_preprocessing, name)

    @property
    def output_spec(self):
        if self._output_spec is None:
            self._output_spec = extract_spec(
                self._test_forward()[0], from_dim=1)
        return self._output_spec
