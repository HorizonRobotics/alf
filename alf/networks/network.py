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
"""Base extension to torch.nn.Module.
   Adapted from tf_agents/tf_agents/networks/network.py
"""

import abc
import copy
import functools
import gin
import inspect
import six
import torch

import torch.nn as nn

import alf
from alf.tensor_specs import TensorSpec
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
                the input.
            name (str):
        """
        super(Network, self).__init__()
        self._name = name
        self._input_tensor_spec = input_tensor_spec
        self._output_spec = None
        self._is_distribution = None
        # if ``singleton_instance`` is True, calling ``copy()`` return ``self``;
        # otherwise a re-created ``Network`` instance will be returned.
        # Default value is ``False`` and can be changed by calling the
        # ``singleton`` method
        self._singleton_instance = False

    def _test_forward(self):
        """Generate a dummy input according to `nested_input_tensor_spec` and
        forward. Can be used to calculate output spec or testing the network.
        """
        inputs = common.zero_tensor_from_nested_spec(
            self._input_tensor_spec, batch_size=1)
        states = common.zero_tensor_from_nested_spec(
            self.state_spec, batch_size=1)
        return self.forward(inputs, states)

    def singleton(self, singleton_instance=True):
        """Change the singleton property to the value given by the input
        argument ``singleton_instance``.
        Args:
            singleton_instance (bool): a flag indicating whether to turn
            the ``self._singleton_instance`` property on or off.
            If ``self._singleton_instance`` is True, calling ``copy()`` will
            return ``self``; otherwise a re-created ``Network`` instance will be
            returned.
        Returns:
            ``self``, which facilitates cascaded calling.
        """
        self._singleton_instance = singleton_instance
        return self

    def copy(self, **kwargs):
        """Create a  copy of this network or return the current instance.

        If ``self._singleton_instance`` is True, calling ``copy()`` will return
        ``self``; otherwise it will re-create and return a new ``Network``
        instance using the original arguments used by the constructor.

        **NOTE** When re-creating ``Network``, Network layer weights are *never*
        copied. This method recreates the ``Network`` instance with the same
        arguments it was initialized with (excepting any new kwargs).

        Args:
            **kwargs: Args to override when recreating this network.  Commonly
                overridden args include 'name'.

        Returns:
            Network:
        """
        if self._singleton_instance:
            return self
        else:
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
            training = self.training
            self.eval()
            self._output_spec = extract_spec(
                self._test_forward()[0], from_dim=1)
            self.train(training)
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
        """Make a parallelized version of this network.

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

    def __init__(self, network, n, name=None):
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
            name(str): a string that will be used as the name of the created
                NaiveParallelNetwork instance. If ``None``, ``naive_parallel_``
                followed by the ``network.name`` will be used by default.
        """
        super().__init__(network.input_tensor_spec,
                         name if name else 'naive_parallel_%s' % network.name)
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
        if self._n > 1:
            output, new_state = alf.nest.map_structure(
                lambda *tensors: torch.cat(tensors, dim=1), *output_states)
        else:
            output, new_state = output_states[0]

        return output, new_state

    @property
    def state_spec(self):
        return self._state_spec


class NetworkWrapper(Network):
    """Wrap module as a Network."""

    def __init__(self, module, input_tensor_spec):
        """
        Args:
            module (Callable): can be called as ``module(input)`` to calculate
                the output.
            input_tensor_spec (TensorSpec): the TensorSpec for the input of ``module``
        """
        super().__init__(input_tensor_spec)
        self._module = module

    def forward(self, x, state=()):
        return self._module(x), state

    def copy(self):
        if self._singleton_instance:
            return self

        module = copy.deepcopy(self._module)
        alf.layers.reset_parameters(module)

        return NetworkWrapper(module, self._input_tensor_spec)


def get_input_tensor_spec(net):
    """Get the input_tensor_spec of net if possible

    Args:
        net (nn.Module):
    Returns:
        nested TensorSpec | None: None if input_tensor_spec cannot be inferred
            from ``net``.
    """
    if isinstance(net, Network):
        return net.input_tensor_spec
    if isinstance(net, alf.layers.FC):
        return alf.TensorSpec((net.input_size, ))
    elif isinstance(net, torch.nn.Sequential):
        return get_input_tensor_spec(net[0])
    else:
        return None


def wrap_as_network(net, input_tensor_spec):
    """Wrap net as a Network if it is not a Network.

    Args:
        net (Network | Callable):
        input_tensor_spec (): if net is not a ``Network``, ``input_tensor_spec``
            must be provided unless net is a ``FC``. In that case, ``input_tensor_spec``
            will be inferred from ``net.input_size`` if it is not provided.
    Returns:
        Network:
    Raises:
        ValueError: if input_tensor_spec is None and cannot be inferred from ``net``
    """
    if isinstance(net, Network):
        return net
    if input_tensor_spec is None:
        input_tensor_spec = get_input_tensor_spec(net)
    if input_tensor_spec is None:
        raise ValueError("input_tensor_spec is undefined for net of "
                         "type: %s" % type(net))
    return NetworkWrapper(net, input_tensor_spec)
