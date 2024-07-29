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
import inspect
import six
import torch
import torch.nn as nn
import typing

import alf
from alf.tensor_specs import TensorSpec
from alf.nest.utils import get_outer_rank
from alf.utils.dist_utils import (DistributionSpec, extract_spec,
                                  distributions_to_params,
                                  params_to_distributions)
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
                "%s.__init__ function accepts *args. This is not allowed." %
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

    def __init__(self, input_tensor_spec, state_spec=(), name="Network"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input.
            state_spec (nested TensorSpec): the (nested) tensor spec of the state
                of the network.
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
        self._state_spec = state_spec

    def _test_forward(self):
        """Generate a dummy input according to `nested_input_tensor_spec` and
        forward. Can be used to calculate output spec or testing the network.
        """
        inputs = common.zero_tensor_from_nested_spec(
            self._input_tensor_spec, batch_size=2)
        states = common.zero_tensor_from_nested_spec(
            self.state_spec, batch_size=2)
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

            def _copy(a):
                if not alf.nest.is_nested(a):
                    if isinstance(a, Network):
                        return a.copy()
                    elif isinstance(a, torch.nn.Module):
                        b = copy.deepcopy(a)
                        alf.layers.reset_parameters(b)
                        return b
                    else:
                        return a
                elif isinstance(a, list) or alf.nest.is_unnamedtuple(a):
                    return type(a)(_copy(value) for value in a)
                elif isinstance(a, dict):
                    return type(a)((k, _copy(v)) for k, v in a.items())
                else:  # namedtuple
                    return type(a)(**dict(
                        (f, _copy(getattr(a, f))) for f in a._fields))

            # we cannot use map_structure to do the copy because map_structure
            # will change the order of the keys in dict. Some Network (e.g. _Sequential)
            # depends on the order of the keys of the argument (element_dict of _Sequential)
            copied_kwargs = _copy(self._saved_kwargs)
            return type(self)(**dict(copied_kwargs, **kwargs))

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
        return self._state_spec

    @property
    def is_rnn(self):
        """Whether this network is a recurrent net."""
        return len(alf.nest.flatten(self.state_spec)) > 0

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
        different independently initialized parameters.

        By default, it creates ``NaiveParallelNetwork``, which simply making
        ``n`` copies of this network and use a loop to call them in ``forward()``.
        If possible, the subclass should override this to generate an optimized
        parallel implementation.

        Args:
            n (int): the number of copies
        Returns:
            Network: A parallel network
        """
        return NaiveParallelNetwork(self, n)


class NaiveParallelNetwork(Network):
    """Naive implementation of parallel network."""

    def __init__(self, network, n, name=None):
        """
        A parallel network has ``n`` copies of network with the same structure but
        different independently initialized parameters.

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
        state_spec = alf.nest.map_structure(
            lambda spec: alf.TensorSpec((n, ) + spec.shape, spec.dtype),
            network.state_spec)
        name = name if name else 'naive_parallel_%s' % network.name
        super().__init__(
            network.input_tensor_spec, state_spec=state_spec, name=name)
        self._networks = nn.ModuleList(
            [network.copy(name=self.name + '_%d' % i) for i in range(n)])
        self._n = n

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
        output_state_spec = None
        for i in range(self._n):
            if outer_rank == 1:
                inp = inputs
            else:
                inp = alf.nest.map_structure(lambda x: x[:, i, ...], inputs)
            s = alf.nest.map_structure(lambda x: x[:, i, ...], state)
            ret = self._networks[i](inp, s)
            if output_state_spec is None:
                output_state_spec = extract_spec(ret)
            output_states.append(ret)

        output_states = distributions_to_params(output_states)
        if self._n > 1:
            output, new_state = alf.nest.utils.stack_nests(
                output_states, dim=1)
        else:
            output, new_state = alf.nest.map_structure(
                lambda x: x.unsqueeze(1), output_states[0])

        output, new_state = params_to_distributions((output, new_state),
                                                    output_state_spec)
        return output, new_state


class NetworkWrapper(Network):
    """Wrap module or function as a Network."""

    def __init__(self,
                 module: typing.Callable,
                 input_tensor_spec: alf.NestedTensorSpec,
                 state_spec: alf.NestedTensorSpec = (),
                 name: str = "NetworkWrapper"):
        """
        Args:
            module: can be called as ``module(input)`` to calculate the output.
                If ``state_spec != ()``, then it's called as ``module(input,state)``
                and its return should be a tuple of ``(output,new_state)``.
            input_tensor_spec: the tensor spec for the input of ``module``
            state_spec: the tensor spec for the state of ``module``
            name: name of the wrapped network
        """
        super().__init__(input_tensor_spec, state_spec, name)
        assert isinstance(
            module,
            typing.Callable), ("module is not Callable: %s" % type(module))
        self._module = module

    def forward(self, x, state=()):
        if state == ():
            return self._module(x), state
        else:
            return self._module(x, state)

    def make_parallel(self, n: int):
        return NetworkWrapper(
            alf.layers.make_parallel_net(self._module, n),
            alf.layers.make_parallel_spec(self.input_tensor_spec, n),
            alf.layers.make_parallel_spec(self.state_spec, n),
            "parallel_" + self.name)


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
        input_tensor_spec (nested TensorSpec): if net is not a ``Network``, ``input_tensor_spec``
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


class BatchSquashNetwork(Network):
    """Wrap a network so that it works on multiple batch dims. Note that the
    output spec of this network is the *same* with that of the wrapped network (
    it won't include batch dims).

    Args:
        network: the network to be wrapped
        batch_dims: how many batch dims to squash before forward
    """

    def __init__(self,
                 network: Network,
                 batch_dims: int = 2,
                 name: str = "BatchSquashNetwork"):
        super().__init__(network.input_tensor_spec, network.state_spec, name)
        assert isinstance(network, Network)
        self._network = network
        self._bs = alf.layers.BatchSquash(batch_dims)

        self._output_spec = network.output_spec

    def forward(self, x, state=()):
        x, state = alf.nest.map_structure(self._bs.flatten, (x, state))
        output, new_state = self._network(x, state)
        return alf.nest.map_structure(self._bs.unflatten, (output, new_state))
