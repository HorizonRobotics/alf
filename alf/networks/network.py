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
    """Base extension to nn.Module to simplify copy operations."""

    def __init__(self, input_tensor_spec, state_spec, name):
        """Creates an instance of `Network`.

        Args:
            input_tensor_spec (nested TensorSpec):  Representing the
                input observations.
            state_spec (nested TensorSpec): Representing the state needed by
                the network. Use () if none.
            name (str): A string representing the name of the network.
        """
        super(Network, self).__init__()
        self._name = name
        self._input_tensor_spec = input_tensor_spec
        self._state_spec = state_spec

    @property
    def name(self):
        return self._name

    @property
    def state_spec(self):
        return self._state_spec

    @property
    def input_tensor_spec(self):
        """Returns the spec of the input to the network ."""
        return self._input_tensor_spec

    def copy(self, **kwargs):
        """Create a shallow copy of this network.

        **NOTE** Network layer weights are *never* copied.  This method recreates
        the `Network` instance with the same arguments it was initialized with
        (excepting any new kwargs).

        Args:
            **kwargs: Args to override when recreating this network. Commonly
            overridden args include 'name'.

        Returns:
            A shallow copy of this network.
        """
        return type(self)(**dict(self._saved_kwargs, **kwargs))

    def __call__(self, inputs, *args, **kwargs):
        return super(Network, self).__call__(inputs, *args, **kwargs)


class DistributionNetwork(Network):
    """Base class for networks which generate Distributions as their output."""

    def __init__(self, input_tensor_spec, state_spec, name):
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=state_spec,
            name=name)
