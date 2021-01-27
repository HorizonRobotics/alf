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
"""Various Network containers."""

import copy
import gin
import torch.nn as nn

import alf
from alf.nest import (flatten, flatten_up_to, map_structure,
                      map_structure_up_to, pack_sequence_as)
from .network import Network, get_input_tensor_spec, wrap_as_network


@gin.configurable
class Sequential(Network):
    """Network composed of a sequence of torch.nn.Module or alf.nn.Network.

    Example:

    .. code-block:: python

        net = Sequential((module1, module2))
        y, new_state = net(x, state)

    is equivalent to the following:

    .. code-block:: python

        z, new_state1 = module1(x, state[0])
        y, new_state2 = module2(z, state[1])
        new_state = (new_state1, new_state2)

    """

    def __init__(self, modules, input_tensor_spec=None, name="Sequential"):
        """

        Args:
            modules (list[nn.Module]): list of ``torch.nn.Module`` or ``alf.nn.Network``.
            input_tensor_spec (TensorSpec): the tensor spec of the input. It must
                be specified if it cannot be inferred from ``modules[0]``.
            name (str):
        """
        state_spec = [()] * len(modules)
        for i, module in enumerate(modules):
            if isinstance(module, Network):
                state_spec[i] = module.state_spec
        if len(flatten(state_spec)) == 0:
            state_spec = ()
        self._state_spec = state_spec
        if input_tensor_spec is None:
            input_tensor_spec = get_input_tensor_spec(modules[0])
        if input_tensor_spec is None:
            raise ValueError(
                "input_tensor_spec needs to "
                "be provided for modules[0] of type %s" % type(modules[0]))
        super().__init__(input_tensor_spec, name)
        self._networks = nn.ModuleList(modules)

    @property
    def state_spec(self):
        return self._state_spec

    def forward(self, input, state=()):
        x = input
        if self._state_spec == ():
            for net in self._networks:
                if isinstance(net, Network):
                    x = net(x)[0]
                else:
                    x = net(x)
            return x, state
        else:
            new_state = [()] * len(self._networks)
            for i, net in enumerate(self._networks):
                if isinstance(net, Network):
                    x, new_state[i] = net(x, state[i])
                else:
                    x = net(x)
            return x, new_state

    def copy(self, name=None):
        """Create a copy of this network or return the current instance.

        If ``self._singleton_instance`` is True, calling ``copy()`` will return
        ``self``; otherwise it will make a copy of all the modules and re-initialize
        their parameters.

        Args:
            name (str): name of the new network. Only used if not self._singleton_instance.
        Returns:
            Sequential:
        """
        if self._singleton_instance:
            return self

        if name is None:
            name = self.name

        new_networks = []
        for n in self._networks:
            if isinstance(n, Network):
                net = n.copy()
            else:
                net = copy.deepcopy(n)
                alf.layers.reset_parameters(net)
            new_networks.append(net)
        return Sequential(new_networks, self._input_tensor_spec, name)

    def __getitem__(self, i):
        return self._networks[i]


class Parallel(Network):
    """Apply each Network in the nest of Network to the corresponding input.

    Example:

    .. code-block:: python

        net = Parallel((module1, module2))
        y, new_state = net(x, state)

    is equivalent to the following:

    .. code-block:: python

        y0, new_state0 = module1(x[0], state[0])
        y1, new_state1 = module2(x[1], state[1])
        y = (y0, y1)
        new_state = (new_state0, new_state1)

    """

    def __init__(self, modules, input_tensor_spec=None, name="Parallel"):
        """
        Args:
            modules (nested nn.Module): a nest of ``torch.nn.Module`` or ``alf.nn.Network``.
            input_tensor_spec (nested TensorSpec): must be provided if it cannot
                be inferred from ``modules``.
            name (str):
        """
        if input_tensor_spec is None:
            input_tensor_spec = map_structure(get_input_tensor_spec, modules)
            specified = all(
                map(lambda s: s is not None, flatten(input_tensor_spec)))
            assert specified, (
                "input_tensor_spec needs "
                "to be specified if it cannot be infered from elements of "
                "networks")
        alf.nest.assert_same_structure_up_to(modules, input_tensor_spec)
        super().__init__(input_tensor_spec, name=name)
        networks = map_structure_up_to(modules, wrap_as_network, modules,
                                       input_tensor_spec)
        self._state_spec = map_structure(lambda net: net.state_spec, networks)
        if len(flatten(self._state_spec)) == 0:
            self._state_spec = ()
        self._networks = networks
        if alf.nest.is_nested(networks):
            # make it a nn.Module so its parameters can be picked up by the framework
            self._nets = alf.nest.utils.make_nested_module(networks)

    @property
    def state_spec(self):
        return self._state_spec

    def forward(self, inputs, state=()):
        if self._state_spec == ():
            output = map_structure_up_to(
                self._networks, lambda net, input: net(input)[0],
                self._networks, inputs)
        else:
            output_and_state = map_structure_up_to(
                self._networks, lambda net, input, s: net(input, s),
                self._networks, inputs, state)
            output = map_structure_up_to(self._networks, lambda os: os[0],
                                         output_and_state)
            state = map_structure_up_to(self._networks, lambda os: os[1],
                                        output_and_state)
        return output, state

    def copy(self, name=None):
        """Create a copy of this network or return the current instance.

        If ``self._singleton_instance`` is True, calling ``copy()`` will return
        ``self``; otherwise it will make a copy of all the modules and re-initialize
        their parameters.

        Args:
            name (str): name of the new network. Only used if not self._singleton_instance.
        Returns:
            Parallel:
        """
        if self._singleton_instance:
            return self

        if name is None:
            name = self.name

        networks = map_structure(lambda net: net.copy(), self._networks)
        return Parallel(networks, self._input_tensor_spec, name)

    @property
    def networks(self):
        return self._networks


class Branch(Network):
    """Apply multiple networks on the same input.

    Example:

    .. code-block:: python

        net = Branch((module1, module2))
        y, new_state = net(x, state)

    is equivalent to the following:

    .. code-block:: python

        y0, new_state0 = module1(x, state[0])
        y1, new_state1 = module2(x, state[1])
        y = (y0, y1)
        new_state = (new_state0, new_state1)

    """

    def __init__(self, modules, input_tensor_spec=None, name="Branch"):
        """
        Args:
            modules (nested nn.Module): a nest of ``torch.nn.Module`` or ``alf.nn.Network``.
            input_tensor_spec (nested TensorSpec): must be provided if it cannot
                be inferred from any one of ``modules``
            name (str):
        """
        if input_tensor_spec is None:
            specs = list(map(get_input_tensor_spec, alf.nest.flatten(modules)))
            specs = list(filter(lambda s: s is not None, specs))
            assert specs, ("input_tensor_spec needs to be specified since it "
                           "cannot be infered from any one of modules")
            for spec in specs:
                assert alf.utils.spec_utils.is_same_spec(spec, specs[0]), (
                    "modules have inconsistent input_tensor_spec: %s vs %s" %
                    (spec, specs[0]))
            input_tensor_spec = specs[0]
        super().__init__(input_tensor_spec, name=name)
        networks = map_structure(
            lambda net: wrap_as_network(net, input_tensor_spec), modules)
        self._state_spec = map_structure(lambda net: net.state_spec, networks)
        if len(flatten(self._state_spec)) == 0:
            self._state_spec = ()
        self._networks = networks
        self._networks_flattened = flatten(networks)
        if alf.nest.is_nested(networks):
            # make it a nn.Module so its parameters can be picked up by the framework
            self._nets = alf.nest.utils.make_nested_module(networks)

    @property
    def state_spec(self):
        return self._state_spec

    def forward(self, inputs, state=()):
        if self._state_spec == ():
            output = list(
                map(lambda net: net(inputs)[0], self._networks_flattened))
            output = pack_sequence_as(self._networks, output)
        else:
            state = flatten_up_to(self._networks, state)
            output_state = list(
                map(lambda net, s: net(inputs, s), self._networks_flattened,
                    state))
            output = pack_sequence_as(self._networks,
                                      [o for o, s in output_state])
            state = pack_sequence_as(self._networks,
                                     [s for o, s in output_state])
        return output, state

    def copy(self, name=None):
        """Create a copy of this network or return the current instance.

        If ``self._singleton_instance`` is True, calling ``copy()`` will return
        ``self``; otherwise it will make a copy of all the modules and re-initialize
        their parameters.

        Args:
            name (str): name of the new network. Only used if not self._singleton_instance.
        Returns:
            Branch:
        """
        if self._singleton_instance:
            return self

        if name is None:
            name = self.name

        networks = map_structure(lambda net: net.copy(), self._networks)
        return Branch(networks, self._input_tensor_spec, name)

    @property
    def networks(self):
        return self._networks
