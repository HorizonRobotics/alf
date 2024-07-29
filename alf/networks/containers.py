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
import torch.nn as nn
from typing import Callable

import alf
from alf.nest import (flatten, flatten_up_to, get_field, map_structure,
                      map_structure_up_to, pack_sequence_as)
from alf.nest.utils import get_nested_field
from alf.utils.spec_utils import is_same_spec
from .network import Network, get_input_tensor_spec, wrap_as_network
from alf.layers import make_parallel_spec


def Sequential(*modules,
               output='',
               input_tensor_spec=None,
               name="Sequential",
               **named_modules):
    """Network composed of a sequence of torch.nn.Module or alf.nn.Network.

    All the modules provided through ``modules`` and ``named_modules`` are calculated
    sequentially in the same order as they appear in the call to ``Sequential``.
    Typically, each module takes the result of the previous module as its input
    (or the input to the Sequential if it is the first module), and the result of
    the last module is the output of the Sequential. But we also allow more
    flexibilities as shown in example 2.

    Example 1:

    .. code-block:: python

        net = Sequential(module1, module2)
        y, new_state = net(x, state)

    is equivalent to the following:

    .. code-block:: python

        z, new_state1 = module1(x, state[0])
        y, new_state2 = module2(z, state[1])
        new_state = (new_state1, new_state2)

    Example 2:

    .. code-block:: python

        net = Sequential(
            module1, a=module2, b=(('input', 'a'), module3), output=('a', 'b'))
        output, new_state = net(input, state)

    is equivalent to the following:

    .. code-block:: python

        _, new_state1 = module1(input, state[0])
        a, new_state2 = module2(_, state[1])
        b, new_state3 = module3((input, a), state[2])
        new_state = (new_state1, new_state2, new_state3)
        output = (a, b)


    Args:
        modules (Callable | (nested str, Callable)):
            The ``Callable`` can be a ``torch.nn.Module``, ``alf.nn.Network``
            or plain ``Callable``. Optionally, their inputs can be specified
            by the first element of the tuple. If input is not provided, it is
            assumed to be the result of the previous module (or input to this
            ``Sequential`` for the first module). If input is provided, it
            should be a nested str. It will be used to retrieve results from
            the dictionary of the current ``named_results``. For modules
            specified by ``modules``, because no ``named_modules`` has been
            invoked, ``named_results`` is ``{'input': input}``.
        named_modules (Callable | (nested str, Callable)):
            The ``Callable`` can be a ``torch.nn.Module``, ``alf.nn.Network``
            or plain ``Callable``. Optionally, their inputs can be specified
            by the first element of the tuple. If input is not provided, it is
            assumed to be the result of the previous module (or input to this
            ``Sequential`` for the first module). If input is provided, it
            should be a nested str. It will be used to retrieve results from
            the dictionary of the current ``named_results``. ``named_results``
            is updated once the result of a named module is calculated.
        output (nested str): if not provided, the result from the last module
            will be used as output. Otherwise, it will be used to retrieve
            results from ``named_results`` after the results of all modules
            have been calculated.
        input_tensor_spec (TensorSpec): the tensor spec of the input. It must
            be specified if it cannot be inferred from ``modules[0]``.
        name (str):
    """
    # The reason that we use a wrapper function for _Sequential is that Network
    # does not allow *args for __init__() (see _NetworkMeta.__new__()). And we
    # want to use *modules here to make the interface consistent with
    # torch.nn.Sequential and alf.layers.Sequential to avoid confusion.
    return _Sequential(
        modules,
        named_modules,
        output=output,
        input_tensor_spec=input_tensor_spec,
        name=name)


class _Sequential(Network):
    def __init__(self,
                 elements=(),
                 element_dict={},
                 output='',
                 input_tensor_spec=None,
                 name='Sequential'):
        state_spec = []
        modules = []
        inputs = []
        outputs = []
        simple = True
        named_elements = list(zip([''] * len(elements), elements)) + list(
            element_dict.items())
        is_nested_str = lambda s: all(
            map(lambda x: type(x) == str, flatten(s)))
        for i, (out, element) in enumerate(named_elements):
            input = ''
            if isinstance(element, tuple) and len(element) == 2:
                input, module = element
            else:
                module = element
            if not (isinstance(module, Callable) and is_nested_str(input)):
                raise ValueError(
                    "Argument %s is not in the form of Callable "
                    "or (nested str, Callable): %s" % (out or str(i), element))
            if isinstance(module, type):
                raise ValueError(
                    "module should not be a type. Did you forget "
                    "to include '()' after it to construct the layer? module=%s"
                    % str(module))
            if isinstance(module, Network):
                state_spec.append(module.state_spec)
            else:
                state_spec.append(())
            inputs.append(input)
            outputs.append(out)
            modules.append(module)
            if out or input:
                simple = False
        if output:
            simple = False
        assert is_nested_str(output), (
            "output should be a nested str: %s" % output)
        if len(flatten(state_spec)) == 0:
            state_spec = ()
        if input_tensor_spec is None and not inputs[0]:
            input_tensor_spec = get_input_tensor_spec(modules[0])
        assert input_tensor_spec is not None, (
            "input_tensor_spec needs to be provided")
        super().__init__(input_tensor_spec, state_spec=state_spec, name=name)
        self._networks = modules
        # pytorch nn.Moddule needs to use ModuleList to keep track of parameters
        self._nets = nn.ModuleList(
            filter(lambda m: isinstance(m, nn.Module), modules))
        if simple:
            self.forward = self._forward_simple
        else:
            self.forward = self._forward_complex
        self._output = output
        self._inputs = inputs
        self._outputs = outputs

    def _forward_simple(self, input, state=()):
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

    def _forward_complex(self, input, state=()):
        x = input
        var_dict = {'input': x}
        if self._state_spec == ():
            for i, net in enumerate(self._networks):
                if self._inputs[i]:
                    x = get_nested_field(var_dict, self._inputs[i])
                if isinstance(net, Network):
                    x = net(x)[0]
                else:
                    x = net(x)
                if self._outputs[i]:
                    var_dict[self._outputs[i]] = x
            new_state = state
        else:
            new_state = [()] * len(self._networks)
            for i, net in enumerate(self._networks):
                if self._inputs[i]:
                    x = get_nested_field(var_dict, self._inputs[i])
                if isinstance(net, Network):
                    x, new_state[i] = net(x, state[i])
                else:
                    x = net(x)
                if self._outputs[i]:
                    var_dict[self._outputs[i]] = x
        if self._output:
            x = get_nested_field(var_dict, self._output)
        return x, new_state

    def __getitem__(self, i):
        return self._networks[i]

    def make_parallel(self, n: int):
        """Create a parallelized version of this network.

        Args:
            n (int): the number of copies
        Returns:
            the parallelized version of this network
        """
        new_networks = []
        new_named_networks = {}
        for net, input, output in zip(self._networks, self._inputs,
                                      self._outputs):
            pnet = alf.layers.make_parallel_net(net, n)
            if not output:
                new_networks.append((input, pnet))
            else:
                new_named_networks[output] = (input, pnet)
        input_spec = make_parallel_spec(self._input_tensor_spec, n)
        return _Sequential(new_networks, new_named_networks, self._output,
                           input_spec, "parallel_" + self.name)


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
                "to be specified if it cannot be inferred from elements of "
                "networks")
        alf.nest.assert_same_structure_up_to(modules, input_tensor_spec)
        networks = map_structure_up_to(modules, wrap_as_network, modules,
                                       input_tensor_spec)
        state_spec = map_structure(lambda net: net.state_spec, networks)
        if len(flatten(state_spec)) == 0:
            state_spec = ()
        super().__init__(input_tensor_spec, state_spec=state_spec, name=name)
        self._networks = networks
        if alf.nest.is_nested(networks):
            # make it a nn.Module so its parameters can be picked up by the framework
            self._nets = alf.nest.utils.make_nested_module(networks)

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

    @property
    def networks(self):
        return self._networks

    def make_parallel(self, n: int):
        """Create a parallelized version of this network.

        Args:
            n (int): the number of copies
        Returns:
            the parallelized version of this network
        """
        networks = map_structure(
            lambda net: alf.layers.make_parallel_net(net, n), self._networks)
        input_spec = make_parallel_spec(self._input_tensor_spec, n)
        return Parallel(networks, input_spec, 'parallel_' + self.name)


def Branch(*modules, input_tensor_spec=None, name="Branch", **named_modules):
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

    Args:
        modules (nested nn.Module | Callable): a nest of ``torch.nn.Module``
            ``alf.nn.Network`` or ``Callable``. Note that ``Branch(module_a, module_b)``
            is equivalent to ``Branch((module_a, module_b))``
        named_modules (nn.Module | Callable): a simpler way of specifying
            a dict of modules. ``Branch(a=model_a, b=module_b)``
            is equivalent to ``Branch(dict(a=module_a, b=module_b))``
        input_tensor_spec (nested TensorSpec): must be provided if it cannot
            be inferred from any one of ``modules``
        name (str):
    """
    # The reason that we use a wrapper function for _Branch is that Network
    # does not allow *args for __init__() (see _NetworkMeta.__new__()).
    return _Branch(
        modules, named_modules, input_tensor_spec=input_tensor_spec, name=name)


class _Branch(Network):
    def __init__(self,
                 modules,
                 named_modules,
                 input_tensor_spec=None,
                 name="Branch"):
        if modules:
            assert not named_modules
            if len(modules) == 1:
                modules = modules[0]
        else:
            modules = named_modules
        if input_tensor_spec is None:
            specs = list(map(get_input_tensor_spec, alf.nest.flatten(modules)))
            specs = list(filter(lambda s: s is not None, specs))
            assert specs, ("input_tensor_spec needs to be specified since it "
                           "cannot be inferred from any one of modules")
            for spec in specs:
                assert alf.utils.spec_utils.is_same_spec(spec, specs[0]), (
                    "modules have inconsistent input_tensor_spec: %s vs %s" %
                    (spec, specs[0]))
            input_tensor_spec = specs[0]
        networks = map_structure(
            lambda net: wrap_as_network(net, input_tensor_spec), modules)
        state_spec = map_structure(lambda net: net.state_spec, networks)
        if len(flatten(state_spec)) == 0:
            state_spec = ()
        super().__init__(input_tensor_spec, state_spec=state_spec, name=name)
        self._networks = networks
        self._networks_flattened = flatten(networks)
        if alf.nest.is_nested(networks):
            # make it a nn.Module so its parameters can be picked up by the framework
            self._nets = alf.nest.utils.make_nested_module(networks)

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

    @property
    def networks(self):
        return self._networks

    def make_parallel(self, n: int):
        """Create a parallelized version of this network.

        Args:
            n (int): the number of copies
        Returns:
            the parallelized version of this network
        """
        networks = map_structure(
            lambda net: alf.layers.make_parallel_net(net, n), self._networks)
        input_spec = make_parallel_spec(self._input_tensor_spec, n)
        return Branch(
            networks,
            input_tensor_spec=input_spec,
            name='parallel_' + self.name)


class Echo(Network):
    """Echo network.

    Echo network uses part of the output of ``block`` of current step as part of
    the input of ``block`` for the next step. In particular, if the input of ``block``
    is a dictionary, it should contains two keys 'input' and 'echo', and 'echo'
    will be taken from the output of the previous step. If the input of ``block``
    is a tuple, the second input will be taken from the output of the previous step.
    If the output is a dictionary, it should contains two keys 'output' and 'echo',
    and 'echo' will be used as the input for the next step. If the output is a tuple,
    the second output will be used as the input for the next step.

    Note that ``block`` itself can be a recurrent network with state.

    Examples:

    .. code-block:: python

        echo = Echo(block)
        output, state = echo(real_input, state)

    is equivalent to the following if the input and output of block are dicts:

    .. code-block:: python

        block_state, echo_input = state
        block_output, block_state = block(dict(input=real_input, echo=echo_input), block_state)
        output = block_output['output']
        echo_output = block_output['echo']
        state = (block_state, echo_output)

    and is equivalent to the following if the input and output of block are tuples:

    .. code-block:: python

        block_state, echo_input = state
        block_output, block_state = block((real_input, echo_input), block_state)
        output, echo_output = block_output
        state = (block_state, echo_output)

    """

    def __init__(self, block, input_tensor_spec=None):
        """
        Args:
            block (Network): the module for performing the actual computation
            input_tensor_spec (nested TensorSpec): If provided, it must match
                the ``block.input_tensor_spec[0]`` or ``block.input_tensor_spec['input']``
        """
        assert isinstance(
            block, Network), ("block must be an instance of "
                              "alf.networks.Network. Got %s" % type(block))
        if (isinstance(block.input_tensor_spec, tuple)
                and len(block.input_tensor_spec) == 2):
            self._is_tuple_input = True
            real_input_spec, echo_input_spec = block.input_tensor_spec
        elif (isinstance(block.input_tensor_spec, dict)
              and len(block.input_tensor_spec) == 2
              and 'input' in block.input_tensor_spec
              and 'echo' in block.input_tensor_spec):
            self._is_tuple_input = False
            real_input_spec = block.input_tensor_spec['input']
            echo_input_spec = block.input_tensor_spec['echo']
        else:
            raise ValueError(
                "block.input_tensor_spec should be a tuple with "
                "two elements or a dict with two keys 'input' and 'echo': %s" %
                block.input_tensor_spec)

        if (isinstance(block.output_spec, tuple)
                and len(block.output_spec) == 2):
            self._is_tuple_output = True
            echo_output_spec = block.output_spec[1]
        elif (isinstance(block.output_spec, dict)
              and len(block.output_spec) == 2 and 'output' in block.output_spec
              and 'echo' in block.output_spec):
            self._is_tuple_output = False
            echo_output_spec = block.output_spec['echo']
        else:
            raise ValueError(
                "block.output_spec should be a tuple with "
                "two elements or a dict with two keys 'output' and 'echo': %s"
                % block.output_spec)
        assert is_same_spec(echo_input_spec, echo_output_spec), (
            "echo input and echo output should have same spec: %s vs. %s" %
            (echo_input_spec, echo_output_spec))

        if input_tensor_spec is not None:
            assert is_same_spec(real_input_spec, input_tensor_spec), (
                "input_tensor_spec is not same as real_input_spec: %s vs. %s" %
                (input_tensor_spec, real_input_spec))

        state_spec = (block.state_spec, echo_input_spec)
        super().__init__(
            input_tensor_spec=real_input_spec, state_spec=state_spec)
        self._block = block

    def forward(self, input, state):
        block_state, echo_state = state
        if self._is_tuple_input:
            block_input = (input, echo_state)
        else:
            block_input = dict(input=input, echo=echo_state)
        block_output, block_state = self._block(block_input, block_state)
        if self._is_tuple_output:
            real_output, echo_output = block_output
        else:
            real_output = block_output['output']
            echo_output = block_output['echo']
        return real_output, (block_state, echo_output)

    def make_parallel(self, n: int):
        """Create a parallelized version of this network.

        Args:
            n (int): the number of copies
        Returns:
            the parallelized version of this network
        """
        return Echo(
            alf.layers.make_parallel_net(self._block),
            make_parallel_spec(self._input_tensor_spec, n))
