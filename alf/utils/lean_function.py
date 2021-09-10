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

import torch
import torch.nn as nn
import types
from typing import Callable

from alf.nest import flatten, pack_sequence_as
from alf.networks import Network


class _LeanFunction(torch.autograd.Function):
    # Reference: Pytorch: Defining new Autograd Functions
    # https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    @staticmethod
    def forward(ctx, func, num_parameters, keywords, *args):
        """
        Args:
            ctx (_LeanFunction): context of the computation. It is the same object
                passed for the corresponding backward().
            func (Callable): func/module to be wrapped
            num_parameters (int): the number of nn.Parameters of func if it is an
                nn.Module. 0 otherwise. If ``func`` is a module, the first
                ``num_parameters`` of arguments in args are the parameters of ``func``.
            keywords (tuple of str): the name of the keys of the keyword arguments
                for ``func``
            args (Any): all the arguments (positional and keyword) for ``func``.
        """
        # The last len(keywords) of args are keyword arguments for func.
        ctx.func = func
        ctx.keywords = keywords
        ctx.parameters = args[:num_parameters]
        args = args[num_parameters:]
        tensors = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
        ctx.args = tuple((isinstance(arg, torch.Tensor),
                          None if isinstance(arg, torch.Tensor) else arg)
                         for arg in args)
        ctx.save_for_backward(*tensors)
        func._inside_lean_function = True
        if keywords:
            num_kwargs = len(keywords)
            kwargs = dict(zip(keywords, args[-num_kwargs:]))
            args = args[:-num_kwargs]
            ret = func(*args, **kwargs)
        else:
            ret = func(*args)
        func._inside_lean_function = False

        # torch.autograd.Function only allows the return value to be a tuple or
        # a tuple of Tensors. So we need to convert output to a tuple of Tensors
        # and convert back in _wrapped(). This is possible for Network because
        # it can get the information about the format of the output. For other
        # types of func, if ret is not a Tensor or tuple of Tensors, pytorch will
        # report an error.
        if isinstance(func, Network):
            ret = tuple(flatten(ret))
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            # saved_tensors is the tensors passed for ctx.save_for_backward
            tensors = list(ctx.saved_tensors)
            func = ctx.func
            parameters = ctx.parameters
            num_parameters = len(parameters)
            args = tuple(
                tensors.pop(0) if arg[0] else arg[1] for arg in ctx.args)
            tensors = tuple(arg for i, arg in enumerate(args)
                            if ctx.needs_input_grad[3 + num_parameters + i])
            keywords = ctx.keywords
            func._inside_lean_function = True
            if keywords:
                num_kwargs = len(keywords)
                kwargs = dict(zip(keywords, args[-num_kwargs:]))
                args = args[:-num_kwargs]
                out = func(*args, **kwargs)
            else:
                out = func(*args)
            func._inside_lean_function = False
        if isinstance(func, Network):
            out = tuple(flatten(out))
        grads = list(
            torch.autograd.grad(out, parameters + tensors, grad_output))
        grads = tuple(
            grads.pop(0) if need else None for need in ctx.needs_input_grad)
        return grads


def lean_function(func: Callable) -> Callable:
    """Wrap ``func`` to save memory for backward.

    The returned function performs same computation as ``func``, but save memory
    by discarding intermediate results. It calculates the gradient by recomputing
    ``func`` using the same input during backward.

    Note: There are several requirements for ``func``:

        1. All the Tensor inputs to ``func`` must be explicitly listed as arguments
          of ``func``. For example, a tuple of Tensors as argument is not allowed.
          Using Tensors outside of ``func`` (e.g., tensors from class member variables)
          is not allowed either unless ``func`` is a ``nn.Module``. On the other
          hand, if ``func`` is a module, its parameters should not be put as arguments
          as they are automatically taken care of.

        2. If ``func`` is not a ``Network``, its return value must be a Tensor
          or a tuple of Tensors. If it is a ``Network``, its return value (output
          and state) must be a nest of Tensors.

        3. ``func```` must be deterministic so that repeated evaluation with the
          same input will get same output.

        It is the responsibility of the user of this function to make sure that
         ``func`` satifisies these requirements. ``lean_function`` will not report
         error if ``func`` does not satisfies these requirements and error will
         be silently ignored.

    Note: pytorch also has a function with similar functionality. See https://pytorch.org/docs/stable/checkpoint.html
    for detail. ``lean_function`` has several advantage over pytorch's implementation:

        1. Keyword arguments are supported.
        2. Both ``torch.autograd.grad`` and ``torch.autograd.backward`` are supported.

    Examples:

    1. Apply to simple function:

    .. code-block:: python

        def myfunc(x, w, b, scale=1.0):
            return torch.sigmoid(scale * (x @ w) + b)

        lean_myfunc = lean_function(myfunc)

        y = lean_myfunc(x, w, b)

    2. Apply to nn.Module:

    .. code-block:: python

        module = alf.layers.FC(3, 5, activation=torch.relu_)
        lean_func = lean_function(module)
        y = lean_func(x)

    3. Apply to a network

    .. code-block:: python

        net = alf.nn.Sequential(
            alf.layers.FC(3, 5, activation=torch.relu_),
            alf.layers.FC(5, 1, activation=torch.sigmoid))
        lean_func = lean_function(net)
        y = lean_func(x)

    Args:
        func: function or module to be wrapped.
    Returns:
        the wrapped function or module. In the case of ``func`` being a ``nn.Module``,
        all the original attributes and methods can still be accessed in the same
        way through the wrapped module.
    """

    def _forward(self, *args, **kwargs):
        if self._inside_lean_function:
            return self._original_forward_for_lean_function(*args, **kwargs)
        else:
            return self._lean_function(*args, **kwargs)

    parameters = ()
    if isinstance(func, nn.Module):
        parameters = tuple(func.parameters())
    if isinstance(func, Network):
        specs = (func.output_spec, func.state_spec)

    def _wrapped(*args, **kwargs):
        # Function.apply does not allow keyword arguments, so we have to convert
        # all keyword arguments to positional arguments
        ret = _LeanFunction.apply(func, len(parameters), tuple(kwargs.keys()),
                                  *parameters, *args, *tuple(kwargs.values()))
        if isinstance(func, Network):
            ret = pack_sequence_as(specs, ret)
        return ret

    if isinstance(func, nn.Module):
        func._lean_function = _wrapped
        func._original_forward_for_lean_function = func.forward
        func._inside_lean_function = False
        func.forward = types.MethodType(_forward, func)
        return func
    else:
        return _wrapped
