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

from absl import logging
import contextlib
import functools
import torch
import torch.nn as nn
import types
from typing import Callable

import alf
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
        assert num_parameters > 0 or len(tensors) > 0, (
            "No Tensor input for %s" % func)
        ctx.device = _infer_device_type(*tensors, *ctx.parameters)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device)
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
        elif isinstance(ret, tuple):
            assert all(isinstance(t, torch.Tensor) for t in ret), (
                "The return value of func must be a Tensor or a tuple of Tensors"
            )
        else:
            assert isinstance(ret, torch.Tensor), (
                "The return value of func must be a Tensor or a tuple of Tensors"
            )
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        device_module = _get_device_module(ctx.device)
        device_autocast_ctx = device_module.amp.autocast(
            **ctx.device_autocast_kwargs) if _supports_autocast(
                ctx.device) else contextlib.nullcontext()

        with torch.enable_grad(), device_autocast_ctx, \
                 torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
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
        3. It returns a decorated function or module so that all the original
           attributes and methods can still be accessed in the same way.

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
        if self._inside_lean_function or not torch.is_grad_enabled():
            return self._original_forward_for_lean_function(*args, **kwargs)
        else:
            return self._lean_function(*args, **kwargs)

    if isinstance(func, Network):
        specs = (func.output_spec, func.state_spec)

    def _wrapped_module(self, *args, **kwargs):
        # Note that the decorated module may be deepcopied. This means that we
        # should not directly access ``func`` or its attributes here, since they
        # will not be copied by copy.deepcopy. Instead, we should use ``self``
        # to access the attributes. ``self`` is passed through ``types.MethodType``
        # can will be correctly copied.
        parameters = tuple(self.parameters())
        # Function.apply does not allow keyword arguments, so we have to convert
        # all keyword arguments to positional arguments
        ret = _LeanFunction.apply(self, len(parameters), tuple(kwargs.keys()),
                                  *parameters, *args, *tuple(kwargs.values()))
        if isinstance(self, Network):
            ret = pack_sequence_as(specs, ret)
        return ret

    def _wrapped_func(func, *args, **kwargs):
        # Function.apply does not allow keyword arguments, so we have to convert
        # all keyword arguments to positional arguments
        ret = _LeanFunction.apply(func, 0, tuple(kwargs.keys()), *args,
                                  *tuple(kwargs.values()))
        return ret

    if isinstance(func, nn.Module):
        func._lean_function = types.MethodType(_wrapped_module, func)
        func._original_forward_for_lean_function = func.forward
        func._inside_lean_function = False
        func.forward = types.MethodType(_forward, func)
        return func
    else:
        return functools.partial(_wrapped_func, func)


# The following functions are copied from torch.utils.checkpoint.py. Since different
# versions of pytorch may have different implementations or lack some functions,
# we copy the functions here


def _get_autocast_kwargs(device="cuda"):
    if device == "cuda":
        device_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
    elif _supports_autocast(device):
        device_module = _get_device_module(device)
        device_autocast_kwargs = {
            "enabled": device_module.is_autocast_enabled(),
            "dtype": device_module.get_autocast_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
    else:
        device_autocast_kwargs = None

    cpu_autocast_kwargs = {
        "enabled": torch.is_autocast_cpu_enabled(),
        "dtype": torch.get_autocast_cpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    return device_autocast_kwargs, cpu_autocast_kwargs


def _supports_autocast(device):
    device_module = _get_device_module(device)
    return device == "cuda" or (hasattr(device_module, "is_autocast_enabled")
                                and hasattr(device_module,
                                            "get_autocast_dtype"))


def _get_device_module(device="cuda"):
    device_module = getattr(torch, device)
    return device_module


def _infer_device_type(*args):
    device_types = list({
        arg.device.type
        for arg in args
        if isinstance(arg, torch.Tensor) and not arg.device.type == "cpu"
    })
    if len(device_types) > 1:
        logging.warning(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices. "
            "Device state will only be saved for devices of a single device type, and the remaining "
            "devices will be ignored. Consequently, if any checkpointed functions involve randomness, "
            "this may result in incorrect gradients. (Note that if CUDA devices are among the devices "
            "detected, it will be prioritized; otherwise, the first device encountered will be selected.)"
        )
    if len(device_types) == 0:
        return alf.get_default_device()
    elif "cuda" in device_types:
        return "cuda"
    else:
        return device_types[0]


class SplitBatchRunner(torch.nn.Module):
    """Split the input into smaller batches and run the model on each batch.

    ``SplitBatchRunner`` uses ``LeanFunction`` to run the model on smaller batches
    so that the intermediate results can be discarded to save memory.

    Note that models using random number generators (e.g. DropOut) are not supported
    for training. For models involving calculating statistics on a batch, using
    SplitBatchRunner may lead to different results.

    Example:

    .. code-block:: python

            model = SplitBatchRunner(model, max_batch_size=256)
            output = model(input, non_batched_inputs={'mask': mask})

    Args:
        model (nn.Module): the model to run
        max_batch_size (int): the maximum batch size to run the model. If negative,
            the model will not be split into smaller batches and the original
            forward of the model will be called.
    """

    def __init__(self, model, max_batch_size):
        super().__init__()
        self._model = lean_function(model)
        self._max_batch_size = max_batch_size

    def forward(self, *args, non_batched_inputs={}, **kwargs):
        """Run the model on the input.

        Args:
            args (tuple): positional arguments for the model
            non_batched_inputs (dict): non-batched inputs for the model
            kwargs (dict): keyword arguments for the model
        """
        batch_size = alf.nest.get_nest_batch_size((args, kwargs))
        if self._max_batch_size <= 0 or batch_size <= self._max_batch_size:
            return self._model._original_forward_for_lean_function(
                *args, **kwargs, **non_batched_inputs)

        outputs = []
        for i in range(0, batch_size, self._max_batch_size):
            batch_args, batch_kwargs = alf.nest.map_structure(
                lambda x: x[i:i + self._max_batch_size], (args, kwargs))
            outputs.append(
                self._model(*batch_args, **batch_kwargs, **non_batched_inputs))

        return alf.nest.map_structure(lambda *x: torch.cat(x, dim=0), *outputs)

    def original_forward(self, *args, **kwargs):
        """Run the model on the input without splitting the batch."""
        return self._model._original_forward_for_lean_function(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)
