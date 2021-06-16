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


class _LeanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, keywords, *args):
        # The last len(keywords) of args are keyword arguments for func.
        ctx.func = func
        ctx.keywords = keywords
        tensors = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
        ctx.args = tuple((isinstance(arg, torch.Tensor),
                          None if isinstance(arg, torch.Tensor) else arg)
                         for arg in args)
        ctx.save_for_backward(*tensors)
        if keywords:
            num_kwargs = len(keywords)
            kwargs = dict(zip(keywords, args[-num_kwargs:]))
            args = args[:-num_kwargs]
            return func(*args, **kwargs)
        else:
            return func(*args)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            tensors = [t.clone() for t in ctx.saved_tensors]
            func = ctx.func
            args = tuple(
                tensors.pop(0) if arg[0] else arg[1] for arg in ctx.args)
            tensors = tuple(arg for i, arg in enumerate(args)
                            if ctx.needs_input_grad[i + 2])
            keywords = ctx.keywords
            if keywords:
                num_kwargs = len(keywords)
                kwargs = dict(zip(keywords, args[-num_kwargs:]))
                args = args[:-num_kwargs]
                out = func(*args, **kwargs)
            else:
                out = func(*args)
        grads = list(torch.autograd.grad(out, tensors, grad_output))
        grads = tuple(
            grads.pop(0) if need else None for need in ctx.needs_input_grad)
        return grads


def lean_function(func):
    """Wrap ``func`` to save memory for backward.

    The returned function performs same computation as ``func``, but save memory
    by discarding intermediate results. It calculates the gradient by recomputing
    ``func`` using the same input during backward.

    Note: All the Tensor inputs to ``func`` must be explicitly listed as arguments
    of ``func``. For example, a tuple of Tensors as argument is not allowed.
    Another requirement for ``func`` is that it must be deterministic so that
    repeated evaluation with the same input will get same output. It is the
    responsibility of the user of this function to make sure that ``func``
    satifisies these requirements. ``lean_function`` will not report error if
    ``func`` does not satisfies these requirements and error will be silently
    ignored.

    Example:

    .. code-block:: python

        def myfunc(x, w, b, scale=1.0):
            return torch.sigmoid(scale * (x @ w) + b)

        lean_myfunc = lean_function(myfunc)

        y = lean_myfunc(x, w, b)

    Args:
        func (Callable): function to be wrapped.
    Returns:
        the wrapped function.
    """

    def wrapped(*args, **kwargs):
        # Function.apply does not allow keyword arguments, so we have to convert
        # all keyword arguments to positional arguments
        return _LeanFunction.apply(func, tuple(kwargs.keys()), *args,
                                   *tuple(kwargs.values()))

    return wrapped
