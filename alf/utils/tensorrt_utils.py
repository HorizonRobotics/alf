# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import os

import torch.onnx
import torch

import numpy as np
import functools
import types
from typing import Tuple, Optional, Callable, Dict, Any
import io
try:
    import onnx
    from onnx import shape_inference
    import onnxruntime.backend as backend
except ImportError:
    onnx = None

import alf
from alf.algorithms.algorithm import Algorithm
from alf.utils import dist_utils

# How to install dependencies (in a virtual env) on the deployment machine:
# ```bash
#   pip install onnx
#
#   pip install tensorrt
# After this, tensorrt .so files will be installed in your virtual env: .../site-packages/tensorrt_lib/
# You need to copy all .so files under it to /usr/local/cuda/targets/x86_64-linux/lib/
# Or alternatively, you can install it system-wide from https://developer.nvidia.com/tensorrt-getting-started

# For cuda 11.x,
#   pip install onnxruntime-gpu
# For cuda 12.x,
#   pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# NOTE: onnxruntime already supports tensorRT backend, please see
# https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
# By default, it will use available providers:
# https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/backend/backend.py#L116
# The available provider list is:
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# The order of elements represents the default priority order of Execution Providers from highest to lowest.


def is_available():
    return onnx is not None


class _OnnxWrapper(torch.nn.Module):
    def __init__(self,
                 module: torch.nn.Module,
                 method: Callable,
                 example_args: Tuple[Any] = (),
                 example_kwargs: Dict[str, Any] = {}):
        """A wrapper class that prepares exporting a ``module.method`` to an ONNX
        model. It transforms the method to a ``nn.Module`` which can be exported
        to ONNX using ``torch.onnx.export``.

        There are mainly several preparation steps:

        1. Put ``module`` in eval mode and set requires_grad to False.
        2. Remove any 'optimizer' attributes in the ``module``, otherwise ``torch.jit.trace``
           will report errors.
        3. ONNX assumes flat arg list. So we need to pack the flat args and kwargs
           back to the original nest structures before calling ``module.method``.
        4. [Hacks]: make sure all inputs participate in the ONNX graph; make sure
           outputs contain no duplicate tensors. The reason is that JIT.trace will
           only trace inputs that are used in the graph, and create inputs for ONNX
           model. If we have a nested input, the nest leaves will be selected by
           JIT.trace depending on their usage. For example, if only TimeStep.observation
           and TimeStep.reward are used, then in the ONNX model, there are only
           two input nodes. When running the ONNX model, we need to manually
           extract the corresponding inputs and send only the two to the model.
           However, it will be difficult for us to know which leaves are actually
           used in advance. So the hack here is to just assume all inputs are used
           so we don't have to resolve the correspondence issue.
        5. Flatten the method output before returning so that ONNX won't flatten it
           according to its own defined order. Later we will recover the original
           output structure using ``self.recover_module_output()``.

        Args:
            module: The module to be exported.
            method: A method in ``module`` to be exported.
            example_args: The example args to be used for ``method``. Should be
                a tuple of args.
            example_kwargs: The example kwargs to be used for ``method``. Should be
                a dict of kargs.
        """
        super().__init__()
        _OnnxWrapper._strip_optimizers(module)
        # freeze the module by not requiring grad and setting eval mode
        for param in module.parameters():
            param.requires_grad = False
        module.eval()
        self._module = module
        self._method = functools.partial(method, self._module)
        self._example_args = example_args
        self._example_kwargs = example_kwargs
        # Sometimes we will return Distributions in the output. In this case,
        # we need to convert them to
        example_output = self._method(*example_args, **example_kwargs)
        self._output_params_spec = dist_utils.extract_spec(
            dist_utils.distributions_to_params(example_output))
        self._output_spec = dist_utils.extract_spec(example_output)

    @staticmethod
    def _strip_optimizers(module):
        """Recursively set all sub-algorithms' optimizers to None as they are not
        supported by onnx (torch.jit.trace).

        This operation is in-place.
        """
        for m in module.modules():
            if isinstance(m, Algorithm):
                m._optimizers = []
                m._default_optimizer = None

    def recover_module_output(self, forward_output):
        """``forward_output`` is a direct return of ``self.forward()``.
        """
        # remove the dummy output as the second one
        forward_output = list(forward_output)[:-1]
        output_nest = alf.nest.py_pack_sequence_as(self._output_params_spec,
                                                   forward_output)
        output = dist_utils.params_to_distributions(output_nest,
                                                    self._output_spec)
        return output

    @torch.no_grad()
    def forward(self, *flat_all_args):
        """onnx will convert whatever arg structure into a tuple. So we need to
        flatten and pack it back into the original structure.

        For now, we assume the inputs contain no keyword arguments.
        """
        ###############################
        # HACK: we generate a dummy output for onnx to record all tensor inputs
        # TODO: find a better way to figure out which args are actually used
        # in the onnx graph, and pass this info to the run() of tensorRT engine.
        dummy_output = sum([
            a.float().mean() for a in flat_all_args
            if isinstance(a, torch.Tensor)
        ])
        ###############################

        flat_args = flat_all_args[:len(alf.nest.flatten(self._example_args))]
        flat_kwargs = flat_all_args[len(flat_args):]

        args = alf.nest.pack_sequence_as(self._example_args, flat_args)
        kwargs = alf.nest.pack_sequence_as(self._example_kwargs, flat_kwargs)

        output = self._method(*args, **kwargs)
        # Need to convert any leaf element of ``alg_step.info`` into tensor
        output_params = dist_utils.distributions_to_params(output)

        ###############################
        # HACK: Sometimes we will return duplicate tensors in the output. In this case,
        # we need to convert duplicate tensors to be different nodes in the graph.
        # Otherwise ONNX will just deduplicate them, which makes the total number
        # of output nodes different from what we've recorded in ``_output_spec``.
        # Using .clone() doesn't help as ONNX will optimize it away.
        # TODO: test whether a tensor is a duplicate. If so, remove it from output_params.
        # And record this information to self and recover at recover_module_output
        output_params = alf.nest.map_structure(
            lambda x: x + torch.zeros_like(x), output_params)
        ###############################

        # We want to use ALF's flatten to avoid ONNX's defined flattening order
        output_params = alf.nest.flatten(output_params)
        return output_params, dummy_output


@alf.configurable(whitelist=['device'])
class TensorRTEngine(torch.nn.Module):
    def __init__(self,
                 module: torch.nn.Module,
                 method: Callable,
                 onnx_file: Optional[str] = None,
                 onnx_verbose: bool = False,
                 device: str = None,
                 example_args: Tuple[Any] = (),
                 example_kwargs: Dict[str, Any] = {}):
        """Class for converting a torch.nn.Module to TensorRT engine for fast
        inference, via ONNX model as the intermediate representation.

        NOTE: if ``tensorrt`` lib is not installed, this backend will fall back
        to use CUDA. If GPU is not available, this backend will fall back to CPU.
        So the class name might not be accurate. But since its main purpose is
        using tensorRT for inference, we keep the name as it is.

        This class is mainly responsible for:

        1. Flatten all args and kwargs before passing it to the engine.
        2. Convert tensor inputs to numpy inputs for the inference engine.
        3. Convert the outputs of the engine to tensor outputs.
        4. Call ``_OnnxWrapper.recover_module_output`` to recover the original
           structure of the output.

        Args:
            module: The module to be converted.
            method: A method in ``module`` to be converted.
            onnx_file: The path to the onnx model. If None, no external file will
                be created. Instead, the onnx model will be created in memory.
            onnx_verbose: If True, the onnx model exporting process will output
                verbose information.
            device: The device used by TensorRT to run the onnx model. If None,
                will set to 'CUDA' if torch.cuda.is_available(), otherwise will be
                set to 'CPU'.
            example_args: The example args to be used for ``method``. Should be
                a tuple of args.
            example_kwargs: The example kwargs to be used for ``method``. Should
                be a dict of kwargs.
        """
        super().__init__()
        self._onnx_wrapper = _OnnxWrapper(module, method, example_args,
                                          example_kwargs)

        flat_all_args = alf.nest.flatten([example_args, example_kwargs])
        flat_all_args = self._dtype_conversions(flat_all_args)

        if onnx_file is None:
            onnx_io = io.BytesIO()
        else:
            onnx_io = onnx_file
        # 'args' must be a tuple of tensors
        torch.onnx.export(
            self._onnx_wrapper,
            args=flat_all_args,
            f=onnx_io,
            # Don't modify the version easily! Other versions might
            # have weird errors.
            opset_version=12,
            verbose=onnx_verbose)
        if isinstance(onnx_io, io.BytesIO):
            onnx_io.seek(0)
        onnx_model = onnx.load(onnx_io)

        # Infer shapes first to avoid the error: "Please run shape inference on the onnx model first."
        # from TensorRT
        onnx_model = shape_inference.infer_shapes(onnx_model)
        if device is None:
            device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        else:
            device = device.upper()
        self._engine = backend.prepare(onnx_model, device=device)

    def _dtype_conversions(
            self, flat_all_args: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        # Convert all uint8 tensors to int32, float64 to float32, because tensorRT
        # does not support uint8 or float64 tensors
        dtype_mappings = {
            torch.uint8: torch.int32,
            torch.float64: torch.float32
        }

        def _convert_if_needed(x):
            if x.dtype in dtype_mappings:
                return x.to(dtype=dtype_mappings[x.dtype])
            return x

        return tuple([_convert_if_needed(a) for a in flat_all_args])

    def forward(self, *args, **kwargs):
        flat_all_args = self._dtype_conversions(
            alf.nest.flatten([args, kwargs]))
        flat_all_args_np = [x.detach().cpu().numpy() for x in flat_all_args]
        # engine accepts a list of args, not a tuple!
        outputs_np = self._engine.run(flat_all_args_np)
        # torch.from_numpy shares the memory with the numpy array
        outputs = alf.nest.map_structure(
            lambda x: torch.from_numpy(x).to(alf.get_default_device()),
            outputs_np)
        return self._onnx_wrapper.recover_module_output(outputs)


def tensorrt_for_inference_if(cond: bool = True):
    """A decorator to compile a method as a tensorRT engine for inference,
    when ``cond`` is true.

    This decorator should be used on a torch.nn.Module member function.

    If will first wrap the object's method to export to onnx, and then create a
    tensorrt engine from the onnx model. The engine will be cached in the object.
    """

    def _decorator(method):

        if not cond:
            return method

        @functools.wraps(method)
        def wrapped(module_to_wrap, *args, **kwargs):

            # The first argument to the method is going to be ``self``, i.e. the
            # instance that the method belongs to. By accessing it we get the
            # reference of the module to wrap.
            assert isinstance(module_to_wrap, torch.nn.Module), (
                f'Cannot apply @tensorrt_for_inference_if on {type(module_to_wrap)}'
            )

            if not hasattr(module_to_wrap, '_tensorrt_engine_map'):
                module_to_wrap._tensorrt_engine_map = {}
            n_args = len(args)
            arg_keys = tuple(kwargs.keys())
            engine_key = (id(method), n_args, arg_keys)

            tensorrt_engine = module_to_wrap._tensorrt_engine_map.get(
                engine_key, None)
            if tensorrt_engine is None:
                tensorrt_engine = TensorRTEngine(
                    module_to_wrap,
                    method,
                    example_args=args,
                    example_kwargs=kwargs)
                module_to_wrap._tensorrt_engine_map[
                    engine_key] = tensorrt_engine
                alf.utils.common.info(
                    "Created a new TensorRT inference engine for "
                    f"'{module_to_wrap.__class__.__name__}.{method.__name__}' "
                    f"with key '{engine_key}'")
            return tensorrt_engine(*args, **kwargs)

        return wrapped

    return _decorator


_tensorrtified_methods = {}


def tensorrtify_method(module, method_name):
    """Convert a module method to use tensorrt inference at runtime. For example,

    .. code-block:: python

        if deploy_mode:
            tensorrtify_method(agent, 'predict_step')

        agent.predict_step(...)  # slow: prepare tensorrt engine for the first time
        agent.predict_step(...)  # fast inference after the first call

    There is also an environment variable "ALF_ENABLE_TENSORRT" to globally turn
    on or off of this function. If off, no method will be changed by this function.

    .. note::

        If the method output contains any value that is not a direct function
        of the input tensors, its value will always be fixed no matter how many
        times the compiled method is called. That's because those "constant" values
        have been harded-coded in the onnx model graph.

        For example, our ``PerfTimer`` will measure the time spent in a code block,
        and fill the time value in the output alg_step.info.
        Since the time spent in the code block is not a function of the input tensors,
        this value will be fixed no matter how many times the method is called.
        (TODO: make PerfTimer a pytorch operator.)

    .. note::

        After converting a method using this function, it's always suggested to
        call the method with the consistently same list of *args and **kwargs.
        For example, we always call ``agent.predict_step(timestep, state=state)``,
        where ``args=(timestep, )`` and ``kwargs={'state': state}``.

        The reason is that tensorrt/onnx has a strict rule for argument number and
        order. when a tensorrt engine is created and stored in a map, we make the
        argument list also part of the key. So if next time the same method is called
        with a different argument list such as ``args=(timestep, state)`` and
        ``kwargs={}``, a new engine will be created again.

    .. note::
        In order to use this function, there are certain limits on how the eager
        mode code should be written. For a complete list of restrictions and solutions,
        please see ``README.md``.

    Args:
        module: a torch.nn.Module
        method_name: the method name of the module
    """
    assert is_available(), 'ONNX/TensorRT is not installed!'

    global _tensorrtified_methods
    key = (module, method_name)
    # Here we check if a previous ``tensorrtify_method`` has already been called
    # for the same module and method. We do not allow tensorrtifying a method
    # a multiple times.
    assert key not in _tensorrtified_methods, (
        f"Method {module}.{method_name} is already tensorrtified: "
        f"{_tensorrtified_methods[key]}")

    method = getattr(module, method_name)
    method = method.__func__  # convert a bound method to an unbound method
    enable_tensorrt = os.environ.get('ALF_ENABLE_TENSORRT', '1') == '1'
    wrapped = tensorrt_for_inference_if(enable_tensorrt)(method)
    setattr(module, method_name, types.MethodType(wrapped, module))

    # add the tensorrtified method to the table
    _tensorrtified_methods[key] = wrapped
