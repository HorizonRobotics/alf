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

from typing import Any, Callable, Optional
import functools

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import alf
from alf.experience_replayers.replay_buffer import ReplayBuffer


class _MethodPerformer(torch.nn.Module):
    """A nn.Module wrapper whose forward() performs a specified method of
    the wrapped module.

    The end goal is to make a TARGET METHOD data distributed.

    We need this delegation so that DDP can then wrap over this module. When DDP
    hijacks the forward() of _MethodPerformer to inject synchronization hooks,
    it effectively does so for the target method of the wrapped module.

    """

    def __init__(self, module: torch.nn.Module, perform: Callable[..., Any]):
        """Constructs a _MethodPerformer.

        Args:

            module: an instance of the module whose method is going to be
                delegated to. The _MethodPerformer instance needs to access and
                inherit the parameters from the module, so that DDP knows what
                parameters to cover.

            perform: the target method of the module.

        """
        super().__init__()

        self._wrapped_module = module  # Register and inherit the parameters
        self._perform = functools.partial(perform, self._wrapped_module)

        # DDP will panic if the wrapped module has member in its state_dict()
        # that is not a Tensor. Here such state_dict members are picked and
        # thrown into _ddp_params_and_buffers_to_ignore. By contract this
        # implicitly instructs DDP wrapper to not include them in its
        # parameter/buffer synchronization.
        self._ddp_params_and_buffers_to_ignore = []
        for name, value in self.state_dict().items():
            if type(value) is not torch.Tensor:
                self._ddp_params_and_buffers_to_ignore.append(name)

        # We also need to ignore all the buffers that is under the replay buffer
        # of the module (e.g. when the module is an Algorithm) for DDP, because
        # we do not want DDP to synchronize replay buffers across processes.
        #
        # Those buffers are not registered in the state_dict() because of Alf's
        # special treatment but can be found under named_buffers(). We do not
        # want DDP to synchronize replay buffers.
        ignored_named_buffers = set()
        for sub_module in module.modules():
            if isinstance(sub_module, ReplayBuffer):
                for _, buf in sub_module.named_buffers():
                    # Find all the buffers that are registered under a
                    # ReplayBuffer submodule.
                    ignored_named_buffers.add(buf)

        for name, buf in self.named_buffers():
            # If the buffer is in the ignored_named_buffers (address-wise equal,
            # i.e. ``is``), add its name to DDP's ignore list.
            if buf in ignored_named_buffers:
                self._ddp_params_and_buffers_to_ignore.append(name)

        # TODO(breakds): In the future when needed, we can do explicit filtering
        # if the wrapped module is an Algorithm. All parameters and buffers that
        # are not within the optimizer can be added to ignore list.

    def forward(self, *args, **kwargs):
        return self._perform(*args, **kwargs)


@alf.configurable
def make_ddp_performer(module: torch.nn.Module,
                       method,
                       ddp_rank: int,
                       find_unused_parameters: bool = False,
                       bucket_cap_mb: int = 25):
    """Creates a DDP wrapped MethodPerformer.

    This function is an alf.configurable and used in the @data_distributed
    series of decorators below. Override this in your configuration with

        alf.config('make_ddp_performer', find_unused_parameters=True)

    to enable ``find_unused_parameters``. This asks DDP to ignore parameters
    that are not used for computing the output of ``forward()`` when waiting for
    synchronization of gradients and parameters upon ``backward()``. Normally
    you do not need to worry about this. It is useful for algorithms such as PPG
    where part of the parameters of the model does NOT ALWAYS contribute to the
    network output.

    """
    print(f'find_unused_parameters={find_unused_parameters}')
    return DDP(
        _MethodPerformer(module=module, perform=method),
        device_ids=[ddp_rank],
        find_unused_parameters=find_unused_parameters,
        bucket_cap_mb=bucket_cap_mb)


def data_distributed(method):
    """This decorator makes a target method of a module capable of being data
    distributed via DDP.

    This is to provide a simple and transparent way to enable DDP for specific
    code logics.

    When the method is wrapped by @data_distributed, the outputs (tensors) of
    this method will have gradient synchronization hooks attached to them. Later
    when those outputs are used in ``backward()`` to compute gradients, the
    hooks will be called to synchronize across all processes. As a result, the
    corresponding parameters receive not only the gradients from this process,
    but also gradients from the other processes. Note that each single process
    will be TRAPPED at the call to the ``backward()`` that involves those output
    tensors, until all processes finished the back propagation and have the
    gradients sync'ed.

    Example usage:

    .. code-block:: python

        class A(nn.Module):
            # ...
            @data_distributed
            def compute_something(self, input):
              return self._network1(input), self._network2(input)
            # ...

    In the above code, after applying the decorator, the method
    ``compute_something`` will be made data distributed if the following
    conditions are met:

    1. Multiple processes within the same process group creates A's instances
       and calls ``compute_something()`` individually.

    2. All such A instances have ``self._ddp_activated_rank`` set to the correct
       rank of the GPU device that belongs to them.

    Otherwise the method ``compute_something()`` will behave normally.

    """
    return data_distributed_when(None)(method)


def data_distributed_when(
        cond: Optional[Callable[[torch.nn.Module], bool]] = None):
    """This is @ data_distributed with an extra conditionon.

    The condition is a function that returns True or False given the wrapped
    module as the input. If the condition evaluates to False, DDP will not be
    activated and the original method will be called.

    """

    def decorator(method):
        @functools.wraps(method)
        def wrapped(*args, **kwargs):
            # The first argument to the method is going to be ``self``, i.e. the
            # instance that the method belongs to. By accessing it we get the
            # reference of the module to wrap.
            module_to_wrap = args[0]
            assert isinstance(module_to_wrap, torch.nn.Module), (
                f'Cannot apply @data_distributed on {type(module_to_wrap)}')

            ddp_rank = getattr(module_to_wrap, '_ddp_activated_rank', -1)

            # Evaluate the condition if it is provided.
            if (cond is not None) and (not cond(module_to_wrap)):
                ddp_rank = -1

            # A ddp_rank of -1 means DDP is not activated for this module. In this
            # case, just perform the normal method call.
            if ddp_rank == -1:
                return method(*args, **kwargs)

            # Create a DDP wrapped _MethodPerformer instance if not yet. All the
            # _MethodPerformer instances are registered in a map called
            # _ddp_performer_map, which belongs to the module to wrap.
            if not hasattr(module_to_wrap, '_ddp_performer_map'):
                setattr(module_to_wrap, '_ddp_performer_map', {})

            performer = module_to_wrap._ddp_performer_map.get(
                method.__name__, None)
            if performer is None:
                performer = make_ddp_performer(module_to_wrap, method,
                                               ddp_rank)
                module_to_wrap._ddp_performer_map[method.__name__] = performer
            return performer(*args[1:], **kwargs)

        return wrapped

    return decorator
