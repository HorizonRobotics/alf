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

from typing import Any, Callable
import functools

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


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
        # implicitly instruct DDP wrapper to not include them in its
        # parameter/buffer synchronization.
        self._ddp_params_and_buffers_to_ignore = []
        for name, value in self.state_dict().items():
            if type(value) is not torch.Tensor:
                self._ddp_params_and_buffers_to_ignore.append(name)

    def forward(self, *args, **kwargs):
        return self._perform(*args, **kwargs)


def data_distributed(method):
    """This decorator makes a target method of a module capable of being data
    distributed via DDP.

    This is to provide a simple and transparent way to enable DDP for specific
    code logics.

    Example usage:

    .. code-block:: python

        class A(nn.Module):
            # ...
            @data_distributed
            def compute_something(self, input):
              return self._network1(input), self._network2(input)
            # ...
        variance_scaling_init(layer.weight.data,
                              nonlinearity=nn.functional.leaky_relu)


    In the above code, after applying the decorator, the method
    ``compute_something`` will be made data distributed if the following
    conditions are met:

    1. Multiple processes within the same process group creates A's instances
       and calls ``compute_something()`` individually.

    2. All such A instances have ``self._ddp_activated_rank`` set to the correct
       rank of the GPU device that belongs to them.

    Otherwise the method ``compute_something()`` will behave normally.

    """

    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        # The first argument to the method is going to be ``self``, i.e. the
        # instance that the method belongs to. By accessing it we get the
        # reference of the module to wrap.
        module_to_wrap = args[0]
        assert isinstance(module_to_wrap, torch.nn.Module), (
            f'Cannot apply @data_distributed on {type(module_to_wrap)}')

        ddp_rank = getattr(module_to_wrap, '_ddp_activated_rank', -1)

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
            performer = DDP(
                _MethodPerformer(module=module_to_wrap, perform=method),
                device_ids=[ddp_rank])
            module_to_wrap._ddp_performer_map[method.__name__] = performer
        return performer(*args[1:], **kwargs)

    return wrapped
