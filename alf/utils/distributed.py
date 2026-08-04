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

from typing import Any, Callable, Iterable, Optional
import functools

from absl import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import alf
from alf.experience_replayers.replay_buffer import ReplayBuffer

FSDP2_OPTIMIZER_STATE = '_fsdp2_optimizer_state'


def _hide_alf_optimizers(module):
    """Temporarily hide optimizer lists from ALF's custom state_dict.

    PyTorch's distributed checkpoint state-dict code expects every returned key
    to name a module attribute. ALF embeds optimizer states under synthetic
    ``_optimizers.N`` keys, so those keys must be hidden while PyTorch gathers
    model state. The returned pairs are restored by the caller.
    """
    hidden = []
    for child in module.modules():
        if hasattr(child, '_optimizers'):
            hidden.append((child, child._optimizers))
            child._optimizers = []
    return hidden


def _restore_alf_optimizers(hidden):
    for module, optimizers in hidden:
        module._optimizers = optimizers


def _hide_replay_buffers(module):
    """Disable replay-buffer state during a model checkpoint operation.

    PyTorch's distributed state-dict loader walks registered buffers directly
    instead of honoring ALF's custom ``state_dict()`` filtering. Mark replay
    buffers non-persistent for the duration as well; their state is handled by
    ALF's separate replay-buffer checkpoint files.
    """
    hidden = []
    for child in module.modules():
        if isinstance(child, ReplayBuffer):
            buffer_persistence = []
            for replay_module in child.modules():
                original = replay_module._non_persistent_buffers_set.copy()
                replay_module._non_persistent_buffers_set.update(
                    replay_module._buffers.keys())
                buffer_persistence.append((replay_module, original))
            hidden.append(
                (child, getattr(child, '_alf_checkpoint_enabled',
                                None), buffer_persistence))
            child._alf_checkpoint_enabled = False
    return hidden


def _restore_replay_buffers(hidden):
    for module, enabled, buffer_persistence in hidden:
        for replay_module, original in buffer_persistence:
            replay_module._non_persistent_buffers_set = original
        if enabled is None:
            del module._alf_checkpoint_enabled
        else:
            module._alf_checkpoint_enabled = enabled


class _MethodPerformer(torch.nn.Module):
    """A nn.Module wrapper whose forward() performs a specified method of
    the wrapped module.

    The end goal is to make a TARGET METHOD data distributed.

    We need this delegation so that DDP can then wrap over this module. When DDP
    hijacks the forward() of _MethodPerformer to inject synchronization hooks,
    it effectively does so for the target method of the wrapped module.

    """

    def __init__(self,
                 module: torch.nn.Module,
                 perform: Callable[..., Any],
                 prepare_for_ddp: bool = True):
        """Constructs a _MethodPerformer.

        Args:

            module: an instance of the module whose method is going to be
                delegated to. The _MethodPerformer instance needs to access and
                inherit the parameters from the module, so that DDP knows what
                parameters to cover.

            perform: the target method of the module.

            prepare_for_ddp: inspect the state dict for values DDP must ignore.
                FSDP2 must skip this because ALF's state dict lazily initializes
                optimizers, which must happen after parameters are sharded.

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
        if prepare_for_ddp:
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

    def set_method(self, perform: Callable[..., Any]):
        """Change the method dispatched by :meth:`forward`.

        FSDP2 shards a module in-place, so all decorated methods of an ALF
        module must share one performer instead of independently wrapping the
        same parameters as DDP does.
        """
        self._perform = functools.partial(perform, self._wrapped_module)


@alf.configurable(whitelist=['find_unused_parameters', 'bucket_cap_mb'])
def make_ddp_performer(module: torch.nn.Module,
                       method,
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
    return DDP(_MethodPerformer(module=module, perform=method),
               device_ids=None,
               find_unused_parameters=find_unused_parameters,
               bucket_cap_mb=bucket_cap_mb)


def _resolve_fsdp2_shard_plan(module: torch.nn.Module,
                              shard_plan: Callable[[torch.nn.Module],
                                                   Iterable[torch.nn.Module]]):
    """Resolve and validate a deterministic, bottom-up shard plan."""
    named_modules = dict(module.named_modules())
    module_names = {child: name for name, child in named_modules.items()}
    selected = list(shard_plan(module))
    resolved = []
    seen = set()
    for child in selected:
        if not isinstance(child, torch.nn.Module):
            raise TypeError(
                "FSDP2 shard_plan must return nn.Module instances; "
                "got %s" % type(child))
        if child is module:
            raise ValueError(
                "FSDP2 shard_plan must not return the root module; "
                "the root performer is sharded automatically")
        if child not in module_names:
            raise ValueError(
                "FSDP2 shard_plan returned a module that is not a "
                "descendant of the distributed module")
        if child not in seen:
            seen.add(child)
            resolved.append((module_names[child], child))

    # Child groups must claim their parameters before parent groups. Sorting by
    # name makes equal-depth ordering deterministic across ranks.
    resolved.sort(key=lambda item: (-item[0].count('.'), item[0]))
    names = [name for name, _ in resolved]
    gathered_names = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_names, names)
    if any(rank_names != names for rank_names in gathered_names):
        raise RuntimeError(
            "FSDP2 shard_plan differs across distributed ranks: "
            "%s" % gathered_names)
    return resolved


@alf.configurable(whitelist=['reshard_after_forward', 'shard_plan'])
def make_fsdp2_performer(
    module: torch.nn.Module,
    method,
    reshard_after_forward: bool = True,
    shard_plan: Optional[Callable[[torch.nn.Module],
                                  Iterable[torch.nn.Module]]] = None):
    """Create an FSDP2 performer for an ALF distributed method.

    Unlike DDP, composable FSDP replaces parameters with sharded ``DTensor``
    parameters in-place. The caller therefore creates exactly one performer
    for a module and reuses it for every distributed method.

    Args:
        module: module owning the method and parameters to shard.
        method: unbound method invoked by the performer.
        reshard_after_forward: whether to release gathered parameters after
            forward. Keeping this configurable permits trading communication
            for memory without changing ALF training code.
        shard_plan: optional callback returning descendant modules that should
            form independent FSDP groups. ALF applies them bottom-up and then
            shards the root performer for all remaining parameters.
    """
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError as e:
        raise RuntimeError("FSDP2 requires a PyTorch version providing "
                           "torch.distributed.fsdp.fully_shard") from e

    if not dist.is_initialized():
        raise RuntimeError("FSDP2 requires an initialized process group")

    performer = _MethodPerformer(module=module,
                                 perform=method,
                                 prepare_for_ddp=False)
    parameters_before_sharding = {
        parameter: name
        for name, parameter in module.named_parameters()
    }
    # DDP broadcasts rank 0's initial module state from its constructor. FSDP2
    # deliberately does not, while ALF may seed workers differently. Preserve
    # the established ALF behavior before replacing parameters with DTensors.
    ignored = set(performer._ddp_params_and_buffers_to_ignore)
    for name, parameter in performer.named_parameters():
        if name not in ignored:
            dist.broadcast(parameter.detach(), src=0)
    for name, buffer in performer.named_buffers():
        if name not in ignored and isinstance(buffer, torch.Tensor):
            dist.broadcast(buffer.detach(), src=0)
    if shard_plan is not None:
        planned_groups = []
        claimed_parameters = set()
        for name, child in _resolve_fsdp2_shard_plan(module, shard_plan):
            group_parameters = [
                parameter for parameter in child.parameters()
                if parameter not in claimed_parameters
            ]
            claimed_parameters.update(group_parameters)
            planned_groups.append(
                (name, child,
                 sum(parameter.numel() for parameter in group_parameters)))
        root_parameters = sum(parameter.numel()
                              for parameter in module.parameters()
                              if parameter not in claimed_parameters)
        for name, child, num_parameters in planned_groups:
            if num_parameters == 0:
                logging.warning("Skipping empty FSDP2 shard group '%s'", name)
                continue
            logging.info("FSDP2 shard group '%s': %d parameters", name,
                         num_parameters)
            fully_shard(child, reshard_after_forward=reshard_after_forward)
        logging.info("FSDP2 root remainder: %d parameters", root_parameters)
    fully_shard(performer, reshard_after_forward=reshard_after_forward)

    # Most ALF optimizers are populated lazily after this first distributed
    # forward. Off-policy setup can populate them earlier while constructing a
    # replay buffer, though. Rebind those groups from the original Parameters
    # to the DTensor Parameters installed by fully_shard().
    if hasattr(module, 'optimizers'):
        sharded_parameters = dict(module.named_parameters())
        replacements = {
            parameter: sharded_parameters[name]
            for parameter, name in parameters_before_sharding.items()
        }
        for optimizer in module.optimizers():
            for group in optimizer.param_groups:
                group['params'] = [
                    replacements.get(parameter, parameter)
                    for parameter in group['params']
                ]
            for parameter, state in list(optimizer.state.items()):
                replacement = replacements.get(parameter)
                if replacement is not None and replacement is not parameter:
                    optimizer.state[replacement] = state
                    del optimizer.state[parameter]
    return performer


def make_distributed_performer(module: torch.nn.Module, method):
    """Create or reuse the performer selected for ``module``."""
    strategy = getattr(module, '_distributed_strategy', 'ddp')
    if strategy == 'ddp':
        return make_ddp_performer(module, method)
    if strategy != 'fsdp2':
        raise ValueError("Unknown distributed strategy: %s" % strategy)

    performer = getattr(module, '_fsdp2_performer', None)
    if performer is None:
        performer = make_fsdp2_performer(module, method)
        # Store this outside nn.Module._modules. Registering the performer on
        # its wrapped module would create a module cycle.
        object.__setattr__(module, '_fsdp2_performer', performer)
    else:
        performer.set_method(method)
    return performer


def is_fsdp2_module(module: torch.nn.Module) -> bool:
    """Whether ``module`` has been activated and wrapped with FSDP2."""
    return (getattr(module, '_distributed_strategy', 'ddp') == 'fsdp2'
            and getattr(module, '_fsdp2_performer', None) is not None)


def fsdp2_full_state_dict(module: torch.nn.Module):
    """Gather a portable full model and optimizer state dict on rank 0.

    All ranks in the process group must call this function. Nonzero ranks
    return an empty dictionary to avoid redundant host memory use.
    """
    if not is_fsdp2_module(module):
        raise ValueError("The module has not been wrapped with FSDP2")

    from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                         get_state_dict)

    performer = module._fsdp2_performer
    # ALF optimizers start with no parameters and are populated lazily by
    # Algorithm._setup_optimizers(). Distributed checkpoint APIs operate on
    # the optimizer directly, so make that normally implicit setup explicit.
    if hasattr(module, '_setup_optimizers'):
        module._setup_optimizers()
    optimizers = module.optimizers()
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    hidden = _hide_alf_optimizers(module)
    hidden_replay_buffers = _hide_replay_buffers(module)
    try:
        model_state, optimizer_state = get_state_dict(performer,
                                                      optimizers,
                                                      options=options)
    finally:
        _restore_replay_buffers(hidden_replay_buffers)
        _restore_alf_optimizers(hidden)
    if dist.get_rank() != 0:
        return {}

    prefix = '_wrapped_module.'
    model_state = {
        name[len(prefix):]: value
        for name, value in model_state.items() if name.startswith(prefix)
    }
    # Honor ALF's custom checkpoint filtering (including disabled replay
    # buffers) instead of blindly saving every state returned by PyTorch.
    allowed_keys = set(module.state_dict().keys())
    model_state = {
        name: value
        for name, value in model_state.items() if name in allowed_keys
    }
    model_state[FSDP2_OPTIMIZER_STATE] = optimizer_state
    return model_state


def load_fsdp2_full_state_dict(module: torch.nn.Module,
                               state_dict,
                               strict: bool = True):
    """Load a state produced by :func:`fsdp2_full_state_dict`."""
    if not is_fsdp2_module(module):
        raise ValueError("The module has not been wrapped with FSDP2")

    from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                         set_model_state_dict,
                                                         set_state_dict)

    state_dict = state_dict.copy()
    optimizer_state = state_dict.pop(FSDP2_OPTIMIZER_STATE, None)
    model_state = {
        '_wrapped_module.' + name: value
        for name, value in state_dict.items()
    }
    options = StateDictOptions(full_state_dict=True, strict=strict)
    # Regular Algorithm.load_state_dict() performs this setup before restoring
    # optimizer state. The FSDP2 checkpoint path bypasses that method and must
    # preserve the same ordering itself.
    if hasattr(module, '_setup_optimizers'):
        module._setup_optimizers()
    optimizers = module.optimizers()
    hidden = _hide_alf_optimizers(module)
    hidden_replay_buffers = _hide_replay_buffers(module)
    try:
        if optimizer_state is None:
            return set_model_state_dict(module._fsdp2_performer,
                                        model_state,
                                        options=options)
        return set_state_dict(module._fsdp2_performer,
                              optimizers,
                              model_state_dict=model_state,
                              optim_state_dict=optimizer_state,
                              options=options)
    finally:
        _restore_replay_buffers(hidden_replay_buffers)
        _restore_alf_optimizers(hidden)


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


def data_distributed_when(cond: Optional[Callable[[torch.nn.Module],
                                                  bool]] = None):
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

            strategy = getattr(module_to_wrap, '_distributed_strategy', 'ddp')
            if strategy == 'ddp':
                # DDP keeps one wrapper per decorated method for backward
                # compatibility with the existing implementation.
                if not hasattr(module_to_wrap, '_ddp_performer_map'):
                    setattr(module_to_wrap, '_ddp_performer_map', {})
                performer = module_to_wrap._ddp_performer_map.get(
                    method.__name__, None)
                if performer is None:
                    performer = make_ddp_performer(module_to_wrap, method)
                    module_to_wrap._ddp_performer_map[
                        method.__name__] = performer
            else:
                performer = make_distributed_performer(module_to_wrap, method)
            return performer(*args[1:], **kwargs)

        return wrapped

    return decorator
