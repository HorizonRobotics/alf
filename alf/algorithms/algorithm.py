# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Algorithm base class."""

from absl import logging
import copy
from collections import OrderedDict
from functools import wraps
import itertools
import json
import os
import six
import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys, _addindent

import alf
from alf.data_structures import AlgStep, namedtuple, LossInfo
from alf.utils import common
from alf.utils import tensor_utils


def _get_optimizer_params(optimizer: torch.optim.Optimizer):
    return set(sum([g['params'] for g in optimizer.param_groups], []))


def _flatten_module(module):
    if isinstance(module, nn.ModuleList):
        return sum(map(_flatten_module, module), [])
    elif isinstance(module, nn.ModuleDict):
        return sum(map(_flatten_module, module.values()), [])
    elif isinstance(module, nn.ParameterList):
        return list(module)
    elif isinstance(module, nn.ParameterDict):
        return list(module.values())
    else:
        # `module` is an nn.Module or nn.Parameter
        return [module]


class Algorithm(nn.Module):
    """Algorithm base class.

    Algorithm is a generic interface for supervised training algorithms.

    User needs to implement predict_step(), train_step() and
    calc_loss()/update_with_gradient().
    """

    def __init__(self,
                 train_state_spec=None,
                 rollout_state_spec=None,
                 predict_state_spec=None,
                 optimizer=None,
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 debug_summaries=False,
                 name="Algorithm"):
        """Create an Algorithm.

        Each algorithm can have a default optimimzer. By default, the parameters
        and/or modules under an algorithm are optimized by the default
        optimizer. One can also specify an optimizer for a set of parameters
        and/or modules using add_optimizer.

        A requirement for this optimizer structure to work is that there is no
        algorithm which is a submodule of a non-algorithm module. Currently,
        this is not checked by the framework. It's up to the user to make sure
        this is true.

        Args:
            train_state_spec (nested TensorSpec): for the network state of
                `train_step()`
            predict_state_spec (nested TensorSpec): for the network state of
                `predict_step()`. If None, it's assume to be same as rollout_state_spec
            optimizer (None|Optimizer): The default optimizer for
                training. See comments above for detail.
            gradient_clipping (float): If not None, serve as a positive threshold
            clip_by_global_norm (bool): If True, use tf.clip_by_global_norm to
                clip gradient. If False, use tf.clip_by_norm for each grad.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this algorithm.
        """
        super(Algorithm, self).__init__()

        self._name = name
        self._train_state_spec = train_state_spec
        self._rollout_state_spec = rollout_state_spec or self._train_state_spec
        self._predict_state_spec = predict_state_spec or self._rollout_state_spec

        self._is_rnn = len(alf.nest.flatten(train_state_spec)) > 0

        self._gradient_clipping = gradient_clipping
        self._clip_by_global_norm = clip_by_global_norm
        self._debug_summaries = debug_summaries
        self._default_optimizer = optimizer
        self._optimizers = []
        self._opt_keys = []
        self._module_to_optimizer = {}
        if optimizer:
            self._optimizers.append(optimizer)

    @property
    def name(self):
        """The name of this algorithm."""
        return self._name

    def add_optimizer(self, optimizer: torch.optim.Optimizer,
                      modules_and_params):
        """Add an optimizer

        Note that the modules and params contained in `modules_and_params`
        should still be the attributes of the algorithm (i.e., they can be
        retrieved in self.children() or self.parameters())

        Args:
            optimizer (Optimizer): optimizer
            modules_and_params (list of Module or Parameter): The modules and
                parameters to be optimized by `optimizer`
        """
        for module in modules_and_params:
            for m in _flatten_module(module):
                self._module_to_optimizer[m] = optimizer
        self._optimizers.append(optimizer)

    def _trainable_attributes_to_ignore(self):
        """Algorithms can overwrite this function to provide which class
        member names should be ignored when getting trainable variables, to
        avoid being assigned with multiple optimizers.

        For example, if in your algorithm you've created a member self._vars
        pointing to the variables of a module for some purpose, you can avoid
        assigning an optimizer to self._vars (because the module will be assigned
        with one) by doing:

            def _trainable_attributes_to_ignore(self):
                return ["_vars"]

        Returns:
            names (list[str]): a list of attribute names to ignore
        """
        return []

    def _get_children(self):
        children = []
        for name, module in self.named_children():
            if name in self._trainable_attributes_to_ignore():
                continue
            children.extend(_flatten_module(module))

        for name, param in self.named_parameters(recurse=False):
            if name in self._trainable_attributes_to_ignore():
                continue
            children.append(param)

        return children

    @property
    def default_optimizer(self):
        """Get the default optimizer for this algorithm."""
        return self._default_optimizer

    def _assert_no_cycle_or_duplicate(self):
        visited = set()
        to_be_visited = [self]
        while to_be_visited:
            node = to_be_visited.pop(0)
            visited.add(node)
            for child in node._get_children():
                assert child not in visited, (
                    "There is a cycle or duplicate in the "
                    "algorithm tree caused by '%s'" % child.name)
                if isinstance(child, Algorithm):
                    to_be_visited.append(child)

    def get_param_name(self, param):
        """Get the name of the parameter.

        Returns:
            string of the name if the parameter can be found.
            None if the parameter cannot be found.
        """
        return self._param_to_name.get(param)

    def _setup_optimizers(self):
        """Setup the param groups for optimizers.

        Returns:
            list of parameters not handled by any optimizers under this algorithm
        """
        self._assert_no_cycle_or_duplicate()
        self._param_to_name = {}

        for name, param in self.named_parameters():
            self._param_to_name[param] = name

        return self._setup_optimizers_()[0]

    def _setup_optimizers_(self):
        """Setup the param groups for optimizers.

        Returns:
            tuple of
                list of parameters not handled by any optimizers under this algorithm
                list of parameters not handled under this algorithm
        """
        default_optimizer = self.default_optimizer
        new_params = []
        self_handled = set()
        handled = set()
        duplicate_error = "Parameter %s is handled by muliple optimizers."

        for child in self._get_children():
            if child in handled:
                continue
            assert id(child) != id(self), "Child should not be self"
            self_handled.add(child)
            assert child not in handled, duplicate_error % child
            handled.add(child)
            if isinstance(child, Algorithm):
                params, child_handled = child._setup_optimizers_()
                for m in child_handled:
                    assert m not in handled, duplicate_error % m
                    handled.add(m)
            elif isinstance(child, nn.Module):
                params = child.parameters()
            elif isinstance(child, nn.Parameter):
                params = [child]
            optimizer = self._module_to_optimizer.get(child, None)
            if optimizer is None:
                new_params.extend(params)
                if default_optimizer is not None:
                    self._module_to_optimizer[child] = default_optimizer
            else:
                existing_params = _get_optimizer_params(optimizer)
                params = list(
                    filter(lambda p: p not in existing_params, params))
                if params:
                    optimizer.add_param_group({'params': params})

        if default_optimizer is not None:
            existing_params = _get_optimizer_params(default_optimizer)
            new_params = list(
                filter(lambda p: p not in existing_params, new_params))
            if new_params:
                default_optimizer.add_param_group({'params': new_params})
            return [], handled
        else:
            return new_params, handled

    def optimizers(self, recurse=True):
        """Get all the optimizers used by this algorithm.

        Args:
            recurse (bool): If True, including all the sub-algorithms
        Returns:
            list of Optimizer
        """
        opts = copy.copy(self._optimizers)
        if recurse:
            for module in self.children():
                if isinstance(module, Algorithm):
                    opts.extend(module.optimizers())
        return opts

    def get_optimizer_info(self):
        """Return the optimizer info for all the modules in a string."""
        unhandled = self._setup_optimizers()

        optimizer_info = []
        if unhandled:
            optimizer_info.append(
                dict(
                    optimizer="None",
                    parameters=[self._param_to_name[p] for p in unhandled]))

        for optimizer in self.optimizers():
            parameters = _get_optimizer_params(optimizer)
            optimizer_info.append(
                dict(
                    optimizer=optimizer.__class__.__name__,
                    hypers=optimizer.defaults,
                    parameters=[self._param_to_name[p] for p in parameters]))
        json_pretty_str_info = json.dumps(obj=optimizer_info, indent=2)

        return json_pretty_str_info

    @property
    def predict_state_spec(self):
        """Returns the RNN state spec for predict_step()."""
        return self._predict_state_spec

    @property
    def rollout_state_spec(self):
        """Returns the RNN state spec for rollout_step()."""
        return self._rollout_state_spec

    @property
    def train_state_spec(self):
        """Returns the RNN state spec for train_step()."""
        return self._train_state_spec

    def convert_train_state_to_predict_state(self, state):
        """Convert RNN state for train_step() to RNN state for predict_step()."""
        alf.nest.assert_same_structure(self._train_state_spec,
                                       self._predict_state_spec)
        return state

    def get_initial_predict_state(self, batch_size):
        return common.zeros_from_spec(self._predict_state_spec, batch_size)

    def get_initial_rollout_state(self, batch_size):
        return common.zeros_from_spec(self._rollout_state_spec, batch_size)

    def get_initial_train_state(self, batch_size):
        return common.zeros_from_spec(self._train_state_spec, batch_size)

    @common.add_method(nn.Module)
    def state_dict(self, destination=None, prefix='', visited=None):
        """Get state dictionary recursively, including both model state
        and optimizers' state (if any). It can handle a number of special cases:
            1) graph with cycle: save all the states and avoid infinite loop
            2) parameter sharing: save only one copy of the shared module/param
            3) optimizers: save the optimizers for all the (sub-)algorithms

        Args:
            destination (OrderedDict): the destination for storing the state
            prefix (str): a string to be added before the name of the items
                (modules, params, algorithms etc) as the key used in the
                state dictionary
            visited (set): a set keeping track of the visited objects
        Returns:
            destination (OrderedDict): the dictionary including both model state
                and optimizers' state (if any)

        """

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(
            version=self._version)

        if visited is None:
            if isinstance(self, Algorithm):
                self._setup_optimizers()
            visited = {self}

        self._save_to_state_dict(destination, prefix, visited)
        opts_dict = OrderedDict()
        for name, child in self._modules.items():
            if child is not None and child not in visited:
                visited.add(child)
                child.state_dict(
                    destination, prefix + name + '.', visited=visited)
        if isinstance(self, Algorithm):
            for i, opt in enumerate(self._optimizers):
                new_key = prefix + '_optimizers.%d' % i
                if new_key not in self._opt_keys:
                    self._opt_keys.append(new_key)
                opts_dict[self._opt_keys[i]] = opt.state_dict()

            destination.update(opts_dict)

        return destination

    def load_state_dict(self, state_dict, strict=True):
        """Load state dictionary for Algorithm

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            If 'strict=True', will keep lists of missing and unexpected keys and
            raise error when any of the lists is non-empty; if `strict=False`,
            missing/unexpected keys will be omitted and no error will be raised.
            ''
        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        self._setup_optimizers()

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix='', visited=None):
            if visited is None:
                visited = {self}
            if isinstance(module, Algorithm):
                for i, opt in enumerate(module._optimizers):
                    opt_key = prefix + '_optimizers.%d' % i
                    if opt_key in state_dict:
                        opt.load_state_dict(state_dict[opt_key])
                        del state_dict[opt_key]
                    elif strict:
                        missing_keys.append(opt_key)

            for name, child in module._modules.items():
                if child is not None and child not in visited:
                    visited.add(child)
                    load(child, prefix + name + '.', visited=visited)

            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                                         True, missing_keys, unexpected_keys,
                                         error_msgs, visited)

        load(self)

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(', '.join(
                        '"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(
                    self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    @common.add_method(nn.Module)
    def _save_to_state_dict(self, destination, prefix, visited=None):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.
        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.
        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
            visited (set): a set keeping track of the visited objects
        """
        if visited is None:
            visited = set()

        for name, param in self._parameters.items():
            if param is not None and param not in visited:
                visited.add(param)
                destination[prefix + name] = param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and buf not in visited:
                visited.add(buf)
                destination[prefix + name] = buf.detach()

    @common.add_method(nn.Module)
    def _load_from_state_dict(self,
                              state_dict,
                              prefix,
                              local_metadata,
                              strict,
                              missing_keys,
                              unexpected_keys,
                              error_msgs,
                              visited=None):
        """Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.
        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.
        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module; if 'strict=True',
                will keep a list of missing and unexpected keys
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
            visited (set): a set keeping track of the visited objects
        """
        if visited is None:
            visited = set()

        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys,
                 unexpected_keys, error_msgs)

        local_name_params = itertools.chain(self._parameters.items(),
                                            self._buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            if param is not None and param not in visited:
                visited.add(param)
            else:
                continue
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append(
                        'size mismatch for {}: copying a param with shape {} from checkpoint, '
                        'the shape in current model is {}.'.format(
                            key, input_param.shape, param.shape))
                    continue

                try:
                    with torch.no_grad():
                        param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append(
                        'While copying the parameter named "{}", '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}, '
                        'an exception occured : {}.'.format(
                            key, param.size(), input_param.size(), ex.args))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split(
                        '.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    @common.add_method(nn.Module)
    def __repr__(self):
        return self._repr()

    @common.add_method(nn.Module)
    def _repr(self, visited=None):
        """Adapted from __repr__() in torch/nn/modules/module.nn. to handle cycles"""

        if visited is None:
            visited = [self]

        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            if module in visited:
                continue
            visited.append(module)
            if isinstance(module, nn.Module):
                mod_str = module._repr(visited)
            else:
                mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    #------------- User need to implement the following functions -------

    # Subclass may override predict_step() for more efficient implementation
    def predict_step(self, inputs, state=None):
        """Predict for one step of inputs.

        Args:
            inputs (nested Tensor): inputs for prediction
            state (nested Tensor): network state (for RNN)

        Returns:
            AlgStep
                output (nested Tensor): prediction result
                state (nested Tensor): should match `predict_state_spec`
        """
        algorithm_step = self.rollout_step(inputs, state)
        return algorithm_step._replace(info=None)

    def rollout_step(self, inputs, state=None):
        """Rollout for one step of inputs.

        Args:
            inputs (nested Tensor): inputs for prediction
            state (nested Tensor): network state (for RNN)

        Returns:
            AlgStep
                output (nested Tensor): prediction result
                state (nested Tensor): should match `rollout_state_spec`
        """
        algorithm_step = self.train_step(inputs, state)
        return algorithm_step._replace(info=None)

    def train_step(self, inputs, state=None):
        """Perform one step of training computation.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            inputs (nested Tensor): inputs for train
            state (nested Tensor): consistent with train_state_spec

        Returns:
            AlgStep
                output (nested Tensor): predict outputs
                state (nested Tensor): should match `train_state_spec`
                info (nested Tensor): information for training. If this is
                    LossInfo, calc_loss() in Algorithm can be used. Otherwise,
                    the user needs to override calc_loss() to calculate loss or
                    override update_with_gradient() to do customized training.
        """
        return AlgStep()

    # Subclass may override update_with_gradient() to allow customized training
    def update_with_gradient(self, loss_info, valid_masks=None, weight=1.0):
        """Complete one iteration of training.

        Update parameters using the gradient with respect to `loss_info`.

        Args:
            loss_info (LossInfo): loss with shape (T, B) (except for
                `loss_info.scalar_loss`)
            valid_masks (tf.Tensor): masks indicating which samples are valid.
                shape=(T, B), dtype=tf.float32
            weight (float): weight for this batch. Loss will be multiplied with
                this weight before calculating gradient
        Returns:
            loss_info (LossInfo): loss information
            params (list[(name, Parameter)]): list of parameters being updated.
        """
        if valid_masks is not None:
            loss_info = alf.nest.map_structure(
                lambda l: torch.mean(l * valid_masks)
                if len(l.shape) == 2 else l, loss_info)
        else:
            loss_info = alf.nest.map_structure(lambda l: torch.mean(l),
                                               loss_info)
        if isinstance(loss_info.scalar_loss, torch.Tensor):
            assert len(loss_info.scalar_loss.shape) == 0
            loss_info = loss_info._replace(
                loss=loss_info.loss + loss_info.scalar_loss)
        loss = weight * loss_info.loss

        unhandled = self._setup_optimizers()
        assert not unhandled, ("Some modules/parameters do not have optimizer"
                               ": %s" % unhandled)
        optimizers = self.optimizers()
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()

        all_params = []
        for i, optimizer in enumerate(optimizers):
            params = []
            for param_group in optimizer.param_groups:
                params.extend(param_group['params'])
            all_params.extend(params)
            if self._gradient_clipping is not None:
                grads = alf.nest.map_structure(lambda p: p.grad, params)
                if self._clip_by_global_norm:
                    _, global_norm = tensor_utils.clip_by_global_norm(
                        grads, self._gradient_clipping, in_place=True)
                    if alf.summary.should_record_summaries():
                        alf.summary.scalar("global_grad_norm/%s" % i,
                                           global_norm)
                else:
                    tensor_utils.clip_by_norms(
                        grads, self._gradient_clipping, in_place=True)
            optimizer.step()

        all_params = [(self._param_to_name[p], p) for p in all_params]
        return loss_info, all_params

    def after_update(self, training_info):
        """Do things after complete one gradient update (i.e. update_with_gradient())

        Args:
            training_info (nested Tensor): information collected for training.
                It is batched from each `info` returned by `rollout_step()` or
                `train_step()`
        Returns:
            None
        """
        pass

    # Subclass may override calc_loss() to allow more sophisticated loss
    def calc_loss(self, training_info):
        """Calculate the loss at each step for each sample.

        Args:
            training_info (nested Tensor): information collected for training.
                It is batched from each `info` returned by `train_step()`
        Returns:
            loss_info (LossInfo): loss at each time step for each sample in the
                batch. The shapes of the tensors in loss_info should be (T, B)
        """
        assert isinstance(training_info, LossInfo), (
            "training_info returned by"
            " train_step() should be LossInfo. Otherwise you need override"
            " calc_loss() to generate LossInfo from training_info")
        return training_info
