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

from abc import abstractmethod
from absl import logging
import copy
import json
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys

import alf
from alf.data_structures import AlgStep, namedtuple, LossInfo
from alf.utils import common


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
        return [module]


class Algorithm(nn.Module):
    """Algorithm base class.

    Algorithm is a generic interface for supervised training algorithms.

    User needs to implement train_step() and calc_loss()/train_complete().

    train_step() is called to generate actions for every environment step.
    It also needs to generate necessary information for training.

    train_complete() is called every train_interval steps (specified in
    PolicyDriver). All the training information collected at each previous
    train_step() are batched and provided as arguments for train_complete().

    The following is the pseudo code to illustrate how Algorithm is used for
    training.

    ```python
    while training not ends:
        training_info = []
        with GradientTape() as tape:
        for i in range(train_interval):
            get inputs
            outputs, state, info = train_step(inputs, state)
            add info to training_info

        train_complete(tape, batched_training_info)
    ```
    """

    def __init__(self,
                 train_state_spec=None,
                 predict_state_spec=None,
                 optimizer=None,
                 trainable_module_sets=None,
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
                `predict()`. If None, it's assume to be same as train_state_spec
            optimizer (None|Optimizer): The default optimizer for
                training. See comments above for detail.
            trainable_module_sets (list[list]): See comments above for detail.
            gradient_clipping (float): If not None, serve as a positive threshold
            clip_by_global_norm (bool): If True, use tf.clip_by_global_norm to
                clip gradient. If False, use tf.clip_by_norm for each grad.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this algorithm.
        """
        super(Algorithm, self).__init__()

        self._name = name
        self._train_state_spec = train_state_spec
        if predict_state_spec is None:
            predict_state_spec = train_state_spec
        self._predict_state_spec = predict_state_spec
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

    def _get_children(self, ignore=True):
        """Get children

        Only exclude the children marked in trainable_attributes_to_ignore
        when ignore is True.
        """
        children = []
        for name, module in self.named_children():
            if name in self._trainable_attributes_to_ignore() and ignore:
                continue
            children.extend(_flatten_module(module))

        for name, param in self.named_parameters(recurse=False):
            if name in self._trainable_attributes_to_ignore() and ignore:
                continue
            children.append(param)

        return children

    @property
    def default_optimizer(self):
        """Get the default optimizer for this algorithm."""
        return self._default_optimizer

    def _assert_no_cycle_or_duplicate(self, ignore=True):
        """Check the existence of cycle.

        Only exclude the attributes marked in trainable_attributes_to_ignore
        when ignore is True.
        """
        visited = set()
        to_be_visited = [self]
        while to_be_visited:
            node = to_be_visited.pop(0)
            visited.add(node)
            for child in node._get_children(ignore):
                assert child not in visited, (
                    "There is a cycle or duplicate in the "
                    "algorithm tree caused by '%s'" % child.name)
                if isinstance(child, Algorithm):
                    to_be_visited.append(child)

    def _setup_optimizers(self):
        """Setup the param groups for optimizers.

        Returns:
            list of parameters not handled by any optimizers under this algorithm
        """
        self._assert_no_cycle_or_duplicate()
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
                optimizer.add_param_group({'params': params})

        if default_optimizer is not None:
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
                dict(optimizer="None", parameters=[id(p) for p in unhandled]))

        for optimizer in self.optimizers():
            parameters = _get_optimizer_params(optimizer)
            optimizer_info.append(
                dict(
                    optimizer=optimizer.__class__.__name__,
                    hypers=optimizer.defaults,
                    # TODO: better name for each parameter
                    parameters=[id(p) for p in parameters]))
        json_pretty_str_info = json.dumps(obj=optimizer_info, indent=2)

        return json_pretty_str_info

    @property
    def predict_state_spec(self):
        """Returns the RNN state spec for predict()."""
        return self._predict_state_spec

    @property
    def train_state_spec(self):
        """Returns the RNN state spec for train_step()."""
        return self._train_state_spec

    def convert_train_state_to_predict_state(self, state):
        """Convert RNN state for train_step() to RNN state for predict()."""
        alf.nest.assert_same_structure(self._train_state_spec,
                                       self._predict_state_spec)
        return state

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Get module dictionary for checkpoint, including both model state
                and optimizers' state (if any)

        Args:
            destination (OrderedDict): the destination for storing the state
        Returns:
            destination (OrderedDict): the dictionary including both model state
                and optimizers' state (if any)

        """
        self._assert_no_cycle_or_duplicate(ignore=False)
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(
            version=self._version)

        self._save_to_state_dict(destination, prefix, keep_vars)
        opts_dict = OrderedDict()
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(
                    destination, prefix + name + '.', keep_vars=keep_vars)
        for i, opt in enumerate(self._optimizers):
            new_key = prefix + '_optimizers.%d' % i
            if new_key not in self._opt_keys:
                self._opt_keys.append(new_key)
            opts_dict[self._opt_keys[i]] = opt.state_dict()

        destination.update(opts_dict)

        return destination

    def load_state_dict(self, state_dict, strict=True):
        """Load state dictionary for Algorithm
        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        self._assert_no_cycle_or_duplicate(ignore=False)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            if isinstance(module, Algorithm):
                for i, opt in enumerate(module._optimizers):
                    opt_key = prefix + '_optimizers.%d' % i
                    if opt_key in state_dict:
                        opt.load_state_dict(state_dict[opt_key])
                        del state_dict[opt_key]
                    elif strict:
                        missing_keys.append(opt_key)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                                         True, missing_keys, unexpected_keys,
                                         error_msgs)

        load(self)
        load = None  # break load->load reference cycle

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

    #------------- User need to implement the following functions -------

    # Subclass may override predict() to allow more efficient implementation
    def predict(self, inputs, state=None):
        """Predict for one step of inputs.

        Args:
            inputs (nested Tensor): inputs for prediction
            state (nested Tensor): network state (for RNN)

        Returns:
            AlgStep
                outputs (nested Tensor): prediction result
                state (nested Tensor): should match `predict_state_spec`
        """
        algorithm_step = self.train_step(inputs, state)
        return algorithm_step._replace(info=None)

    @abstractmethod
    def train_step(self, inputs, state=None):
        """Perform one step of predicting and training computation.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            inputs (nested Tensor): inputs for train
            state (nested Tensor): consistent with train_state_spec

        Returns:
            AlgStep
                output (nested Tensor): predict outputs
                state (nested Tensor): should match `predict_state_spec`
                info (nested Tensor): information for training. If this is
                    LossInfo, calc_loss() in Algorithm can be used. Otherwise,
                    the user needs to override calc_loss() to calculate loss or
                    override train_complete() to do customized training.
        """
        pass

    # Subclass may override train_complete() to allow customized training
    def train_complete(self, training_info, valid_masks=None, weight=1.0):
        """Complete one iteration of training.

        `train_complete` should calculate gradients and update parameters using
        those gradients.

        Args:
            tape (tf.GradientTape): the tape which are used for calculating
                gradient. All the previous `train_interval` `train_step()`
                are called under the context of this tape.
            training_info (nested Tensor): information collected for training.
                It is batched from each `info` returned bt `train_step()`
            valid_masks (tf.Tensor): masks indicating which samples are valid.
                shape=(T, B), dtype=tf.float32
            weight (float): weight for this batch. Loss will be multiplied with
                this weight before calculating gradient
        Returns:
            loss_info (LossInfo): loss information
            params (list[Parameter]): list of parameters being updated.
        """
        loss_info = self.calc_loss(training_info)
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
                if self._clip_by_global_norm:
                    # TODO: implement alf.clip_by_global_norm
                    global_norm = alf.clip_by_global_norm(
                        params, self._gradient_clipping)
                    if common.should_record_summaries():
                        alf.summary.scalar("global_grad_norm/%s" % i,
                                           global_norm)
                else:
                    # TODO: implement alf.clip_gradient_norms
                    alf.clip_gradient_norms(params, self._gradient_clipping)
            optimizer.step()

        self.after_train(training_info)

        return loss_info, all_params

    def after_train(self, training_info):
        """Do things after complete one iteration of training, such as update
        target network.

        Args:
            training_info (nested Tensor): information collected for training.
                It is batched from each `info` returned bt `train_step()`
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
