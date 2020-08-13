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
from alf.data_structures import AlgStep, namedtuple, LossInfo, StepType
from alf.experience_replayers.experience_replay import (
    OnetimeExperienceReplayer, SyncExperienceReplayer)
from alf.utils.checkpoint_utils import is_checkpoint_enabled
from alf.utils import (common, dist_utils, math_ops, spec_utils, summary_utils,
                       tensor_utils)
from alf.utils.summary_utils import record_time
from alf.utils.math_ops import add_ignore_empty
from .config import TrainerConfig
from .data_transformer import IdentityDataTransformer


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
    """Algorithm base class. ``Algorithm`` is a generic interface for supervised
    training algorithms. The key interface functions are:

    1. ``predict_step()``: one step of computation of action for evaluation.
    2. ``rollout_step()``: one step of computation for rollout. It is used for
       collecting experiences during training. Different from ``predict_step``,
       ``rollout_step`` may include addtional computations for training. An
       algorithm could immediately use the collected experiences to update
       parameters after one rollout (multiple rollout steps) is performed; or it
       can put these collected experiences into a replay buffer.
    3. ``train_step()``: only used by algorithms that put experiences into
       replay buffers. The training data are sampled from the replay buffer
       filled by ``rollout_step()``.
    4. ``train_from_unroll()``: perform a training iteration from the unrolled
       result.
    5. ``train_from_replay_buffer()``: perform a training iteration from a
       replay buffer.
    6. ``update_with_gradient()``: Do one gradient update based on the loss. It
       is used by the default ``train_from_unroll()`` and
       ``train_from_replay_buffer()`` implementations. You can override to
       implement your own ``update_with_gradient()``.
    7. ``calc_loss()``: calculate loss based the ``experience`` and the
       ``train_info`` collected from ``rollout_step()`` or ``train_step()``. It
       is used by the default implementations of ``train_from_unroll()`` and
       ``train_from_replay_buffer()``. If you want to use these two functions,
       you need to implement ``calc_loss()``.
    8. ``after_update()``: called by ``train_iter()`` after every call to
       ``update_with_gradient()``, mainly for some postprocessing steps such as
       copying a training model to a target model in SAC or DQN.
    9. ``after_train_iter()``: called by ``train_iter()`` after every call to
       ``train_from_unroll()`` (on-policy training iter) or
       ``train_from_replay_buffer`` (off-policy training iter). It's mainly for
       training additional modules that have their own training logic (e.g.,
       on/off-policy, replay buffers, etc). Other things might also be possible
       as long as they should be done once every training iteration.

    .. note::
        A base (non-RL) algorithm will not directly interact with an
        environment. The interation loop will always be driven by an
        ``RLAlgorithm`` that outputs actions and gets rewards. So a base
        (non-RL) algorithm is always attached to an ``RLAlgorithm`` and cannot
        change the timing of (when to launch) a training iteration. However, it
        can have its own logic of a training iteration (e.g.,
        ``train_from_unroll()`` and ``train_from_replay_buffer()``) which can be
        triggered by a parent ``RLAlgorithm`` inside its ``after_train_iter()``.
    """

    def __init__(self,
                 train_state_spec=(),
                 rollout_state_spec=None,
                 predict_state_spec=None,
                 optimizer=None,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="Algorithm"):
        """Each algorithm can have a default optimimzer. By default, the parameters
        and/or modules under an algorithm are optimized by the default
        optimizer. One can also specify an optimizer for a set of parameters
        and/or modules using add_optimizer.

        A requirement for this optimizer structure to work is that there is no
        algorithm which is a submodule of a non-algorithm module. Currently,
        this is not checked by the framework. It's up to the user to make sure
        this is true.

        Args:
            train_state_spec (nested TensorSpec): for the network state of
                ``train_step()``.
            rollout_state_spec (nested TensorSpec): for the network state of
                ``predict_step()``. If None, it's assumed to be the same as
                ``train_state_spec``.
            predict_state_spec (nested TensorSpec): for the network state of
                ``predict_step()``. If None, it's assume to be same as
                ``rollout_state_spec``.
            optimizer (None|Optimizer): The default optimizer for
                training. See comments above for detail.
            config (TrainerConfig): config for training. ``config`` only needs to
                be provided to the algorithm which performs a training iteration
                by itself.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this algorithm.
        """
        super(Algorithm, self).__init__()

        self._name = name
        self._config = config

        self._train_state_spec = train_state_spec
        if rollout_state_spec is not None:
            self._rollout_state_spec = rollout_state_spec
        else:
            self._rollout_state_spec = self._train_state_spec
        if predict_state_spec is not None:
            self._predict_state_spec = predict_state_spec
        else:
            self._predict_state_spec = self._rollout_state_spec

        self._initial_train_states = {}
        self._initial_rollout_states = {}
        self._initial_predict_states = {}
        self._initial_transform_states = {}

        self._experience_spec = None
        self._train_info_spec = None
        self._processed_experience_spec = None

        if config and config.data_transformer:
            self._data_transformer = config.data_transformer
        else:
            self._data_transformer = IdentityDataTransformer()
        self._num_earliest_frames_ignored = self._data_transformer.stack_size - 1
        self._transform_state_spec = self._data_transformer.state_spec

        self._observers = []
        self._metrics = []
        self._exp_replayer = None
        self._exp_replayer_type = None

        self._use_rollout_state = False
        if config:
            self.use_rollout_state = config.use_rollout_state
            if config.temporally_independent_train_step is None:
                config.temporally_independent_train_step = (len(
                    alf.nest.flatten(self.train_state_spec)) == 0)

        self._is_rnn = len(alf.nest.flatten(train_state_spec)) > 0

        self._debug_summaries = debug_summaries
        self._default_optimizer = optimizer
        self._optimizers = []
        self._opt_keys = []
        self._module_to_optimizer = {}
        if optimizer:
            self._optimizers.append(optimizer)

    def is_rl(self):
        """Always returns False for non-RL algorithms."""
        return False

    @property
    def name(self):
        """The name of this algorithm."""
        return self._name

    def _set_children_property(self, property_name, value):
        """Set the property named ``property_name`` in child Algorithm to
        ``value``.
        """
        for child in self._get_children():
            if isinstance(child, Algorithm):
                child.__setattr__(property_name, value)

    def need_full_rollout_state(self):
        """Whether ``AlgStep.state`` from ``rollout_step`` should be full.

        If True, it means that ``rollout_step()`` should return the complete state
        for ``train_step()``.
        """
        return self._is_rnn and self._use_rollout_state

    @property
    def use_rollout_state(self):
        """If True, when off-policy training, the RNN states will be taken
        from the replay buffer; otherwise they will be set to 0.

        In the case of True, the ``train_state_spec`` of an algorithm should always
        be a subset of the ``rollout_state_spec``.
        """
        return self._use_rollout_state

    @use_rollout_state.setter
    def use_rollout_state(self, flag):
        self._use_rollout_state = flag
        self._set_children_property('use_rollout_state', flag)

    def set_exp_replayer(self,
                         exp_replayer: str,
                         num_envs,
                         max_length: int,
                         prioritized_sampling=False):
        """Set experience replayer.

        Args:
            exp_replayer (str): type of experience replayer. One of ("one_time",
                "uniform")
            num_envs (int): the total number of environments from all batched
                environments.
            max_length (int): the maximum number of steps the replay
                buffer store for each environment.
            prioritized_sampling (bool): Use prioritized sampling if this is True.
        """
        assert exp_replayer in ("one_time", "uniform"), (
            "Unsupported exp_replayer: %s" % exp_replayer)
        self._exp_replayer_type = exp_replayer
        self._exp_replayer_num_envs = num_envs
        self._exp_replayer_length = max_length
        self._prioritized_sampling = prioritized_sampling

    def _set_exp_replayer(self, sample_exp):
        """Initialize the experience replayer for the very first time given a
        sample experience which is used to infer the specs for the buffer
        initialization.

        Args:
            sample_exp (nested Tensor):
        """
        self._experience_spec = dist_utils.extract_spec(sample_exp, from_dim=1)
        self._exp_contains_step_type = ('step_type' in dict(
            alf.nest.extract_fields_from_nest(sample_exp)))

        if self._exp_replayer_type == "one_time":
            self._exp_replayer = OnetimeExperienceReplayer()
        elif self._exp_replayer_type == "uniform":
            exp_spec = dist_utils.to_distribution_param_spec(
                self._experience_spec)
            self._exp_replayer = SyncExperienceReplayer(
                exp_spec,
                self._exp_replayer_num_envs,
                self._exp_replayer_length,
                prioritized_sampling=self._prioritized_sampling,
                num_earliest_frames_ignored=self._num_earliest_frames_ignored,
                name="exp_replayer")
        else:
            raise ValueError("invalid experience replayer name")
        self._observers.append(self._exp_replayer.observe)

    def observe_for_replay(self, exp):
        r"""Record an experience in a replay buffer.

        Args:
            exp (nested Tensor): exp (nested Tensor): The shape is
                :math:`[B, \ldots]`, where :math:`B` is the batch size of the
                batched environment.
        """
        if not self._use_rollout_state:
            exp = exp._replace(state=())
        elif id(self.rollout_state_spec) != id(self.train_state_spec):
            # Prune exp's state (rollout_state) according to the train state spec
            exp = exp._replace(
                state=alf.nest.prune_nest_like(
                    exp.state, self.train_state_spec, value_to_match=()))

        if self._exp_replayer is None and self._exp_replayer_type:
            self._set_exp_replayer(exp)

        exp = dist_utils.distributions_to_params(exp)
        for observer in self._observers:
            observer(exp)

    def observe_for_metrics(self, time_step):
        r"""Observe a time step for recording environment metrics.

        Args:
            time_step (TimeStep): the current time step during ``unroll()``.
        """
        for metric in self._metrics:
            metric(time_step)

    def transform_timestep(self, time_step, state):
        """Transform time_step.

        ``transform_timestep`` is called for all raw time_step got from
        the environment before passing to ``predict_step`` and ``rollout_step``. For
        off-policy algorithms, the replay buffer stores raw time_step. So when
        experiences are retrieved from the replay buffer, they are tranformed by
        ``transform_timestep`` in ``OffPolicyAlgorithm`` before passing to
        ``_update()``.

        The transformation should be stateless. By default, only observation
        is transformed.

        Args:
            time_step (TimeStep or Experience): time step
            state (nested Tensor): state of the transformer(s)
        Returns:
            TimeStep or Experience: transformed time step
        """
        return self._data_transformer.transform_timestep(time_step, state)

    def transform_experience(self, experience):
        """Transform an Experience structure.

        This is used on the experience data retrieved from replay buffer.

        Args:
            experience (Experience): the experience retrieved from replay buffer.
                Note that ``experience.batch_info``, ``experience.replay_buffer``
                need to be set.
        Returns:
            Experience: transformed experience
        """
        return self._data_transformer.transform_experience(experience)

    def preprocess_experience(self, experience):
        """This function is called on the experiences obtained from a replay
        buffer. An example usage of this function is to calculate advantages and
        returns in ``PPOAlgorithm``.

        The shapes of tensors in experience are assumed to be :math:`(B, T, ...)`.

        Args:
            experience (nest): original experience
        Returns:
            processed experience
        """
        return experience

    def summarize_train(self, experience, train_info, loss_info, params):
        """Generate summaries for training & loss info after each gradient update.
        The default implementation of this function only summarizes params
        (with grads) and the loss. An algorithm can override this for additional
        summaries. See ``RLAlgorithm.summarize_train()`` for an example.

        Args:
            experience (nested Tensor): samples used for the most recent
                ``update_with_gradient()``. By default it's not summarized.
            train_info (nested Tensor): ``AlgStep.info`` returned by either
                ``rollout_step()`` (on-policy training) or ``train_step()``
                (off-policy training). By default it's not summarized.
            loss_info (LossInfo): loss
            params (list[Parameter]): list of parameters with gradients
        """
        if self._config.summarize_grads_and_vars:
            summary_utils.summarize_variables(params)
            summary_utils.summarize_gradients(params)
        if self._debug_summaries:
            summary_utils.summarize_loss(loss_info)

    def add_optimizer(self, optimizer: torch.optim.Optimizer,
                      modules_and_params):
        """Add an optimizer.

        Note that the modules and params contained in ``modules_and_params``
        should still be the attributes of the algorithm (i.e., they can be
        retrieved in ``self.children()`` or ``self.parameters()``).

        Args:
            optimizer (Optimizer): optimizer
            modules_and_params (list of Module or Parameter): The modules and
                parameters to be optimized by ``optimizer``.
        """
        assert optimizer is not None, "You shouldn't add a None optimizer!"
        for module in modules_and_params:
            for m in _flatten_module(module):
                self._module_to_optimizer[m] = optimizer
        self._optimizers.append(optimizer)

    def _trainable_attributes_to_ignore(self):
        """Algorithms can overwrite this function to provide which class
        member names should be ignored when getting trainable variables, to
        avoid being assigned with multiple optimizers.

        For example, if in your algorithm you've created a member ``self._vars``
        pointing to the variables of a module for some purpose, you can avoid
        assigning an optimizer to ``self._vars`` (because the module will be assigned
        with one) by doing:

        .. code-block:: python

            def _trainable_attributes_to_ignore(self):
                return ["_vars"]

        Returns:
            list[str]: a list of attribute names to ignore.
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
            string: the name if the parameter can be found; otherwise ``None``.
        """
        return self._param_to_name.get(param)

    def _setup_optimizers(self):
        """Setup the param groups for optimizers.

        Returns:
            list: a list of parameters not handled by any optimizers under this
            algorithm.
        """
        self._assert_no_cycle_or_duplicate()
        self._param_to_name = {}

        for name, param in self.named_parameters():
            self._param_to_name[param] = name

        return self._setup_optimizers_()[0]

    def _setup_optimizers_(self):
        """Setup the param groups for optimizers.

        Returns:
            tuple:

            - list of parameters not handled by any optimizers under this algorithm
            - list of parameters not handled under this algorithm
        """
        default_optimizer = self.default_optimizer
        new_params = []
        handled = set()
        duplicate_error = "Parameter %s is handled by muliple optimizers."

        def _add_params_to_optimizer(params, opt):
            existing_params = _get_optimizer_params(opt)
            params = list(filter(lambda p: p not in existing_params, params))
            if params:
                opt.add_param_group({'params': params})

        for child in self._get_children():
            if child in handled:
                continue
            assert id(child) != id(self), "Child should not be self"
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
                _add_params_to_optimizer(params, optimizer)

        if default_optimizer is not None:
            _add_params_to_optimizer(new_params, default_optimizer)
            return [], handled
        else:
            return new_params, handled

    def optimizers(self, recurse=True, include_ignored_attributes=False):
        """Get all the optimizers used by this algorithm.

        Args:
            recurse (bool): If True, including all the sub-algorithms
            include_ignored_attributes (bool): If True, still include all child
                attributes without ignoring any.
        Returns:
            list: list of ``Optimizer``s.
        """
        opts = copy.copy(self._optimizers)
        if recurse:
            if include_ignored_attributes:
                children = self.children()
            else:
                children = self._get_children()
            for module in children:
                if isinstance(module, Algorithm):
                    opts.extend(
                        module.optimizers(recurse, include_ignored_attributes))
        return opts

    def get_optimizer_info(self):
        """Return the optimizer info for all the modules in a string.

        TODO: for a subalgorithm that's an ignored attribute, its optimizer info
        won't be obtained.
        """
        unhandled = self._setup_optimizers()

        optimizer_info = []
        if unhandled:
            optimizer_info.append(
                dict(
                    optimizer="None",
                    parameters=[self._param_to_name[p] for p in unhandled]))

        for optimizer in self.optimizers(include_ignored_attributes=True):
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
        """Returns the RNN state spec for ``predict_step()``."""
        return self._predict_state_spec

    @property
    def rollout_state_spec(self):
        """Returns the RNN state spec for ``rollout_step()``."""
        return self._rollout_state_spec

    @property
    def train_state_spec(self):
        """Returns the RNN state spec for ``train_step()``."""
        return self._train_state_spec

    @property
    def train_info_spec(self):
        """The spec for the ``AlgStep.info`` returned from ``train_step()``."""
        assert self._train_info_spec is not None, (
            "train_step() has not been called. train_info_spec is not available."
        )
        return self._train_info_spec

    @property
    def experience_spec(self):
        """Spec for experience."""
        assert self._experience_spec is not None, (
            "observe() has not been called. experience_spec is not avaialble.")
        return self._experience_spec

    @property
    def processed_experience_spec(self):
        """Spec for processed experience.

        Returns:
            TensorSpec: Spec for the experience returned by ``preprocess_experience()``.
        """
        assert self._processed_experience_spec is not None, (
            "preprocess_experience() has not been used. processed_experience_spec"
            "is not available")
        return self._processed_experience_spec

    def convert_train_state_to_predict_state(self, state):
        """Convert RNN state for ``train_step()`` to RNN state for
        ``predict_step()``."""
        alf.nest.assert_same_structure(self._train_state_spec,
                                       self._predict_state_spec)
        return state

    def get_initial_transform_state(self, batch_size):
        r = self._initial_transform_states.get(batch_size)
        if r is None:
            r = spec_utils.zeros_from_spec(self._transform_state_spec,
                                           batch_size)
            self._initial_transform_states[batch_size] = r
        return r

    def get_initial_predict_state(self, batch_size):
        r = self._initial_predict_states.get(batch_size)
        if r is None:
            r = spec_utils.zeros_from_spec(self._predict_state_spec,
                                           batch_size)
            self._initial_predict_states[batch_size] = r
        return r

    def get_initial_rollout_state(self, batch_size):
        r = self._initial_rollout_states.get(batch_size)
        if r is None:
            r = spec_utils.zeros_from_spec(self._rollout_state_spec,
                                           batch_size)
            self._initial_rollout_states[batch_size] = r
        return r

    def get_initial_train_state(self, batch_size):
        r = self._initial_train_states.get(batch_size)
        if r is None:
            r = spec_utils.zeros_from_spec(self._train_state_spec, batch_size)
            self._initial_train_states[batch_size] = r
        return r

    @common.add_method(nn.Module)
    def state_dict(self, destination=None, prefix='', visited=None):
        """Get state dictionary recursively, including both model state
        and optimizers' state (if any). It can handle a number of special cases:

        - graph with cycle: save all the states and avoid infinite loop
        - parameter sharing: save only one copy of the shared module/param
        - optimizers: save the optimizers for all the (sub-)algorithms

        Args:
            destination (OrderedDict): the destination for storing the state.
            prefix (str): a string to be added before the name of the items
                (modules, params, algorithms etc) as the key used in the
                state dictionary.
            visited (set): a set keeping track of the visited objects.

        Returns:
            OrderedDict: the dictionary including both model state and optimizers'
            state (if any).
        """

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(
            version=self._version)

        if visited is None:
            visited = {self}

        if not is_checkpoint_enabled(self):
            return destination

        self._save_to_state_dict(destination, prefix, visited)
        opts_dict = OrderedDict()
        for name, child in self._modules.items():
            if child is not None and child not in visited:
                visited.add(child)
                child.state_dict(
                    destination, prefix + name + '.', visited=visited)
        if isinstance(self, Algorithm):
            self._setup_optimizers()
            for i, opt in enumerate(self._optimizers):
                new_key = prefix + '_optimizers.%d' % i
                if new_key not in self._opt_keys:
                    self._opt_keys.append(new_key)
                opts_dict[self._opt_keys[i]] = opt.state_dict()

            destination.update(opts_dict)

        return destination

    @common.add_method(nn.Module)
    def load_state_dict(self, state_dict, strict=True):
        """Load state dictionary for the algorithm.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in ``state_dict`` match the keys returned by this module's
                ``torch.nn.Module.state_dict`` function. If ``strict=True``, will
                keep lists of missing and unexpected keys and raise error when
                any of the lists is non-empty; if ``strict=False``, missing/unexpected
                keys will be omitted and no error will be raised.
                (Default: ``True``)

        Returns:
            namedtuple:

            - missing_keys: a list of str containing the missing keys.
            - unexpected_keys: a list of str containing the unexpected keys.
        """
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def _load(module, prefix='', visited=None):
            if visited is None:
                visited = {self}
            if not is_checkpoint_enabled(module):
                return
            if isinstance(module, Algorithm):
                module._setup_optimizers()
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
                    _load(child, prefix + name + '.', visited=visited)

            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            if type(module)._load_from_state_dict in (
                    Algorithm._load_from_state_dict,
                    nn.Module._load_from_state_dict):
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys,
                    unexpected_keys, error_msgs, visited)
            else:
                # Some pytorch modules (e.g. BatchNorm layers) override
                # _load_from_state_dict, which uses the original
                # Module._load_from_state_dict. So we have to handle them
                # differently. Not using `visited` should not cause a problem
                # because those modules are not implemented by ALF and will not
                # have cycle through them.
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys,
                    unexpected_keys, error_msgs)

        _load(self)

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
        r"""Saves module state to ``destination`` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in ``torch.nn.Module.state_dict``. In rare cases, subclasses
        can achieve class-specific behavior by overriding this method with custom
        logic.

        Args:
            destination (dict): a dict where state will be stored.
            prefix (str): the prefix for parameters and buffers used in this
                module.
            visited (set): a set keeping track of the visited objects.
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
        """Copies parameters and buffers from ``state_dict`` into only
        this module, but not its descendants. This is called on every submodule
        in ``torch.nn.Module.load_state_dict``. Metadata saved for this
        module in input ``state_dict`` is provided as ``local_metadata``.
        For state dicts without metadata, ``local_metadata`` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at ``local_metadata.get("version", None)``.

        .. note::

            ``state_dict`` is not the same object as the input ``state_dict`` to
            ``torch.nn.Module.load_state_dict``. So it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module.
            local_metadata (dict): a dict containing the metadata for this module.
            strict (bool): whether to strictly enforce that the keys in
                ``state_dict`` with ``prefix`` match the names of
                parameters and buffers in this module; if ``strict=True``,
                will keep a list of missing and unexpected keys.
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list.
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list.
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                ``torch.nn.Module.load_state_dict``.
            visited (set): a set keeping track of the visited objects.
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
            inputs (nested Tensor): inputs for prediction.
            state (nested Tensor): network state (for RNN).

        Returns:
            AlgStep:
            - output (nested Tensor): prediction result.
            - state (nested Tensor): should match ``predict_state_spec``.
        """
        algorithm_step = self.rollout_step(inputs, state)
        return algorithm_step._replace(info=None)

    def rollout_step(self, inputs, state=None):
        """Rollout for one step of inputs.

        Args:
            inputs (nested Tensor): inputs for prediction.
            state (nested Tensor): network state (for RNN).

        Returns:
            AlgStep:
            - output (nested Tensor): prediction result.
            - state (nested Tensor): should match ``rollout_state_spec``.
        """
        algorithm_step = self.train_step(inputs, state)
        return algorithm_step._replace(info=None)

    def train_step(self, inputs, state=None):
        """Perform one step of training computation.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            inputs (nested Tensor): inputs for train.
            state (nested Tensor): consistent with ``train_state_spec``.

        Returns:
            AlgStep:
            - output (nested Tensor): predict outputs.
            - state (nested Tensor): should match ``train_state_spec``.
            - info (nested Tensor): information for training. If this is
              ``LossInfo``, ``calc_loss()`` in ``Algorithm`` can be used.
              Otherwise, the user needs to override ``calc_loss()`` to
              calculate loss or override ``update_with_gradient()`` to do
              customized training.
        """
        return AlgStep()

    # Subclass may override update_with_gradient() to allow customized training
    def update_with_gradient(self,
                             loss_info,
                             valid_masks=None,
                             weight=1.0,
                             batch_info=None):
        """Complete one iteration of training.

        Update parameters using the gradient with respect to ``loss_info``.

        Args:
            loss_info (LossInfo): loss with shape :math:`(T, B)` (except for
                ``loss_info.scalar_loss``)
            valid_masks (tf.Tensor): masks indicating which samples are valid.
                (``shape=(T, B), dtype=tf.float32``)
            weight (float): weight for this batch. Loss will be multiplied with
                this weight before calculating gradient.
            batch_info (BatchInfo): information about this batch returned by
                ``ReplayBuffer.get_batch()``
        Returns:
            tuple:
            - loss_info (LossInfo): loss information.
            - params (list[(name, Parameter)]): list of parameters being updated.
        """
        masks = None
        if (batch_info is not None and batch_info.importance_weights != ()
                and self._config.priority_replay_beta != 0):
            masks = batch_info.importance_weights.pow(
                -self._config.priority_replay_beta).unsqueeze(0)

        if valid_masks is not None:
            if masks is not None:
                masks = masks * valid_masks
            else:
                masks = valid_masks

        if masks is not None:
            loss_info = alf.nest.map_structure(
                lambda l: torch.mean(l * masks) if len(l.shape) == 2 else l,
                loss_info)
        else:
            loss_info = alf.nest.map_structure(lambda l: torch.mean(l),
                                               loss_info)
        if isinstance(loss_info.scalar_loss, torch.Tensor):
            assert len(loss_info.scalar_loss.shape) == 0
            loss_info = loss_info._replace(
                loss=add_ignore_empty(loss_info.loss, loss_info.scalar_loss))
        loss = weight * loss_info.loss

        unhandled = self._setup_optimizers()
        unhandled = [self._param_to_name[p] for p in unhandled]
        assert not unhandled, ("'%s' has some modules/parameters do not have "
                               "optimizer: %s" % (self.name, unhandled))
        optimizers = self.optimizers()
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()

        all_params = []
        for optimizer in optimizers:
            params = []
            for param_group in optimizer.param_groups:
                params.extend(param_group['params'])
            assert params, (
                "The recorded optimizer '" + optimizer.name +
                "' haven't been used for learning any parameters! Please check."
            )
            all_params.extend(params)
            optimizer.step()

        all_params = [(self._param_to_name[p], p) for p in all_params]
        return loss_info, all_params

    def after_update(self, experience, train_info):
        """Do things after completing one gradient update (i.e. ``update_with_gradient()``).
        This function can be used for post-processings following one minibatch
        update, such as copy a training model to a target model in SAC, DQN, etc.

        Args:
            experience (nest): experiences collected for the most recent
                ``update_with_gradient()``.
            train_info (nest): information collected for training.
                It is batched from each ``AlgStep.info`` returned by ``rollout_step()``
                or ``train_step()``.
        """
        pass

    def after_train_iter(self, experience, train_info=None):
        """Do things after completing one training iteration (i.e. ``train_iter()``
        that consists of one or multiple gradient updates). This function can
        be used for training additional modules that have their own training logic
        (e.g., on/off-policy, replay buffers, etc). These modules should be added
        to ``_trainable_attributes_to_ignore`` in the parent algorithm.

        Other things might also be possible as long as they should be done once
        every training iteration.

        This function will serve the same purpose with ``after_update`` if there
        is always only one gradient update in each training iteration. Otherwise
        it's less frequently called than ``after_update``.

        Args:
            experience (nest): experience collected during ``unroll()``.
                Note that it won't contain the field ``rollout_info`` because this
                is the info collected just from the unroll but not from a replay
                buffer. And ``rollout_info`` has been assigned to ``train_info``.
            train_info (nest): information collected during ``unroll()``. If it's
                ``None``, then only off-policy training is allowed. Currently
                this arg is ``None`` when:

                - This function is called by ``_train_iter_on_policy``, because
                  it's not recomended to backprop on the same graph twice.
                - This function is called by ``_train_iter_off_policy`` with
                  ``config.unroll_with_grad=False``.
        """
        pass

    # Subclass may override calc_loss() to allow more sophisticated loss
    def calc_loss(self, experience, train_info):
        """Calculate the loss at each step for each sample.

        Args:
            experience (Experience): experiences collected from the most recent
                ``unroll()`` or from a replay buffer. It's used for the most
                recent ``update_with_gradient()``.
            train_info (nest): information collected for training. It is batched
                from each ``AlgStep.info`` returned by ``rollout_step()``
                (on-policy training) or ``train_step()`` (off-policy training).
        Returns:
            LossInfo: loss at each time step for each sample in the
                batch. The shapes of the tensors in loss info should be
                :math:`(T, B)`.
        """
        assert isinstance(train_info, LossInfo), (
            "train_info returned by"
            " train_step() should be LossInfo. Otherwise you need override"
            " calc_loss() to generate LossInfo from train_info")
        return train_info

    def train_from_unroll(self, experience, train_info):
        """Train given the info collected from ``unroll()``. This function can
        be called by any child algorithm that doesn't have the unroll logic but
        has a different training logic with its parent (e.g., off-policy).

        Args:
            experience (Experience): collected during ``unroll()``.
            train_info (nest): ``AlgStep.info`` returned by ``rollout_step()``.

        Returns:
            int: number of steps that have been trained
        """
        if self.is_rl():
            valid_masks = (experience.step_type != StepType.LAST).to(
                torch.float32)
        else:
            valid_masks = None
        experience = experience._replace(rollout_info_field='rollout_info')
        loss_info = self.calc_loss(experience, train_info)
        loss_info, params = self.update_with_gradient(loss_info, valid_masks)
        self.after_update(experience, train_info)
        self.summarize_train(experience, train_info, loss_info, params)
        return torch.tensor(alf.nest.get_nest_shape(experience)).prod()

    @common.mark_replay
    def train_from_replay_buffer(self, update_global_counter=False):
        """This function can be called by any algorithm that has its own
        replay buffer configured. There are several parameters specified in
        ``self._config`` that will affect how the training is performed:

        - ``initial_collect_steps``: only start replaying and training after so
          many time steps have been stored in the replay buffer
        - ``mini_batch_size``: the batch size of a minibatch
        - ``mini_batch_length``: the temporal extension of a minibatch. An
          algorithm can sample temporally correlated experiences for training
          stateful models by setting this value greater than 1.
        - ``num_updates_per_train_iter``: how many updates to perform in each
          training iteration. Its behavior might be different depending on the
          value of ``config.whole_replay_buffer_training``:

          - If ``True``, each update will scan over the entire buffer to get
            chopped minibatches and a random experience shuffling is performed
            before each update;
          - If ``False``, each update will sample a new minibatch from the replay
            buffer.

        - ``whole_replay_buffer_training``: a very special case where all data in
          the replay buffer will be used for training (e.g., PPO). In this case,
          for every update in ``num_updates_per_train_iter``, the data will
          be shuffled and divided into
          ``buffer_size//(mini_batch_size * mini_batch_length)`` "mini-updates".

        Args:
            update_global_counter (bool): controls whether this function changes
                the global counter for summary. If there are multiple
                algorithms, then only the parent algorithm should change this
                quantity and child algorithms should disable the flag. When it's
                ``True``, it will affect the counter only if
                ``config.update_counter_every_mini_batch=True``.
        """
        config: TrainerConfig = self._config

        if self._exp_replayer.total_size < config.initial_collect_steps:
            # returns 0 if haven't started training yet; throughput will be 0
            return 0

        # TODO: If this function can be called asynchronously, and using
        # prioritized replay, then make sure replay and train below is atomic.
        with record_time("time/replay"):
            mini_batch_size = config.mini_batch_size
            if mini_batch_size is None:
                mini_batch_size = self._exp_replayer.batch_size
            if config.whole_replay_buffer_training:
                experience = self._exp_replayer.replay_all()
                if config.clear_replay_buffer:
                    self._exp_replayer.clear()
                num_updates = config.num_updates_per_train_iter
                batch_info = None
            else:
                experience, batch_info = self._exp_replayer.replay(
                    sample_batch_size=(
                        mini_batch_size * config.num_updates_per_train_iter),
                    mini_batch_length=config.mini_batch_length)
                num_updates = 1

        with record_time("time/train"):
            return self._train_experience(
                experience, batch_info, num_updates, mini_batch_size,
                config.mini_batch_length,
                (config.update_counter_every_mini_batch
                 and update_global_counter))

    def _train_experience(self, experience, batch_info, num_updates,
                          mini_batch_size, mini_batch_length,
                          update_counter_every_mini_batch):
        """Train using experience."""
        experience = dist_utils.params_to_distributions(
            experience, self.experience_spec)
        experience = self._add_batch_info(experience, batch_info)
        if self._exp_replayer_type != "one_time":
            # The experience put in one_time replayer is already transformed
            # in unroll().
            experience = self.transform_experience(experience)
        experience = self.preprocess_experience(experience)
        experience = self._clear_batch_info(experience)
        if self._processed_experience_spec is None:
            self._processed_experience_spec = dist_utils.extract_spec(
                experience, from_dim=2)
        experience = dist_utils.distributions_to_params(experience)

        length = alf.nest.get_nest_size(experience, dim=1)
        mini_batch_length = (mini_batch_length or length)
        if batch_info is not None:
            assert mini_batch_length == length, (
                "mini_batch_length (%s) is "
                "different from length (%s). Not supported." %
                (mini_batch_length, length))

        if mini_batch_length > length:
            common.warning_once(
                "mini_batch_length=%s is set to a smaller length=%s" %
                (mini_batch_length, length))
            mini_batch_length = length
        elif length % mini_batch_length:
            common.warning_once(
                "length=%s not a multiple of mini_batch_length=%s" %
                (length, mini_batch_length))
            length = length // mini_batch_length * mini_batch_length
            experience = alf.nest.map_structure(lambda x: x[:, :length, ...],
                                                experience)
            common.warning_once(
                "Experience length has been cut to %s" % length)

        if len(alf.nest.flatten(self.train_state_spec)) > 0:
            if not self._use_rollout_state:
                # If not using rollout states, then we will assume zero initial
                # training states. To have a proper state warm up,
                # mini_batch_length should be greater than 1. Otherwise the states
                # are always 0s.
                if mini_batch_length == 1:
                    logging.fatal(
                        "Should use TrainerConfig.use_rollout_state=True "
                        "for training from a replay buffer when minibatch_length==1, "
                        "otherwise the initial states are always zeros!")
                else:
                    # In this case, a state warm up is recommended. For example,
                    # having mini_batch_length>1 and discarding first several
                    # steps when computing losses. For a warm up, make sure to
                    # leave a mini_batch_length > 1 if any recurrent model is to
                    # be trained.
                    common.warning_once(
                        "Consider using TrainerConfig.use_rollout_state=True "
                        "for training from a replay buffer.")
            elif mini_batch_length == 1:
                # If using rollout states and mini_batch_length=1, there will be
                # no gradient flowing in any recurrent matrix. Only the output
                # layers on top of the recurrent output will be trained.
                common.warning_once(
                    "Using rollout states but with mini_batch_length=1. In "
                    "this case, any recurrent model can't be properly trained!"
                )
            else:
                # If using rollout states with mini_batch_length>1. In theory,
                # any recurrent model can be properly trained. With a greater
                # mini_batch_length, the temporal correlation can be better
                # captured.
                pass

        experience = alf.nest.map_structure(
            lambda x: x.reshape(-1, mini_batch_length, *x.shape[2:]),
            experience)

        batch_size = alf.nest.get_nest_batch_size(experience)

        def _make_time_major(nest):
            """Put the time dim to axis=0."""
            return alf.nest.map_structure(lambda x: x.transpose(0, 1), nest)

        for u in range(num_updates):
            if mini_batch_size < batch_size:
                indices = torch.randperm(batch_size)
                experience = alf.nest.map_structure(lambda x: x[indices],
                                                    experience)
                if batch_info is not None:
                    batch_info = alf.nest.map_structure(
                        lambda x: x[indices], batch_info)
            for b in range(0, batch_size, mini_batch_size):
                if update_counter_every_mini_batch:
                    alf.summary.increment_global_counter()
                is_last_mini_batch = (u == num_updates - 1
                                      and b + mini_batch_size >= batch_size)
                do_summary = (is_last_mini_batch
                              or update_counter_every_mini_batch)
                alf.summary.enable_summary(do_summary)
                batch = alf.nest.map_structure(
                    lambda x: x[b:min(batch_size, b + mini_batch_size)],
                    experience)
                if batch_info:
                    binfo = alf.nest.map_structure(
                        lambda x: x[b:min(batch_size, b + mini_batch_size)],
                        batch_info)
                else:
                    binfo = None
                batch = _make_time_major(batch)
                exp, train_info, loss_info, params = self._update(
                    batch,
                    binfo,
                    weight=alf.nest.get_nest_size(batch, 1) / mini_batch_size)
                if do_summary:
                    self.summarize_train(exp, train_info, loss_info, params)

        train_steps = batch_size * mini_batch_length * num_updates
        return train_steps

    def _collect_train_info_sequentially(self, experience):
        batch_size = alf.nest.get_nest_size(experience, dim=1)
        initial_train_state = self.get_initial_train_state(batch_size)
        if self._use_rollout_state:
            policy_state = alf.nest.map_structure(lambda state: state[0, ...],
                                                  experience.state)
        else:
            policy_state = initial_train_state

        num_steps = alf.nest.get_nest_size(experience, dim=0)
        info_list = []
        for counter in range(num_steps):
            exp = alf.nest.map_structure(lambda ta: ta[counter], experience)
            exp = dist_utils.params_to_distributions(
                exp, self.processed_experience_spec)
            if self._exp_contains_step_type:
                policy_state = common.reset_state_if_necessary(
                    policy_state, initial_train_state,
                    exp.step_type == StepType.FIRST)
            elif policy_state != ():
                common.warning_once(
                    "Policy state is non-empty but the experience doesn't "
                    "contain the 'step_type' field. No way to reinitialize "
                    "the state but will simply keep updating it.")
            policy_step = self.train_step(exp, policy_state)
            if self._train_info_spec is None:
                self._train_info_spec = dist_utils.extract_spec(
                    policy_step.info)
            info_list.append(
                dist_utils.distributions_to_params(policy_step.info))
            policy_state = policy_step.state

        info = alf.nest.utils.stack_nests(info_list)
        info = dist_utils.params_to_distributions(info, self.train_info_spec)
        return info

    def _collect_train_info_parallelly(self, experience):
        shape = alf.nest.get_nest_shape(experience)
        length, batch_size = shape[:2]

        exp = alf.nest.map_structure(lambda x: x.reshape(-1, *x.shape[2:]),
                                     experience)

        if self._use_rollout_state:
            policy_state = exp.state
        else:
            size = alf.nest.get_nest_size(exp, dim=0)
            policy_state = self.get_initial_train_state(size)

        exp = dist_utils.params_to_distributions(
            exp, self.processed_experience_spec)
        policy_step = self.train_step(exp, policy_state)

        if self._train_info_spec is None:
            self._train_info_spec = dist_utils.extract_spec(policy_step.info)
        info = dist_utils.distributions_to_params(policy_step.info)
        info = alf.nest.map_structure(
            lambda x: x.reshape(length, batch_size, *x.shape[1:]), info)
        info = dist_utils.params_to_distributions(info, self.train_info_spec)
        return info

    def _add_batch_info(self, experience, batch_info):
        if batch_info is not None:
            experience = experience._replace(
                batch_info=batch_info,
                replay_buffer=self._exp_replayer.replay_buffer)
        return experience._replace(rollout_info_field='rollout_info')

    def _clear_batch_info(self, experience):
        return experience._replace(
            batch_info=(), replay_buffer=(), rollout_info_field=())

    def _update(self, experience, batch_info, weight):
        length = alf.nest.get_nest_size(experience, dim=0)
        if self._config.temporally_independent_train_step or length == 1:
            train_info = self._collect_train_info_parallelly(experience)
        else:
            train_info = self._collect_train_info_sequentially(experience)

        experience = dist_utils.params_to_distributions(
            experience, self.processed_experience_spec)

        experience = self._add_batch_info(experience, batch_info)
        loss_info = self.calc_loss(experience, train_info)
        if loss_info.priority != ():
            priority = (loss_info.priority**self._config.priority_replay_alpha
                        + self._config.priority_replay_eps)
            self._exp_replayer.update_priority(batch_info.env_ids,
                                               batch_info.positions, priority)

        if self.is_rl():
            valid_masks = (experience.step_type != StepType.LAST).to(
                torch.float32)
        else:
            valid_masks = None
        loss_info, params = self.update_with_gradient(loss_info, valid_masks,
                                                      weight, batch_info)
        self.after_update(experience, train_info)

        return experience, train_info, loss_info, params
