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
from contextlib import nullcontext
import functools
import itertools
import json
import numpy as np
import os
import psutil
from typing import Dict
import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys, _addindent

import alf
from alf.data_structures import AlgStep, LossInfo, StepType, TimeStep
from alf.experience_replayers.replay_buffer import BatchInfo, ReplayBuffer
from alf.optimizers.utils import GradientNoiseScaleEstimator
from alf.utils.checkpoint_utils import (is_checkpoint_enabled,
                                        extract_sub_state_dict_from_checkpoint)
from alf.utils import common, dist_utils, spec_utils, summary_utils
from alf.utils.summary_utils import record_time
from alf.utils.math_ops import add_ignore_empty
from alf.utils.distributed import data_distributed_when
from alf.utils import tensor_utils
from .algorithm_interface import AlgorithmInterface
from .config import TrainerConfig
from .data_transformer import IdentityDataTransformer


def _get_optimizer_params(optimizer: torch.optim.Optimizer):
    return sum([g['params'] for g in optimizer.param_groups], [])


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


class Algorithm(AlgorithmInterface):
    """Base implementation for AlgorithmInterface."""

    def __init__(self,
                 train_state_spec=(),
                 rollout_state_spec=None,
                 predict_state_spec=None,
                 is_on_policy=None,
                 optimizer=None,
                 checkpoint=None,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="Algorithm"):
        """Each algorithm can have a default optimimzer. By default, the parameters
        and/or modules under an algorithm are optimized by the default
        optimizer. One can also specify an optimizer for a set of parameters
        and/or modules using add_optimizer. You can find out which parameter is
        handled by which optimizer using ``get_optimizer_info()``.

        A requirement for this optimizer structure to work is that there is no
        algorithm which is a submodule of a non-algorithm module. Currently,
        this is not checked by the framework. It's up to the user to make sure
        this is true.

        Args:
            train_state_spec (nested TensorSpec): for the network state of
                ``train_step()``.
            rollout_state_spec (nested TensorSpec): for the network state of
                ``rollout_step()``. If None, it's assumed to be the same as
                ``train_state_spec``.
            predict_state_spec (nested TensorSpec): for the network state of
                ``predict_step()``. If None, it's assume to be same as
                ``rollout_state_spec``.
            is_on_policy (None|bool):
            optimizer (None|Optimizer): The default optimizer for
                training. See comments above for detail.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the
                - "prefix" is the prefix to the contents in the checkpoint
                to be loaded. It can be a multi-step path denoted by "A.B.C".
                If the checkpoint comes from a previous ALF training
                session, the standard prefix starts with "alg" (e.g. "alg._sub_alg1").
                If prefix is omitted, the effects is the same as providing "alg",
                which will load the full 'alg' part of the checkpoint.
                - "path" is the full path to the checkpoint file saved
                by ALF, e.g. "/path_to_experiment/train/algorithm/ckpt-100".
                Therefore, an example value for ``checkpoint`` is
                "alg._sub_alg1@/path_to_experiment/train/algorithm/ckpt-100".
            config (TrainerConfig): config for training. ``config`` only needs to
                be provided to the algorithm which performs a training iteration
                by itself.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this algorithm.
        """
        super(Algorithm, self).__init__()

        self._name = name
        self._config = config
        self._proc = psutil.Process(os.getpid())

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
        self._replay_buffer = None

        self._ddp_activated_rank = -1

        # These 3 parameters are only set when ``set_replay_buffer()`` is called.
        self._replay_buffer_num_envs = None
        self._replay_buffer_max_length = None
        self._prioritized_sampling = None

        self._use_rollout_state = False
        # See the ``force_params_visible_to_parent`` property below for details.
        self._force_params_visible_to_parent = False
        self._grad_scaler = None
        self._temporally_independent_train_step = None
        if config:
            self._temporally_independent_train_step = config.temporally_independent_train_step
            self.use_rollout_state = config.use_rollout_state
            if config.enable_amp and torch.cuda.is_available():
                self._grad_scaler = torch.cuda.amp.GradScaler()
        if self._temporally_independent_train_step is None:
            self._temporally_independent_train_step = (len(
                alf.nest.flatten(self.train_state_spec)) == 0)

        self._is_rnn = len(alf.nest.flatten(train_state_spec)) > 0

        self._debug_summaries = debug_summaries
        self._default_optimizer = optimizer
        self._optimizers = []
        self._module_to_optimizer = {}
        self._path = ''
        if optimizer:
            self._optimizers.append(optimizer)
        self._is_on_policy = is_on_policy
        self._gns_estimator = None
        if config:
            self._rl_train_every_update_steps = config.rl_train_every_update_steps
            self._rl_train_after_update_steps = config.rl_train_after_update_steps
            if config.summarize_gradient_noise_scale:
                self._gns_estimator = GradientNoiseScaleEstimator()

        self._checkpoint = checkpoint
        self._checkpoint_pre_loaded = False

    def __init_subclass__(cls, *args, **kwargs):
        """This function is called at the creation of sub-classes of this class.
        Here we customize the ``__init__``function of the input ``cls`` to have
        a post init call when the input ``cls`` is the sub-class.
        """
        super().__init_subclass__(*args, **kwargs)

        # use ``functools.warps`` to keep the signature of the original
        # ``__init__`` function, to ensure alf.config work correctly.
        @functools.wraps(cls.__init__)
        def new_init(self, *args, init=cls.__init__, **kwargs):
            init(self, *args, **kwargs)
            if cls is type(self):
                self._post_init()

        cls.__init__ = new_init

    def _post_init(self):
        """This function will be called automatically in the sub-class
        at the end of the __init__ function, in order to activate _post_init
        functionalities.
        Algorithms can overwrite this function to provide customized post init
        behaviors.
        """
        self._preload_checkpoint()

    def _preload_checkpoint(self):
        """Preload checkpoint to the algorithm, based on the specified ``checkpoint``.
        """

        if self._checkpoint is not None:
            prefix_and_path = self._checkpoint.split('@')
            assert len(prefix_and_path) in [1,
                                            2], ("invalid checkpoint: "
                                                 "{}").format(prefix_and_path)

            if len(prefix_and_path) == 1:
                # only path is provided
                checkpoint_path = prefix_and_path[0]
                checkpoint_prefix = 'alg'
            else:
                checkpoint_prefix, checkpoint_path = prefix_and_path

            assert 'alg' in checkpoint_prefix, "wrong prefix"

            stat_dict = extract_sub_state_dict_from_checkpoint(
                checkpoint_prefix, checkpoint_path)

            status = self.load_state_dict(stat_dict, strict=True)
            # Currently, optimizers are not handled by this function
            missing_keys = list(
                filter(lambda k: k.find('_optimizers.') < 0,
                       status.missing_keys))
            assert not missing_keys and not status.unexpected_keys, (
                "\033[1;31m Checkpoint mismatches with the model: \033[1;0m \n"
                +
                "\033[1;31m Missing-keys \033[1;0m (keys in model but not in checkpoint): {}\n"
                .format(missing_keys) +
                "\033[1;31m Unexpected-keys \033[1;0m (keys in checkpoint but not in model): {}"
                .format(status.unexpected_keys))
            self._checkpoint_pre_loaded = True
            common.info(
                'in-algorithm checkpoint loaded: {}'.format(prefix_and_path))

    @property
    def pre_loaded(self):
        """A property indicating whether a checkpoint for the current instance
        has been pre-loaded, by specifying ``checkpoint_prefix@checkpoint_path``
        where ``checkpoint_prefix@`` is optional.
        """
        return self._checkpoint_pre_loaded

    def forward(self, *input):
        raise RuntimeError("forward() should not be called")

    @property
    def path(self):
        return self._path

    def set_path(self, path):
        self._path = path

    @property
    def on_policy(self):
        return self._is_on_policy

    @property
    def has_offline(self):
        """Whether has offline data for RL algorithms. Always return False
        for non-RL algorithms.
        """
        if self.is_rl():
            return self._has_offline
        else:
            return False

    def set_on_policy(self, is_on_policy):
        if self.on_policy is not None:
            assert self.on_policy == is_on_policy, (
                "set_on_policy() can"
                "only be called to change is_on_policy if is_on_policy is None."
            )
        self._is_on_policy = is_on_policy

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

    def activate_ddp(self, rank: int):
        """Prepare the Algorithm with DistributedDataParallel wrapper

        Note that Algorithm does not need to remember the rank of the device.

        Args:
            rank (int): DDP wrapper needs to know on which GPU device this
                module's parameters and buffers are supposed to be.
        """
        self._ddp_activated_rank = rank

    @use_rollout_state.setter
    def use_rollout_state(self, flag):
        self._use_rollout_state = flag
        self._set_children_property('use_rollout_state', flag)

    @property
    def force_params_visible_to_parent(self) -> bool:
        """Whether the already optimizer-handled parameters are seen by the paranet
        algorithm.

        Normally, when the parameters of this algorithm is handled by its
        optimizer, ``_setup_optimizers_`` will prevent the parent algorithm's
        optimizer to see and more importantly, handle them. Setting this value
        to true will force the parameters to be seen and handled by the parent
        algorithm, even if they are already handled by this algorithm.

        Note that parameters ignored by ``_trainable_attributes_to_ignore()``
        will stay invisible to the parent algorithm.

        It is by default False, and can be changed with the following setter.

        """
        return self._force_params_visible_to_parent

    @force_params_visible_to_parent.setter
    def force_params_visible_to_parent(self, flag: bool):
        self._force_params_visible_to_parent = flag

    def set_replay_buffer(self,
                          num_envs,
                          max_length: int,
                          prioritized_sampling=False):
        """Set the parameters for the replay buffer.

        Args:
            num_envs (int): the total number of environments from all batched
                environments.
            max_length (int): the maximum number of steps the replay
                buffer store for each environment.
            prioritized_sampling (bool): Use prioritized sampling if this is True.
        """
        self._replay_buffer_num_envs = num_envs
        self._replay_buffer_max_length = max_length
        self._prioritized_sampling = prioritized_sampling

    def _set_replay_buffer(self, sample_exp):
        """Initialize the replay buffer for the very first time given a
        sample experience which is used to infer the specs for the buffer
        initialization.

        Args:
            sample_exp (nested Tensor):
        """
        if (self._replay_buffer_num_envs is None
                or self._replay_buffer_max_length is None
                or self._prioritized_sampling is None):
            # Do not even create the replay buffer if the required
            # parameters are not set by set_replay_buffer
            common.warning_once(
                'Experience replayer must be initialized first by calling '
                'set_replay_buffer() before observe_for_replay() is called! '
                'Skipping ...')
            return

        self._experience_spec = dist_utils.extract_spec(sample_exp, from_dim=1)
        self._exp_contains_step_type = (getattr(sample_exp, 'step_type', None)
                                        is not None)

        exp_spec = dist_utils.to_distribution_param_spec(self._experience_spec)
        self._replay_buffer = ReplayBuffer(
            data_spec=exp_spec,
            num_environments=self._replay_buffer_num_envs,
            max_length=self._replay_buffer_max_length,
            prioritized_sampling=self._prioritized_sampling,
            num_earliest_frames_ignored=self._num_earliest_frames_ignored,
            name=f'{self._name}_replay_buffer')
        self._observers.append(lambda exp: self._replay_buffer.add_batch(
            exp, exp.env_id))

    def observe_for_replay(self, exp):
        r"""Record an experience in a replay buffer.

        Args:
            exp (nested Tensor): exp (nested Tensor): The shape is
                :math:`[B, \ldots]`, where :math:`B` is the batch size of the
                batched environment.
        """
        exp = common.prune_exp_replay_state(exp, self._use_rollout_state,
                                            self.rollout_state_spec,
                                            self.train_state_spec)

        if self._replay_buffer is None:
            self._set_replay_buffer(exp)

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
        experiences are retrieved from the replay buffer, they are transformed by
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
            params (list[Parameter]|None): list of parameters with gradients
        """
        if self._config.summarize_grads_and_vars and params is not None:
            summary_utils.summarize_variables(params)
            summary_utils.summarize_gradients(params)
        if self._debug_summaries:
            summary_utils.summarize_loss(loss_info)
            obs = alf.nest.find_field(experience, "observation")
            if len(obs) == 1:
                summary_utils.summarize_nest("observation", obs[0])

        if alf.summary.should_record_summaries():
            mem = self._proc.memory_info().rss
            mem += sum(child.memory_info().rss
                       for child in self._proc.children(recursive=True))
            alf.summary.scalar(name='memory/cpu', data=mem // 1e6)

        if torch.cuda.is_available() and alf.summary.should_record_summaries():
            mem = torch.cuda.memory_allocated() // 1e6
            alf.summary.scalar(name='memory/gpu_allocated', data=mem)
            mem = torch.cuda.memory_reserved() // 1e6
            alf.summary.scalar(name='memory/gpu_reserved', data=mem)
            mem = torch.cuda.max_memory_allocated() // 1e6
            alf.summary.scalar(name='memory/max_gpu_allocated', data=mem)
            mem = torch.cuda.max_memory_reserved() // 1e6
            alf.summary.scalar(name='memory/max_gpu_reserved', data=mem)
            torch.cuda.reset_peak_memory_stats()
            # TODO: consider using torch.cuda.empty_cache() to save memory.

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
        member names should be ignored when getting trainable parameters, to
        avoid being assigned to the default optimizer.

        For example, if in your algorithm you've created a member ``self._vars``
        pointing to the parameters of a module for some purpose, you can avoid
        assigning an optimizer to ``self._vars`` (because the module will be assigned
        with one) by doing:

        .. code-block:: python

            def _trainable_attributes_to_ignore(self):
                return ["_vars"]

        Returns:
            list[str]: a list of attribute names to ignore.
        """
        return []

    def _get_children(self, include_ignored_attributes=False):
        children = []
        if include_ignored_attributes:
            to_ignore = []
        else:
            to_ignore = self._trainable_attributes_to_ignore()
        for name, module in self.named_children():
            if name in to_ignore:
                continue
            children.extend(_flatten_module(module))

        for name, param in self.named_parameters(recurse=False):
            if name in to_ignore:
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

    def compute_paras_statistics(self) -> Dict[str, np.ndarray]:
        """Compute some simple statistics of the algorithm's parameters.

        This function uses L1, L2, mean, std as the statistics.

        Returns:
            Dict[np.ndarray]: a dict of 1D numpy arrays, each containing simple
                parameter statistics, which can be used as a proxy for checking
                the consistency between two parameter set. The keys are parameter
                names of the module.
        """

        def _stats_per_para(para):
            l2_norm = tensor_utils.global_norm([para])
            l1_norm = para.abs().sum()
            mean = para.mean()
            if para.numel() == 1:
                std = torch.zeros(()).to(para)
            else:
                std = torch.std(para)
            stat = torch.stack(
                [l2_norm / para.numel(), l1_norm / para.numel(), mean, std])
            return stat.cpu().numpy()

        stats = {}
        for name, para in self.named_parameters():
            stats[name] = _stats_per_para(para.detach())
        return stats

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

        return self._setup_optimizers_(self._param_to_name)[0]

    def _setup_optimizers_(self, param_to_name):
        """Setup the param groups for optimizers.

        Returns:
            tuple:

            - list of parameters not handled by any optimizers under this algorithm
            - list of parameters handled under this algorithm
        """
        default_optimizer = self.default_optimizer
        # The reason of using dict instead of set to hold the parameters is that
        # dict is guaranteed to preserve the insertion order so that we can get
        # deterministic ordering of the parameters.
        new_params = dict()
        handled = dict()
        duplicate_error = "Parameter %s is handled by multiple optimizers."

        def _add_params_to_optimizer(params, opt):
            existing_params = set(_get_optimizer_params(opt))
            added_param_list = list(
                filter(lambda p: p not in existing_params, params))
            if added_param_list:
                opt.add_param_group({'params': added_param_list})
            return added_param_list

        # Iterate over all the child modules and add their parameters
        # into either
        #
        # 1. ``handled``: when the module is already assigned to an
        #    optimizer to handle in this algorithm.
        #
        # 2. or ``new_params``: when the module does not have an
        #    assigned optimizer in this algorithm (yet).
        #
        # For case 2, after the loop the ``new_params`` will be assigned to the
        # default optimizer if there is one. Otherwise, they will be left as
        # "unhandled".
        for child in self._get_children():
            if child in handled:
                continue
            assert id(child) != id(self), "Child should not be self"
            if isinstance(child, Algorithm):
                params, child_handled = child._setup_optimizers_(param_to_name)
                if child.force_params_visible_to_parent:
                    params += child_handled
                else:
                    for m in child_handled:
                        assert m not in handled, duplicate_error % param_to_name.get(
                            m)
                        handled[m] = 1
            elif isinstance(child, nn.Module):
                params = list(child.parameters())
            elif isinstance(child, nn.Parameter):
                params = [child]
            optimizer = self._module_to_optimizer.get(child, None)
            if optimizer is None:
                new_params.update((p, 1) for p in params)
                if default_optimizer is not None:
                    self._module_to_optimizer[child] = default_optimizer
            else:
                for m in params:
                    assert m not in handled, duplicate_error % param_to_name.get(
                        m)
                params = _add_params_to_optimizer(params, optimizer)
                handled.update((p, 1) for p in params)

        for p in handled:
            if p in new_params:
                del new_params[p]
        if default_optimizer is not None:
            added_param_list = _add_params_to_optimizer(
                new_params.keys(), default_optimizer)
            # In this case, all parameters are handled (specifically, the
            # new_params which are not handled before are assigned to the default
            # optimizer). Therefore, return [] as "unhandled parameters".
            return [], list(handled.keys()) + added_param_list
        else:
            return list(new_params.keys()), list(handled.keys())

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
            children = self._get_children(include_ignored_attributes)
            for module in children:
                if isinstance(module, Algorithm):
                    opts.extend(
                        module.optimizers(recurse, include_ignored_attributes))
        return opts

    def get_optimizer_info(self):
        """Return the optimizer info for all the modules in a string.

        TODO: for a subalgorithm that's an ignored attribute, its optimizer info
        won't be obtained.

        Returns:
            str: the json string of the information about all the optimizers.
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
                    parameters=sorted(
                        [self._param_to_name[p] for p in parameters])))
        json_pretty_str_info = json.dumps(obj=optimizer_info, indent=2)

        return json_pretty_str_info

    def get_unoptimized_parameter_info(self):
        """Return the information about the parameters not being optimized.

        Note: the difference of this with the parameters contained in the optimizer
        'None' from ``get_optimizer_info()`` is that ``get_optimizer_info()`` does not
        traverse all the parameters (e.g., parameters in list, tuple, dict, or set).

        Returns:
            str: path of all parameters not being optimized
        """
        self._setup_optimizers()
        optimized_parameters = []
        for optimizer in self.optimizers(include_ignored_attributes=True):
            optimized_parameters.extend(_get_optimizer_params(optimizer))
        optimized_parameters = set(optimized_parameters)
        all_parameters = common.get_all_parameters(self)
        unoptimized_parameters = []
        for name, p in all_parameters:
            if p not in optimized_parameters:
                unoptimized_parameters.append(name)
        return json.dumps(obj=sorted(unoptimized_parameters), indent=2)

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
            "observe() has not been called. experience_spec is not available.")
        return self._experience_spec

    @property
    def processed_experience_spec(self):
        """Spec for processed experience.

        Returns:
            TensorSpec: Spec for the experience returned by ``preprocess_experience()``.
        """
        assert self._processed_experience_spec is not None, (
            "preprocess_experience() has not been used. processed_experience_spec "
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
    def state_dict(self, destination=None, prefix='', visited=None, **kwargs):
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
            kwargs: additional keyword arguments for backward compatibility (
                e.g., 'keep_vars' for torch<0.4). They are not used in newer
                pytorch versions.

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
                opts_dict[new_key] = opt.state_dict()

            destination.update(opts_dict)

        return destination

    @common.add_method(nn.Module)
    def load_state_dict(self, state_dict, strict=True, skip_preloded=True):
        """Load state dictionary for the algorithm.

        Args:
            state_dict (dict): a dict containing parameters and persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in ``state_dict`` match the keys returned by this module's
                ``torch.nn.Module.state_dict`` function. If ``strict=True``, will
                keep lists of missing and unexpected keys; if ``strict=False``,
                missing/unexpected keys will be omitted. (Default: ``True``)
            skip_preloded (bool): whether to skip the modules that support
                pre-loading and have been pre-loaded. Currently only Algorithm
                and its derivatives support pre-loading. (Default: ``True``)
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

        def _load(module, visited, prefix=''):
            if not is_checkpoint_enabled(module):
                return
            if isinstance(module, Algorithm):
                if skip_preloded and module.pre_loaded:
                    return
                module._setup_optimizers()
                for i, opt in enumerate(module._optimizers):
                    opt_key = prefix + '_optimizers.%d' % i
                    if opt_key in state_dict:
                        opt.load_state_dict(state_dict[opt_key])
                        del state_dict[opt_key]
                    elif strict:
                        missing_keys.append(opt_key)

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

            for name, child in module._modules.items():
                if child is not None and child not in visited:
                    visited.add(child)
                    _load(child, visited, prefix + name + '.')

        _load(self, visited={self})

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
                        'an exception occurred : {}.'.format(
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
        """Adapted from __repr__() in torch/nn/modules/module.nn. to handle cycles."""

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
            valid_masks (Tensor): masks indicating which samples are valid.
                (``shape=(T, B), dtype=torch.float32``)
            weight (float): weight for this batch. Loss will be multiplied with
                this weight before calculating gradient.
            batch_info (BatchInfo): information about this batch returned by
                ``ReplayBuffer.get_batch()``
        Returns:
            tuple:
            - loss_info (LossInfo): loss information.
            - params (list[(name, Parameter)]): list of parameters being updated.
        """

        if self._debug_summaries:
            summary_utils.summarize_per_category_loss(loss_info)

        loss_info = self._aggregate_loss(loss_info, valid_masks, batch_info)

        all_params, gns = self._backward_and_gradient_update(
            loss_info.loss * weight)

        loss_info = loss_info._replace(gns=gns)
        loss_info = alf.nest.map_structure(torch.mean, loss_info)

        return loss_info, all_params

    def _aggregate_loss(self, loss_info, valid_masks=None, batch_info=None):
        """Computed aggregated loss.

        Args:
            loss_info (LossInfo): loss with shape :math:`(T, B)` (except for
                ``loss_info.scalar_loss``)
            valid_masks (Tensor): masks indicating which samples are valid.
                (``shape=(T, B), dtype=torch.float32``)
            batch_info (BatchInfo): information about this batch returned by
                ``ReplayBuffer.get_batch()``
        Returns:
            loss_info (LossInfo): loss information, with the aggregated loss
                in the ``loss`` field (i.e. ``loss_info.loss``).
        """
        masks = None
        if (batch_info is not None and batch_info.importance_weights != ()
                and self._config.priority_replay):
            if (loss_info.loss == () or loss_info.loss.ndim != 2
                    or loss_info.scalar_loss != ()):
                common.warning_once(
                    "The importance_weights of priority "
                    "sampling cannot be applied to LossInfo.scalar_loss or "
                    "LossInfo.loss whose ndim is not 2.")
            masks = batch_info.importance_weights.pow(
                -self._config.priority_replay_beta()).unsqueeze(0)
            if self._config is not None and self._config.normalize_importance_weights_by_max:
                masks = masks / masks.max()

        if valid_masks is not None:
            if masks is not None:
                masks = masks * valid_masks
            else:
                masks = valid_masks

        if masks is not None:
            loss_info = alf.nest.map_structure(
                lambda l: l * masks if l.ndim == 2 else l, loss_info)
        if isinstance(loss_info.scalar_loss, torch.Tensor):
            assert len(loss_info.scalar_loss.shape) == 0
            loss_info = loss_info._replace(
                loss=add_ignore_empty(loss_info.loss, loss_info.scalar_loss))
        return loss_info

    def _backward_and_gradient_update(self, loss):
        """Do backward and gradient update to all the trainable parameters.

        Args:
            loss (Tensor): an aggregated scalar loss
        Returns:
            params (list[(name, Parameter)]): list of parameters being updated.
        """
        unhandled = self._setup_optimizers()
        unhandled = [self._param_to_name[p] for p in unhandled]
        assert not unhandled, ("'%s' has some modules/parameters do not have "
                               "optimizer: %s" % (self.name, unhandled))

        optimizers = self.optimizers()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

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

        simple_gns = ()
        if self._debug_summaries and self._gns_estimator is not None:
            simple_gns = self._gns_estimator(loss, all_params)

        if isinstance(loss, torch.Tensor):
            with record_time("time/backward"):
                if self._grad_scaler is not None:
                    alf.summary.scalar("optimizer/grad_scale",
                                       self._grad_scaler.get_scale())
                    loss = self._grad_scaler.scale(loss)
                loss.mean().backward()

        for optimizer in optimizers:
            if self._grad_scaler is not None:
                # For ALF optimizers, gradient clipping is performed inside
                # optimizer.step, so we don't need to explicitly unscale grad
                # as the pytorch tutorial https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self._grad_scaler.step(optimizer)
            else:
                optimizer.step()

        if self._grad_scaler is not None:
            self._grad_scaler.update()

        all_params = [(self._param_to_name[p], p) for p in all_params]
        unused_parameters = [p[0] for p in all_params if p[1].grad is None]
        if unused_parameters:
            common.warning_once(
                "Find parameters without gradients, please double check: %s",
                unused_parameters)
        return all_params, simple_gns

    # Subclass may override calc_loss() to allow more sophisticated loss
    def calc_loss(self, info):
        """Calculate the loss at each step for each sample.

        Args:
            info (nest): information collected for training. It is batched
                from each ``AlgStep.info`` returned by ``rollout_step()``
                (on-policy training) or ``train_step()`` (off-policy training).
        Returns:
            LossInfo: loss at each time step for each sample in the
                batch. The shapes of the tensors in loss info should be
                :math:`(T, B)`.
        """
        assert isinstance(info, LossInfo), (
            "info returned by"
            " train_step() should be LossInfo. Otherwise you need to override"
            " calc_loss() to generate LossInfo from info")
        return info

    # offline related functions
    def train_step_offline(self, inputs, state, rollout_info, pre_train=False):
        """Perform one step of offline training computation.

        It is called to calculate output for every time step for a batch of
        experience from offline replay buffer. It also needs to generate
        necessary information for ``calc_loss_offline()``.
        By default, this function calls ``train_step`` as its default
        implementation.

        Args:
            inputs (nested Tensor): inputs for train.
            state (nested Tensor): consistent with ``train_state_spec``.
            rollout_info (nested Tensor): info from ``rollout_step()``. It is
                retrieved from replay buffer.
            pre_train (bool): whether in pre_training phase. This flag
                can be used for algorithms that need to implement different
                training procedures at different phases.
        Returns:
            AlgStep:
            - output (nested Tensor): prediction result.
            - state (nested Tensor): should match ``train_state_spec``.
            - info (nested Tensor): information for training. It will temporally
              batched and passed as ``info`` for calc_loss(). If this is
              ``LossInfo``, ``calc_loss()`` in ``Algorithm`` can be used.
              Otherwise, the user needs to override ``calc_loss()`` to
              calculate loss or override ``update_with_gradient()`` to do
              customized training.
        """
        try:
            return self.train_step(inputs, state, rollout_info)
        except:
            # the default train_step is not compatible with the
            # offline data, need to implement ``train_step_offline``
            # in subclass
            logging.exception('need to implement train_step_offline function')

    def calc_loss_offline(self, info_offline, pre_train=False):
        """Calculate the hybrid loss at each step for each sample.
        By default, this function calls ``calc_loss`` as its default
        implementation.

        Args:
            info_offline (nest): information collected for training from the
                offline training branch. It is returned by
                ``train_step_offline()`` (hybrid off-policy training).
            pre_train (bool): whether in pre_training phase. This flag
                can be used for algorithms that need to implement different
                training procedures at different phases.
        Returns:
            LossInfo: loss at each time step for each sample in the
                batch. The shapes of the tensors in loss info should be
                :math:`(T, B)`.
        """
        try:
            return self.calc_loss(info_offline)
        except:
            # the default calc_loss is not compatible with the
            # offline data, need to implement ``calc_loss_offline``
            # in subclass
            logging.exception('need to implement calc_loss_offline function')

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
        if self.is_rl() and self._config.mask_out_loss_for_last_step:
            valid_masks = (experience.step_type != StepType.LAST).to(
                torch.float32)
        else:
            valid_masks = None
        experience = experience._replace(rollout_info_field='rollout_info')
        loss_info = self.calc_loss(train_info)
        loss_info, params = self.update_with_gradient(loss_info, valid_masks)
        self.after_update(experience.time_step, train_info)
        self.summarize_train(experience, train_info, loss_info, params)
        shape = alf.nest.get_nest_shape(experience)
        return shape[0] * shape[1]

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
          If ``mini_batch_length`` is None, then ``unroll_length`` will be used
          for this calculation.


        If ``has_offline`` is True (e.g., by  specifying a valid replay
        buffer path to ``offline_buffer_dir`` in the config), it will
        enter the hybrid training mode, i.e., by sampling from both the
        original replay buffer and the offline buffer for training.

        Args:
            update_global_counter (bool): controls whether this function changes
                the global counter for summary. If there are multiple
                algorithms, then only the parent algorithm should change this
                quantity and child algorithms should disable the flag. When it's
                ``True``, it will affect the counter only if
                ``config.update_counter_every_mini_batch=True``.
        """
        config: TrainerConfig = self._config

        # returns 0 if haven't started training yet, when ``_replay_buffer`` is
        # not None and the number of samples in the buffer is less than
        # ``initial_collect_steps``; throughput will be 0 in this phase.
        # Note that the conditional that ``_replay_buffer`` is not None is
        # required here since in the case of offline pre-training when online RL
        # training is not started yet, ``_replay_buffer`` will be None since it
        # is only lazily created later when online RL training started.
        if (self._replay_buffer and
                self._replay_buffer.total_size < config.initial_collect_steps):
            return 0

        def _replay():
            # a local function to sample batch of experience from the
            # ``_replay_buffer`` for training.
            # TODO: If this function can be called asynchronously, and using
            # prioritized replay, then make sure replay and train below is atomic.
            with record_time("time/replay"):
                mini_batch_size = config.mini_batch_size
                if mini_batch_size is None:
                    mini_batch_size = self._replay_buffer.num_environments
                if config.whole_replay_buffer_training:
                    experience, batch_info = self._replay_buffer.gather_all(
                        ignore_earliest_frames=True)
                    num_updates = config.num_updates_per_train_iter
                else:
                    assert config.mini_batch_length is not None, (
                        "No mini_batch_length is specified for off-policy training"
                    )
                    experience, batch_info = self._replay_buffer.get_batch(
                        batch_size=(mini_batch_size *
                                    config.num_updates_per_train_iter),
                        batch_length=config.mini_batch_length)
                    num_updates = 1
            return experience, batch_info, num_updates, mini_batch_size

        if not self.has_offline:
            experience, batch_info, num_updates, mini_batch_size = _replay()
            with record_time("time/train"):
                return self._train_experience(
                    experience,
                    batch_info,
                    num_updates,
                    mini_batch_size,
                    config.mini_batch_length,
                    (config.update_counter_every_mini_batch
                     and update_global_counter),
                    whole_replay_buffer_training=config.
                    whole_replay_buffer_training)
        else:
            # hybrid training scheme
            global_step = alf.summary.get_global_counter()
            if ((global_step >= self._rl_train_after_update_steps) and
                (global_step % self._rl_train_every_update_steps == 0)):
                self._RL_train = True
            else:
                self._RL_train = False

            if global_step >= self._rl_train_after_update_steps:
                # flag used for indicating initial pre-training phase
                self._pre_train = False
            else:
                self._pre_train = True

            if self._RL_train:
                experience, batch_info, num_updates, mini_batch_size = _replay(
                )
            else:
                experience = None
                batch_info = None
                num_updates = 1
                mini_batch_size = config.mini_batch_size

            with record_time("time/offline_replay"):
                offline_experience, offline_batch_info = self._offline_replay_buffer.get_batch(
                    batch_size=(
                        mini_batch_size * config.num_updates_per_train_iter),
                    batch_length=config.mini_batch_length)
            # train hybrid
            with record_time("time/offline_train"):
                return self._train_hybrid_experience(
                    experience, batch_info, offline_experience,
                    offline_batch_info, num_updates, mini_batch_size,
                    config.mini_batch_length,
                    (config.update_counter_every_mini_batch
                     and update_global_counter))

    def _train_experience(self,
                          experience,
                          batch_info,
                          num_updates,
                          mini_batch_size,
                          mini_batch_length,
                          update_counter_every_mini_batch,
                          whole_replay_buffer_training: bool = False):
        """Train using experience."""
        (experience, processed_exp_spec, batch_info, length, mini_batch_length,
         batch_size) = self._prepare_experience_data(
             experience, self.experience_spec, batch_info, mini_batch_length,
             self._replay_buffer, whole_replay_buffer_training)

        if self._processed_experience_spec is None:
            self._processed_experience_spec = processed_exp_spec

        if whole_replay_buffer_training:
            # Special treatment for whole_replay_buffer_training.

            # Treatment 1: Adjust batch_info. This is better explained by
            # walking through an example.
            #
            # Suppose we gather_all() gives us env_ids = [0, 1] and positions =
            # [7, 7] for batch_info. This means that before the experience is
            # chopped by mini_batch_length (reshape), there are two trajectories
            # for env 0 and env 1. Assuming there are 12 steps in each of the
            # trajectories, and the mini_batch_length is 3.
            #
            # After the choppping (reshape), it is expected to have 8
            # trajectories in experience (because each original trajectory is
            # now chopped into 4 new trajectories of length 3), and therefore we
            # would like to adjust the batch_info to correctly reflect that. The
            # result would be
            #
            # env_ids =   [0,  0,  0,  0,  1,  1,  1,  1]
            # positions = [7, 10, 13, 16,  7, 10, 13, 16]
            num_envs = batch_info.env_ids.shape[0]
            num_mini_batches_per_original_traj = length // mini_batch_length
            batch_info = BatchInfo(
                env_ids=batch_info.env_ids.repeat_interleave(
                    num_mini_batches_per_original_traj),
                positions=batch_info.positions.repeat_interleave(
                    num_mini_batches_per_original_traj) + torch.arange(
                        0, length, mini_batch_length).repeat(num_envs),
                replay_buffer=batch_info.replay_buffer)

            # Treatment 2: Adjust the mini_batch_size.
            #
            # In the case when batch_size is not a multiple of mini_batch_size,
            # we want to avoid having a tiny mini batch in the end. For example,
            # when batch_size is 264 and mini_batch_size is 32, if
            # mini_batch_size is not adjusted, there will be 8 mini batches of
            # size 32 and 1 mini batch of size 8.
            #
            # In the above example, we will try to squeeze all the experience
            # into 8 mini batches by adjusting the mini_batch_size to 33.
            if batch_size % mini_batch_size > 0:
                num_batches_desired = batch_size // mini_batch_size
                if num_batches_desired > 0:
                    mini_batch_size = np.ceil(
                        batch_size / num_batches_desired).astype(int)

        if self._config.empty_cache:
            torch.cuda.empty_cache()

        indices = None
        for u in range(num_updates):
            if mini_batch_size < batch_size:
                indices = torch.randperm(
                    batch_size, device=experience.step_type.device)
            for b in range(0, batch_size, mini_batch_size):

                is_last_mini_batch = (u == num_updates - 1
                                      and b + mini_batch_size >= batch_size)
                do_summary = (is_last_mini_batch
                              or update_counter_every_mini_batch)

                mini_batch_list, mini_batch_info_list = \
                    self._extract_mini_batch_and_info_from_experience(
                                            indices,
                                            [experience],
                                            [batch_info],
                                            batch_size,
                                            b,
                                            mini_batch_size,
                                            update_counter_every_mini_batch,
                                            do_summary)

                exp, train_info, loss_info, params = self._update(
                    mini_batch_list[0],
                    mini_batch_info_list[0],
                    weight=alf.nest.get_nest_size(mini_batch_list[0], 1) /
                    mini_batch_size)
                if do_summary:
                    self.summarize_train(exp, train_info, loss_info, params)
                # These are no longer used, release them to reduce memory usage.
                del exp, train_info, loss_info, params

        train_steps = batch_size * mini_batch_length * num_updates
        return train_steps

    def _prepare_experience_data(self,
                                 experience,
                                 experience_spec,
                                 batch_info,
                                 mini_batch_length,
                                 replay_buffer,
                                 whole_replay_buffer_training=False):
        # Apply transformation and enrichment to the experience.
        experience = dist_utils.params_to_distributions(
            experience, experience_spec)
        experience = alf.data_structures.add_batch_info(
            experience, batch_info, replay_buffer)
        with alf.device(experience.step_type.device.type):
            experience = self.transform_experience(experience)

        # TODO(breakds): Create a cleaner and more readable function to prepare
        # experience that better handles the similar and distinct part of
        # "whole_replay_buffer_training or not", including correctly set
        #
        # 1. shape of experience
        # 2. mini_batch_length
        # 3. mini_batch_size
        # 4. batch_info
        #
        # So that we do not have to explicitly (and more importantly implicitly)
        # condition on whole_replay_buffer_training in a lot of isolated places.

        # using potentially updated batch_info after data_transformers
        if experience.batch_info != ():
            batch_info = experience.batch_info

        experience = alf.data_structures.clear_batch_info(experience)

        with summary_utils.record_time("time/preprocess_experience"):
            time_step, rollout_info = self.preprocess_experience(
                experience.time_step, experience.rollout_info, batch_info)
        experience = experience._replace(
            time_step=time_step, rollout_info=rollout_info)

        processed_exp_spec = dist_utils.extract_spec(experience, from_dim=2)

        experience = dist_utils.distributions_to_params(experience)

        length = alf.nest.get_nest_size(experience, dim=1)
        mini_batch_length = (mini_batch_length or length)
        if not whole_replay_buffer_training:
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

        return (experience, processed_exp_spec, batch_info, length,
                mini_batch_length, batch_size)

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
            policy_step = self.train_step(exp.time_step, policy_state,
                                          exp.rollout_info)
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
        policy_step = self.train_step(exp.time_step, policy_state,
                                      exp.rollout_info)

        if self._train_info_spec is None:
            self._train_info_spec = dist_utils.extract_spec(policy_step.info)
        info = dist_utils.distributions_to_params(policy_step.info)
        info = alf.nest.map_structure(
            lambda x: x.reshape(length, batch_size, *x.shape[1:]), info)
        info = dist_utils.params_to_distributions(info, self.train_info_spec)
        return info

    @data_distributed_when(lambda algorithm: not algorithm.on_policy)
    def _compute_train_info_and_loss_info(self, experience):
        """Compute train_info and loss_info based on the experience.

        This function has data distributed support if the algorithm is
        off-policy. This means that if the Algorithm instance has DDP activated
        and is off-policy, the output will have a hook to synchronize gradients
        across processes upon the call to the backward() that involves the output
        (i.e. train_info and loss_info).

        """
        length = alf.nest.get_nest_size(experience, dim=0)
        if self._temporally_independent_train_step or length == 1:
            train_info = self._collect_train_info_parallelly(experience)
        else:
            train_info = self._collect_train_info_sequentially(experience)
        loss_info = self.calc_loss(train_info)

        return train_info, loss_info

    def _update_priority(self, loss_info, batch_info,
                         replay_buffer: ReplayBuffer):
        """Update the priority of the ``replay buffer`` based on the ``priority``
        field of loss_info.
        """
        if loss_info.priority != ():
            priority = (loss_info.priority + self._config.priority_replay_eps
                        )**self._config.priority_replay_alpha()
            replay_buffer.update_priority(batch_info.env_ids,
                                          batch_info.positions, priority)
            if self._debug_summaries and alf.summary.should_record_summaries():
                with alf.summary.scope("PriorityReplay"):
                    summary_utils.add_mean_hist_summary(
                        "new_priority", priority)
                    summary_utils.add_mean_hist_summary(
                        "old_importance_weight", batch_info.importance_weights)
        else:
            assert batch_info is None or batch_info.importance_weights == (), (
                "Priority replay is enabled. But priority is not calculated.")

    def _update(self, experience, batch_info, weight):
        """
            experience (Experience): experience from the online buffer used for
                gradient update.
            batch_info (BatchInfo): information about the batch of data from
                the online buffer
            weight (float): weight for this batch. Loss will be multiplied with
                this weight before calculating gradient.
        """
        with torch.cuda.amp.autocast(self._config.enable_amp):
            train_info, loss_info = self._compute_train_info_and_loss_info(
                experience)

        self._update_priority(loss_info, batch_info, self._replay_buffer)

        if self.is_rl() and self._config.mask_out_loss_for_last_step:
            valid_masks = (experience.step_type != StepType.LAST).to(
                torch.float32)
        else:
            valid_masks = None
        loss_info, params = self.update_with_gradient(loss_info, valid_masks,
                                                      weight, batch_info)
        self.after_update(experience.time_step, train_info)

        return experience, train_info, loss_info, params

    def _collect_train_info_offline(self, experience, pre_train):
        shape = alf.nest.get_nest_shape(experience)
        length, batch_size = shape[:2]

        exp = alf.nest.map_structure(lambda x: x.reshape(-1, *x.shape[2:]),
                                     experience)

        if self._use_rollout_state:
            policy_state = exp.state
        else:
            size = alf.nest.get_nest_size(exp, dim=0)
            policy_state = self.get_initial_train_state(size)

        policy_step = self.train_step_offline(exp.time_step, policy_state,
                                              exp.rollout_info, pre_train)

        info = dist_utils.distributions_to_params(policy_step.info)
        info = alf.nest.map_structure(
            lambda x: x.reshape(length, batch_size, *x.shape[1:]), info)

        return info

    def _extract_mini_batch_and_info_from_experience(
            self, indices, experience_list, batch_info_list, batch_size,
            mini_batch_start_position, mini_batch_size,
            update_counter_every_mini_batch, do_summary):
        """Extract mini-batch and the corresponding batch info from experience.
        This function also convert the mini-batch to be time-major and to be on
        the default device.

        Args:
            indices (tensor|None): indices of the shape [batch_size]. Typically
                it is a randomly permuted version of the sequential indices
                to enable the extraction of a random mini-batch from the batch.
                If None, the natural sequential indices will be used.
                ``indices`` together with ``mini_batch_start_position`` and
                ``mini_batch_size`` will determine the segment of indices
                used for extracting the mini-batches.
            experience_list ([Experience]): list of experiences from which the
                mini-batch will be extracted, one for each experience.
            batch_info_list (BatchInfo): list of batch information about each
                of the experience in ``experience_list``.
            batch_size (int): size of batch. Currently assumes all the
                experiences (if not None) has the same batch size.
            mini_batch_start_position (int): the starting position in the
                ``indices`` for extracting the mini-batch
            mini_batch_size (int): size of the mini-batch to be extracted
                from the experiences.
            update_counter_every_mini_batch (bool): whether to update the global
                counter for each mini_batch.
            do_summary (bool): whether to enable summary.
        """
        if update_counter_every_mini_batch:
            alf.summary.increment_global_counter()

        alf.summary.enable_summary(do_summary)

        if indices is None:
            batch_indices = slice(
                mini_batch_start_position,
                min(batch_size, mini_batch_start_position + mini_batch_size))
        else:
            batch_indices = indices[mini_batch_start_position:min(
                batch_size, mini_batch_start_position + mini_batch_size)]

        def _make_time_major(nest):
            """Put the time dim to axis=0."""
            return alf.nest.map_structure(lambda x: x.transpose(0, 1), nest)

        mini_batch_list = []
        mini_batch_info_list = []
        for experience, batch_info in zip(experience_list, batch_info_list):
            if experience is not None:
                batch = alf.nest.map_structure(lambda x: x[batch_indices],
                                               experience)
                if batch_info:
                    binfo = alf.nest.map_structure(
                        lambda x: x[batch_indices] if isinstance(
                            x, torch.Tensor) else x, batch_info)
                else:
                    binfo = None
                batch = _make_time_major(batch)
                batch = alf.nest.utils.convert_device(batch)
            else:
                batch = None
                binfo = None

            mini_batch_list.append(batch)
            mini_batch_info_list.append(binfo)

        return mini_batch_list, mini_batch_info_list

    def _train_hybrid_experience(
            self, experience, batch_info, offline_experience,
            offline_batch_info, num_updates, mini_batch_size,
            mini_batch_length, update_counter_every_mini_batch):
        """Train using both experience (if available) and offline_experience.
        We assume that experience can be None.
        """
        if experience is not None:
            (experience, processed_exp_spec, batch_info, length,
             mini_batch_length, batch_size) = self._prepare_experience_data(
                 experience, self.experience_spec, batch_info,
                 mini_batch_length, self._replay_buffer)

            self._processed_experience_spec = processed_exp_spec

        if self._pre_train:
            context = common.pretrain_context()
        else:
            context = nullcontext()
        with context:
            # TODO: use a different mini_batch_length for offline training
            (offline_experience, _, offline_batch_info, length,
             mini_batch_length, batch_size) = self._prepare_experience_data(
                 offline_experience, self._offline_experience_spec,
                 offline_batch_info, mini_batch_length,
                 self._offline_replay_buffer)

        indices = None
        for u in range(num_updates):
            if mini_batch_size < batch_size:
                # here we use the cpu version of torch.randperm(n) to generate
                # the permuted indices, as the cuda version of torch.randperm(n)
                # seems to have a bug when n is a large number, generating
                # negative or very large values that cause out of bound kernel
                # error: https://github.com/pytorch/pytorch/issues/59756
                indices = alf.nest.utils.convert_device(
                    torch.randperm(batch_size, device='cpu'))

            for b in range(0, batch_size, mini_batch_size):

                is_last_mini_batch = (u == num_updates - 1
                                      and b + mini_batch_size >= batch_size)
                do_summary = (is_last_mini_batch
                              or update_counter_every_mini_batch)

                mini_batch_list, mini_batch_info_list = \
                    self._extract_mini_batch_and_info_from_experience(
                                            indices,
                                            [experience, offline_experience],
                                            [batch_info, offline_batch_info],
                                            batch_size,
                                            b,
                                            mini_batch_size,
                                            update_counter_every_mini_batch,
                                            do_summary)

                batch, offline_batch = mini_batch_list
                binfo, offline_binfo = mini_batch_info_list

                (exp, train_info, loss_info, offline_exp, offline_train_info,
                 offline_loss_info, params) = self._hybrid_update(
                     batch,
                     binfo,
                     offline_batch,
                     offline_binfo,
                     weight=alf.nest.get_nest_size(offline_batch, 1) /
                     mini_batch_size)
                if do_summary:
                    if exp:
                        self.summarize_train(exp, train_info, loss_info,
                                             params)
                    if offline_exp:
                        with alf.summary.scope("offline"):
                            self.summarize_train(offline_exp,
                                                 offline_train_info,
                                                 offline_loss_info, None)

        train_steps = 2 * batch_size * mini_batch_length * num_updates
        return train_steps

    def _hybrid_update(self, experience, batch_info, offline_experience,
                       offline_batch_info, weight):
        """
            experience (Experience): experience from the online buffer used for
                gradient update.
            batch_info (BatchInfo): information about the batch of data from
                the online buffer
            offline_experience (Experience): experience from offline replay
                buffer used for gradient update.
            offline_batch_info (BatchInfo): information about the batch of data
                from the offline buffer
            weight (float): weight for this batch. Loss will be multiplied with
                this weight before calculating gradient.
        """

        length = alf.nest.get_nest_size(offline_experience, dim=0)

        if self._RL_train:
            with torch.cuda.amp.autocast(self._config.enable_amp):
                train_info, loss_info = self._compute_train_info_and_loss_info(
                    experience)
                self._update_priority(loss_info, batch_info,
                                      self._replay_buffer)
        else:
            train_info = None
            loss_info = None

        if self._pre_train:
            context = common.pretrain_context()
        else:
            context = nullcontext()

        with context:
            offline_train_info = self._collect_train_info_offline(
                offline_experience, self._pre_train)
            offline_loss_info = self.calc_loss_offline(offline_train_info,
                                                       self._pre_train)

            self._update_priority(offline_loss_info, offline_batch_info,
                                  self._offline_replay_buffer)

            if self.is_rl():
                offline_valid_masks = (offline_experience.step_type !=
                                       StepType.LAST).to(torch.float32)
            else:
                offline_valid_masks = None

            if self._debug_summaries:
                with alf.summary.scope("offline"):
                    summary_utils.summarize_per_category_loss(
                        offline_loss_info)

            offline_loss_info = self._aggregate_loss(
                offline_loss_info, offline_valid_masks, offline_batch_info)

        if loss_info is not None:
            if self.is_rl():
                valid_masks = (experience.step_type != StepType.LAST).to(
                    torch.float32)
            else:
                valid_masks = None

            if self._debug_summaries:
                summary_utils.summarize_per_category_loss(loss_info)

            loss_info = self._aggregate_loss(loss_info, valid_masks,
                                             batch_info)
            # TODO: merge loss infos into one for summarization
            loss_info = loss_info._replace(
                loss=loss_info.loss + offline_loss_info.loss)

        else:
            loss_info = offline_loss_info

        params = self._backward_and_gradient_update(loss_info.loss * weight)

        if self._RL_train:
            # for now, there is no need to do a hybrid after update
            self.after_update(experience.time_step, train_info)

        loss_info = alf.nest.map_structure(torch.mean, loss_info)
        offline_loss_info = alf.nest.map_structure(torch.mean,
                                                   offline_loss_info)

        return experience, train_info, loss_info, offline_experience, \
                offline_train_info, offline_loss_info, params


class Loss(Algorithm):
    """Algorithm that uses its input as loss.

    It can be subclassed to customize calc_loss().
    """

    def __init__(self, loss_weight=1.0, name="LossAlg"):
        super().__init__(name=name)
        self._loss_weight = loss_weight

    def predict_step(self, inputs, state=None):
        return AlgStep()

    def rollout_step(self, inputs, state=None):
        if self.on_policy:
            return AlgStep(info=inputs)
        else:
            return AlgStep()

    def train_step(self, inputs, state=None, rollout_info=None):
        return AlgStep(info=inputs)

    def calc_loss(self, info):
        return LossInfo(loss=self._loss_weight * info, extra=info)
