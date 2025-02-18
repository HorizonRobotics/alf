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
"""Various functions used by different alf modules."""

from absl import flags
from absl import logging
import contextlib
import copy
from fasteners.process_lock import InterProcessLock
from functools import wraps
import gin
import glob
import math
import numpy as np
import os
import pathlib
import pprint
import random
import shutil
import socket
import subprocess
import sys
import time
import torch
import torch.distributions as td
import torch.nn as nn
import traceback
import types
from typing import Callable, List, Dict, Union

import alf
from alf.algorithms.config import TrainerConfig
import alf.nest as nest
from alf.utils.schedulers import Scheduler, as_scheduler
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils.spec_utils import zeros_from_spec as zero_tensor_from_nested_spec
from alf.utils.per_process_context import PerProcessContext
from . import dist_utils, gin_utils


def add_method(cls):
    """A decorator for adding a method to a class (cls).
    Example usage:

    .. code-block:: python

        class A:
            pass
        @add_method(A)
        def new_method(self):
            print('new method added')
        # now new_method() is added to class A and is ready to be used
        a = A()
        a.new_method()
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func

    return decorator


def as_list(x):
    """Convert ``x`` to a list.

    It performs the following conversion:

    .. code-block:: python

        None => []
        list => x
        tuple => list(x)
        other => [x]

    Args:
        x (any): the object to be converted
    Returns:
        list:
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def tuplify2d(x):
    """Convert ``x`` to a tuple of length two.

    It performs the following conversion:

    .. code-block:: python

        x => x if isinstance(x, tuple) and len(x) == 2
        x => (x, x) if not isinstance(x, tuple)

    Args:
        x (any): the object to be converted
    Returns:
        tuple:
    """
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    return (x, x)


class Periodically(nn.Module):
    def __init__(self, body, period, name='periodically'):
        """Periodically performs the operation defined in body.

        Args:
            body (Callable): callable to be performed every time
                an internal counter is divisible by the period.
            period (int): inverse frequency with which to perform the operation.
            name (str): name of the object.

        Raises:
            TypeError: if body is not a callable.
        """
        super().__init__()
        if not callable(body):
            raise TypeError('body must be callable.')
        self._body = body
        self._period = period
        self._counter = 0
        self._name = name

    def forward(self):
        self._counter += 1
        if self._counter % self._period == 0:
            self._body()
        elif self._period is None:
            return


@alf.configurable
class TargetUpdater(nn.Module):
    r"""Performs a soft update of the target model parameters.

    For each weight :math:`w_s` in the model, and its corresponding
    weight :math:`w_t` in the target_model, a soft update is:

    .. math::

        w_t = (1 - \tau) * w_t + \tau * w_s.

    Note: we only perform soft updates for parameters and always copy buffers.

    Args:
        models (Network | list[Network] | Parameter | list[Parameter] ): the
            current model or parameter.
        target_models (Network | list[Network] | Parameter | list[Parameter]):
            the model or parameter to be updated.
        tau: A float scalar or scheduler in :math:`[0, 1]`. Default :math:`\tau=1.0`
            means hard update.
        period: Step interval or scheduler at which the target model is updated.
        init_copy (bool): If True, also copy ``models`` to ``target_models`` in the
            beginning.
        delayed_update: if True, ``target_models`` is updated using recent_models
            every ``period`` steps. If ``tau`` is 1, the recent_models is ``models``
            ``period`` steps before. If ``tau`` is not 1, recent_models is
            an exponential moving average of ``models`` with rate ``tau``.
            The use of delayed_update may help to improve the stability of TD
            learning when a small ``period`` is used.
    """

    def __init__(self,
                 models,
                 target_models,
                 tau: Union[float, Scheduler] = 1.0,
                 period: Union[int, Scheduler] = 1,
                 init_copy=True,
                 delayed_update: bool = False):
        super().__init__()
        models = as_list(models)
        target_models = as_list(target_models)
        assert len(models) == len(target_models), (
            "The length of models and "
            "target_models are different: %s vs. %s" % (len(models),
                                                        len(target_models)))
        for model, target_model in zip(models, target_models):
            self._validate(model, target_model)
        self._models = models
        self._target_models = target_models
        if delayed_update:
            self._recent_models = list(
                map(self._make_copy, models, target_models))
        self._tau = as_scheduler(tau)
        self._period = as_scheduler(period)
        self._delayed_update = delayed_update
        self._counter = 0
        if init_copy:
            for model, target_model in zip(models, target_models):
                self._copy_model_or_parameter(model, target_model)

    def _make_copy(self, s, t):
        if isinstance(s, nn.Parameter):
            if id(s) == id(t):
                return s
            else:
                return copy.deepcopy(s)
        else:
            module = nn.ParameterList()
            for ws, wt in zip(s.parameters(), t.parameters()):
                if id(ws) == id(wt):
                    module.append(ws)
                else:
                    module.append(copy.deepcopy(ws))
            for i, (ws, wt) in enumerate(zip(s.buffers(), t.buffers())):
                if id(ws) == id(wt):
                    module.register_buffer("b%s" % i, ws)
                else:
                    module.register_buffer("b%s" % i, copy.deepcopy(ws))
            return module

    def _validate(self, s, t):
        def _error_msg(ns, nt):
            return ("The corresponding parameter/buffer of the source model "
                    "and the target model have different name: %s vs %s" %
                    (ns, nt))

        def _warning_msg(n):
            warning(
                "The corresponding parameter/buffer %s of the source model "
                "and the target model are same object. They will be ignored by "
                "TargetUpdater." % n)

        if isinstance(s, nn.Parameter):
            if id(s) == id(t):
                warning("target and the source parameter are same object. It "
                        "will be ignored by the TargetUpdater.")
        else:
            sparams = list(s.named_parameters())
            tparams = list(t.named_parameters())
            assert len(sparams) == len(tparams), (
                "The source model and the "
                "target models have different number of parameters: %s vs. %s"
                % (len(sparams), len(tparams)))
            for (ns, ws), (nt, wt) in zip(sparams, tparams):
                assert ns == nt, _error_msg(ns, nt)
                if id(ws) == id(wt):
                    _warning_msg(ns)
            sbuffers = list(s.named_buffers())
            tbuffers = list(t.named_buffers())
            assert len(sbuffers) == len(tbuffers), (
                "The source model and the "
                "target models have different number of buffers: %s vs. %s" %
                (len(sbuffers), len(tbuffers)))
            for (ns, ws), (nt, wt) in zip(sbuffers, tbuffers):
                assert ns == nt, _error_msg(ns, nt)
                if id(ws) == id(wt):
                    _warning_msg(ns)

    def _copy_model_or_parameter(self, s, t):
        if isinstance(s, nn.Parameter):
            if id(s) != id(t):
                t.data.copy_(s)
        else:
            for ws, wt in zip(s.parameters(), t.parameters()):
                if id(ws) != id(wt):
                    wt.data.copy_(ws)
            for ws, wt in zip(s.buffers(), t.buffers()):
                if id(ws) != id(wt):
                    wt.copy_(ws)

    def _lerp_model_or_parameter(self, s, t, tau):
        if isinstance(s, nn.Parameter):
            if id(s) != id(t):
                t.data.lerp_(s, tau)
        else:
            for ws, wt in zip(s.parameters(), t.parameters()):
                if id(ws) != id(wt):
                    wt.data.lerp_(ws, tau)
            for ws, wt in zip(s.buffers(), t.buffers()):
                if id(ws) != id(wt):
                    wt.copy_(ws)

    def forward(self):
        self._counter += 1
        period = self._period()
        tau = self._tau()
        if self._counter >= period:
            if self._delayed_update:
                for model, target_model in zip(self._recent_models,
                                               self._target_models):
                    self._copy_model_or_parameter(model, target_model)
            elif tau != 1.0:
                for model, target_model in zip(self._models,
                                               self._target_models):
                    self._lerp_model_or_parameter(model, target_model, tau)
            else:
                for model, target_model in zip(self._models,
                                               self._target_models):
                    self._copy_model_or_parameter(model, target_model)
        if self._delayed_update:
            if tau != 1.0:
                for model, target_model in zip(self._models,
                                               self._recent_models):
                    self._lerp_model_or_parameter(model, target_model, tau)
            elif self._counter >= period:
                for model, target_model in zip(self._models,
                                               self._recent_models):
                    self._copy_model_or_parameter(model, target_model)

        if self._counter >= period:
            self._counter = 0


@alf.configurable
class PeriodicReset(nn.Module):
    r"""Performs a periodic reset to model parameters.

    For each weight :math:`m` in the model, reset (reinitialization) is done periodically
    according to the specified schedule:

    .. math::

        m <-- re-initialize.

    Note:
        1) we reinitialize Network parameters and always reset buffers. For non-Network
        parameters, we record their initial value and reassign those values when reset.
        2) for a ``Network`` instance, if it implements a member function ``reset_parameters``,
        then this function will be used to reset the network parameter, which provides
        the user more flexibilities to achieve customized reset behaviors (e.g. only reset
        certain layers). If ``reset_parameters`` does not exist, then all the network
        parameters will be reset.

    Args:
        models (Network | list[Network] | Parameter | list[Parameter] ): the
            models or parameters that will be reset periodically according to schedule.
        period: Step interval or scheduler at which the models or parameters will be reset.
            If negative, no reset is done.
        reset_buffers: if True, the torch.buffer instances will also be reset
        post_processings: a list of callables to be applied after reset. For example,
            initial parameter copy to the target critic in some RL algorithms.
    """

    def __init__(self,
                 models,
                 period: Union[int, Scheduler] = 1,
                 reset_buffers: bool = True,
                 post_processings=List[Callable]):
        super().__init__()
        models = as_list(models)
        self._models = models
        self._period = as_scheduler(period)
        self._counter = 0
        self._reset_buffers = reset_buffers
        self._post_processings = post_processings
        # record the initial values of torch.nn.Parameter instances in ``models``
        self._init_param_values = {
            id(p): p.data.clone()
            for p in models if isinstance(p, torch.nn.Parameter)
        }

    def _copy_model_or_parameter(self, s, t):
        if isinstance(t, nn.Parameter):
            t.data.copy_(s)
        else:
            for ws, wt in zip(s.parameters(), t.parameters()):
                wt.data.copy_(ws)
            if self._reset_buffers:
                for ws, wt in zip(s.buffers(), t.buffers()):
                    wt.copy_(ws)

    def forward(self):
        self._counter += 1
        period = self._period()
        if period < 0:
            return
        if self._counter >= period:
            for i, m in enumerate(self._models):
                if isinstance(m, alf.networks.Network):
                    if callable(getattr(m, 'reset_parameters', None)):
                        m.reset_parameters()
                    else:
                        self._copy_model_or_parameter(m.copy(), m)
                elif isinstance(m, torch.nn.Parameter):
                    self._copy_model_or_parameter(
                        self._init_param_values[id(m)], m)
            for c in self._post_processings:
                c()

            self._counter = 0


def expand_dims_as(x, y, end=True):
    """Expand the shape of ``x`` with extra singular dimensions.

    The result is broadcastable to the shape of ``y``.

    Args:
        x (Tensor): source tensor
        y (Tensor): target tensor. Only its shape will be used.
        end (bool): If True, the extra dimensions are at the end of ``x``;
            otherwise they are at the beginning.
    Returns:
        ``x`` with extra singular dimensions.
    """
    assert x.ndim <= y.ndim
    k = y.ndim - x.ndim
    if k == 0:
        return x
    else:
        if end:
            assert x.shape == y.shape[:x.ndim]
            return x.reshape(*x.shape, *([1] * k))
        else:
            assert x.shape == y.shape[k:]
            return x.reshape(*([1] * k), *x.shape)


def reset_state_if_necessary(state, initial_state, reset_mask):
    """Reset state to initial state according to ``reset_mask``.

    Args:
        state (nested Tensor): the current batched states
        initial_state (nested Tensor): batched initial states
        reset_mask (nested Tensor): with ``shape=(batch_size,), dtype=torch.bool``
    Returns:
        nested Tensor
    """
    if torch.any(reset_mask):
        return alf.nest.map_structure(
            lambda i_s, s: torch.where(
                expand_dims_as(reset_mask, i_s), i_s.to(s.dtype), s),
            initial_state, state)
    else:
        return state


def run_under_record_context(func,
                             summary_dir,
                             summary_interval,
                             flush_secs,
                             summarize_first_interval=True,
                             summary_max_queue=10):
    """Run ``func`` under summary record context.

    Args:
        func (Callable): the function to be executed.
        summary_dir (str): directory to store summary. A directory starting with
            ``~/`` will be expanded to ``$HOME/``.
        summary_interval (int): how often to generate summary based on the
            global counter
        flush_secs (int): flush summary to disk every so many seconds
        summarize_first_interval (bool): whether to summarize every step of
            the first interval (default True). It might be better to turn
            this off for an easier post-processing of the curve.
        summary_max_queue (int): the largest number of summaries to keep in a queue;
            will flush once the queue gets bigger than this. Defaults to 10.
    """
    # For DDP training, we only do summary on one of the ranks.
    # Since rank-0 does more work than other rank (e.g. Evaluation),
    # we do summary on rank-1 to reduce the load of rank-0
    if PerProcessContext(
    ).is_distributed and PerProcessContext().ddp_rank != 1:
        func()
        return

    summary_dir = os.path.expanduser(summary_dir)
    summary_writer = alf.summary.create_summary_writer(
        summary_dir, flush_secs=flush_secs, max_queue=summary_max_queue)
    global_step = alf.summary.get_global_counter()

    def _cond():
        # We always write summary in the initial `summary_interval` steps
        # because there might be important changes at the beginning.
        return (alf.summary.is_summary_enabled() and
                ((global_step < summary_interval and summarize_first_interval)
                 or global_step % summary_interval == 0))

    with alf.summary.push_summary_writer(summary_writer):
        with alf.summary.record_if(_cond):
            func()

    summary_writer.close()


@alf.configurable
def cast_transformer(observation, dtype=torch.float32):
    """Cast observation

    Args:
         observation (nested Tensor): observation
         dtype (Dtype): The destination type.
    Returns:
        casted observation
    """

    def _cast(obs):
        if isinstance(obs, torch.Tensor):
            return obs.type(dtype)
        return obs

    return alf.nest.map_structure(_cast, observation)


@alf.configurable
def image_scale_transformer(observation, fields=None, min=-1.0, max=1.0):
    """Scale image to min and max (0->min, 255->max).

    Args:
        observation (nested Tensor): If observation is a nested structure, only
            ``namedtuple`` and ``dict`` are supported for now.
        fields (list[str]): the fields to be applied with the transformation. If
            None, then ``observation`` must be a ``Tensor`` with dtype ``uint8``.
            A field str can be a multi-step path denoted by "A.B.C".
        min (float): normalize minimum to this value
        max (float): normalize maximum to this value
    Returns:
        Transformed observation
    """

    def _transform_image(obs):
        assert isinstance(obs, torch.Tensor), str(type(obs)) + ' is not Tensor'
        assert obs.dtype == torch.uint8, "Image must have dtype uint8!"
        obs = obs.type(torch.float32)
        return ((max - min) / 255.) * obs + min

    fields = fields or [None]
    for field in fields:
        observation = nest.transform_nest(
            nested=observation, field=field, func=_transform_image)
    return observation


def _markdownify_gin_config_str(string, description=''):
    """Convert an gin config string to markdown format.

    Args:
        string (str): the string from ``gin.operative_config_str()``.
        description (str): Optional long-form description for this config str.
    Returns:
        string: the markdown version of the config string.
    """

    # This function is from gin.tf.utils.GinConfigSaverHook
    # TODO: Total hack below. Implement more principled formatting.
    def _process(line):
        """Convert a single line to markdown format."""
        if not line.startswith('#'):
            return '    ' + line

        line = line[2:]
        if line.startswith('===='):
            return ''
        if line.startswith('None'):
            return '    # None.'
        if line.endswith(':'):
            return '#### ' + line
        return line

    output_lines = []

    if description:
        output_lines.append("    # %s\n" % description)

    for line in string.splitlines():
        procd_line = _process(line)
        if procd_line is not None:
            output_lines.append(procd_line)

    return '\n'.join(output_lines)


def get_gin_confg_strs():
    """
    Obtain both the operative and inoperative config strs from gin.

    The operative configuration consists of all parameter values used by
    configurable functions that are actually called during execution of the
    current program, and inoperative configuration consists of all parameter
    configured but not used by configurable functions. See
    ``gin.operative_config_str()`` and ``gin_utils.inoperative_config_str`` for
    more detail on how the config is generated.

    Returns:
        tuple:
        - md_operative_config_str (str): a markdown-formatted operative str
        - md_inoperative_config_str (str): a markdown-formatted inoperative str
    """
    operative_config_str = gin.operative_config_str()
    md_operative_config_str = _markdownify_gin_config_str(
        operative_config_str,
        'All parameter values used by configurable functions that are actually called'
    )
    md_inoperative_config_str = gin_utils.inoperative_config_str()
    if md_inoperative_config_str:
        md_inoperative_config_str = _markdownify_gin_config_str(
            md_inoperative_config_str,
            "All parameter values configured but not used by program. The configured "
            "functions are either not called or called with explicit parameter values "
            "overriding the config.")
    return md_operative_config_str, md_inoperative_config_str


def summarize_gin_config():
    """Write the operative and inoperative gin config to Tensorboard summary.
    """
    md_operative_config_str, md_inoperative_config_str = get_gin_confg_strs()
    alf.summary.text('gin/operative_config', md_operative_config_str)
    if md_inoperative_config_str:
        alf.summary.text('gin/inoperative_config', md_inoperative_config_str)


def copy_gin_configs(root_dir, gin_files):
    """Copy gin config files to root_dir

    Args:
        root_dir (str): directory path
        gin_files (None|list[str]): list of file paths
    """
    root_dir = os.path.expanduser(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    for f in gin_files:
        shutil.copyfile(f, os.path.join(root_dir, os.path.basename(f)))


def get_gin_file():
    """Get the gin configuration file.

    If ``FLAGS.gin_file`` is not set, find gin files under ``FLAGS.root_dir`` and
    returns them. If there is no 'gin_file' flag defined, return ''.

    Returns:
        the gin file(s)
    """
    if hasattr(flags.FLAGS, "gin_file"):
        gin_file = flags.FLAGS.gin_file
        if gin_file is None:
            root_dir = os.path.expanduser(flags.FLAGS.root_dir)
            gin_file = glob.glob(os.path.join(root_dir, "*.gin"))
            assert gin_file, "No gin files are found! Please provide"
        return gin_file
    else:
        return ''


ALF_CONFIG_FILE = 'alf_config.py'


def get_conf_file(root_dir=None):
    """Get the configuration file.

    If ``FLAGS.conf`` is not set, find alf_config.py or configured.gin under
    ``root_dir`` and returns it. If root_dir is None, use ``FLAGS.root_dir``
    as ``root_dir``. If there is no 'root_dir' flag defined, return None.

    Args:
        root_dir (str): when None, FLAGS.root_dir is used to find the conf file.

    Returns:
        str: the name of the conf file. None if there is no conf file
    """
    conf_file = getattr(flags.FLAGS, 'conf', None)
    if conf_file is not None:
        return conf_file
    conf_file = getattr(flags.FLAGS, 'gin_file', None)
    if conf_file is not None:
        return conf_file

    if root_dir is None:
        if not hasattr(flags.FLAGS, 'root_dir'):
            return None
        root_dir = os.path.expanduser(flags.FLAGS.root_dir)
    conf_file = os.path.join(root_dir, ALF_CONFIG_FILE)
    if os.path.exists(conf_file):
        return conf_file
    gin_file = glob.glob(os.path.join(root_dir, "*.gin"))
    if not gin_file:
        return None
    assert len(
        gin_file) == 1, "Multiple *.gin files are found in %s" % root_dir
    return gin_file[0]


def parse_conf_file(conf_file, create_env=True):
    """Parse config from file.

    It also looks for FLAGS.gin_param and FLAGS.conf_param for extra configs.

    Note: a global environment will be created (which can be obtained by
    alf.get_env()) and random seed will be initialized by this function using
    common.set_random_seed().

    Args:
        conf_file (str): the full path to the config file
        create_env (bool): whether to create env. If True, create the env if it is not
            created yet.
    """
    if conf_file.endswith(".gin"):
        gin_params = getattr(flags.FLAGS, 'gin_param', None)
        gin.parse_config_files_and_bindings([conf_file], gin_params)
        ml_type = alf.get_config_value('TrainerConfig.ml_type')
        if ml_type == 'rl':
            # Create the global environment and initialize random seed
            alf.get_env()
    else:
        conf_params = getattr(flags.FLAGS, 'conf_param', None)
        alf.parse_config(conf_file, conf_params, create_env)


def get_epsilon_greedy(config: TrainerConfig):
    if config is not None:
        return config.epsilon_greedy
    else:
        return alf.get_config_value('TrainerConfig.epsilon_greedy')


def summarize_config():
    """Write config to TensorBoard."""

    def _format(configs):
        paragraph = pprint.pformat(dict(configs))
        return "    ".join((os.linesep + paragraph).splitlines(keepends=True))

    conf_file = get_conf_file()
    if conf_file is None or conf_file.endswith('.gin'):
        return summarize_gin_config()

    operative_configs = alf.get_operative_configs()
    inoperative_configs = alf.get_inoperative_configs()
    alf.summary.text('config/operative_config', _format(operative_configs))
    if inoperative_configs:
        alf.summary.text('config/inoperative_config',
                         _format(inoperative_configs))


def read_conf_file(root_dir: str) -> str:
    """Read the content of the conf file.

    Args:
        root_dir: alf log directory path
    Returns:
        the content of the conf file as a str. ``None`` if conf file is not
        specified through commandline and cannot be found in root_dir
    """
    conf_file = get_conf_file()
    if conf_file is None:
        return None
    with open(conf_file, 'r') as f:
        content = f.read()
    return content


def write_config(root_dir: str):
    """Write config to a file under directory ``root_dir``

    Configs from FLAGS.conf_param are also recorded.

    Args:
        root_dir: directory path
    """
    conf_file = get_conf_file()
    if conf_file is None or conf_file.endswith('.gin'):
        return write_gin_configs(root_dir, 'configured.gin')

    root_dir = os.path.expanduser(root_dir)
    alf.save_config(os.path.join(root_dir, ALF_CONFIG_FILE))


def get_initial_policy_state(batch_size, policy_state_spec):
    """
    Return zero tensors as the initial policy states.

    Args:
        batch_size (int): number of policy states created
        policy_state_spec (nested structure): each item is a tensor spec for
            a state
    Returns:
        state (nested structure): each item is a tensor with the first dim equal
            to ``batch_size``. The remaining dims are consistent with
            the corresponding state spec of ``policy_state_spec``.
    """
    return zero_tensor_from_nested_spec(policy_state_spec, batch_size)


def get_initial_time_step(env, first_env_id=0):
    """Return the initial time step.

    Args:
        env (AlfEnvironment):
        first_env_id (int): the environment ID for the first sample in this
            batch.
    Returns:
        TimeStep: the init time step with actions as zero tensors.
    """
    time_step = env.current_time_step()
    return time_step._replace(env_id=time_step.env_id + first_env_id)


_env = None


def set_global_env(env):
    """Set global env."""
    global _env
    _env = env


@alf.configurable
def get_raw_observation_spec(field=None):
    """Get the ``TensorSpec`` of observations provided by the global environment.

    Args:
        field (str): a multi-step path denoted by "A.B.C".
    Returns:
        nested TensorSpec: a spec that describes the observation.
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    specs = _env.observation_spec()
    if field:
        for f in field.split('.'):
            specs = specs[f]
    return specs


_transformed_observation_spec = None


def set_transformed_observation_spec(spec):
    """Set the spec of the observation transformed by data transformers."""
    global _transformed_observation_spec
    _transformed_observation_spec = spec


@alf.configurable
def get_observation_spec(field=None):
    """Get the spec of observation transformed by data transformers.

    The data transformers are specified by ``TrainerConfig.data_transformer_ctor``.

    Args:
        field (str): a multi-step path denoted by "A.B.C".
    Returns:
        nested TensorSpec: a spec that describes the observation.
    """
    assert _transformed_observation_spec is not None, (
        "This function should be "
        "called after the global variable _transformed_observation_spec is set"
    )

    specs = _transformed_observation_spec
    if field:
        for f in field.split('.'):
            specs = specs[f]
    return specs


@alf.configurable
def get_states_shape():
    """Get the tensor shape of internal states of the agent provided by
      the global environment.

      Returns:
        0 if internal states is not part of observation; otherwise a
        ``torch.Size``. We don't raise error so this code can serve to check
        whether ``env`` has states input.
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    if isinstance(_env.observation_spec(),
                  dict) and ('states' in _env.observation_spec()):
        return _env.observation_spec()['states'].shape
    else:
        return 0


@alf.configurable
def get_action_spec():
    """Get the specs of the tensors expected by ``step(action)`` of the global
    environment.

    Returns:
        nested TensorSpec: a spec that describes the shape and dtype of each tensor
        expected by ``step()``.
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    return _env.action_spec()


@alf.configurable
def get_reward_spec():
    """Get the specs of the reward tensors of the global environment.
    Returns:
        nested TensorSpec: a spec that describes the shape and dtype of each reward
        tensor.
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    return _env.reward_spec()


def get_env():
    assert _env, "set a global env by `set_global_env` before using the function"
    return _env


@alf.configurable
def get_vocab_size():
    """Get the vocabulary size of observations provided by the global environment.

    Returns:
        int: size of the environment's/teacher's vocabulary. Returns 0 if
        language is not part of observation. We don't raise error so this code
        can serve to check whether the env has language input
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    if isinstance(_env.observation_spec(),
                  dict) and ('sentence' in _env.observation_spec()):
        # return _env.observation_spec()['sentence'].shape[0]
        # is the sequence length of the sentence.
        return _env.observation_spec()['sentence'].maximum + 1
    else:
        return 0


@alf.configurable
def active_action_target_entropy(active_action_portion=0.2, min_entropy=0.3):
    """Automatically compute target entropy given the action spec. Currently
    support discrete actions only.

    The general idea is that we assume :math:`Nk` actions having uniform probs
    for a good policy. Thus the target entropy should be :math:`log(Nk)`, where
    :math:`N` is the total number of discrete actions and k is the active action
    portion.

    TODO: incorporate this function into ``EntropyTargetAlgorithm`` if it proves
    to be effective.

    Args:
        active_action_portion (float): a number in :math:`(0, 1]`. Ideally, this
            value should be greater than ``1/num_actions``. If it's not, it will
            be ignored.
        min_entropy (float): the minimum possible entropy. If the auto-computed
            entropy is smaller than this value, then it will be replaced.

    Returns:
        float: the target entropy for ``EntropyTargetAlgorithm``.
    """
    assert active_action_portion <= 1.0 and active_action_portion > 0
    action_spec = get_action_spec()
    assert action_spec.is_discrete(
        action_spec), "only support discrete actions!"
    num_actions = action_spec.maximum - action_spec.minimum + 1
    return max(math.log(num_actions * active_action_portion), min_entropy)


def write_gin_configs(root_dir, gin_file):
    """
    Write a gin configuration to a file. Because the user can

    1) manually change the gin confs after loading a conf file into the code, or
    2) include a gin file in another gin file while only the latter might be
       copied to ``root_dir``.

    So here we just dump the actual used gin conf string to a file.

    Args:
        root_dir (str): directory path
        gin_file (str): a single file path for storing the gin configs. Only
            the basename of the path will be used.
    """
    root_dir = os.path.expanduser(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    file = os.path.join(root_dir, os.path.basename(gin_file))

    md_operative_config_str, md_inoperative_config_str = get_gin_confg_strs()
    config_str = md_operative_config_str + '\n\n' + md_inoperative_config_str

    # the mark-down string can just be safely written as a python file
    with open(file, "w") as f:
        f.write(config_str)


@logging.skip_log_prefix
def warning_once(msg, *args):
    """Generate warning message ``msg % args`` once.

    Note that the current implementation resembles that of the ``log_every_n()```
    function in ``logging`` but reduces the calling stack by one to ensure
    the multiple warning once messages generated at difference places can be
    displayed correctly.

    Args:
        msg: str, the message to be logged.
        *args: The args to be substituted into the msg.
    """
    caller = logging.get_absl_logger().findCaller()
    count = logging._get_next_log_count_per_token(caller)
    logging.log_if(logging.WARNING, "\033[1;31m" + msg + "\033[1;0m",
                   count == 0, *args)


@logging.skip_log_prefix
def warning(msg, *args):
    """Generate warning message ``msg % args``.

    Args:
        msg: str, the message to be logged.
        *args: The args to be substituted into the msg.
    """
    logging.log(logging.WARNING, "\033[1;31m" + msg + "\033[1;0m", *args)


@logging.skip_log_prefix
def info(msg, *args):
    """Generate info message ``msg % args``.

    Args:
        msg: str, the message to be logged.
        *args: The args to be substituted into the msg.
    """
    logging.log(logging.INFO, "\033[1;34m" + msg + "\033[1;0m", *args)


@logging.skip_log_prefix
def info_once(msg, *args):
    """Generate info message ``msg % args`` once.

    Args:
        msg: str, the message to be logged.
        *args: The args to be substituted into the msg.
    """
    caller = logging.get_absl_logger().findCaller()
    count = logging._get_next_log_count_per_token(caller)
    logging.log_if(logging.INFO, "\033[1;34m" + msg + "\033[1;0m", count == 0,
                   *args)


def set_random_seed(seed):
    """Set a seed for deterministic behaviors.

    Note: If someone runs an experiment with a pre-selected manual seed, he can
    definitely reproduce the results with the same seed; however, if he runs the
    experiment with seed=None and re-run the experiments using the seed previously
    returned from this function (e.g. the returned seed might be logged to
    Tensorboard), and if cudnn is used in the code, then there is no guarantee
    that the results will be reproduced with the recovered seed.

    Args:
        seed (int|None): seed to be used. If None, a default seed based on
            pid and time will be used.
    Returns:
        The seed being used if ``seed`` is None.
    """
    if seed is None:
        seed = abs(hash(str(os.getpid()) + '|' + str(time.time())))
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        force_torch_deterministic = getattr(flags.FLAGS,
                                            'force_torch_deterministic', True)
        # causes RuntimeError: scatter_add_cuda_kernel does not have a deterministic implementation
        torch.use_deterministic_algorithms(force_torch_deterministic)
    seed %= 2**32
    random.seed(seed)
    # sometime the seed passed in can be very big, but np.random.seed
    # only accept seed smaller than 2**32
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def log_metrics(metrics, prefix=''):
    """Log metrics through logging.
    Args:
        metrics (list[alf.metrics.StepMetric]): list of metrics to be logged
        prefix (str): prefix to the log segment
    """
    log = [f'{m.name} = {m.result()}' for m in metrics]
    logging.info('%s \n\t\t %s', prefix, '\n\t\t '.join(log))
    log = [f'{m.name}.std = {m.std()}' for m in metrics]
    logging.info('%s \n\t\t %s', prefix, '\n\t\t '.join(log))


def detach(nests: alf.nest.Nest):
    """Detach nested Tensors or Distributions

    Args:
        nests: tensors or distributions to be detached
    Returns:
        detached Tensors/Distributions with same structure as nests
    """

    def _detach_dist_or_tensor(dist_or_tensor):
        if isinstance(dist_or_tensor, td.Distribution):
            builder, params = dist_utils._get_builder(dist_or_tensor)
            return builder(**detach(params))
        else:
            return dist_or_tensor.detach()

    return nest.map_structure(_detach_dist_or_tensor, nests)


# A catch all mode.  Currently includes on-policy training on unrolled experience.
EXE_MODE_OTHER = 0
# Unroll during training
EXE_MODE_ROLLOUT = 1
# Replay, policy evaluation on experience and training
EXE_MODE_REPLAY = 2
# Evaluation / testing or playing a learned model
EXE_MODE_EVAL = 3
# pretrain mode
EXE_MODE_PRETRAIN = 4

# Global execution mode to track where the program is in the RL training process.
# This is used currently for observation normalization to only update statistics
# during training (vs unroll).  This is also used in tensorboard plotting of
# network output values, evaluation of the same network during rollout vs eval vs
# replay will be plotted to different graphs.
_exe_mode = EXE_MODE_OTHER
_exe_mode_strs = ["other", "rollout", "replay", "eval", "pretrain"]


def set_exe_mode(mode):
    """Mark whether the current code belongs to unrolling or training. This flag
    might be used to change the behavior of some functions accordingly.

    Args:
        training (bool): True for training, False for unrolling
    Returns:
        the old exe mode
    """
    global _exe_mode
    old = _exe_mode
    _exe_mode = mode
    return old


def exe_mode_name():
    """return the execution mode as string.
    """
    return _exe_mode_strs[_exe_mode]


def is_replay():
    """Return a bool value indicating whether the current code belongs to
    replaying. Replaying implies off-policy training.

    Any code under ``train_from_replay_buffer()`` of any algorithm is classified
    as replaying. This phase starts from experience sampling from the replay buffer,
    all the way to the parameter update.
    """
    return _exe_mode == EXE_MODE_REPLAY


def is_rollout():
    """Return a bool value indicating whether the current code belongs to
    unrolling. For on-policy algorithms, unrolling could be treated as part of
    training as it usually generates training info for calculating the loss.

    Any code under ``unroll()`` of the root RL algorithm is classified as unrolling.
    This is the phase of collecting experiences for training.
    """
    return _exe_mode == EXE_MODE_ROLLOUT


def is_eval():
    """Return a bool value indicating whether the current code belongs to
    evaluation or playing a learned model.
    """
    return _exe_mode == EXE_MODE_EVAL


def is_pretrain():
    """Return a bool value indicating whether the current code belongs to
    pre-train. The code within a function that is decorated by ``mark_pretrain``
    is flagged as ``pretrain``. A code block that is within a ``pretrain_context``
    is also flagged as ``pretrain``.
    """
    return _exe_mode == EXE_MODE_PRETRAIN


def is_training(alg):
    """Return a bool value indicating whether the current code is in a training
    phase, for either an on-policy or an off-policy algorithm.

    A training phase is defined as the rollout phase for an on-policy algorithm,
    or the replay phase for an off-policy algorithm.

    .. note::

        Currently this function returns False for the code under ``train_from_unroll()``.

    Args:
        alg (Algorithm): the algorithm to be decided
    """
    return (alg.on_policy and is_rollout()) or is_replay()


def mark_eval(func):
    """A decorator that will automatically mark the ``_exe_mode`` flag when
    entering/exiting a evaluation/test function.

    Args:
        func (Callable): a function
    """

    def _func(*args, **kwargs):
        old_mode = _exe_mode
        set_exe_mode(EXE_MODE_EVAL)
        ret = func(*args, **kwargs)
        set_exe_mode(old_mode)
        return ret

    return _func


def mark_replay(func):
    """A decorator that will automatically mark the ``_exe_mode`` flag when
    entering/exiting a experience replay function.

    Args:
        func (Callable): a function
    """

    def _func(*args, **kwargs):
        old_mode = _exe_mode
        set_exe_mode(EXE_MODE_REPLAY)
        ret = func(*args, **kwargs)
        set_exe_mode(old_mode)
        return ret

    return _func


def mark_rollout(func):
    """A decorator that will automatically mark the ``_exe_mode`` flag when
    entering/exiting a rollout function.

    Args:
        func (Callable): a function
    """

    def _func(*args, **kwargs):
        old_mode = _exe_mode
        set_exe_mode(EXE_MODE_ROLLOUT)
        ret = func(*args, **kwargs)
        set_exe_mode(old_mode)
        return ret

    return _func


def mark_pretrain(func):
    """A decorator that will automatically mark the ``_exe_mode`` flag when
    entering/exiting a pretrain function.

    Args:
        func (Callable): a function
    """

    def _func(*args, **kwargs):
        old_mode = _exe_mode
        set_exe_mode(EXE_MODE_PRETRAIN)
        ret = func(*args, **kwargs)
        set_exe_mode(old_mode)
        return ret

    return _func


class eval_context(object):
    """A context manager that will automatically mark the ``_exe_mode`` flag
    as ``EXE_MODE_EVAL`` when entering a context and revert to the original
    ``_exe_mode`` when exiting the context.
    """

    def __init__(self):
        self._old_mode = _exe_mode

    def __enter__(self):
        set_exe_mode(EXE_MODE_EVAL)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            raise exc_type(exc_value).with_traceback(traceback)
        set_exe_mode(self._old_mode)
        return True


class replay_context(object):
    """A context manager that will automatically mark the ``_exe_mode`` flag
    as ``EXE_MODE_REPLAY`` when entering a context and revert to the original
    ``_exe_mode`` when exiting the context.
    """

    def __init__(self):
        self._old_mode = _exe_mode

    def __enter__(self):
        set_exe_mode(EXE_MODE_REPLAY)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            raise exc_type(exc_value).with_traceback(traceback)
        set_exe_mode(self._old_mode)
        return True


class rollout_context(object):
    """A context manager that will automatically mark the ``_exe_mode`` flag
    as ``EXE_MODE_ROLLOUT`` when entering a context and revert to the original
    ``_exe_mode`` when exiting the context.
    """

    def __init__(self):
        self._old_mode = _exe_mode

    def __enter__(self):
        set_exe_mode(EXE_MODE_ROLLOUT)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            raise exc_type(exc_value).with_traceback(traceback)
        set_exe_mode(self._old_mode)
        return True


class pretrain_context(object):
    """A context manager that will automatically mark the ``_exe_mode`` flag
    as ``EXE_MODE_PRETRAIN`` when entering a context and revert to the original
    ``_exe_mode`` when exiting the context.
    """

    def __init__(self):
        self._old_mode = _exe_mode

    def __enter__(self):
        set_exe_mode(EXE_MODE_PRETRAIN)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            raise exc_type(exc_value).with_traceback(traceback)
        set_exe_mode(self._old_mode)
        return True


@alf.configurable
def flattened_size(spec):
    """Return the size of the vector if spec.shape is flattened.

    It's same as np.prod(spec.shape)
    Args:
        spec (alf.TensorSpec): a TensorSpec object
    Returns:
        np.int64: the size of flattened shape
    """
    # np.prod(()) == 1.0, need to convert to np.int64
    return np.int64(np.prod(spec.shape))


def is_inside_docker_container():
    """Return whether the current process is running inside a docker container.

    See discussions at `<https://stackoverflow.com/questions/23513045/how-to-check-if-a-process-is-running-inside-docker-container>`_
    """
    return os.path.exists("/.dockerenv")


def check_numerics(nested):
    """Assert all the tensors in nested are finite.

    Args:
        nested (nested Tensor): nested Tensor to be checked.
    """
    nested_finite = alf.nest.map_structure(
        lambda x: torch.all(torch.isfinite(x)), nested)
    if not all(alf.nest.flatten(nested_finite)):
        bad = alf.nest.map_structure(lambda x, finite: () if finite else x,
                                     nested, nested_finite)
        assert all(alf.nest.flatten(nested_finite)), (
            "Some tensor in nested is not finite: %s" % bad)


def get_all_parameters(obj):
    """Get all the parameters under ``obj`` and its descendents.

    Note: This function assumes all the parameters can be reached through tuple,
    list, dict, set, nn.Module or the attributes of an object. If a parameter is
    held in a strange way, it will not be included by this function.

    Args:
        obj (object): will look for parameters under this object.
    Returns:
        list: list of (path, Parameters)
    """
    from alf.environments.alf_environment import AlfEnvironment
    all_parameters = []
    memo = set()
    unprocessed = [(obj, '')]
    # BFS for all subobjects
    while unprocessed:
        obj, path = unprocessed.pop(0)
        if isinstance(obj, types.ModuleType):
            # Do not traverse into a module. There are too much stuff inside a
            # module.
            continue
        if isinstance(obj, nn.Parameter):
            all_parameters.append((path, obj))
            continue
        if isinstance(obj, (torch.Tensor, np.ndarray, AlfEnvironment)):
            # We don't expect parameters in these objects, so we skip them
            # to save time.
            continue
        if path:
            path += '.'
        if nest.is_namedtuple(obj):
            for name, value in nest.extract_fields_from_nest(obj):
                if id(value) not in memo:
                    unprocessed.append((value, path + str(name)))
                    memo.add(id(value))
        elif isinstance(obj, dict):
            # The keys of a generic dict are not necessarily str, and cannot be
            # handled by nest.extract_fields_from_nest.
            for name, value, in obj.items():
                if id(value) not in memo:
                    unprocessed.append((value, path + str(name)))
                    memo.add(id(value))
        elif isinstance(obj, (tuple, list, set)):
            for i, value in enumerate(obj):
                if id(value) not in memo:
                    unprocessed.append((value, path + str(i)))
                    memo.add(id(value))
        elif isinstance(obj, nn.Module):
            for name, m in obj.named_children():
                if id(m) not in memo:
                    unprocessed.append((m, path + name))
                    memo.add(id(m))
            for name, p in obj.named_parameters():
                if id(p) not in memo:
                    unprocessed.append((p, path + name))
                    memo.add(id(p))
        attribute_names = dir(obj)
        for name in attribute_names:
            if name.startswith('__') and name.endswith('__'):
                # Ignore system attributes,
                continue
            # We want to skip property functions, which may raise exception
            # when called in a wrong context (e.g. Algorithm.experience_spec)
            if isinstance(getattr(type(obj), name, None), property):
                continue
            attr = None
            try:
                attr = getattr(obj, name)
            except:
                # handle the case that the attribute cannot be accessed
                pass

            if attr is None or id(attr) in memo:
                continue
            unprocessed.append((attr, path + name))
            memo.add(id(attr))
    return all_parameters


def snapshot_repo_roots() -> Dict[str, str]:
    """Return a dict of repo root dirs for snapshot. The paths should be defined
    by a special environment variable ``ALF_SNAPSHOT_REPO_ROOTS``, in the following
    format:

    .. code-block:: bash

        export ALF_SNAPSHOT_REPO_ROOTS="<module_name1>=<repo_root1>:<module_name2>=<repo_root2>:..."

    where pairs of "<module_name>=<repo_root>" are separated by ":". Note that
    ``<repo_root>`` should be the parent dir of the module package dir.

    Returns:
        dict[str]: a dict of ``{module_name: repo_root}``, excluding the alf repo
            itself.
    """
    repo_roots_envar = os.getenv('ALF_SNAPSHOT_REPO_ROOTS')
    repo_roots = {}
    if repo_roots_envar is not None:
        pairs = repo_roots_envar.split(':')
        for p in pairs:
            assert '=' in p, (
                "Each repo str must be in the format '<module>=<repo_root>'! "
                f"Got {p}")
            module, repo_root = p.split('=')
            repo_roots[module] = str(pathlib.Path(repo_root).absolute())
    return repo_roots


def generate_alf_snapshot(alf_root: str, conf_file: str, dest_path: str):
    """Given a destination path, copy the local ALF root dir to the path. To
    save disk space, only ``*.py`` files will be copied.

    This function can be used to generate a snapshot of the repo so that the
    exactly same code status will be recovered when later playing a trained
    model or launching a grid-search job in the waiting queue.

    Args:
        alf_root: the parent path of the 'alf' module
        conf_file: the alf config file
        dest_path: the path to generate a snapshot of ALF repo
    """

    def _is_subdir(path, directory):
        relative = os.path.relpath(path, directory)
        return not relative.startswith(os.pardir)

    def rsync(src, target, includes, excludes=[]):
        args = [
            'rsync',
            '-rI',
        ]
        args += ['--exclude=%s' % e for e in excludes]
        args += ['--include=*/'] + ['--include=%s' % i for i in includes]
        args += ['--exclude=*']
        args += [src, target]
        # shell=True preserves string arguments
        subprocess.check_call(
            " ".join(args), stdout=sys.stdout, stderr=sys.stdout, shell=True)

    includes = [
        "*.py", "*.gin", "*.so", "*.json", "*.xml", "*.cpp", "*.c", "*.cc",
        "*.hpp", "*.h", "*.stl", "*.png", "*.txt", "*.yaml"
    ]
    excludes = ['*/build/']
    repo_roots = {**snapshot_repo_roots(), **{'alf': alf_root}}
    for name, root in repo_roots.items():
        assert not _is_subdir(dest_path, root), (
            "Snapshot path '%s' is not allowed under any repo root '%s'! " %
            (dest_path, root) + "Use a different one!")
        # Only copy the module dir because the root dir might contain many
        # other modules in the case where repo is pip installed in 'site-packages'.
        rsync(root + f'/{name}', dest_path, includes, excludes)
        # compress the snapshot repo into a ".tar.gz" file
        os.system(
            f"cd {dest_path}; tar -czf {name}.tar.gz {name}; rm -rf {name}")
        info(f"Generated a snapshot {name}@{root}")


def unzip_alf_snapshot(root_dir: str):
    """Restore an ALF snapshot from a job directory by unzipping the snapshot
    'tar.gz' files.

    Args:
        root_dir: the tensorboard job directory
    """
    module_names = []
    for zipped_repo in glob.glob(f"{root_dir}/*.tar.gz"):
        # assuming all '*.tar.gz' under root_dir are repo snapshots
        name = os.path.basename(zipped_repo).split('.')[0]
        info("=== Using an ALF snapshot at '%s' ===", zipped_repo)
        os.system(f"rm -rf {root_dir}/{name}")
        os.system(f"cd {root_dir}; tar -xzf {name}.tar.gz")
        module_names.append(name)
    return module_names


def get_alf_snapshot_env_vars(root_dir):
    """Given a ``root_dir``, return modified env variable dict so that ``PYTHONPATH``
    points to the ALF snapshot under this directory.
    """
    module_names = unzip_alf_snapshot(root_dir)
    python_path = os.environ.get("PYTHONPATH", "")
    for name in module_names:
        assert not is_repo_root(os.getcwd(), name), (
            "Using a snapshot is not allowed under a valid repo root: " +
            "'%s' (contains '%s')!" % (os.getcwd(), name) +
            " Try running the command in a different directory.")
        root = root_dir
        if name == "alf":
            legacy_alf_root = os.path.join(root, "alf")
            if os.path.isfile(os.path.join(legacy_alf_root, "alf")):
                # legacy alf repo path for backward compatibility
                # legacy tb dirs: root_dir/alf/alf/__init__.py
                root = legacy_alf_root
            alf_examples = os.path.join(root, "alf/examples")
            python_path = ":".join([root, alf_examples, python_path])
        else:
            python_path = ":".join([root, python_path])
    env_vars = copy.copy(os.environ)
    env_vars.update({"PYTHONPATH": python_path})
    return env_vars


def abs_path(path):
    """Given any path, return the absolute path with expanding the user.
    """
    return os.path.realpath(os.path.expanduser(path))


_alf_root = None


def alf_root():
    """Get the ALF root path."""
    global _alf_root
    if _alf_root is None:
        # alf.__file__==<ALF_ROOT>/alf/__init__.py
        _alf_root = str(pathlib.Path(alf.__file__).parent.parent.absolute())
    return _alf_root


def is_repo_root(dir, module_name):
    """Given a directory, check if it is a valid repo root. Currently the way
    of checking is to see if there is valid ``__init__.py`` under it.
    """
    return os.path.isfile(os.path.join(dir, f'{module_name}/__init__.py'))


def compute_summary_or_eval_interval(config, summary_or_eval_calls=100):
    """Automatically compute a summary or eval interval according to the config
    and the expected total number of summary or eval calls. This function can
    avoid manually computing the interval value when an expected number of calls
    is in mind.

    .. warning::
        This function might not work for algorithms that change the global
        counter themselves, e.g., ``LMAlgorithm``.

    Args:
        config (TrainerConfig): the configuration object for training
        summary_or_eval_calls (int): the expected number of summary
            or eval calls throughout the training process. This number can control
            the time consumed on summary or eval. Note that this number might not
            be exactly satisfied eventually, if the calculated interval has been
            rounded up.

    Returns:
        int: summary or eval interval
    """
    # Do not support this for now because the summary global counter will have
    # a different value with the iteration number.
    assert not config.update_counter_every_mini_batch, (
        "This function currently doesn't support update_counter_every_mini_batch=True!"
    )

    if config.num_iterations > 0:
        num_iterations = config.num_iterations
    # this condition is exclusive with the above
    else:
        assert config.num_env_steps
        # the rollout env is always created with ``nonparallel=False``
        num_envs = alf.get_config_value(
            "create_environment.num_parallel_environments")
        num_iterations = config.num_env_steps / (
            num_envs * config.unroll_length)

    interval = math.ceil(num_iterations / summary_or_eval_calls)
    info_once("A summary or eval interval=%d is calculated" % interval)
    return interval


def call_stack() -> List[str]:
    """Return a list of strings showing the current function call stacks for
    debugging.
    """
    return [line.strip() for line in traceback.format_stack()]


@contextlib.contextmanager
def get_unused_port(start, end=65536, n=1):
    """Get an unused port in the range [start, end) .

    Args:
        start (int) : port range start
        end (int): port range end
        n (int): get ``n`` consecutive unused ports
    Raises:
        socket.error: if no unused port is available
    """
    process_locks = []
    unused_ports = []
    try:
        for port in range(start, end):
            process_locks.append(
                InterProcessLock(path='/tmp/socialbot/{}.lock'.format(port)))
            if not process_locks[-1].acquire(blocking=False):
                process_locks[-1].lockfile.close()
                process_locks.pop()
                for process_lock in process_locks:
                    process_lock.release()
                process_locks = []
                continue
            try:
                with contextlib.closing(socket.socket()) as sock:
                    sock.bind(('', port))
                    unused_ports.append(port)
                    if len(unused_ports) == 2:
                        break
            except socket.error:
                for process_lock in process_locks:
                    process_lock.release()
                process_locks = []
        if len(unused_ports) < n:
            raise socket.error("No unused port in [{}, {})".format(start, end))
        if n == 1:
            yield unused_ports[0]
        else:
            yield unused_ports
    finally:
        if process_locks:
            for process_lock in process_locks:
                process_lock.release()


def prune_exp_replay_state(
        exp: 'Experience', use_rollout_state: bool,
        rollout_state_spec: alf.NestedTensorSpec,
        train_state_spec: alf.NestedTensorSpec) -> 'Experience':
    """Prune an experience's state in the replay buffer to save memory.

    The basic idea is to remove state components that are not needed by training.

    Args:
        exp: The experience whose state is to be pruned.
        use_rollout_state: Whether to use rollout state as the initial training state.
        rollout_state_spec: The rollout state spec.
        train_state_spec: The training state spec.

    Returns:
        An experience whose state is pruned.
    """
    if not use_rollout_state:
        exp = exp._replace(state=())
    elif id(rollout_state_spec) != id(train_state_spec):
        # Prune exp's state (rollout_state) according to the train state spec
        exp = exp._replace(
            state=alf.nest.prune_nest_like(
                exp.state, train_state_spec, value_to_match=()))
    return exp


def save_video(frames: List[np.ndarray],
               output_file: str = "output.mp4",
               fps: int = 30):
    """
    Saves a list of RGB NumPy arrays as a video file.

    Args:
        frames: list of RGB images (H, W, 3).
        output_file: output video file path.
        fps (int): frames per second.
    """
    import cv2
    if not frames:
        raise ValueError("The frame list is empty.")

    # Get video dimensions from the first frame
    height, width, _ = frames[0].shape

    # Define the video writer (MP4 codec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write frames to the video file
    for frame in frames:
        out.write(cv2.cvtColor(
            frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    # Release the writer
    out.release()
