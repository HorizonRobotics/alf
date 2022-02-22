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
import collections
from collections import OrderedDict
import copy
import functools
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
import subprocess
import sys
import time
import torch
import torch.distributions as td
import torch.nn as nn
import traceback
import types
from typing import Callable, List

import alf
from alf.algorithms.config import TrainerConfig
import alf.nest as nest
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


def get_target_updater(models, target_models, tau=1.0, period=1, copy=True):
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
        tau (float): A float scalar in :math:`[0, 1]`. Default :math:`\tau=1.0`
            means hard update.
        period (int): Step interval at which the target model is updated.
        copy (bool): If True, also copy ``models`` to ``target_models`` in the
            beginning.
    Returns:
        Callable: a callable that performs a soft update of the target model parameters.
    """
    models = as_list(models)
    target_models = as_list(target_models)

    def _copy_model_or_parameter(s, t):
        if isinstance(s, nn.Parameter):
            t.data.copy_(s)
        else:
            for ws, wt in zip(s.parameters(), t.parameters()):
                wt.data.copy_(ws)
            for ws, wt in zip(s.buffers(), t.buffers()):
                wt.copy_(ws)

    def _lerp_model_or_parameter(s, t):
        if isinstance(s, nn.Parameter):
            t.data.lerp_(s, tau)
        else:
            for ws, wt in zip(s.parameters(), t.parameters()):
                wt.data.lerp_(ws, tau)
            for ws, wt in zip(s.buffers(), t.buffers()):
                wt.copy_(ws)

    if copy:
        for model, target_model in zip(models, target_models):
            _copy_model_or_parameter(model, target_model)

    def update():
        if tau != 1.0:
            for model, target_model in zip(models, target_models):
                _lerp_model_or_parameter(model, target_model)
        else:
            for model, target_model in zip(models, target_models):
                _copy_model_or_parameter(model, target_model)

    return Periodically(update, period, 'periodic_update_targets')


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
        initial_state (nested Tensor): batched intitial states
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
    # Disable summary if in distributed mode and the running process isn't the
    # master process (i.e. rank = 0)
    if PerProcessContext().ddp_rank > 0:
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
        Transfromed observation
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


def get_conf_file():
    """Get the configuration file.

    If ``FLAGS.conf`` is not set, find alf_config.py or configured.gin under
    ``FLAGS.root_dir`` and returns it. If there is no 'conf' flag defined,
    return None.

    Returns:
        str: the name of the conf file. None if there is no conf file
    """
    if not hasattr(flags.FLAGS, "conf") and not hasattr(
            flags.FLAGS, "gin_file"):
        return None

    conf_file = getattr(flags.FLAGS, 'conf', None)
    if conf_file is not None:
        return conf_file
    conf_file = getattr(flags.FLAGS, 'gin_file', None)
    if conf_file is not None:
        return conf_file

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


def parse_conf_file(conf_file):
    """Parse config from file.

    It also looks for FLAGS.gin_param and FLAGS.conf_param for extra configs.

    Note: a global environment will be created (which can be obtained by
    alf.get_env()) and random seed will be initialized by this function using
    common.set_random_seed().

    Args:
        conf_file (str): the full path to the config file
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
        alf.parse_config(conf_file, conf_params)


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


def write_config(root_dir):
    """Write config to a file under directory ``root_dir``

    Configs from FLAGS.conf_param are also recorded.

    Args:
        root_dir (str): directory path
    """
    conf_file = get_conf_file()
    if conf_file is None or conf_file.endswith('.gin'):
        return write_gin_configs(root_dir, 'configured.gin')

    root_dir = os.path.expanduser(root_dir)
    alf_config_file = os.path.join(root_dir, ALF_CONFIG_FILE)
    os.makedirs(root_dir, exist_ok=True)
    pre_configs = alf.config_util.get_handled_pre_configs()
    config = ''
    if pre_configs:
        config += "########### pre-configs ###########\n\n"
        config += "import alf\n"
        config += "alf.pre_config({\n"
        for config_name, config_value in pre_configs:
            if isinstance(config_value, str):
                config += "    '%s': '%s',\n" % (config_name, config_value)
            else:
                config += "    '%s': %s,\n" % (config_name, config_value)
        config += "})\n\n"
        config += "########### end pre-configs ###########\n\n"
    f = open(conf_file, 'r')
    config += f.read()
    f.close()
    f = open(alf_config_file, 'w')
    f.write(config)
    f.close()


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
    Write a gin configration to a file. Because the user can

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
    """Generate warning message once.

    Note that the current implementation resembles that of the ``log_every_n()```
    function in ``logging`` but reduces the calling stack by one to ensure
    the multiple warning once messages generated at difference places can be
    displayed correctly.

    Args:
        msg: str, the message to be logged.
        *args: The args to be substitued into the msg.
    """
    caller = logging.get_absl_logger().findCaller()
    count = logging._get_next_log_count_per_token(caller)
    logging.log_if(logging.WARNING, msg, count == 0, *args)


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
        seed = int(np.uint32(hash(str(os.getpid()) + '|' + str(time.time()))))
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    random.seed(seed)
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
    log = ['{0} = {1}'.format(m.name, m.result()) for m in metrics]
    logging.info('%s \n\t\t %s', prefix, '\n\t\t '.join(log))


def create_ou_process(action_spec, ou_stddev, ou_damping):
    """Create nested zero-mean Ornstein-Uhlenbeck processes.

    The temporal update equation is:

    .. code-block:: python

        x_next = (1 - damping) * x + N(0, std_dev)

    Note: if ``action_spec`` is nested, the returned nested OUProcess will not bec
    checkpointed.

    Args:
        action_spec (nested BountedTensorSpec): action spec
        ou_damping (float): Damping rate in the above equation. We must have
            :math:`0 <= damping <= 1`.
        ou_stddev (float): Standard deviation of the Gaussian component.
    Returns:
        nested ``OUProcess`` with the same structure as ``action_spec``.
    """

    def _create_ou_process(action_spec):
        return dist_utils.OUProcess(action_spec.zeros(), ou_damping, ou_stddev)

    ou_process = alf.nest.map_structure(_create_ou_process, action_spec)
    return ou_process


def detach(nests):
    """Detach nested Tensors.

    Args:
        nests (nested Tensor): tensors to be detached
    Returns:
        detached Tensors with same structure as nests
    """
    return nest.map_structure(lambda t: t.detach(), nests)


# A catch all mode.  Currently includes on-policy training on unrolled experience.
EXE_MODE_OTHER = 0
# Unroll during training
EXE_MODE_ROLLOUT = 1
# Replay, policy evaluation on experience and training
EXE_MODE_REPLAY = 2
# Evaluation / testing or playing a learned model
EXE_MODE_EVAL = 3

# Global execution mode to track where the program is in the RL training process.
# This is used currently for observation normalization to only update statistics
# during training (vs unroll).  This is also used in tensorboard plotting of
# network output values, evaluation of the same network during rollout vs eval vs
# replay will be plotted to different graphs.
_exe_mode = EXE_MODE_OTHER
_exe_mode_strs = ["other", "rollout", "replay", "eval"]


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
    unrolling or training.
    """
    return _exe_mode == EXE_MODE_REPLAY


def is_rollout():
    """Return a bool value indicating whether the current code belongs to
    unrolling or training.
    """
    return _exe_mode == EXE_MODE_ROLLOUT


def is_eval():
    """Return a bool value indicating whether the current code belongs to
    evaluation or playing a learned model.
    """
    return _exe_mode == EXE_MODE_EVAL


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
        obj (object): will look for paramters under this object.
    Returns:
        list: list of (path, Parameters)
    """
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
        if isinstance(obj, torch.Tensor):
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
            attr = None
            try:
                attr = getattr(obj, name)
            except:
                # some attrbutes are property function, which may raise exception
                # when called in a wrong context (e.g. Algorithm.experience_spec)
                pass
            if attr is None or id(attr) in memo:
                continue
            unprocessed.append((attr, path + name))
            memo.add(id(attr))
    return all_parameters


def generate_alf_root_snapshot(alf_root, dest_path):
    """Given a destination path, copy the local ALF root dir to the path. To
    save disk space, only ``*.py`` files will be copied.

    This function can be used to generate a snapshot of the repo so that the
    exactly same code status will be recovered when later playing a trained
    model or launching a grid-search job in the waiting queue.

    Args:
        alf_root (str): the path to the ALF repo
        dest_path (str): the path to generate a snapshot of ALF repo
    """

    def _is_subdir(path, directory):
        relative = os.path.relpath(path, directory)
        return not relative.startswith(os.pardir)

    def rsync(src, target, includes):
        args = ['rsync', '-rI', '--include=*/']
        args += ['--include=%s' % i for i in includes]
        args += ['--exclude=*']
        args += [src, target]
        # shell=True preserves string arguments
        subprocess.check_call(
            " ".join(args), stdout=sys.stdout, stderr=sys.stdout, shell=True)

    assert not _is_subdir(dest_path, alf_root), (
        "Snapshot path '%s' is not allowed under ALF root! Use a different one!"
        % dest_path)

    # these files are important for code status
    includes = ["*.py", "*.gin", "*.so", "*.json"]
    rsync(alf_root, dest_path, includes)

    # rename ALF repo to a unified dir name 'alf'
    alf_dirname = os.path.basename(alf_root)
    if alf_dirname != "alf":
        os.system("mv %s/%s %s/alf" % (dest_path, alf_dirname, dest_path))


def get_alf_snapshot_env_vars(root_dir):
    """Given a ``root_dir``, return modified env variable dict so that ``PYTHONPATH``
    points to the ALF snapshot under this directory.
    """
    alf_repo = os.path.join(root_dir, "alf")
    alf_examples = os.path.join(alf_repo, "alf/examples")
    python_path = os.environ.get("PYTHONPATH", "")
    python_path = ":".join([alf_repo, alf_examples, python_path])
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


def is_alf_root(dir):
    """Given a directory, check if it is a valid ALF root. Currently the way
    of checking is to see if there is valid ``__init__.py`` under it.
    """
    return os.path.isfile(os.path.join(dir, 'alf/__init__.py'))


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
    if config.num_env_steps > 0:
        # the rollout env is always creatd with ``nonparallel=False``
        num_envs = alf.get_config_value(
            "create_environment.num_parallel_environments")
        num_iterations = config.num_env_steps / (
            num_envs * config.unroll_length)

    interval = math.ceil(num_iterations / summary_or_eval_calls)
    logging.info("A summary or eval interval=%d is calculated" % interval)
    return interval


def call_stack() -> List[str]:
    """Return a list of strings showing the current function call stacks for
    debugging.
    """
    return [line.strip() for line in traceback.format_stack()]
