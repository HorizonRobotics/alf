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
import functools
from functools import wraps
import gin
import glob
import math
import numpy as np
import os
import random
import shutil
import time
import torch
import torch.distributions as td
import torch.nn as nn
from typing import Callable

import alf
import alf.nest as nest
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils.spec_utils import zeros_from_spec as zero_tensor_from_nested_spec
from alf.utils.normalizers import WindowNormalizer, EMNormalizer, AdaptiveNormalizer
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

    Args:
        models (Network | list[Network]): the current model.
        target_models (Network | list[Network]): the model to be updated.
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

    if copy:
        for model, target_model in zip(models, target_models):
            for ws, wt in zip(model.parameters(), target_model.parameters()):
                wt.data.copy_(ws)

    def update():
        for model, target_model in zip(models, target_models):
            for ws, wt in zip(model.parameters(), target_model.parameters()):
                wt.data.lerp_(ws, tau)

    return Periodically(update, period, 'periodic_update_targets')


def expand_dims_as(x, y):
    """Expand the shape of ``x`` with extra singular dimensions.

    The result is broadcastable to the shape of ``y``.

    Args:
        x (Tensor): source tensor
        y (Tensor): target tensor. Only its shape will be used.
    Returns:
        ``x`` with extra singular dimensions.
    """
    assert x.ndim <= y.ndim
    assert x.shape == y.shape[:len(x.shape)]
    k = y.ndim - x.ndim
    if k == 0:
        return x
    else:
        return x.reshape(*x.shape, *([1] * k))


def reset_state_if_necessary(state, initial_state, reset_mask):
    """Reset state to initial state according to ``reset_mask``.

    Args:
        state (nested Tensor): the current batched states
        initial_state (nested Tensor): batched intitial states
        reset_mask (nested Tensor): with ``shape=(batch_size,), dtype=tf.bool``
    Returns:
        nested Tensor
    """
    return alf.nest.map_structure(
        lambda i_s, s: torch.where(expand_dims_as(reset_mask, i_s), i_s, s),
        initial_state, state)


def run_under_record_context(func,
                             summary_dir,
                             summary_interval,
                             flush_secs,
                             summary_max_queue=10):
    """Run ``func`` under summary record context.

    Args:
        func (Callable): the function to be executed.
        summary_dir (str): directory to store summary. A directory starting with
            ``~/`` will be expanded to ``$HOME/``.
        summary_interval (int): how often to generate summary based on the
            global counter
        flush_secs (int): flush summary to disk every so many seconds
        summary_max_queue (int): the largest number of summaries to keep in a queue;
            will flush once the queue gets bigger than this. Defaults to 10.
    """
    summary_dir = os.path.expanduser(summary_dir)
    summary_writer = alf.summary.create_summary_writer(
        summary_dir, flush_secs=flush_secs, max_queue=summary_max_queue)
    global_step = alf.summary.get_global_counter()

    def _cond():
        # We always write summary in the initial `summary_interval` steps
        # because there might be important changes at the beginning.
        return (alf.summary.is_summary_enabled()
                and (global_step < summary_interval
                     or global_step % summary_interval == 0))

    with alf.summary.push_summary_writer(summary_writer):
        with alf.summary.record_if(_cond):
            func()

    summary_writer.close()


@gin.configurable
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


@gin.configurable
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
        observation = nest.utils.transform_nest(
            nested=observation, field=field, func=_transform_image)
    return observation


@gin.configurable
def scale_transformer(observation, scale, dtype=torch.float32, fields=None):
    """Scale observation

    Args:
         observation (nested Tensor): observation to be scaled
         scale (float): scale factor
         dtype (Dtype): The destination type.
         fields (list[str]): fields to be scaled, A field str is a multi-level
                path denoted by "A.B.C".
    Returns:
        scaled observation
    """

    def _scale_obs(obs):
        return (obs * scale).type(dtype)

    fields = fields or [None]
    for field in fields:
        observation = nest.utils.transform_nest(
            nested=observation, field=field, func=_scale_obs)
    return observation


@gin.configurable
def reward_clipping(r, minmax=(-1, 1)):
    """
    Clamp immediate rewards to the range :math:`[min, max]`.

    Can be used as a reward shaping function passed to an algorithm
    (e.g. ``ActorCriticAlgorithm``).
    """
    assert minmax[0] <= minmax[1], "range error"
    return torch.clamp(r, minmax[0], minmax[1])


@gin.configurable
def reward_scaling(r, scale=1):
    """
    Scale immediate rewards by a factor of ``scale``.

    Can be used as a reward shaping function passed to an algorithm
    (e.g. ``ActorCriticAlgorithm``).
    """
    return r * scale


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


@gin.configurable
class ObservationNormalizer(nn.Module):
    def __init__(self,
                 observation_spec,
                 clipping=0.,
                 window_size=10000,
                 update_rate=1e-4,
                 speed=8.0,
                 mode="adaptive"):
        """Create an observation normalizer with optional value clipping to be
        used as the ``observation_transformer`` of an algorithm. It will be called
        before both ``rollout_step()`` and ``train_step()``.

        The normalizer by default doesn't automatically update the mean and std.
        Instead, it will check when ``self.forward()`` is called, whether an
        algorithm is unrolling or training. It only updates the mean and std
        during unroll. This is the suggested way of using an observation
        normalizer (i.e., update the stats when encountering new data for the
        first time). This same strategy has been used by OpenAI's baselines for
        training their Robotics environments.

        Args:
            observation_spec (TensorSpec): the observation spec
            clipping (float): a floating value for clipping the normalized
                observation into ``[-clipping, clipping]``. Only valid if it's
                greater than 0.
            window_size (int): the window size of ``WindowNormalizer``.
            update_rate (float): the update rate of ``EMNormalizer``.
            speed (float): the speed of updating for ``AdaptiveNormalizer``.
            mode (str): a value in ["adaptive", "window", "em"] indicates which
                normalizer to use.
        """
        super().__init__()
        self._clipping = float(clipping)
        if mode == "adaptive":
            self._normalizer = AdaptiveNormalizer(
                tensor_spec=observation_spec,
                speed=float(speed),
                auto_update=False)
        elif mode == "window":
            self._normalzier = WindowNormalizer(
                tensor_spec=observation_spec,
                window_size=int(window_size),
                auto_update=False)
        elif mode == "em":
            self._normalizer = EMNormalizer(
                tensor_spec=observation_spec,
                update_rate=float(update_rate),
                auto_update=False)
        else:
            raise ValueError("Unsupported mode: " + mode)

    def forward(self, observation):
        """Normalize a given observation. If during unroll, then first update
        the normalizer. The normalizer won't be updated in other circumstances.
        """
        if not is_training():
            self._normalizer.update(observation)
        return self._normalizer.normalize(observation, self._clipping)


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


def transpose2(x, dim1, dim2):
    """Transpose two axes ``dim1`` and ``dim2`` of a tensor."""
    perm = list(range(len(x.shape)))
    perm[dim1] = dim2
    perm[dim2] = dim1
    return tf.transpose(x, perm)


_env = None


def set_global_env(env):
    """Set global env."""
    global _env
    _env = env


@gin.configurable
def get_observation_spec(field=None):
    """Get the ``TensorSpec`` of observations provided by the global environment.

    This spec is used for creating models only! All ``uint8`` dtype will be converted
    to torch.float32 as a temporary solution, to be consistent with
    ``image_scale_transformer()``. See

    https://github.com/HorizonRobotics/alf/pull/239#issuecomment-544644558

    Args:
        field (str): a multi-step path denoted by "A.B.C".
    Returns:
        nested TensorSpec: a spec that describes the observation.
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    specs = _env.observation_spec()
    specs = nest.map_structure(
        lambda spec: (TensorSpec(spec.shape, torch.float32)
                      if spec.dtype == torch.uint8 else spec), specs)

    if field:
        for f in field.split('.'):
            specs = specs[f]
    return specs


@gin.configurable
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


@gin.configurable
def get_action_spec():
    """Get the specs of the tensors expected by ``step(action)`` of the global
    environment.

    Returns:
        nested TensorSpec: a spec that describes the shape and dtype of each tensor
        expected by ``step()``.
    """
    assert _env, "set a global env by `set_global_env` before using the function"
    return _env.action_spec()


def get_env():
    assert _env, "set a global env by `set_global_env` before using the function"
    return _env


@gin.configurable
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


@gin.configurable
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
    assert tensor_spec.is_discrete(
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


def warning_once(msg, *args):
    """Generate warning message once.

    Args:
        msg: str, the message to be logged.
        *args: The args to be substitued into the msg.
    """
    logging.log_every_n(logging.WARNING, msg, 1 << 62, *args)


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


_training = False


def set_training(training=True):
    """Mark whether the current code belongs to unrolling or training. This flag
    might be used to change the behavior of some functions accordingly.

    Args:
        training (bool): True for training, False for unrolling
    """
    global _training
    _training = training


def is_training():
    """Return a bool value indicating whether the current code belongs to
    unrolling or training.
    """
    return _training


def mark_training(train_func):
    """A decorator that will automatically mark the ``_training`` flag when
    entering/exiting a training function.

    Args:
        train_func (Callable): a training function
    """

    def _train_func(*args, **kwargs):
        set_training(True)
        ret = train_func(*args, **kwargs)
        set_training(False)
        return ret

    return _train_func


@gin.configurable
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
