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

import glob
import os
import shutil

from absl import flags
import gin
import tensorflow as tf

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.specs import tensor_spec
from tf_agents.utils import common as tfa_common
from tf_agents.trajectories import trajectory
from alf.utils import summary_utils, gin_utils


def zero_tensor_from_nested_spec(nested_spec, batch_size):
    def _zero_tensor(spec):
        if batch_size is None:
            shape = spec.shape
        else:
            spec_shape = tf.convert_to_tensor(value=spec.shape, dtype=tf.int32)
            shape = tf.concat(([batch_size], spec_shape), axis=0)
        dtype = spec.dtype
        return tf.zeros(shape, dtype)

    return tf.nest.map_structure(_zero_tensor, nested_spec)


def set_per_process_memory_growth(flag=True):
    """Set if memory growth should be enabled for a PhysicalDevice.

    With memory growth set to True, tf will not allocate all memory on the
    device upfront.

    Args:
        flag (bool): True if do not allocate memory upfront.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, flag)
            except RuntimeError as e:
                # Memory growth must be set at program startup
                print(e)


def as_list(x):
    """Convert x to a list.

    It performs the following conversion:
        None => []
        list => x
        tuple => list(x)
        other => [x]
    Args:
        x (any): the object to be converted
    Returns:
        a list.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def get_target_updater(models, target_models, tau=1.0, period=1):
    """Performs a soft update of the target model parameters.

    For each weight w_s in the model, and its corresponding
    weight w_t in the target_model, a soft update is:
    w_t = (1 - tau) * w_t + tau * w_s

    Args:
        models (Network | list[Network]): the current model.
        target_models (Network | list[Network]): the model to be updated.
        tau (float): A float scalar in [0, 1]. Default `tau=1.0` means hard
            update.
        period (int): Step interval at which the target model is updated.

    Returns:
        A callable that performs a soft update of the target model parameters.
    """
    models = as_list(models)
    target_models = as_list(models)

    def update():
        for model, target_model in zip(models, target_models):
            tfa_common.soft_variables_update(model.variables,
                                             target_model.variables, tau)

    return tfa_common.Periodically(update, period, 'periodic_update_targets')


def add_nested_summaries(prefix, data):
    """Add summary about loss_info

    Args:
        prefix (str): the prefix of the names of the summaries
        data (namedtuple): data to be summarized
    """
    for field in data._fields:
        elem = getattr(data, field)
        name = prefix + '/' + field
        if isinstance(elem, tuple) and hasattr(elem, '_fields'):
            add_nested_summaries(name, elem)
        elif isinstance(elem, tf.Tensor):
            tf.summary.scalar(name, elem)


def add_loss_summaries(loss_info: LossInfo):
    """Add summary about loss_info

    Args:
        loss_info (LossInfo): loss_info.extra must be a namedtuple
    """
    tf.summary.scalar('loss', data=loss_info.loss)
    if not loss_info.extra:
        return
    if getattr(loss_info.extra, '_fields') is None:
        return
    add_nested_summaries('loss', loss_info.extra)


def add_action_summaries(actions, action_specs):
    """Generate histogram summaries for actions.

    Actions whose rank is more than 1 will be skipped.

    Args:
        actions (nested Tensor): actions to be summarized
        action_specs (nested TensorSpec): spec for the actions
    """
    action_specs = tf.nest.flatten(action_specs)
    actions = tf.nest.flatten(actions)

    for i, (action, action_spec) in enumerate(zip(actions, action_specs)):
        if len(action_spec.shape) > 1:
            continue

        if tensor_spec.is_discrete(action_spec):
            summary_utils.histogram_discrete(
                name="action/%s" % i,
                data=action,
                bucket_min=action_spec.minimum,
                bucket_max=action_spec.maximum)
        else:
            if len(action_spec.shape) == 0:
                action_dim = 1
            else:
                action_dim = action_spec.shape[-1]
            action = tf.reshape(action, (-1, action_dim))

            def _get_val(a, i):
                return a if len(a.shape) == 0 else a[i]

            for a in range(action_dim):
                # TODO: use a descriptive name for the summary
                summary_utils.histogram_continuous(
                    name="action/%s/%s" % (i, a),
                    data=action[:, a],
                    bucket_min=_get_val(action_spec.minimum, a),
                    bucket_max=_get_val(action_spec.maximum, a))


def get_distribution_params(nested_distribution):
    """Get the params for an optionally nested action distribution.

    Only returns parameters that have tf.Tensor values.

    Args:
        nested_distribution (nested tf.distribution.Distribution):
            The distributions whose parameter tensors to extract.
    Returns:
        A nest of distribution parameters. Each leaf is a dict corresponding to
        one distribution, with keys as parameter name and values as tensors
        containing parameter values.
    """

    def _tensor_parameters_only(params):
        return {
            k: params[k]
            for k in params if isinstance(params[k], tf.Tensor)
        }

    return tf.nest.map_structure(
        lambda single_dist: _tensor_parameters_only(single_dist.parameters),
        nested_distribution)


def expand_dims_as(x, y):
    """Expand the shape of `x` with extra singular dimensions.

    The result is broadcastable to the shape of `y`
    Args:
        x (Tensor): source tensor
        y (Tensor): target tensor. Only its shape will be used.
    Returns
        x with extra singular dimensions.
    """
    assert len(x.shape) <= len(y.shape)
    assert x.shape == y.shape[:len(x.shape)]
    k = len(y.shape) - len(x.shape)
    if k == 0:
        return x
    else:
        return tf.reshape(x, x.shape.concatenate((1, ) * k))


def reset_state_if_necessary(state, initial_state, reset_mask):
    """Reset state to initial state according to reset_mask

    Args:
      state (nested Tensor): the current batched states
      initial_state (nested Tensor): batched intitial states
      reset_mask (nested Tensor): with shape=(batch_size,), dtype=tf.bool
    Returns:
      nested Tensor
    """
    return tf.nest.map_structure(
        lambda i_s, s: tf.where(expand_dims_as(reset_mask, i_s), i_s, s),
        initial_state, state)


def run_under_record_context(func, summary_dir, summary_interval,
                             flush_millis):
    """Run `func` under summary record context.

    Args:
        func (Callable): the function to be executed.
        summary_dir (str): directory to store summary. A directory starting with
            "~/" will be expanded to "$HOME/"
        summary_interval (int): how often to generate summary based on the
            global counter
        flush_millis (int): flush summary to disk every so many milliseconds
    """
    summary_dir = os.path.expanduser(summary_dir)
    summary_writer = tf.summary.create_file_writer(
        summary_dir, flush_millis=flush_millis)
    summary_writer.set_as_default()
    global_step = get_global_counter()
    with tf.summary.record_if(lambda: tf.equal((global_step + 1) %
                                               summary_interval, 0)):
        func()


from tensorflow.python.ops.summary_ops_v2 import should_record_summaries


def get_global_counter(default_counter=None):
    """Get the global counter.

    Args:
        default_counter (Variable): If not None, this counter will be returned.
    Returns:
        If default_counter is not None, it will be returned. Otherwise,
        If tf.summary.experimental.get_step() is not None, it will be returned.
        Othewise, a counter will be created and returned.
        tf.summary.experimental.set_step() will be set to the created counter.

    """
    if default_counter is None:
        default_counter = tf.summary.experimental.get_step()
        if default_counter is None:
            default_counter = tf.Variable(
                0, dtype=tf.int64, trainable=False, name="global_counter")
            tf.summary.experimental.set_step(default_counter)
    return default_counter


@gin.configurable
def image_scale_transformer(observation, min=-1.0, max=1.0):
    """Scale image to min and max (0->min, 255->max)

    Note: it treats an observation with len(shape)==4 as image
    Args:
        observation (nested Tensor): observations
        min (float): normalize minimum to this value
        max (float): normalize maximum to this value
    Returns:
        Transfromed observation
    """

    def _transform_image(obs):
        # tf_agent changes all gym.spaces.Box observation to tf.float32.
        # See _spec_from_gym_space() in tf_agents/environments/gym_wrapper.py
        if len(obs.shape) == 4:
            if obs.dtype == tf.uint8:
                obs = tf.cast(obs, tf.float32)
            return ((max - min) / 255.) * obs + min
        else:
            return obs

    return tf.nest.map_structure(_transform_image, observation)


@gin.configurable
def reward_clipping(r, minmax=(-1, 1)):
    """
    Clamp immediate rewards to the range [`min`, `max`].

    Can be used as a reward shaping function passed to an algorithm
    (e.g. ActorCriticAlgorithm).
    """
    assert minmax[0] <= minmax[1], "range error"
    return tf.clip_by_value(r, minmax[0], minmax[1])


@gin.configurable
def reward_scaling(r, scale=1):
    """
    Scale immediate rewards by a factor of `scale`.

    Can be used as a reward shaping function passed to an algorithm
    (e.g. ActorCriticAlgorithm).
    """
    return r * scale


def _markdownify_gin_config_str(string, description=''):
    """Convert an gin config string to markdown format.
    
    Args:
        string (str): the string from gin.operative_config_str()
        description (str): Optional long-form description for this config_str
    Returns:
        The string of the markdown version of the config string.
    """

    # This function is from gin.tf.utils.GinConfigSaverHook
    # TODO: Total hack below. Implement more principled formatting.
    def process(line):
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
        procd_line = process(line)
        if procd_line is not None:
            output_lines.append(procd_line)

    return '\n'.join(output_lines)


def summarize_gin_config():
    """Write the operative and inoperative gin config to Tensorboard summary.
    
    The operative configuration consists of all parameter values used by
    configurable functions that are actually called during execution of the
    current program, and inoperative configuration consists of all parameter
    configured but not used by configurable functions. See `gin.operative_config_str()`
    and `gin_utils.inoperative_config_str` for more detail on how the config is generated.
    """
    operative_config_str = gin.operative_config_str()
    md_config_str = _markdownify_gin_config_str(
        operative_config_str,
        'All parameter values used by configurable functions that are actually called'
    )
    tf.summary.text('gin/operative_config', md_config_str)
    inoperative_config_str = gin_utils.inoperative_config_str()
    if inoperative_config_str:
        md_config_str = _markdownify_gin_config_str(
            inoperative_config_str,
            "All parameter values configured but not used by program. The configured "
            "functions are either not called or called with explicit parameter values "
            "overriding the config.")
        tf.summary.text('gin/inoperative_config', md_config_str)


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
    
    If FLAGS.gin_file is not set, find gin files under FLAGS.root_dir and
    returns them.
    Returns:
        the gin file(s)
    """
    gin_file = flags.FLAGS.gin_file
    if gin_file is None:
        root_dir = os.path.expanduser(flags.FLAGS.root_dir)
        gin_file = glob.glob(os.path.join(root_dir, "*.gin"))
        assert gin_file, "No gin files are found! Please provide"
    return gin_file


def tensor_extend(x, y):
    """Extending tensor with new_slice.

    new_slice.shape should be same as tensor.shape[1:]
    Args:
        x (Tensor): tensor to be extended
        y (Tensor): the tensor which will be appended to `x`
    Returns:
        the extended tensor. Its shape is (x.shape[0]+1, x.shape[1:])
    """
    return tf.concat([x, tf.reshape(y, [1] + y.shape.as_list())], axis=0)


def tensor_extend_zero(x):
    """Extending tensor with zeros.

    new_slice.shape should be same as tensor.shape[1:]
    Args:
        x (Tensor): tensor to be extended
    Returns:
        the extended tensor. Its shape is (x.shape[0]+1, x.shape[1:])
    """
    return tf.concat(
        [x, tf.zeros([1] + x.shape.as_list()[1:], dtype=x.dtype)], axis=0)


def explained_variance(ypred, y):
    """Computes fraction of variance that ypred explains about y.
    
    Adapted from baselines.ppo2 explained_variance()

    Interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    
    Args:
        ypred (Tensor): prediction for y
        y (Tensor): target
    Returns:
        1 - Var[y-ypred] / Var[y]
    """
    ypred = tf.reshape(ypred, [-1])
    y = tf.reshape(y, [-1])
    _, vary = tf.nn.moments(y, axes=(0, ))
    return 1 - tf.nn.moments(y - ypred, axes=(0, ))[1] / (vary + 1e-30)
