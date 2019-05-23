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
"""Various functions used by different alf modules"""

import os

import tensorflow as tf

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.utils import common as tfa_common

scalar_spec = tf.TensorSpec(shape=(), dtype=tf.float32)


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


def get_target_updater(model, target_model, tau=1.0, period=1):
    """Performs a soft update of the target model parameters.

    For each weight w_s in the model, and its corresponding
    weight w_t in the target_model, a soft update is:
    w_t = (1 - tau) * w_t + tau * w_s

    Args:
        model (Network): the current model.
        target_model (Network): the model to be updated.
        tau (float): A float scalar in [0, 1]. Default `tau=1.0` means hard
            update.
        period (int): Step interval at which the target model is updated.

    Returns:
        A callable that performs a soft update of the target model parameters.
    """
    with tf.name_scope('update_targets'):

        def update():
            return tfa_common.soft_variables_update(
                model.variables, target_model.variables, tau)

        return tfa_common.Periodically(update, period,
                                       'periodic_update_targets')


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
        if len(action_spec.shape) == 0:
            action_dim = 1
        else:
            action_dim = action_spec.shape[-1]
        action = tf.reshape(action, (-1, action_dim))
        for a in range(action_dim):
            # TODO: use a descriptive name for the summary
            tf.summary.histogram("action/%s/%s" % (i, a), action[:, a])


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
    with tf.summary.record_if(
            lambda: tf.equal((global_step + 1) % summary_interval, 0)):
        func()


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
