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

import tensorflow as tf

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.utils import common as tfa_common


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


def reset_state_if_necessary(state, initial_state, reset_mask):
    """Reset state to initial state according to reset_mask
    
    Args:
      state (nested Tensor): the current batched states
      initial_state (nested Tensor): batched intitial states
      reset_mask: nested Tensor with shape=(batch_size,), dtype=tf.bool
    Returns:
      nested Tensor
    """
    return tf.nest.map_structure(lambda i_s, s: tf.where(reset_mask, i_s, s),
                                 initial_state, state)
