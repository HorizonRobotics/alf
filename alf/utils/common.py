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


def get_target_updater(model, target_model, tau=1.0, period=1):
    """Performs a soft update of the target model parameters.

    For each weight w_s in the model, and its corresponding
    weight w_t in the target_model, a soft update is:
    w_t = (1 - tau) * w_t + tau * w_s

    Args:
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


def add_loss_summaries(loss_info: LossInfo, step):
    """Add summary about loss_info

    Args:
        loss_info (LossInfo): loss_info.extra must be a namedtuple
    """
    tf.summary.scalar('loss', data=loss_info.loss, step=step)
    if not loss_info.extra:
        return
    if getattr(loss_info.extra, '_fields') is None:
        return
    for field in loss_info.extra._fields:
        tf.summary.scalar(
            'loss/' + field, data=getattr(loss_info.extra, field), step=step)


def get_distribution_params(nested_distribution):
    """Get the params for an optionally nested action distribution.

    Only returns parameters that have tf.Tensor values.

    Args:
      nested_distribution: The nest of distributions whose parameter tensors to
        extract.
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
