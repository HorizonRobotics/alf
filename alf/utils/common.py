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
