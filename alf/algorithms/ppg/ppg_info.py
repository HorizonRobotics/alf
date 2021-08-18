# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

# Import annotations to enable type hints of PPGTrainInfo inside PPGTrainInfo
from __future__ import annotations
from alf.data_structures import namedtuple

# Data structure to store the information produced by agent
# interacting with the environment.
PPGRolloutInfo = namedtuple(
    'PPGRolloutInfo',
    [
        # produced by the policy head
        'action_distribution',
        # Sampled from the action distribution produced by the policy head
        'action',
        # estimated value function by the value head
        'value',
        # estimated value function by the auxiliary value head
        'aux',
        'step_type',
        'discount',
        'reward',
        'reward_weights',
    ],
    default_value=())


class PPGTrainInfo(
        namedtuple(
            'PPGTrainInfo',
            PPGRolloutInfo._fields +
            ('advantages', 'rollout_action_distribution', 'returns'),
            default_value=())):
    """Data structure that stores extra derived information for training
    in addition to the original rollout information.

    Such extra information is derived during training updates, in
    ``preprocess_experience()`` and used across calls to
    ``train_step()``.

    It is designed as a separate class (as opposite to be merged into
    PPGRolloutInfo) becase we want to make it explicit about what are
    derived compared to the rollout information during training.

    """

    def absorbed(self, rollout_info: PPGRolloutInfo) -> PPGTrainInfo:
        """Combines the PPGTrainInfo and the PPGRolloutInfo.

        This function generate a new PPGTrainInfo instead of updating
        ``self`` in place.

        In ``train_step()`, we would like to keep the derived information in
        PPGTrainInfo while updating most of the shared fields (with
        PPGRolloutInfo) from evaluation of the updated network. This function
        makes it easy to do that.

        It is also used in ``preprocess_experience()`` where we need to combine
        the PPGRolloutInfo from the experience and the newly computed derived
        information.

        Args:

            rollout_info (PPGRolloutInfo): the result of rollout or evaluation
                that needs to be combined with ``self``

        Returns:

            A new PPGTrainInfo that combines the useful part from both parties.
        """
        return self._replace(
            step_type=rollout_info.step_type,
            reward=rollout_info.reward,
            discount=rollout_info.discount,
            action_distribution=rollout_info.action_distribution,
            value=rollout_info.value,
            aux=rollout_info.aux,
            reward_weights=rollout_info.reward_weights)
