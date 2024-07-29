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
from typing import Optional

from alf.data_structures import namedtuple, TimeStep, AlgStep
from alf.utils import common, dist_utils

from .disjoint_policy_value_network import DisjointPolicyValueNetwork

# Data structure to store the information produced by agent
# interacting with the environment.
PPGRolloutInfo = namedtuple(
    'PPGRolloutInfo',
    [
        # produced by the policy head
        'action_distribution',
        # Sampled from the action distribution produced by the policy head
        'action',
        # Log probability of the action at the rollout time
        'log_prob',
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
            PPGRolloutInfo._fields + ('rollout_action_distribution',
                                      'rollout_value', 'rollout_log_prob'),
            default_value=())):
    """Data structure that stores extra derived information for training
    in addition to the original rollout information.

    Such extra information is derived during training updates and used
    across calls to ``train_step()``.

    It is designed as a separate class (as opposite to be merged into
    PPGRolloutInfo) because we want to make it explicit about what are
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


def ppg_network_forward(network: DisjointPolicyValueNetwork,
                        inputs: TimeStep,
                        state,
                        require_aux: bool = True,
                        epsilon_greedy: Optional[float] = None) -> AlgStep:
    """Evaluates the network forward pass for roll out or training
    The signature mimics ``rollout_step()`` of ``Algorithm`` completedly.
    Args:
    
        network: the network whose forward pass is to be performed.
        inputs: carries the observation that is needed as input to the
            network.
        state (nested Tesnor): carries the state for RNN-based network
        require_aux: whether to compute and return auxiliary estimation.
            See DisjointPolicyValueNetwork.forward() for details.
        epsilon_greedy: if set to None, the action will be sampled
            strictly based on the action distribution. If set to a
            value in [0, 1], epsilon-greedy sampling will be used to
            sample the action from the action distribution, and the
            float value determines the chance of action sampling
            instead of taking argmax.
    """
    (action_distribution, value, aux), state = network(
        inputs.observation, state=state, require_aux=require_aux)

    if epsilon_greedy is not None:
        action = dist_utils.epsilon_greedy_sample(action_distribution,
                                                  epsilon_greedy)
        log_prob = ()
    else:
        action, log_prob = dist_utils.sample_action_distribution(
            action_distribution, return_log_prob=True)

    return AlgStep(
        output=action,
        state=state,
        info=PPGRolloutInfo(
            action_distribution=action_distribution,
            action=common.detach(action),
            log_prob=common.detach(log_prob),
            value=value,
            aux=aux,
            step_type=inputs.step_type,
            discount=inputs.discount,
            reward=inputs.reward,
            reward_weights=()))
