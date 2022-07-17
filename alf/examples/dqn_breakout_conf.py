# Copyright (c) 2022 Horizon Robotics. All Rights Reserved.
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

# NOTE: for lower bound value target improvement, add these flags:
# --conf_param='ReplayBuffer.keep_episodic_info=True'
# --conf_param='ReplayBuffer.record_episodic_return=True'
# --conf_param='LowerBoundedTDLoss.lb_target_q=True'

import alf
from alf.algorithms.dqn_algorithm import DqnAlgorithm
from alf.utils.schedulers import LinearScheduler

# Much of the network and critic loss parameters are the same as in sac_breakout.
from alf.examples.sac_breakout_conf import q_network_cls, critic_loss_ctor, \
    critic_optimizer

alf.config(
    'DqnAlgorithm',
    q_network_cls=q_network_cls,
    rollout_epsilon_greedy=LinearScheduler(
        progress_type="percent", schedule=[(0, 1.), (0.1, 0.1), (1., 0.1)]),
    critic_loss_ctor=critic_loss_ctor,
    q_optimizer=critic_optimizer)

alf.config('Agent', rl_algorithm_cls=DqnAlgorithm)
