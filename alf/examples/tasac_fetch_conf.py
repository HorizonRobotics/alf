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

import sys
sys.path.append("./benchmarks/fetch")

from functools import partial

import alf
from alf.algorithms.tasac_algorithm import TasacAlgorithm
from alf.utils import dist_utils

import sac_conf
import fetch_conf

alf.config(
    'TasacAlgorithmBase',
    actor_network_cls=fetch_conf.actor_distribution_network_cls,
    critic_network_cls=fetch_conf.critic_network_cls,
    target_update_tau=0.05,
    target_update_period=40,
    target_entropy=(partial(
        dist_utils.calc_default_target_entropy, min_prob=0.05),
                    partial(
                        dist_utils.calc_default_target_entropy, min_prob=0.2)))

alf.config(
    'Agent', rl_algorithm_cls=TasacAlgorithm, optimizer=fetch_conf.optimizer)

alf.config('TASACTDLoss', gamma=0.98)

# training config
alf.config('TrainerConfig', mini_batch_length=4)
