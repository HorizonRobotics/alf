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

from functools import partial

import alf
from alf.algorithms.td_loss import TDLoss
from alf.algorithms.seditor_algorithm import SEditorAlgorithm
from alf.utils import dist_utils

from alf.examples.safety.sac import sac_safety_gym_conf

alf.config(
    'SEditorAlgorithm',
    actor_network_cls=sac_safety_gym_conf.actor_network_cls,
    critic_network_cls=sac_safety_gym_conf.critic_network_cls,
    target_entropy=(partial(
        dist_utils.calc_default_target_entropy, min_prob=0.1),
                    partial(
                        dist_utils.calc_default_target_entropy,
                        min_prob=0.05)),
    communicate_steps=1,
    train_da_alpha=True,
    initial_alpha=(1e-3, 1.),
    target_update_tau=0.005)

alf.config('Agent', rl_algorithm_cls=SEditorAlgorithm)
