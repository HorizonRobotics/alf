# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import alf
from alf.algorithms.seditor_algorithm import SEditorAlgorithm

from alf.examples import sac_safety_gym_conf
"""Follow instructions in ``sac_safety_gym_conf.py`` and ``suite_safety_gym.py``
for env installation.
"""

alf.config(
    'SEditorAlgorithm',
    actor_network_ctor=sac_safety_gym_conf.actor_network_cls,
    critic_network_ctor=sac_safety_gym_conf.critic_network_cls,
    target_update_tau=0.005)

alf.config('Agent', rl_algorithm_cls=SEditorAlgorithm)
