# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.utils.schedulers import LinearScheduler

from alf.examples import sac_bipedal_walker_conf

optimizer = alf.optimizers.Adam(
    lr=5e-4,
    capacity_ratio=LinearScheduler(
        progress_type="percent", schedule=[(0, 0.1), (1., 1)]),
    masked_out_value=0)
alf.config("Agent", optimizer=optimizer)

# training config
alf.config("TrainerConfig", evaluate=True, async_eval=True, eval_interval=1)
