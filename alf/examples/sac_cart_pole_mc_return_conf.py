# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from alf.algorithms.sac_algorithm import SacAlgorithm

alf.import_config("sac_cart_pole_conf.py")

default_return = -1000
use_mc_return = True
mini_batch_length = 2  # set to a value > 2 for multi-step learning

alf.config("ReplayBuffer",
           keep_episodic_info=True,
           record_episodic_return=True,
           default_return=default_return)
alf.config("TDLoss", default_return=default_return)
alf.config("SacAlgorithm", use_mc_return=use_mc_return)

alf.config('TrainerConfig', mini_batch_length=mini_batch_length)
