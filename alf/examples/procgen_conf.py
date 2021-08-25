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
import alf
from alf.environments import suite_gym
from alf.algorithms.data_transformer import ImageScaleTransformer

alf.config('create_environment', env_load_fn=suite_gym.load)
alf.config('suite_gym.load', max_episode_steps=12500)

# Configure the data transformers
alf.config('ImageScaleTransformer', min=0.0, max=1.0)

alf.config('TrainerConfig', data_transformer_ctor=[ImageScaleTransformer])
