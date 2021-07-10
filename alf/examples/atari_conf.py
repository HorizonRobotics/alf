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
from alf.environments import suite_gym, gym_wrappers
from alf.algorithms.data_transformer import FrameStacker, ImageScaleTransformer, RewardClipping

alf.config('create_environment', env_load_fn=suite_gym.load)
alf.config('DMAtariPreprocessing', frame_skip=4)
alf.config(
    'suite_gym.load',
    gym_env_wrappers=[gym_wrappers.DMAtariPreprocessing],
    # Default max episode steps for all games
    #
    # Per DQN paper setting, 18000 frames assuming frameskip = 4
    max_episode_steps=4500)

# Configure the data transformers
alf.config('FrameStacker', stack_size=4)
alf.config('ImageScaleTransformer', min=0.0)
alf.config('RewardClipping', minmax=(-1, 1))

alf.config(
    'TrainerConfig',
    data_transformer_ctor=[
        FrameStacker, ImageScaleTransformer, RewardClipping
    ],
    epsilon_greedy=0.1)
