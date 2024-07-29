# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.examples import sac_bipedal_walker_conf

# This is a minimal example of how to use the hybrid training pipeline.
# In the simplest case where the offline replay buffer has all the necessary
# data required for off-policy training, we only need to provide a valid replay
# buffer path to offline_buffer_dir, which will automatically invoke the offline
# training branch apart from the normal RL training.

# training config
alf.config(
    "TrainerConfig",
    mini_batch_size=4096 // 2,
    # we can control when to start the online RL training with this
    rl_train_after_update_steps=0,
    # we can control the update frequency of the online RL training wrt offline
    # RL by setting rl_train_every_update_steps.
    # E.g., rl_train_every_update_steps=2 means we only do one online RL training
    # every two offline RL training passes.
    rl_train_every_update_steps=1,
    offline_buffer_dir=
    "/media/DATA/data/pytorch_alf/bipedal/data_collection/train/algorithm/ckpt-112030-replay_buffer"
)
