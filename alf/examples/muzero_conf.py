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
from alf.algorithms.agent import Agent
from alf.algorithms.muzero_algorithm import MuzeroAlgorithm
from alf.experience_replayers.replay_buffer import ReplayBuffer

alf.config('Agent', rl_algorithm_cls=MuzeroAlgorithm)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    mini_batch_length=1,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    use_rollout_state=True,
    debug_summaries=True,
    summarize_grads_and_vars=True)

alf.config('ReplayBuffer', keep_episodic_info=True)

alf.config("summarize_variables", with_histogram=False)
alf.config("summarize_gradients", with_histogram=False)
