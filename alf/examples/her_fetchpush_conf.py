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
from alf.algorithms.data_transformer import HindsightExperienceTransformer, \
    ObservationNormalizer
from alf.algorithms.ddpg_algorithm import DdpgAlgorithm
from alf.environments import suite_robotics
from alf.nest.utils import NestConcat

from alf.examples import ddpg_fetchpush_conf

alf.config('suite_robotics.load', concat_desired_goal=False)
alf.config('ActorNetwork', preprocessing_combiner=NestConcat())
alf.config(
    'CriticNetwork',
    observation_preprocessing_combiner=NestConcat(),
    action_preprocessing_combiner=NestConcat())

alf.config('ReplayBuffer', keep_episodic_info=True)
alf.config('HindsightExperienceTransformer', her_proportion=0.8)
alf.config(
    'TrainerConfig',
    data_transformer_ctor=[
        HindsightExperienceTransformer, ObservationNormalizer
    ])

alf.config('DdpgAlgorithm', action_l2=0.05)

# Finer grain tensorboard summaries plus local action distribution
# TrainerConfig.summarize_action_distributions=True
# TrainerConfig.summary_interval=1
# TrainerConfig.update_counter_every_mini_batch=True
# TrainerConfig.summarize_grads_and_vars=1
# TrainerConfig.summarize_output=True
# summarize_gradients.with_histogram=False
# summarize_variables.with_histogram=False
