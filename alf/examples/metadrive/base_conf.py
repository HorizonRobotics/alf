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
from alf.data_structures import Experience
from alf.environments import suite_metadrive
from alf.utils import summary_utils

# Environment Configuration
alf.config(
    'create_environment',
    env_load_fn=suite_metadrive.load,
    num_parallel_environments=12)

alf.config(
    'suite_metadrive.load',
    scenario_num=5000,
    crash_penalty=50.0,
    success_reward=200.0,
    traffic_density=0.1)

# The following config will create customized summaries of the env_info from the
# MetaDrive environment via ``custom_summary`` of ``summarize_rollout``. The
# default env_info summarization will only take care of the ones specified in
# the ``fields`` of ``AverageEnvInfoMetric``.


def summarize_metadrive(experience: Experience):
    with alf.summary.scope("MetaDrive"):
        env_info = experience.time_step.env_info
        summary_utils.add_mean_hist_summary("velocity@step",
                                            env_info["velocity@step"])
        summary_utils.add_mean_hist_summary("abs_steering@step",
                                            env_info["abs_steering@step"])


alf.config("alf.metrics.metrics.AverageEnvInfoMetric", fields=["reach_goal"])
alf.config("RLAlgorithm.summarize_rollout", custom_summary=summarize_metadrive)
