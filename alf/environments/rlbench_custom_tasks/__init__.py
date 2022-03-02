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

import os
from os.path import dirname, join, abspath
import importlib

import gym
from gym.envs.registration import register

TASKS_PATH = join(dirname(abspath(__file__)), '.')
TASKS = [
    t for t in os.listdir(TASKS_PATH)
    if t != '__init__.py' and t.endswith('.py')
]

for task_file in TASKS:
    task_name = task_file.split('.py')[0]
    mod = importlib.import_module(
        'alf.environments.rlbench_custom_tasks.%s' % task_name)
    class_name = ''.join([w[0].upper() + w[1:] for w in task_name.split('_')])
    task_class = getattr(mod, class_name)
    register(
        id='%s-state-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'state'
        })
    register(
        id='%s-vision-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'vision'
        })
    register(
        id='%s-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'custom'
        })
