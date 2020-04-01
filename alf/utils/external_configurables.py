# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Make various external gin-configurable objects."""

import gin
import gym
import torch

import alf

alf.optimizers.Adam = gin.external_configurable(alf.optimizers.Adam,
                                                'alf.optimizers.Adam')
alf.optimizers.AdamW = gin.external_configurable(alf.optimizers.AdamW,
                                                 'alf.optimizers.AdamW')
alf.optimizers.SGD = gin.external_configurable(alf.optimizers.SGD,
                                               'alf.optimizers.SGD')

# This allows the environment creation arguments to be configurable by supplying
# gym.envs.registration.EnvSpec.make.ARG_NAME=VALUE
gym.envs.registration.EnvSpec.make = gin.external_configurable(
    gym.envs.registration.EnvSpec.make, 'gym.envs.registration.EnvSpec.make')

# Activation functions.
gin.external_configurable(torch.exp, 'torch.exp')
gin.external_configurable(torch.tanh, 'torch.tanh')
gin.external_configurable(torch.relu, 'torch.relu')
gin.external_configurable(torch.nn.functional.elu, 'torch.nn.functional.elu')
gin.external_configurable(torch.nn.functional.softsign,
                          'torch.nn.functional.softsign')

# gin.external_configurable(tf.keras.initializers.GlorotUniform,
#                           'tf.keras.initializers.GlorotUniform')
