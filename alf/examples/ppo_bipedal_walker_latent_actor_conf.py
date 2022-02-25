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

from functools import partial

import torch

import alf
from alf.examples import ppo_bipedal_walker_conf
from alf.networks import LatentActorDistributionNetwork
from alf.algorithms.entropy_target_algorithm import SGDEntropyTargetAlgorithm

alf.config(
    "ActorCriticAlgorithm",
    actor_network_ctor=partial(
        LatentActorDistributionNetwork, scale_distribution=True))

alf.config(
    "RealNVPNetwork",
    fc_layer_params=(128, ),
    num_layers=5,
    activation=torch.relu_)

alf.config(
    'Agent',
    entropy_target_cls=partial(SGDEntropyTargetAlgorithm, initial_alpha=1.))
