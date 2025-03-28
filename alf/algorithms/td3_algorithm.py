# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Twin Delayed Deep Deterministic Policy Gradient (TD3)."""

import functools
import numpy as np

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.ddpg_algorithm import DdpgAlgorithm
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops, spec_utils


@alf.configurable
class Td3Algorithm(DdpgAlgorithm):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3). 

    Reference:
    Fujimoto et at "Addressing Function Approximation Error in Actor-Critic Methods"
    https://arxiv.org/abs/1802.09477
    """

    def __init__(self,
                 target_policy_noise_scale=0.2,
                 target_policy_noise_clip=0.5,
                 name="Td3Algorithm",
                 **kwargs):
        """
        Refer to DdpgAlgorithm for more details for kwargs

        Args:
            target_policy_noise_scale (float): target policy smoothing noise
            target_policy_noise_clip (float): target policy smoothing noise clip
        """
        super().__init__(name=name, **kwargs)
        self._target_policy_noise_scale = target_policy_noise_scale
        self._target_policy_noise_clip = target_policy_noise_clip
