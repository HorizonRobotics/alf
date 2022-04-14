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
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.dynamic_action_repeat_agent import DynamicActionRepeatAgent
from alf.algorithms.data_transformer import UntransformedTimeStep
from alf.examples import sac_bipedal_walker_conf
from alf.utils import dist_utils

alf.config(
    "SacAlgorithm",
    q_network_cls=partial(
        alf.networks.QNetwork,
        preprocessing_combiner=alf.layers.NestConcat(),
        fc_layer_params=sac_bipedal_walker_conf.hidden_layers),
    target_entropy=(partial(
        dist_utils.calc_default_target_entropy, min_prob=0.2),
                    partial(
                        dist_utils.calc_default_target_entropy, min_prob=0.1)))

alf.config(
    "DynamicActionRepeatAgent", K=5, rl_algorithm_cls=SacAlgorithm, gamma=0.99)

alf.config(
    "TrainerConfig",
    data_transformer_ctor=[UntransformedTimeStep],
    algorithm_ctor=partial(
        DynamicActionRepeatAgent,
        optimizer=sac_bipedal_walker_conf.optimizer,
        reward_normalizer_ctor=sac_bipedal_walker_conf.reward_normalizer))
