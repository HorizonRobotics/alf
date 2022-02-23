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

import alf
from alf.nest.utils import NestConcat
from alf.examples import sac_conf
from alf.networks import UnitNormalActorDistributionNetwork
from alf.networks import LatentActorDistributionNetwork
from alf.networks import RealNVPNetwork

alf.config(
    "create_environment", env_name="Pendulum-v0", num_parallel_environments=1)

alf.config("CriticNetwork", joint_fc_layer_params=(100, 100))
alf.config("RealNVPNetwork", fc_layer_params=(64, 64), num_layers=5)

alf.config("Agent", optimizer=alf.optimizers.AdamTF(lr=5e-4))
alf.config(
    "SacAlgorithm",
    actor_network_cls=partial(
        LatentActorDistributionNetwork,
        prior_actor_distribution_network_ctor=
        UnitNormalActorDistributionNetwork,
        normalizing_flow_network_ctor=RealNVPNetwork,
        scale_distribution=True),
    target_update_tau=0.005)

alf.config(
    "TrainerConfig",
    initial_collect_steps=1000,
    mini_batch_length=2,
    unroll_length=1,
    mini_batch_size=64,
    num_updates_per_train_iter=1,
    num_iterations=20000,
    num_checkpoints=5,
    evaluate=True,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=100,
    eval_interval=500,
    replay_buffer_length=100000)
