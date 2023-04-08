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
alf.import_config("sac_conf.py")
from alf.networks import ActorDistributionNetwork, QNetwork
from alf.utils.losses import element_wise_squared_loss
from alf.algorithms.sac_algorithm import SacAlgorithm

# environment config
alf.config(
    'create_environment', env_name="CartPole-v0", num_parallel_environments=8)

# algorithm config
alf.config('QNetwork', fc_layer_params=(100, ))
# note that for discrete action space we do not need the actor network as a
# discrete action can be sampled from the Q values.
alf.config(
    'SacAlgorithm',
    q_network_cls=QNetwork,
    actor_optimizer=alf.optimizers.Adam(lr=1e-3, name='actor'),
    critic_optimizer=alf.optimizers.Adam(lr=1e-3, name='critic'),
    alpha_optimizer=alf.optimizers.Adam(lr=1e-3, name='alpha'),
    target_update_tau=0.01)

alf.config(
    'OneStepTDLoss', td_error_loss_fn=element_wise_squared_loss, gamma=0.98)

# training config
alf.config(
    'TrainerConfig',
    initial_collect_steps=1000,
    mini_batch_length=2,
    mini_batch_size=64,
    unroll_length=1,
    num_updates_per_train_iter=1,
    num_iterations=10000,
    num_checkpoints=5,
    evaluate=False,
    eval_interval=100,
    debug_summaries=True,
    summary_interval=100,
    replay_buffer_length=100000)
