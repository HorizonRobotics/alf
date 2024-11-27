# Copyright (c) 2024 Horizon Robotics and Hobot Contributors. All Rights Reserved.
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
from functools import partial
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.distributed_off_policy_algorithm import (
    DistributedUnroller, DistributedTrainer)
from alf.networks import QNetwork

alf.config('make_ddp_performer', find_unused_parameters=True)

mode = alf.define_config('mode', 0)  # 0: trainer, 1: unroller

alf.config(
    'create_environment', env_name="CartPole-v0", num_parallel_environments=1)

alf.config(
    'TrainerConfig',
    whole_replay_buffer_training=False,
    initial_collect_steps=10000,
    mini_batch_size=32,
    num_checkpoints=1,
    num_iterations=100000,
    debug_summaries=True,
    num_updates_per_train_iter=1)  # fixed

core_alg_ctor = partial(
    SacAlgorithm,
    q_network_cls=partial(QNetwork, fc_layer_params=(128, )),
    target_update_tau=0.005)

if mode == 0:
    alg_ctor = partial(
        DistributedTrainer,
        max_utd_ratio=10,
        optimizer=alf.optimizers.Adam(lr=1e-3),
        core_alg_ctor=core_alg_ctor)
    alf.config(
        'TrainerConfig',
        algorithm_ctor=alg_ctor,
        mini_batch_length=2,
        replay_buffer_length=100000,
        unroll_length=-1,  # with assertion
        summary_interval=100,  # can only summarize training statistics
        evaluate=False)  # no evaluation on the trainer
else:
    alg_ctor = partial(
        DistributedUnroller,
        pull_params_every_n_iters=20,
        core_alg_ctor=core_alg_ctor)
    alf.config(
        'TrainerConfig',
        algorithm_ctor=alg_ctor,
        summary_interval=100,  # can only summarize rollout statistics
        unroll_length=10,  # How often to request a parameter update from trainer
        async_eval=False,
        eval_interval=600,
        evaluate=False)  # evaluation on the client (optional)
