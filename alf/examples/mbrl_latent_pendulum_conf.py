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
from alf.algorithms.agent import Agent
from alf.algorithms.mbrl_algorithm import LatentMbrlAlgorithm
from alf.algorithms.planning_algorithm import CEMPlanAlgorithm, RandomShootingAlgorithm
from alf.algorithms.predictive_representation_learner import PredictiveRepresentationLearner, SimpleDecoder
from alf.networks.encoding_networks import EncodingNetwork, LSTMEncodingNetwork
from alf.optimizers import Adam
from alf.utils.math_ops import identity


def define_config(name, default_value):
    alf.define_config(name, default_value)
    return alf.get_config_value('_CONFIG._USER.' + name)


alf.config(
    "create_environment", env_name="Pendulum-v0", num_parallel_environments=1)

alf.config('ReplayBuffer', keep_episodic_info=True)

decoder_net_ctor = partial(
    EncodingNetwork,
    last_layer_size=1,
    last_activation=identity,
    last_kernel_initializer=torch.nn.init.zeros_,
    output_tensor_spec=alf.TensorSpec(()),
)

alf.config(
    "PredictiveRepresentationLearner",
    num_unroll_steps=25,
    encoding_net_ctor=partial(
        EncodingNetwork, activation=torch.relu_, fc_layer_params=(100, 100)),
    decoder_ctor=[
        partial(
            SimpleDecoder,
            decoder_net_ctor=decoder_net_ctor,
            target_field="reward",
            summarize_each_dimension=True)
    ],
    dynamics_net_ctor=partial(LSTMEncodingNetwork, hidden_size=(100, 100)))

planner_type = define_config("planner_type", "random_shooting")

assert planner_type in ["cem", "random_shooting"
                        ], (f"Unrecognized planner type '{planner_type}'.")

if planner_type == "cem":
    planner_ctor = partial(
        CEMPlanAlgorithm,
        population_size=400,
        planning_horizon=25,
        elite_size=40,
        max_iter_num=5,
        epsilon=0.01,
        tau=0.9,
        scalar_var=1.0)
elif planner_type == "random_shooting":
    planner_ctor = partial(
        RandomShootingAlgorithm, population_size=5000, planning_horizon=25)

alf.config("LatentMbrlAlgorithm", planner_module_ctor=planner_ctor)

optimizer = Adam(lr=1e-4)

alf.config(
    "Agent",
    optimizer=optimizer,
    representation_learner_cls=PredictiveRepresentationLearner,
    rl_algorithm_cls=LatentMbrlAlgorithm)

alf.config(
    "TrainerConfig",
    algorithm_ctor=Agent,
    unroll_length=1,
    mini_batch_size=32,
    mini_batch_length=4,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    num_updates_per_train_iter=5,
    update_counter_every_mini_batch=False,
    num_iterations=10_000,
    num_checkpoints=5,
    evaluate=False,
    enable_amp=False,
    debug_summaries=True,
    summary_interval=10,
    replay_buffer_length=100_000,
    initial_collect_steps=200,
    summarize_grads_and_vars=True)
