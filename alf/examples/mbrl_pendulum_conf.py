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
from alf.algorithms.dynamics_learning_algorithm import DeterministicDynamicsAlgorithm, StochasticDynamicsAlgorithm
from alf.algorithms.mbrl_algorithm import MbrlAlgorithm
from alf.algorithms.planning_algorithm import CEMPlanAlgorithm, RandomShootingAlgorithm
from alf.algorithms.reward_learning_algorithm import FixedRewardFunction
from alf.examples.mbrl_pendulum import reward_function_for_pendulum
from alf.initializers import variance_scaling_init
from alf.networks.dynamics_networks import DynamicsNetwork
from alf.optimizers import AdamW
from alf.utils.math_ops import clipped_exp, swish


def define_config(name, default_value):
    alf.define_config(name, default_value)
    return alf.get_config_value('_CONFIG._USER.' + name)


alf.config(
    "create_environment", env_name="Pendulum-v0", num_parallel_environments=1)

# +------------------------------+
# | Dynamics Learning            |
# +------------------------------+

stochastic_dynamics = define_config("stochastic_dynamics", False)

alf.config(
    "DynamicsNetwork",
    activation=swish,
    joint_fc_layer_params=(500, 500, 500),
    kernel_initializer=variance_scaling_init)

alf.config("NormalProjectionNetwork", std_transform=clipped_exp)

if stochastic_dynamics:
    alf.config("DynamicsNetwork", prob=1)
    dynamics_algorithm_ctor = partial(
        StochasticDynamicsAlgorithm,
        num_replicas=1,
        dynamics_network_ctor=DynamicsNetwork)
else:
    alf.config("DynamicsNetwork", prob=0)
    dynamics_algorithm_ctor = partial(
        DeterministicDynamicsAlgorithm,
        num_replicas=1,
        dynamics_network_ctor=DynamicsNetwork)

# +------------------------------+
# | Planner                      |
# +------------------------------+

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

# +------------------------------+
# | MBRL Algorithm               |
# +------------------------------+

alf.config(
    "MbrlAlgorithm",
    dynamics_module_ctor=dynamics_algorithm_ctor,
    reward_module=FixedRewardFunction(reward_function_for_pendulum),
    planner_module_ctor=planner_ctor,
    particles_per_replica=1,
    dynamics_optimizer=AdamW(lr=1e-3, weight_decay=1e-4))

alf.config(
    "TrainerConfig",
    algorithm_ctor=MbrlAlgorithm,
    unroll_length=200,
    mini_batch_size=32,
    mini_batch_length=1,
    whole_replay_buffer_training=True,
    clear_replay_buffer=False,
    num_updates_per_train_iter=5,
    update_counter_every_mini_batch=False,
    num_iterations=50,  # num of interactions of unroll_length with env
    num_checkpoints=5,
    use_rollout_state=True,
    evaluate=False,
    enable_amp=False,
    debug_summaries=True,
    summary_interval=1,
    replay_buffer_length=100_000,
    initial_collect_steps=200,
    summarize_grads_and_vars=True)
