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
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.trac_algorithm import TracAlgorithm
from alf.algorithms.data_transformer import RewardScaling

# environment config
alf.config(
    'create_environment', env_name="CartPole-v0", num_parallel_environments=8)

# reward scaling
alf.config('TrainerConfig', data_transformer_ctor=RewardScaling)
alf.config('RewardScaling', scale=0.01)

# algorithm config
alf.config('ActorDistributionNetwork', fc_layer_params=(100, ))
alf.config('ValueNetwork', fc_layer_params=(100, ))
alf.config(
    'ActorCriticAlgorithm',
    optimizer=alf.optimizers.Adam(lr=1e-3, gradient_clipping=10.0))
alf.config(
    'ActorCriticLoss',
    entropy_regularization=1e-4,
    gamma=0.98,
    use_gae=True,
    use_td_lambda_return=True)

# training config
alf.config(
    'TrainerConfig',
    unroll_length=10,
    algorithm_ctor=TracAlgorithm,
    num_iterations=2500,
    num_checkpoints=5,
    evaluate=True,
    eval_interval=500,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    summary_interval=5,
    epsilon_greedy=0.1)
