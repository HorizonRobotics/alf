# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

"""Simple Concurrent DQN Demo Configuration.

This demonstrates SimpleConcurrentAlgorithm with DQN on the NoisyArray environment.
The algorithm creates multiple independent DQN copies that learn concurrently.
"""

import alf
from alf.algorithms.dqn_algorithm import DqnAlgorithm
from alf.algorithms.simple_concurrent_algorithm import SimpleConcurrentAlgorithm
from alf.environments.parallel_environment import ParallelAlfEnvironment
from alf.networks import QNetwork
from alf.optimizers import AdamTF
from alf.utils.schedulers import LinearScheduler

# Environment configuration
alf.config(
    "create_environment",
    env_name="CartPole-v1",
    num_parallel_environments=2,  # Must be multiple of num_copies (2)
)  # Use parallel environments

# Q-Network configuration
q_network_cls = lambda input_tensor_spec, action_spec: QNetwork(
    input_tensor_spec=input_tensor_spec,
    action_spec=action_spec,
    fc_layer_params=(64, 64),
)

# DQN algorithm configuration
alf.config(
    "DqnAlgorithm",
    q_network_cls=q_network_cls,
    rollout_epsilon_greedy=LinearScheduler(
        progress_type="percent", schedule=[(0, 0.9), (0.1, 0.1), (1.0, 0.05)]
    ),
    q_optimizer=AdamTF(lr=1e-3),
)

# SimpleConcurrentAlgorithm configuration
alf.config(
    "SimpleConcurrentAlgorithm", algorithm_ctor=DqnAlgorithm, num_copies=2
)  # 2 independent DQN copies

# Training configuration
alf.config(
    "TrainerConfig",
    algorithm_ctor=SimpleConcurrentAlgorithm,
    num_iterations=2000,
    unroll_length=1,
    mini_batch_length=2,
    mini_batch_size=64,
    num_updates_per_train_iter=1,
    initial_collect_steps=100,
    replay_buffer_length=10000,
    evaluate=True,
    eval_interval=200,
    num_eval_episodes=5,
    debug_summaries=True,
    random_seed=42,
)
