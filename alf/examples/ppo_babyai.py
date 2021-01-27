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
import torch

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.encoding_algorithm import EncodingAlgorithm
from alf.algorithms.ppo_algorithm import PPOAlgorithm, PPOLoss
from alf.algorithms.trac_algorithm import TracAlgorithm
from alf.environments import suite_babyai, alf_wrappers

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    unroll_length=64,
    mini_batch_size=32,
    num_iterations=50000,
    num_updates_per_train_iter=2,
    eval_interval=1000,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=100,
    use_rollout_state=True,
)

alf.config(
    'suite_babyai.load',
    mode='sent',
    alf_env_wrappers=[alf_wrappers.ActionObservationWrapper])

alf.config(
    'create_environment',
    env_load_fn=suite_babyai.load,
    env_name=[
        "BabyAI-GoToObj-v0", "BabyAI-GoToRedBallGrey-v0",
        "BabyAI-GoToRedBall-v0", "BabyAI-GoToLocal-v0", "BabyAI-PickupLoc-v0"
    ],
    num_parallel_environments=64,
    batched_wrappers=(alf_wrappers.CurriculumWrapper, ),
)

observation_spec = alf.get_observation_spec()
action_spec = alf.get_action_spec()
mission_spec = observation_spec['observation']['mission']
vocab_size = mission_spec.maximum + 1

encoding_dim = 128
activation = torch.relu_
fc_layers_params = (256, 256)

observation_preprocessors = {
    "image":
        torch.nn.Sequential(
            alf.layers.Permute(2, 0, 1), alf.layers.Cast(),
            alf.layers.Conv2D(3, encoding_dim, kernel_size=3),
            alf.layers.Conv2D(encoding_dim, encoding_dim, kernel_size=3),
            alf.layers.Reshape((encoding_dim, -1)), alf.layers.Transpose()),
    "direction":
        torch.nn.Sequential(
            torch.nn.Embedding(4, encoding_dim),
            alf.layers.Reshape((1, encoding_dim))),
    "mission":
        torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, encoding_dim),
            alf.layers.Reshape((-1, encoding_dim)))
}
input_preprocessors = {
    'observation':
        observation_preprocessors,
    'prev_action':
        torch.nn.Sequential(
            torch.nn.Embedding(7, encoding_dim),
            alf.layers.Reshape((1, encoding_dim)))
}

encoder_cls = partial(
    alf.networks.TransformerNetwork,
    input_preprocessors=input_preprocessors,
    memory_size=8,
    core_size=1,
    num_prememory_layers=0,
    num_memory_layers=4,
    num_attention_heads=3,
    d_ff=encoding_dim,
    centralized_memory=True)

repr_learner_cls = partial(EncodingAlgorithm, encoder_cls=encoder_cls)

actor_network_ctor = partial(
    alf.networks.ActorDistributionNetwork,
    fc_layer_params=fc_layers_params,
    activation=activation,
    discrete_projection_net_ctor=alf.networks.CategoricalProjectionNetwork)

value_network_ctor = partial(
    alf.networks.ValueNetwork,
    fc_layer_params=fc_layers_params,
    activation=activation)

alf.config(
    'PPOLoss',
    entropy_regularization=0.0,
    gamma=0.99,
    normalize_advantages=True,
    td_lambda=0.95,
    td_error_loss_fn=alf.utils.losses.element_wise_squared_loss,
    check_numerics=True)

alf.config(
    'PPOAlgorithm',
    actor_network_ctor=actor_network_ctor,
    value_network_ctor=value_network_ctor,
    loss_class=PPOLoss)

alf.config(
    'TracAlgorithm',
    ac_algorithm_cls=PPOAlgorithm,
    action_dist_clip_per_dim=0.01)

alf.config('EntropyTargetAlgorithm', initial_alpha=0.001)

alf.config(
    'Agent',
    representation_learner_cls=repr_learner_cls,
    optimizer=alf.optimizers.AdamTF(lr=1e-4),
    rl_algorithm_cls=TracAlgorithm,
    enforce_entropy_target=True)
