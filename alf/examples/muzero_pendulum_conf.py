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

from alf.algorithms.muzero_representation_learner import MuzeroRepresentationImpl
import math

import alf
import alf.examples.muzero_conf
from alf.utils import dist_utils
from alf.algorithms.data_transformer import RewardNormalizer
from alf.algorithms.mcts_models import SimpleMCTSModel
from alf.algorithms.mcts_algorithm import MCTSAlgorithm, VisitSoftmaxTemperatureByProgress
from alf.optimizers import AdamTF
from alf.networks import StableNormalProjectionNetwork, TruncatedProjectionNetwork


def define_config(name, default_value):
    alf.define_config(name, default_value)
    return alf.get_config_value('_CONFIG._USER.' + name)


train_repr_prediction = define_config('train_repr_prediction', False)
initial_scale = define_config('initial_scale', 1.0)

alf.config(
    "create_environment", env_name="Pendulum-v0", num_parallel_environments=1)

alf.config(
    "StableNormalProjectionNetwork",
    max_std=1000.0,
    state_dependent_std=True,
    scale_distribution=True,
    dist_squashing_transform=dist_utils.Softsign())

alf.config(
    "TruncatedProjectionNetwork",
    state_dependent_scale=True,
    scale_bias_initializer_value=math.log(math.exp(initial_scale) - 1),
    min_scale=0.05,
    max_scale=10.0,
    dist_ctor=dist_utils.TruncatedNormal)

alf.config(
    "SimplePredictionNet",
    continuous_projection_net_ctor=TruncatedProjectionNetwork)

alf.config(
    "SimpleMCTSModel",
    train_repr_prediction=train_repr_prediction,
    initial_alpha=0.005,
    num_sampled_actions=20)

alf.config(
    "MCTSAlgorithm",
    num_simulations=10,
    root_dirichlet_alpha=0.5,
    root_exploration_fraction=0.,
    pb_c_init=0.5,
    pb_c_base=19652,
    discount=0.99,
    is_two_player_game=False,
    visit_softmax_temperature_fn=VisitSoftmaxTemperatureByProgress(),
    act_with_exploration_policy=True,
    learn_with_exploration_policy=True,
    search_with_exploration_policy=True,
    unexpanded_value_score='mean',
    expand_all_children=False,
    expand_all_root_children=True)

reward_transformer = RewardNormalizer(update_mode="rollout")

alf.config(
    "MuzeroRepresentationImpl",
    model_ctor=SimpleMCTSModel,
    num_unroll_steps=5,
    td_steps=10,
    reward_transformer=reward_transformer,
    train_repr_prediction=train_repr_prediction,
    reanalyze_ratio=1.0,
    reanalyze_algorithm_ctor=MCTSAlgorithm,
    target_update_period=1,
    target_update_tau=0.01)

alf.config(
    "MuzeroAlgorithm",
    discount=0.99,
    representation_learner_ctor=MuzeroRepresentationImpl,
    mcts_algorithm_ctor=MCTSAlgorithm,
    reward_transformer=reward_transformer)

alf.config("Agent", optimizer=AdamTF(lr=5e-4))

# training config
alf.config(
    "TrainerConfig",
    unroll_length=10,
    mini_batch_size=256,
    num_updates_per_train_iter=10,
    num_iterations=1000,
    num_checkpoints=5,
    evaluate=False,
    summary_interval=0,
    num_summaries=100,
    replay_buffer_length=100000,
    initial_collect_steps=1000)
