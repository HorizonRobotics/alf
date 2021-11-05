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

import functools
import alf
from alf.algorithms.agent import Agent
from alf.algorithms.encoding_algorithm import EncodingAlgorithm
from alf.algorithms.mcts_algorithm import MCTSAlgorithm, VisitSoftmaxTemperatureByMoves
from alf.algorithms.mcts_models import SimpleMCTSModel
from alf.algorithms.muzero_algorithm import MuzeroAlgorithm
from alf.environments import suite_simple
from alf.networks import EncodingNetwork
from alf.optimizers import Adam

from alf.examples import muzero_conf

alf.config(
    'create_environment',
    env_name='StochasticWithRiskyBranch',
    env_load_fn=suite_simple.load,
    num_parallel_environments=32)

alf.config(
    'VisitSoftmaxTemperatureByMoves',
    move_temperature_pairs=[(0, 1.0), (10, 0.0001)])
alf.config(
    'MCTSAlgorithm',
    num_simulations=200,
    discount=1.0,
    root_dirichlet_alpha=0.25,
    root_exploration_fraction=0.25,
    pb_c_init=1.25,
    pb_c_base=19652,
    is_two_player_game=False,
    visit_softmax_temperature_fn=VisitSoftmaxTemperatureByMoves())

alf.config(
    'MuzeroAlgorithm',
    mcts_algorithm_ctor=MCTSAlgorithm,
    model_ctor=SimpleMCTSModel,
    num_unroll_steps=2,
    td_steps=-1)

alf.config(
    'EncodingNetwork',
    fc_layer_params=(10, ),
    input_preprocessors=alf.layers.Reshape([-1]))
alf.config('EncodingAlgorithm', encoder_cls=EncodingNetwork)

alf.config(
    'Agent', representation_learner_cls=EncodingAlgorithm, optimizer=Adam())

alf.config(
    'TrainerConfig',
    data_transformer_ctor=None,
    unroll_length=10,
    mini_batch_size=512,
    num_updates_per_train_iter=10,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    num_iterations=4000,
    num_checkpoints=1,
    evaluate=True,
    num_eval_episodes=100,
    eval_interval=1000,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=10,
    replay_buffer_length=200)
