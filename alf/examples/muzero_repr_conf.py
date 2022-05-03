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

import alf
from alf.algorithms.mcts_models import SimpleMCTSModel
from alf.algorithms.muzero_representation_learner import LinearTdStepFunc, MuzeroRepresentationImpl, MuzeroRepresentationLearner

alf.config('TrainerConfig', use_rollout_state=True)
alf.config('ReplayBuffer', keep_episodic_info=True)

alf.config(
    "MCTSModel",
    predict_reward_sum=True,
    policy_loss_weight=1.0,
    value_loss_weight=0.05,
    repr_prediction_loss_weight=40.0,
    reward_loss_weight=2.0)

alf.config(
    "SimpleMCTSModel",
    train_repr_prediction=True,
    train_game_over_function=True,
    train_policy=False,
    initial_alpha=0.)

alf.config(
    "MuzeroRepresentationImpl",
    model_ctor=SimpleMCTSModel,
    reanalyze_td_steps_func=LinearTdStepFunc(
        max_bootstrap_age=1.2, min_td_steps=1),
    train_repr_prediction=True,
    train_game_over_function=True,
    train_policy=False,
    reanalyze_ratio=0.0,
    target_update_period=400,
    target_update_tau=1.0)

alf.config("MuzeroRepresentationLearner", impl_cls=MuzeroRepresentationImpl)

alf.config(
    'Agent',
    representation_learner_cls=MuzeroRepresentationLearner,
    representation_use_rl_state=True)
