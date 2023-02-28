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
import math
from pathlib import Path

import torch

import alf
from alf import layers
from alf import networks
from alf.algorithms.data_transformer import RewardNormalizer
from alf.algorithms.mcts_models import SimpleMCTSModel
from alf.algorithms.muzero_representation_learner import MuzeroRepresentationImpl
from alf.algorithms.planning_algorithm import MPPIPlanner
from alf.networks.encoding_networks import EncodingNetwork, LSTMEncodingNetwork
from alf.networks.projection_networks import BetaProjectionNetwork, StableNormalProjectionNetwork, TruncatedProjectionNetwork
from alf.algorithms.td_mpc_algorithm import TdMpcAlgorithm
from alf.optimizers import Adam
from alf.tensor_specs import TensorSpec
from alf.utils import dist_utils

from alf.examples import muzero_conf
from alf.utils.math_ops import identity
from alf.utils.schedulers import LinearScheduler
from alf.utils.summary_utils import summarize_tensor_gradients

alf.config('Agent', rl_algorithm_cls=TdMpcAlgorithm)


def define_config(name, default_value):
    alf.define_config(name, default_value)
    return alf.get_config_value('_CONFIG._USER.' + name)


# Environment Options
num_envs = define_config('num_envs', 1)
unroll_length = define_config('unroll_length', 1)
num_iterations = define_config('num_iterations', 10_000)
discount = define_config('discount', 0.99)

alf.config("AverageDiscountedReturnMetric", discount=discount)

alf.config(
    "create_environment",
    env_name="Pendulum-v0",
    num_parallel_environments=num_envs)

# Training Options
mini_batch_size = define_config('mini_batch_size', 32)
lr = define_config('lr', 1e-4)
initial_collect_steps = define_config('initial_collect_steps', 200)

# Muzero Model Options
num_unroll_steps = define_config('num_unroll_steps', 8)
alf.config('multi_quantile_huber_loss', delta=0.0)

# Extra Meta Options

value_target = define_config("value_target", "reanalyze")
num_td_steps = define_config("num_td_steps",
                             10 if value_target.startswith("TD") else 0)
reanalyze = value_target == "reanalyze"
reanalyze_horizon = define_config("reanalyze_horizon", 5)
reanalyze_td_steps = define_config("reanalyze_td_steps", 3)
reanalyze_num_iters = define_config('reanalyze_num_iters', 3)
target_update_tau = define_config("target_update_tau", 0.005)
value_loss_weight = define_config("value_loss_weight", 0.005)
reward_loss_weight = define_config("reward_loss_weight", 4.0)

# MPPI Options
mppi_horizon = define_config('mppi_horizon', 25)
mppi_num_trajs = define_config('mppi_num_trajs', 100)
mppi_num_elites = define_config('mppi_num_elites', 10)
mppi_num_iters = define_config('mppi_num_iters', 5)

mppi_temperature = define_config(
    'mppi_temperature',
    LinearScheduler(
        progress_type="percent", schedule=[(0.0, 0.5), (1.0, 0.02)]))

mppi_guided_ratio = define_config('mppi_guided_ratio', 0.05)
mppi_use_value = define_config("mppi_use_value", True)
mppi_use_reward = define_config("mppi_use_reward", True)

# +------------------------------------------------------------+
# | Models                                                     |
# +------------------------------------------------------------+

# inherit_path = define_config(
#     'inherit_path', "/home/breakds/tmp/checkpoints/muzero.pendulum.ckpt-1000")
inherit_path = define_config('inherit_path', "")

alf.config(
    "Trainer.inherit_parameters",
    enable=inherit_path != "",
    path=Path(inherit_path))

policy_loss_weight = define_config('policy_loss_weight', 0.2)
normalize_advantages = define_config('normalize_advantages', False)

alf.config(
    "TruncatedProjectionNetwork",
    state_dependent_scale=True,
    scale_bias_initializer_value=math.log(math.exp(1.0) - 1),
    min_scale=0.05,
    max_scale=10.0,
    dist_ctor=dist_utils.TruncatedNormal)


def create_encoding_net(observation_spec: TensorSpec):
    return EncodingNetwork(
        observation_spec, fc_layer_params=(100, 100), activation=torch.relu_)


def create_prediction_net(latent_spec, action_spec):
    def _summarize_grad(x, name):
        if not x.requires_grad:
            return x
        if alf.summary.should_record_summaries():
            return summarize_tensor_gradients(
                "SimpleMCTSModel/" + name, x, clone=True)
        else:
            return x

    value_head = layers.Sequential(
        partial(_summarize_grad, name='value_grad'),
        # Actual value head
        layers.FC(100, 1, kernel_initializer=torch.nn.init.xavier_uniform_),
        layers.Reshape(()))

    reward_head = layers.Sequential(
        partial(_summarize_grad, name='reward_grad'),
        # Actual reward head
        layers.FC(100, 1, kernel_initializer=torch.nn.init.zeros_),
        layers.Reshape(()))

    if mppi_guided_ratio > 0:
        policy_head = layers.Sequential(
            partial(_summarize_grad, name='policy_grad'),
            BetaProjectionNetwork(
                input_size=100, action_spec=action_spec,
                min_concentration=1.0))
    else:
        policy_head = lambda x: ()

    return alf.nn.Branch(
        value_head,
        reward_head,
        policy_head,  # policy
        lambda x: (),  # game over
        input_tensor_spec=latent_spec)


alf.config(
    "MCTSModel",
    predict_reward_sum=False,
    policy_loss_weight=policy_loss_weight,
    value_loss_weight=value_loss_weight,
    repr_prediction_loss_weight=20.0,
    reward_loss_weight=reward_loss_weight,
    initial_loss_weight=1.0,
    use_pg_loss=False,
    normalize_advantages=normalize_advantages,
    ppo_clipping=0.2)

alf.config(
    "SimpleMCTSModel",
    # NOTE(breakds): dynamics is still the simple dynamics
    encoding_net_ctor=create_encoding_net,
    prediction_net_ctor=create_prediction_net,
    num_sampled_actions=20,
    train_reward_function=True,
    train_game_over_function=False,
    train_repr_prediction=False,
    train_policy=mppi_guided_ratio > 0,
    initial_alpha=0.005)

alf.config(
    "MPPIPlanner",
    horizon=mppi_horizon,
    discount=discount,
    num_elites=mppi_num_elites,
    num_trajs=mppi_num_trajs,
    num_iters=mppi_num_iters,
    momentum=0.1,
    temperature=mppi_temperature,
    policy_guided_ratio=mppi_guided_ratio,
    min_std=
    0.01,  # TODO(breakds): Schedule this to be smaller with more progress
    use_value=mppi_use_value,
    use_reward=mppi_use_reward,
)

reward_transformer = RewardNormalizer(update_mode="rollout")

alf.config(
    "MuzeroRepresentationImpl",
    model_ctor=SimpleMCTSModel,
    num_unroll_steps=num_unroll_steps,
    td_steps=num_td_steps,
    reward_transformer=reward_transformer,
    reanalyze_algorithm_ctor=partial(
        MPPIPlanner, horizon=reanalyze_horizon, num_iters=reanalyze_num_iters),
    reanalyze_td_steps=reanalyze_td_steps,
    recurrent_gradient_scaling_factor=1.0,
    train_reward_function=True,
    train_game_over_function=False,
    train_repr_prediction=False,
    train_policy=mppi_guided_ratio > 0,
    reanalyze_ratio=1.0 if reanalyze else 0.0,
    target_update_period=1,
    target_update_tau=target_update_tau,
    priority_func=
    "lambda loss_info: loss_info.extra['value'].clamp(0.0, 16.0).sqrt().sum(dim=0)"
)

alf.config(
    "TdMpcAlgorithm",
    discount=discount,
    enable_amp=False,
    representation_learner_ctor=MuzeroRepresentationImpl,
    planner_algorithm_ctor=MPPIPlanner,
    reward_transformer=reward_transformer)

# --- END of Model ---

optimizer = Adam(lr=lr)

alf.config("Agent", optimizer=optimizer)

alf.config(
    "TrainerConfig",
    wandb_project="alf.tdmpc_pendulum",
    unroll_length=unroll_length,
    mini_batch_size=mini_batch_size,
    num_updates_per_train_iter=5,
    update_counter_every_mini_batch=False,
    num_iterations=num_iterations,
    num_checkpoints=5,
    evaluate=False,
    enable_amp=False,
    debug_summaries=True,
    summary_interval=10,
    replay_buffer_length=100_000 // num_envs,
    initial_collect_steps=initial_collect_steps,
    summarize_grads_and_vars=True)
