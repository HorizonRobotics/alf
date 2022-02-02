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
"""Configuration for training Atari games using MuZero.

The structure of the big model (use_small_net=False, use_bn=True, lstm_reward=True,
sum_over_reward_prediction=False, predict_reward_sum=True, train_repr_prediction=True)
largely follows the model structure of EfficientZero (arXiv:2111.00210)

There are several important algorithmic differences with EfficientZero:

1. Different loss for reward and value.
    EfficientZero uses:
    - rv_loss = DiscreteRegressionLoss(SqrtLinearTransform(0.001), inverse_after_mean=True)
    We use:
    - rv_loss = OrderedDiscreteRegressionLoss(Sqrt1pTransform(), inverse_after_mean=False)
2. different MCTS.
    EfficientZero uses:
    - num_parallel_sims = 1
    - (act/learn/search)_with_exploration_policy = False
    - root_exploration_fraction = 0.25
    We use:
    - num_parallel_sims = 2
    - (act/learn/search)_with_exploration_policy = True
    - root_exploration_fraction = 0
3. different BatchNorm:
    EfficientZero uses torch.nn.BatchNorm.
    We use customized BatchNorm (alf.layers.BatchNorm) which can correctly handle
    state.
3. different optimizer. EfficientZero uses SGD. We use ADAMTF.

There are many other hyper-parameter differences not listed above.
"""

import torch
import torch.nn as nn
from functools import partial
from alf.environments import alf_wrappers

import alf
import alf.examples.atari_conf
import alf.examples.muzero_conf
from alf.utils.schedulers import LinearScheduler, StepScheduler
from alf.algorithms.muzero_algorithm import LinearTdStepFunc
from alf.algorithms.mcts_models import SimpleMCTSModel
from alf.algorithms.mcts_algorithm import MCTSAlgorithm, VisitSoftmaxTemperatureByProgress
from alf.optimizers import SGD, Adam, AdamTF, AdamW
from alf.algorithms.data_transformer import RewardClipping

from alf.algorithms.data_transformer import FrameStacker
from alf.utils.summary_utils import summarize_tensor_gradients
from alf.utils import losses

# MuzeroAlgorithm does not suppport the RewardClipping configured in atari_conf
alf.config('TrainerConfig', data_transformer_ctor=[FrameStacker])


def define_config(name, default_value):
    alf.define_config(name, default_value)
    return alf.get_config_value('_CONFIG._USER.' + name)


train_repr_prediction = define_config('train_repr_prediction', True)
train_game_over_function = define_config('train_game_over_function', False)
num_envs = define_config('num_envs', 2)
unroll_length = define_config('unroll_length', 1)
use_small_net = define_config('use_small_net', False)
discount = define_config('discount', 0.988)
initial_lr = define_config('initial_lr', 2e-4)
norm_type = define_config('norm_type', alf.layers)
optimizer_type = define_config('optimizer_type', 'ADAMTF')
terminal_on_life_loss = define_config('terminal_on_life_loss', True)
use_bn = define_config('use_bn', True)
pred_use_bn = define_config('pred_use_bn', False)
lstm_reward = define_config('lstm_reward', True)
sum_over_reward_prediction = define_config('sum_over_reward_prediction', False)
weight_decay = define_config('weight_decay', 0.1)
rv_weight_decay = define_config('rv_weight_decay', 4e-6)
rv_bias_decay = define_config('rv_bias_decay', 0)
action_weight_decay = define_config('action_weight_decay', None)
action_bias_decay = define_config('action_bias_decay', None)

rv_loss = define_config(
    'rv_loss',
    losses.OrderedDiscreteRegressionLoss(
        transform=alf.math.Sqrt1pTransform(), inverse_after_mean=False))
rv_bias_zero_init = define_config('rv_bias_zero_init', False)

# This option can make the optimization of the parameters of quantile regression
# invariant to the number of quantiles.
scale_grad_by_num_quantiles = define_config('scale_grad_by_num_quantiles',
                                            False)

# alf.math.identity, alf.layers.BatchNorm2d(64), alf.math.normalize_min_max
dyn_state_normalizer = define_config('dyn_state_normalizer', alf.math.identity)

# If 1, not use quantile regression
num_quantiles = define_config('num_quantiles', 256)
alf.config('multi_quantile_huber_loss', delta=0.0)

reward_transformer = RewardClipping() if num_quantiles == 1 else None

alf.config(
    "AverageDiscountedReturnMetric",
    discount=discount,
    reward_transformer=reward_transformer)

if use_small_net:
    alf.config('DMAtariPreprocessing', screen_size=84, gray_scale=True)
else:
    alf.config('DMAtariPreprocessing', screen_size=96, gray_scale=False)

if terminal_on_life_loss:
    alf.config(
        'suite_gym.load',
        alf_env_wrappers=[alf_wrappers.AtariTerminalOnLifeLossWrapper])

alf.config(
    "create_environment",
    env_name="BreakoutNoFrameskip-v4",
    num_parallel_environments=num_envs)

alf.config("alf.batch_norm.BatchNorm1d", affine=True)
alf.config("alf.batch_norm.BatchNorm2d", affine=True)
alf.config(
    "layers.ResidueBlock",
    with_batch_normalization=use_bn,
    bn_ctor=norm_type.BatchNorm2d)
alf.config("layers.Conv2D", use_bn=use_bn, bn_ctor=norm_type.BatchNorm2d)
alf.config("layers.FC", bn_ctor=norm_type.BatchNorm1d)


def create_representation_net(observation_spec):
    in_channels = observation_spec.shape[0]
    return alf.nn.Sequential(
        alf.layers.Scale(1. / 255.),
        alf.layers.Conv2D(
            in_channels, 32, kernel_size=3, strides=2, padding=1),
        alf.layers.ResidueBlock(32, 32, 3, 1),
        alf.layers.ResidueBlock(32, 64, 3, 2),
        alf.layers.ResidueBlock(64, 64, 3, 1),
        nn.AvgPool2d(2),
        alf.layers.ResidueBlock(64, 64, 3, 1),
        nn.AvgPool2d(2),
        alf.layers.ResidueBlock(64, 64, 3, 1),
        alf.math.identity if use_bn else alf.math.normalize_min_max,
        input_tensor_spec=observation_spec,
    )


def create_dynamics_net(input_tensor_spec):
    state_spec, action_spec = input_tensor_spec
    plane_size = state_spec.shape[1:]
    num_planes = state_spec.shape[0]
    num_actions = action_spec.maximum + 1.
    return alf.nn.Sequential(
        lambda x: torch.cat([
            x[0], (x[1] / num_actions).reshape(-1, 1, 1, 1).expand(
                -1, 1, *plane_size)
        ],
                            dim=1),
        a=alf.layers.Conv2D(
            num_planes + 1, 64, 3, padding=1, activation=alf.math.identity),
        b=(('input.0', 'a'), lambda x: (x[0] + x[1]).relu_()),
        c=alf.layers.ResidueBlock(64, 64, 3, 1),
        d=dyn_state_normalizer,
        input_tensor_spec=input_tensor_spec,
    )


def create_representation_net_small(observation_spec):
    in_channels = observation_spec.shape[0]
    return alf.nn.Sequential(
        alf.layers.Scale(1. / 255.),
        alf.layers.Conv2D(
            in_channels,
            32,
            kernel_size=8,
            strides=4,
        ),
        alf.layers.Conv2D(32, 64, kernel_size=4, strides=2),
        alf.layers.Conv2D(64, 64, kernel_size=3, strides=1),
        input_tensor_spec=observation_spec,
    )


def create_dynamics_net_small(input_tensor_spec):
    state_spec, action_spec = input_tensor_spec
    plane_size = state_spec.shape[1:]
    num_planes = state_spec.shape[0]
    num_actions = action_spec.maximum + 1.
    return alf.nn.Sequential(
        lambda x: torch.cat([
            x[0], (x[1] / num_actions).reshape(-1, 1, 1, 1).expand(
                -1, 1, *plane_size)
        ],
                            dim=1),
        a=alf.layers.Conv2D(
            num_planes + 1, 64, 3, padding=1, activation=alf.math.identity),
        b=(('input.0', 'a'), lambda x: (x[0] + x[1]).relu_()),
        c=dyn_state_normalizer,
        input_tensor_spec=input_tensor_spec,
    )


class SumNet(alf.nn.Network):
    def __init__(self, input_tensor_spec):
        super().__init__(input_tensor_spec, input_tensor_spec)

    def forward(self, inputs, state):
        state = inputs + state
        return state, state


def _get_rv_bias_initializer():
    if rv_bias_zero_init:
        return None
    return rv_loss.initialize_bias


@alf.configurable
def create_prediction_net(state_spec, action_spec, initial_game_over_bias=-5):
    dim = 32

    def _summarize_grad(x, name):
        if not x.requires_grad:
            return x
        if alf.summary.should_record_summaries():
            return summarize_tensor_gradients(
                "SimpleMCTSModel/" + name, x, clone=True)
        else:
            return x

    def _make_trunk(lstm=False):
        if lstm:
            if use_small_net:
                return [
                    alf.layers.Reshape(-1),
                    alf.nn.LSTMCell(7 * 7 * 64, 512),
                    alf.layers.FC(
                        512, dim, activation=torch.relu_, use_bn=pred_use_bn)
                ]
            else:
                return [
                    alf.layers.Conv2D(64, 16, 1),
                    alf.layers.Reshape(-1),
                    alf.nn.LSTMCell(6 * 6 * 16, 512),
                    alf.layers.FC(
                        512, dim, activation=torch.relu_, use_bn=pred_use_bn)
                ]
        else:
            if use_small_net:
                return [
                    alf.layers.Reshape(-1),
                    alf.layers.FC(
                        7 * 7 * 64,
                        1024,
                        activation=torch.relu_,
                        use_bn=use_bn),
                    alf.layers.FC(
                        1024, dim, activation=torch.relu_, use_bn=pred_use_bn),
                ]
            else:
                return [
                    alf.layers.ResidueBlock(64, 64, 3, 1),
                    alf.layers.Conv2D(64, 16, 3, padding=1),
                    alf.layers.Reshape(-1),
                    alf.layers.FC(
                        6 * 6 * 16,
                        dim,
                        activation=torch.relu_,
                        use_bn=pred_use_bn),
                ]

    if num_quantiles == 1:
        reshape_layer = [alf.layers.Reshape(())]
        reward_spec = alf.TensorSpec(())
    else:
        reshape_layer = []
        reward_spec = alf.TensorSpec((num_quantiles, ))

    def _scale_grad(scale):
        if scale == 1.0:
            return []
        else:
            return [lambda x: torch.lerp(x.detach(), x, scale)]

    value_net = alf.layers.Sequential(
        partial(_summarize_grad, name='value_grad'),
        *_make_trunk(),
        *_scale_grad(1 / num_quantiles if scale_grad_by_num_quantiles else 1),
        alf.layers.FC(
            dim,
            num_quantiles,
            weight_opt_args=dict(weight_decay=rv_weight_decay),
            bias_opt_args=dict(weight_decay=rv_bias_decay),
            bias_initializer=_get_rv_bias_initializer(),
            kernel_initializer=torch.nn.init.zeros_),
        *_scale_grad(num_quantiles if scale_grad_by_num_quantiles else 1),
        *reshape_layer,
    )

    reward_net = [
        partial(_summarize_grad, name='reward_grad'),
        *_make_trunk(lstm_reward),
        *_scale_grad(1 / num_quantiles if scale_grad_by_num_quantiles else 1),
        alf.layers.FC(
            dim,
            num_quantiles,
            weight_opt_args=dict(weight_decay=rv_weight_decay),
            bias_opt_args=dict(weight_decay=rv_bias_decay),
            bias_initializer=_get_rv_bias_initializer(),
            kernel_initializer=torch.nn.init.zeros_),
        *_scale_grad(num_quantiles if scale_grad_by_num_quantiles else 1),
    ] + reshape_layer
    if sum_over_reward_prediction:
        reward_net.append(SumNet(reward_spec))
    reward_net = alf.nn.Sequential(*reward_net, input_tensor_spec=state_spec)

    action_net = alf.layers.Sequential(
        partial(_summarize_grad, name='policy_grad'),
        *_make_trunk(),
        alf.nn.CategoricalProjectionNetwork(
            dim,
            action_spec,
            logits_init_output_factor=0.0,
            weight_opt_args=dict(weight_decay=action_weight_decay),
            bias_opt_args=dict(weight_decay=action_bias_decay)),
    )
    game_over_net = alf.layers.Sequential(
        *_make_trunk(),
        alf.layers.FC(
            dim,
            1,
            kernel_initializer=torch.nn.init.zeros_,
            bias_init_value=initial_game_over_bias),
        alf.layers.Reshape(()),
    ) if train_game_over_function else lambda x: ()

    return alf.nn.Branch(
        value_net,
        reward_net,
        action_net,
        game_over_net,
        input_tensor_spec=state_spec,
    )


@alf.configurable
def repr_projection_net_ctor(input_tensor_spec,
                             hidden_size=512,
                             output_size=1024,
                             last_use_bn=False):
    return alf.nn.Sequential(
        alf.layers.Reshape(-1),
        alf.layers.FC(
            input_tensor_spec.numel,
            hidden_size,
            activation=torch.relu_,
            use_bn=True),
        alf.layers.FC(
            hidden_size, hidden_size, activation=torch.relu_, use_bn=True),
        alf.layers.FC(hidden_size, output_size, use_bn=last_use_bn),
        input_tensor_spec=input_tensor_spec)


@alf.configurable
def repr_prediction_net_ctor(input_tensor_spec,
                             hidden_size=512,
                             output_size=1024):
    return alf.nn.Sequential(
        alf.layers.FC(
            input_tensor_spec.numel,
            hidden_size,
            activation=torch.relu_,
            use_bn=True), alf.layers.FC(hidden_size, output_size))


if use_small_net:
    encoding_net_ctor = create_representation_net_small
    dynamics_net_ctor = create_dynamics_net_small
else:
    encoding_net_ctor = create_representation_net
    dynamics_net_ctor = create_dynamics_net

AdamOptimizers = {'ADAM': Adam, 'ADAMTF': AdamTF, 'ADAMW': AdamW}
lr_schedule = StepScheduler("percent", [(0.45, initial_lr),
                                        (0.9, 0.1 * initial_lr),
                                        (1.0, 0.01 * initial_lr)])

alf.config(
    "MCTSModel",
    value_loss=rv_loss,
    reward_loss=rv_loss,
    predict_reward_sum=True,
    policy_loss_weight=1.0,
    value_loss_weight=0.05,
    repr_prediction_loss_weight=40.0,
    reward_loss_weight=2.0)

alf.config(
    "SimpleMCTSModel",
    encoding_net_ctor=encoding_net_ctor,
    dynamics_net_ctor=dynamics_net_ctor,
    prediction_net_ctor=create_prediction_net,
    train_repr_prediction=train_repr_prediction,
    train_game_over_function=train_game_over_function,
    repr_projection_net_ctor=repr_projection_net_ctor,
    repr_prediction_net_ctor=repr_prediction_net_ctor,
    initial_alpha=0.)

alf.config(
    "MCTSAlgorithm",
    discount=discount,
    num_simulations=52,
    num_parallel_sims=2,
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.,
    pb_c_init=0.5,
    pb_c_base=19652,
    is_two_player_game=False,
    visit_softmax_temperature_fn=VisitSoftmaxTemperatureByProgress(
        [(0.5, 1.0), (0.75, 0.5), (1, 0.25)]),
    act_with_exploration_policy=True,
    learn_with_exploration_policy=True,
    search_with_exploration_policy=True,
    unexpanded_value_score='mean',
    expand_all_children=False,
    expand_all_root_children=False,
    max_unroll_length=5,
    learn_policy_temperature=1.0)

alf.config(
    "MuzeroAlgorithm",
    enable_amp=True,
    mcts_algorithm_ctor=MCTSAlgorithm,
    model_ctor=SimpleMCTSModel,
    # The following line can be commented out if GPU memory is large enough
    reanalyze_batch_size=1280 if use_small_net else 640,
    num_unroll_steps=5,
    td_steps=10,
    reward_transformer=reward_transformer,
    reanalyze_td_steps_func=  #LinearMaxAgeTdStepFunc(),
    LinearTdStepFunc(max_bootstrap_age=1.2, min_td_steps=1),
    train_repr_prediction=train_repr_prediction,
    train_game_over_function=train_game_over_function,
    reanalyze_ratio=1.0,
    target_update_period=400,
    target_update_tau=1.0)

opt_kwargs = dict(
    lr=lr_schedule,
    weight_decay=weight_decay,
    gradient_clipping=1e9,
    clip_by_global_norm=True)

if optimizer_type in AdamOptimizers:
    optimizer = AdamOptimizers[optimizer_type](
        betas=(0.9, 0.999), eps=1e-7, **opt_kwargs)
else:
    optimizer = SGD(momentum=0.9, **opt_kwargs)

alf.config("Agent", optimizer=optimizer)

# training config
alf.config(
    "TrainerConfig",
    random_seed=2,
    unroll_length=unroll_length,
    mini_batch_size=256,
    num_updates_per_train_iter=5,
    update_counter_every_mini_batch=False,
    priority_replay=True,
    priority_replay_alpha=1.2,
    priority_replay_beta=LinearScheduler("percent", [(0.0, 0.4), (1.0, 1.0)]),
    num_iterations=0,
    num_env_steps=100000,
    num_checkpoints=1,
    evaluate=False,
    enable_amp=False,
    debug_summaries=True,
    summary_interval=int(100 // (unroll_length * num_envs)),
    replay_buffer_length=100000 // num_envs,
    initial_collect_steps=2000,
    summarize_grads_and_vars=True)
