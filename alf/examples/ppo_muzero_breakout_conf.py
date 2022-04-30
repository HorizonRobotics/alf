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
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.data_transformer import FrameStacker, UntransformedTimeStep
from alf.optimizers import AdamTF
from alf.algorithms.muzero_representation_learner import LinearTdStepFunc, MuzeroRepresentationImpl, MuzeroRepresentationLearner, MuzeroRepresentationTrainingOptions
from alf.algorithms.mcts_models import SimpleMCTSModel
from alf.utils.summary_utils import summarize_tensor_gradients
from alf.networks.network import Network
from alf.tensor_specs import TensorSpec
import functools

import torch

import alf
from alf import layers
from alf.algorithms.agent import Agent
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.utils.losses import AsymmetricSimSiamLoss, OrderedDiscreteRegressionLoss

from alf.examples import atari_conf, ppo_conf, muzero_repr_conf

discount = 0.999
num_envs = 64
num_quantiles = 256

LATENT_SIZE = 64 * 7 * 7

alf.config("AverageDiscountedReturnMetric", discount=discount)

alf.config(
    'TrainerConfig',
    data_transformer_ctor=[UntransformedTimeStep, FrameStacker])

# From OpenAI gym wiki:
# "v0 vs v4: v0 has repeat_action_probability of 0.25
#  (meaning 25% of the time the previous action will be used instead of the new action),
#   while v4 has 0 (always follow your issued action)
# Because we already implements frame_skip in AtariPreprocessing, we should always
# use 'NoFrameSkip' Atari environments from OpenAI gym
alf.config(
    'create_environment',
    env_name='BreakoutNoFrameskip-v4',
    num_parallel_environments=num_envs)

# +------------------------------------------------------------+
# | MuZero Representation Configurations                       |
# +------------------------------------------------------------+

rv_loss = OrderedDiscreteRegressionLoss(
    transform=alf.math.Sqrt1pTransform(), inverse_after_mean=False)


def create_representation_net_small(observation_spec):
    in_channels = observation_spec.shape[0]
    return alf.nn.Sequential(
        layers.Scale(1. / 255.),
        layers.Conv2D(
            in_channels,
            32,
            kernel_size=8,
            strides=4,
        ),
        layers.Conv2D(32, 64, kernel_size=4, strides=2),
        layers.Conv2D(64, 64, kernel_size=3, strides=1),
        torch.nn.GroupNorm(1, 64),
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
        a=layers.Conv2D(
            num_planes + 1, 64, 3, padding=1, activation=alf.math.identity),
        b=(('input.0', 'a'), lambda x: (x[0] + x[1]).relu_()),
        c=torch.nn.GroupNorm(1, 64),
        input_tensor_spec=input_tensor_spec,
    )


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
            return [
                layers.Reshape(-1),
                alf.nn.LSTMCell(LATENT_SIZE, 512),
                layers.FC(512, dim, activation=torch.relu_, use_bn=False)
            ]
        else:
            return [
                layers.Reshape(-1),
                layers.FC(
                    LATENT_SIZE, 1024, activation=torch.relu_, use_bn=False),
                layers.FC(1024, dim, activation=torch.relu_, use_bn=False),
            ]

    value_net = layers.Sequential(
        functools.partial(_summarize_grad, name='value_grad'),
        *_make_trunk(),
        layers.FC(
            dim,
            num_quantiles,
            bias_initializer=rv_loss.initialize_bias,
            kernel_initializer=torch.nn.init.zeros_),
    )

    reward_net = [
        functools.partial(_summarize_grad, name='reward_grad'),
        *_make_trunk(lstm=True),
        layers.FC(
            dim,
            num_quantiles,
            bias_initializer=rv_loss.initialize_bias,
            kernel_initializer=torch.nn.init.zeros_),
    ]
    reward_net = alf.nn.Sequential(*reward_net, input_tensor_spec=state_spec)

    # NOTE: The action net is not used when train_policy is set to False
    action_net = layers.Sequential(
        functools.partial(_summarize_grad, name='policy_grad'), *_make_trunk(),
        alf.nn.CategoricalProjectionNetwork(
            dim, action_spec, logits_init_output_factor=0.0))

    game_over_net = layers.Sequential(
        *_make_trunk(),
        layers.FC(
            dim,
            1,
            kernel_initializer=torch.nn.init.zeros_,
            bias_init_value=initial_game_over_bias), layers.Reshape(()))

    return alf.nn.Branch(
        value_net,
        reward_net,
        action_net,
        game_over_net,
        input_tensor_spec=state_spec,
    )


alf.config(
    "MCTSModel",
    value_loss=rv_loss,
    reward_loss=rv_loss,
    repr_loss=AsymmetricSimSiamLoss(
        input_size=LATENT_SIZE,
        proj_hidden_size=512,
        pred_hidden_size=512,
        output_size=1024))

alf.config(
    "SimpleMCTSModel",
    encoding_net_ctor=create_representation_net_small,
    dynamics_net_ctor=create_dynamics_net_small,
    prediction_net_ctor=create_prediction_net)

alf.config(
    "MuzeroRepresentationImpl",
    reanalyze_batch_size=1280,
    num_unroll_steps=5,
    td_steps=10,
    discount=discount,
    enable_amp=True,
    reanalyze_algorithm_ctor=PPOAlgorithm,
    reanalyze_ratio=1.0,
    optimizer=AdamTF(lr=1e-4, betas=(0.9, 0.999), eps=1e-7))

alf.config(
    "MuzeroRepresentationLearner",
    training_options=MuzeroRepresentationTrainingOptions(
        interval=1,
        mini_batch_length=1,
        mini_batch_size=256,
        num_updates_per_train_iter=10,
        replay_buffer_length=100000 // num_envs,
        initial_collect_steps=2000,
        priority_replay=True,
        priority_replay_alpha=1.2,
        priority_replay_beta=0.0))

# +------------------------------------------------------------+
# | Policy Algorithm Configurations: PPO                       |
# +------------------------------------------------------------+


def memoized(func):
    memoized_result = None

    @functools.wraps(func)
    def _func(*args, **kwargs):
        nonlocal memoized_result
        if memoized_result is None:
            memoized_result = func(*args, **kwargs)
        return memoized_result

    return _func


@memoized
def create_actor_network(input_tensor_spec, action_spec):
    return alf.nn.Sequential(
        layers.Reshape((-1, )),
        ActorDistributionNetwork(
            input_tensor_spec=TensorSpec(
                shape=(LATENT_SIZE, ), dtype=torch.float32),
            action_spec=action_spec,
            fc_layer_params=(128, )),
        input_tensor_spec=input_tensor_spec)


@memoized
def create_value_network(input_tensor_spec):
    return alf.nn.Sequential(
        layers.Reshape((-1, )),
        ValueNetwork(
            input_tensor_spec=TensorSpec(
                shape=(LATENT_SIZE, ), dtype=torch.float32),
            fc_layer_params=(128, )),
        input_tensor_spec=input_tensor_spec)


alf.config('CategoricalProjectionNetwork', logits_init_output_factor=1e-10)

alf.config(
    'PPOLoss',
    entropy_regularization=1e-2,
    gamma=discount,
    td_loss_weight=0.1,
    normalize_advantages=False)

alf.config(
    'ActorCriticAlgorithm',
    actor_network_ctor=create_actor_network,
    value_network_ctor=create_value_network)

alf.config(
    'Agent',
    representation_learner_cls=MuzeroRepresentationLearner,
    optimizer=AdamTF(lr=1e-3))

alf.config(
    'TrainerConfig',
    unroll_length=8,
    mini_batch_size=64,
    mini_batch_length=None,
    enable_amp=True,
    num_updates_per_train_iter=3,
    algorithm_ctor=Agent,
    num_iterations=0,
    num_env_steps=5_000_000,
    evaluate=False,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=50)
