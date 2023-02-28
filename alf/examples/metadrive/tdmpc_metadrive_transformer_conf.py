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

from pathlib import Path
from alf.optimizers import AdamTF
from alf.algorithms.muzero_representation_learner import LinearTdStepFunc, MuzeroRepresentationImpl
from alf.algorithms.td_mpc_algorithm import TdMpcAlgorithm
from alf.algorithms.mcts_models import SimpleMCTSModel
from alf.algorithms.planning_algorithm import MPPIPlanner
from alf.utils.schedulers import StepScheduler
from alf.networks.projection_networks import MixtureProjectionNetwork, StableNormalProjectionNetwork, BetaProjectionNetwork
from functools import partial
from alf.utils.summary_utils import summarize_tensor_gradients
from typing import NamedTuple, Optional, Tuple

import torch

import alf
from alf import layers
from alf.utils import losses
from alf.examples.metadrive import base_conf
from alf.examples import muzero_conf

alf.config('Agent', rl_algorithm_cls=TdMpcAlgorithm)


def define_config(name, default_value):
    alf.define_config(name, default_value)
    return alf.get_config_value('_CONFIG._USER.' + name)


# Environment Options
num_envs = define_config('num_envs', 32)
unroll_length = define_config('unroll_length', 10)
num_env_steps = define_config('num_env_steps', 8_000_000)
discount = define_config('discount', 0.995)

alf.config(
    'create_environment',
    env_name='Vectorized',
    num_parallel_environments=num_envs)

alf.config(
    'metadrive.sensors.VectorizedObservation',
    segment_resolution=2.0,
    polyline_size=4,
    polyline_limit=128,
    history_window_size=24)

alf.config("AverageDiscountedReturnMetric", discount=discount)

# High-level Options
policy_type = define_config("policy_type", "sequence")
assert policy_type in ["single_step", "sequence", "decision"]
policy_mixture = define_config("policy_mixture", 3)

# Training Options
mini_batch_size = define_config('mini_batch_size', 160)
initial_lr = define_config('initial_lr', 1e-3)
weight_decay = define_config('weight_decay', 0.0)

# Transformer Encoder Options
model_size = define_config("model_size", "big")
d_model = define_config('d_model', {
    "big": 128,
    "mid": 64,
    "small": 48,
}[model_size])
num_heads = define_config('num_heads', {
    "big": 8,
    "mid": 4,
    "small": 3,
}[model_size])
ending_embedding_dim = define_config("ending_embedding_dim", 1)
latent_layer_norm = define_config("latent_layer_norm", True)

# Muzero Model Options
rv_loss = define_config(
    'rv_loss',
    losses.OrderedDiscreteRegressionLoss(
        transform=alf.math.Sqrt1pTransform(), inverse_after_mean=False))
rv_bias_zero_init = define_config('rv_bias_zero_init', False)
alf.config('multi_quantile_huber_loss', delta=0.0)
rv_weight_decay = define_config('rv_weight_decay', 4e-6)
rv_bias_decay = define_config('rv_bias_decay', 0)
num_quantiles = define_config('num_quantiles', 32)
initial_collect_steps = define_config('initial_collect_steps', 5000)
num_unroll_steps = define_config('num_unroll_steps', 8)

# MPPI Options
mppi_horizon = define_config('mppi_horizon', 8)
mppi_num_trajs = define_config('mppi_num_trajs', 100)
mppi_num_elites = define_config('mppi_num_elites', 10)
mppi_num_iters = define_config('mppi_num_iters', 5)
mppi_temperature = define_config('mppi_temperature', 0.02)
mppi_guided_ratio = define_config('mppi_guided_ratio', 0.2)
mppi_use_elite_as_policy_target = define_config(
    "mppi_use_elite_as_policy_target", True)
mppi_weighted_policy_target = define_config("mppi_weighted_policy_target",
                                            False)
non_commit_ratio = define_config("non_commit_ratio", 0.25)
rollout_commit_prob = (1.0 - non_commit_ratio) / (non_commit_ratio *
                                                  (mppi_horizon - 1))

# Reanalyze Options
reanalyze_horizon = define_config(
    "reanalyze_horizon", mppi_horizon if policy_type == "sequence" else 5)
reanalyze_td_steps = define_config("reanalyze_td_steps", 5)
reanalyze_num_iters = define_config('reanalyze_num_iters', 3)
reanalyze_num_trajs = define_config('reanalyze_num_trajs', 50)
reanalyze_num_elites = define_config('reanalyze_num_elites', 5)

# Loss Options
value_loss_weight = define_config("value_loss_weight", 0.5)
reward_loss_weight = define_config("reward_loss_weight", 2.0)
policy_loss_weight = define_config('policy_loss_weight', 0.2)
game_over_loss_weight = define_config('game_over_loss_weight', 20.0)
emphasize_beyond_episode_end = define_config("emphasize_beyond_episode_end",
                                             0.0)

# +----------------------------------------+
# | Experiment Specific Configurations     |
# +----------------------------------------+

detach_policy = define_config('detach_policy', False)
initial_alpha = define_config('initial_alpha', 0.0)

# Freeze or not?
#
# When `freeze_network` is set to True, everything except for the
# policy head and value head.
freeze_network = define_config('freeze_network', False)
# Here if ``inherit_path`` is set to "", there is not inheriting at
# all, and ``inherit_policy`` will be ignored as well.
inherit_path = define_config('inherit_path', "")
inherit_policy = define_config('inherit_policy', False)

# +----------------------------------------+
# | Transformer Encoding Cofngiuration     |
# +----------------------------------------+


class EmbeddingConfig(NamedTuple):
    """Specifies the embedding FCs for an input before it is fed to the transformer.
    """
    d_input: int = 1  # The dimension of the input
    hidden: Tuple = ()  # The size of the hidden for each FC layer

    def make_embedding_net(self, d_output: int):
        assert all([isinstance(h, int) and h > 0 for h in self.hidden])

        stack = []
        d_input = self.d_input
        for h in self.hidden:
            stack.append(layers.FC(d_input, h, activation=torch.relu_))
            d_input = h
        stack.append(layers.FC(d_input, d_output, activation=torch.relu_))

        return torch.nn.Sequential(*stack)


class ObservationCombiner(torch.nn.Module):
    """This module combines the vectorized map, agents and ego inputs so as to
    prepare them to feed the transformer.
    The input should be a tuple of 3, (map_sequence, ego, agents), where
    1. ``map_sequence`` is of shape [B, L, d_map_feature], where B is the batch
       size, L is the maximum number of polylines, and d_map_feature is the size
       of the polyline feature.
    2. ``ego`` is a single vector encodes the ego car trajectory and is of shape
       [B, d_ego_feature].
    3. ``agents`` is of shape [B, A, d_agent_feature], where A is the maximum
       number of agents being encoded, and d_agent_feature is the size of the
       per-agent feature.
    All the 3 inputs above will go through a fully connected layer respectively
    and generate a set of L vectors (map), a set of a single vector(ego), and a
    set of A vectors (agents) respectively, where all vectors will have the
    shape [d_model,]. The 3 sets will be combined (concatenated).
    """

    def __init__(self,
                 map_feature: EmbeddingConfig,
                 ego_feature: EmbeddingConfig,
                 agent_feature: EmbeddingConfig,
                 respect_ending: bool = False):
        super().__init__()

        self._ending_embedder = None
        if respect_ending:
            self._ending_embedder = torch.nn.Embedding(3, d_model)
        self._map_eb = map_feature.make_embedding_net(d_model)
        self._agent_eb = agent_feature.make_embedding_net(d_model)
        self._ego_eb = ego_feature.make_embedding_net(d_model)

    def forward(self, inputs):
        map_sequence, ego, agents, ending = inputs

        x0 = self._ego_eb(ego)  # [B, d_model]

        if self._ending_embedder is not None:
            x0 = x0 + self._ending_embedder(ending)

        x0 = x0.unsqueeze(1)  # [B, 1, d_model]

        # The input ``sequence`` is [B, L, d_map_feature]
        x1 = self._map_eb(map_sequence)  # [B, L, d_model]

        x2 = agents.view(*agents.shape[:2], -1)
        x2 = self._agent_eb(x2)  # [B, A, d_model]

        return torch.cat([x0, x1, x2], dim=1)


class MaskedTransformer(torch.nn.Module):
    def __init__(self, num_layers, num_other_entities):
        super().__init__()
        self._memory_size = 1 + num_other_entities

        tf_layers = []
        for i in range(num_layers):
            tf_layers.append(
                layers.TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    memory_size=self._memory_size,
                    positional_encoding='none'))
        self._tf_layers = torch.nn.ModuleList(tf_layers)

    def forward(self, inputs):
        x, map_mask, agent_mask = inputs
        B = x.shape[0]
        mask = torch.hstack((torch.zeros(B, 1, dtype=bool), map_mask,
                             agent_mask))
        for layer in self._tf_layers:
            x = layer(memory=x, mask=mask)
        return x


def create_representation_net(observation_spec):
    num_other_entities = observation_spec['map'].shape[0] + observation_spec[
        'agents'].shape[0]

    # yapf: disable
    stack = [
        alf.nn.Branch(
            alf.nn.Sequential(
                lambda x: (x['map'],
                           x['ego'],
                           x['agents'], x['ending']),
                ObservationCombiner(
                    map_feature=EmbeddingConfig(
                        d_input=observation_spec['map'].shape[-1]),
                    ego_feature=EmbeddingConfig(
                        d_input=observation_spec['ego'].shape[-1],
                        hidden=(32, )),
                    agent_feature=EmbeddingConfig(
                        d_input=observation_spec['agents'].shape[1] *
                        observation_spec['agents'].shape[2],
                        hidden=(32, )),
                    respect_ending=ending_embedding_dim > 0),
                input_tensor_spec=observation_spec),
            lambda x: (~x['map_mask'], ~x['agent_mask'])),
        lambda x: (x[0], x[1][0], x[1][1]),
        MaskedTransformer(num_layers=3, num_other_entities=num_other_entities),
        # Take the corresponding transformer output of the first vector in the
        # sequence (corresponding to "ego") as the final output of the encoder.
        lambda x: x[:, 0, :],
        *([torch.nn.LayerNorm(d_model)] if latent_layer_norm else []),
        *([lambda x: x.detach()] if freeze_network else [])
    ]
    # yapf: enable

    return alf.nn.Sequential(*stack, input_tensor_spec=observation_spec)


# +-------------------------------------------------------+
# | MuZero Representation Specific Configuration          |
# +-------------------------------------------------------+


def create_ego_centric_dynamics(input_tensor_spec):
    _, action_spec = input_tensor_spec

    action_embedding = EmbeddingConfig(
        d_input=action_spec.shape[-1], hidden=(128, ))
    transition = EmbeddingConfig(d_input=d_model * 2, hidden=(512, ))

    return alf.nn.Sequential(
        layers.Branch(
            lambda x: x[0],  # Pick representation (CLS token)
            layers.Sequential(
                lambda x: x[1],  # Pick action
                action_embedding.make_embedding_net(d_model))),
        lambda x: torch.cat(x, dim=1),
        transition.make_embedding_net(d_model),
        *([torch.nn.LayerNorm(d_model)] if latent_layer_norm else []),
        *([lambda x: x.detach()] if freeze_network else []),
        input_tensor_spec=input_tensor_spec)


def _get_rv_bias_initializer():
    if rv_bias_zero_init:
        return None
    return rv_loss.initialize_bias


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

    def _make_trunk():
        return [
            layers.Reshape(-1),
            layers.FC(d_model, 1024, activation=torch.relu_),
            layers.FC(1024, dim, activation=torch.relu_),
        ]

    if num_quantiles == 1:
        reshape_layer = [layers.Reshape(())]
        reward_spec = alf.TensorSpec(())
    else:
        reshape_layer = []
        reward_spec = alf.TensorSpec((num_quantiles, ))

    value_net = layers.Sequential(
        partial(_summarize_grad, name='value_grad'),
        *_make_trunk(),
        # The parameters of the last FC is initialized such that the initial
        # expectation from the discrete distribution is close to 0. But fp16 is
        # not accurate enough to make it close to 0. So we explicitly disable
        # AMP.
        layers.AMPWrapper(
            False,
            layers.FC(
                dim,
                num_quantiles,
                weight_opt_args=dict(weight_decay=rv_weight_decay),
                bias_opt_args=dict(weight_decay=rv_bias_decay),
                bias_initializer=_get_rv_bias_initializer(),
                kernel_initializer=torch.nn.init.zeros_)),
        *reshape_layer,
    )

    reward_net = [
        *([] if freeze_network else
          [partial(_summarize_grad, name='reward_grad')]),
        # Since we are not predicting reward sum, no need to use LSTM here.
        *_make_trunk(),
        # The parameters of the last FC is initialized such that the initial
        # expectation from the discrete distribution is close to 0. But fp16 is
        # not accurate enough to make it close to 0. So we explicitly disable
        # AMP.
        layers.AMPWrapper(
            False,
            layers.FC(
                dim,
                num_quantiles,
                weight_opt_args=dict(weight_decay=rv_weight_decay),
                bias_opt_args=dict(weight_decay=rv_bias_decay),
                bias_initializer=_get_rv_bias_initializer(),
                kernel_initializer=torch.nn.init.zeros_)),
        *reshape_layer,
        *([lambda x: x.detach()] if freeze_network else [])
    ]

    reward_net = alf.nn.Sequential(*reward_net, input_tensor_spec=state_spec)

    if mppi_guided_ratio > 0:
        policy_action_spec = action_spec
        if policy_type == "sequence":
            policy_action_spec = action_spec.replace(
                shape=(action_spec.shape[0] * mppi_horizon,
                       *action_spec.shape[1:]))
        component_ctor = partial(BetaProjectionNetwork, min_concentration=1.0)
        proj_net_ctor = component_ctor if policy_mixture == 1 else partial(
            MixtureProjectionNetwork,
            num_components=policy_mixture,
            component_ctor=component_ctor)
        proj_net = proj_net_ctor(dim, policy_action_spec)

        action_net = layers.Sequential(
            *([lambda x: x.detach()] if detach_policy
              else [partial(_summarize_grad, name='policy_grad')]),
            *_make_trunk(), proj_net)
    else:
        action_net = lambda x: ()

    game_over_net = layers.Sequential(
        *([] if freeze_network else
          [partial(_summarize_grad, name='game_over_grad')]), *_make_trunk(),
        layers.FC(
            dim,
            1,
            kernel_initializer=torch.nn.init.zeros_,
            bias_init_value=initial_game_over_bias), layers.Reshape(()),
        *([lambda x: x.detach()] if freeze_network else []))

    return alf.nn.Branch(
        value_net,
        reward_net,
        action_net,
        game_over_net,
        input_tensor_spec=state_spec,
    )


alf.config(
    "Trainer.inherit_parameters",
    enable=inherit_path != "",
    path=Path(inherit_path),
    ignored_parameter_prefixes=[
        "_rl_algorithm._repr_learner._model._prediction_net._nets.2",
        "_rl_algorithm._repr_learner._target_model._prediction_net._nets.2"
    ] if not inherit_policy else [])

lr_schedule = StepScheduler("percent", [(0.6, initial_lr),
                                        (0.8, 0.25 * initial_lr),
                                        (1.0, 0.05 * initial_lr)])

alf.config(
    "MCTSModel",
    value_loss=rv_loss,
    reward_loss=rv_loss,
    repr_loss=losses.AsymmetricSimSiamLoss(
        input_size=d_model,
        proj_hidden_size=512,
        pred_hidden_size=512,
        output_size=1024),
    # NOTE(breakds): When predict reward sum is turned on, the learned model
    # does not do well in replan-play, but works well in commit-play. Disabling
    # makes replan-play great again.
    predict_reward_sum=False,
    policy_loss_weight=policy_loss_weight,
    value_loss_weight=value_loss_weight,
    repr_prediction_loss_weight=20.0,
    reward_loss_weight=reward_loss_weight,
    game_over_loss_weight=game_over_loss_weight,
    initial_loss_weight=1.0,
    emphasize_beyond_episode_end=emphasize_beyond_episode_end,
    use_pg_loss=False,
    ppo_clipping=0.2)

alf.config(
    "SimpleMCTSModel",
    num_sampled_actions=16,
    encoding_net_ctor=create_representation_net,
    dynamics_net_ctor=create_ego_centric_dynamics,
    prediction_net_ctor=partial(create_prediction_net),
    train_repr_prediction=True,
    train_game_over_function=True,
    train_policy=mppi_guided_ratio > 0,
    initial_alpha=initial_alpha)

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
    use_elite_as_policy_target=mppi_use_elite_as_policy_target,
    weighted_policy_target=mppi_weighted_policy_target,
    min_std=0.1,  # TODO(breakds): Schedule this to be smaller with more progress
    rollout_commit_prob=rollout_commit_prob,
)

alf.config(
    "MuzeroRepresentationImpl",
    model_ctor=SimpleMCTSModel,
    num_unroll_steps=num_unroll_steps,
    td_steps=10,  # Not used as reanalyze ratio is 1.0
    reanalyze_algorithm_ctor=partial(
        MPPIPlanner,
        horizon=reanalyze_horizon,
        num_iters=reanalyze_num_iters,
        num_elites=reanalyze_num_elites,
        num_trajs=reanalyze_num_trajs),
    reanalyze_td_steps=reanalyze_td_steps,
    reanalyze_td_steps_func=  #LinearMaxAgeTdStepFunc(),
    LinearTdStepFunc(max_bootstrap_age=0.8, min_td_steps=1),
    reanalyze_batch_size=None,
    train_repr_prediction=True,
    train_game_over_function=True,
    random_action_after_episode_end=True,
    train_policy=mppi_guided_ratio > 0,
    reanalyze_ratio=1.0,
    target_update_period=10,
    target_update_tau=0.5)

alf.config(
    "TdMpcAlgorithm",
    discount=discount,
    enable_amp=True,
    representation_learner_ctor=MuzeroRepresentationImpl,
    planner_algorithm_ctor=MPPIPlanner)

opt_kwargs = dict(
    lr=lr_schedule,
    weight_decay=weight_decay,
    gradient_clipping=1e9,
    clip_by_global_norm=True)

optimizer = AdamTF(betas=(0.9, 0.999), eps=1e-7, **opt_kwargs)

alf.config("Agent", optimizer=optimizer)

alf.config(
    "TrainerConfig",
    wandb_project="alf.tdmpc_metadrive",
    unroll_length=unroll_length,
    mini_batch_size=mini_batch_size,
    num_updates_per_train_iter=3,
    update_counter_every_mini_batch=False,
    priority_replay=True,
    priority_replay_alpha=1.2,
    priority_replay_beta=0,
    num_iterations=0,
    num_env_steps=num_env_steps,
    num_checkpoints=10,
    evaluate=False,
    enable_amp=False,
    debug_summaries=True,
    summary_interval=int(10000 // (unroll_length * num_envs)),
    replay_buffer_length=1_000_000 // num_envs,
    initial_collect_steps=initial_collect_steps,
    summarize_grads_and_vars=True)
