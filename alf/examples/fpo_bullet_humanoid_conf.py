"""
FPO (Flow Policy Optimization) configuration for Bullet Humanoid
This replaces PPO with DiffusionFPOAlgorithm and uses FlowMatchingActorNetwork
"""

import alf
import torch
from functools import partial

from alf.optimizers import AdamTF
from alf.algorithms.agent import Agent
from alf.algorithms.diffusion_fpo_algorithm import DiffusionFPOAlgorithm
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.networks.value_networks import ValueNetwork
from alf.networks.flow_matching_actor_network import FlowMatchingActorNetwork
from alf.networks.flow_matching_trajectory_head import FlowMatchingTrajectoryHead
from alf.utils.losses import element_wise_squared_loss
import alf.nest.utils

# Import pybullet_envs to register HumanoidBulletEnv-v0
import pybullet_envs  # noqa: F401

# Environment configuration
alf.config('create_environment', env_name="HumanoidBulletEnv-v0", num_parallel_environments=96)
alf.config('suite_gym.wrap_env', clip_action=False)

# Observation: [376] (state vector)
# Action: [17] (continuous torques for 17 joints)
# For FPO, we treat single-step actions as 1-step trajectories

# FlowMatchingTrajectoryHead configuration
# Treat single action [17] as 1-step trajectory [1, 17]
head_ctor = partial(
    FlowMatchingTrajectoryHead,
    num_poses=1,  # Single step trajectory
    d_ffn=256,
    d_model=256,
    action_dim=17,  # Humanoid action dimension
    diffusion_steps=20,
    dit_depth=3,
    time_sample_alpha=0.0,
    pretrained_checkpoint=None,  # No pretrained checkpoint for Bullet Humanoid
    enable_trajectory_scaling=False,
)

# FlowMatchingActorNetwork configuration
# Note: encoder_input_dim should match the actual observation dimension after preprocessing
# HumanoidBulletEnv-v0 outputs [376] but may be preprocessed to a different dimension
# If observation is [376], set encoder_input_dim=376; if preprocessed to [44], use encoder_input_dim=44
actor_network_ctor = partial(
    FlowMatchingActorNetwork,
    head_ctor=head_ctor,
    flow_matching_steps=5,
    encoder_input_dim=44,  # Match HumanoidBulletEnv-v0 observation dimension
    encoder_output_dim=256,  # Encoded feature dimension for DiT
    encoder_hidden_dims=(128,),  # Hidden layer dimensions
    encoder_activation=torch.nn.ReLU,
)

# Value network configuration
# Note: Need preprocessing_combiner for nested observation specs
value_network_ctor = partial(
    ValueNetwork,
    preprocessing_combiner=alf.nest.utils.NestConcat(),
    fc_layer_params=(32, 32, 32),
    activation=torch.tanh,
)

# Optimizer configuration
# Note: AdamTF is not configurable via alf.config, so we instantiate it directly
optimizer = AdamTF(
    lr=3e-4,
    gradient_clipping=0.5,
    clip_by_global_norm=True,
)

# PPOAlgorithm configuration (for advantage computation only)
# DiffusionFPOAlgorithm uses PPO internally for GAE advantages when use_ppo_advantages=True
alf.config(
    'PPOLoss',
    entropy_regularization=0,
    normalize_scalar_advantages=True,
    gamma=0.99,
    td_lambda=0.95,
    td_loss_weight=0.5,
    td_error_loss_fn=element_wise_squared_loss,
    advantage_clip=10.0,  # CRITICAL: Clip advantages to prevent explosion
)

# DiffusionFPOAlgorithm configuration
# CRITICAL: Must train value network (td_loss_weight > 0) to prevent value divergence
# Without value network training, advantages explode and training fails
alf.config(
    'DiffusionFPOAlgorithm',
    actor_network_ctor=actor_network_ctor,
    value_network_ctor=value_network_ctor,
    optimizer=optimizer,
    n_mc_samples=8,
    average_losses_before_exp=True,
    clip_epsilon=0.05,
    discretize_t_for_training=True,
    use_ppo_advantages=True,
    reward_mode="advantage",  # Use GAE advantages
    td_loss_weight=0.5,  # CRITICAL: Enable TD loss to train value network (was 0.0)
    fpo_loss_weight=1.0,  # Enable FPO loss
    imitation_loss_weight=0.0,  # Disable imitation loss (FPO only)
    warmup_value_network_iterations=0,
    cfm_loss_mode="eps_mse",
    use_noise_std_for_importance_ratio=True,  # Set to True for PHC-style importance ratio
    noise_std=1.0,  # Noise standard deviation for CFM loss (PHC uses 0.05), the larger gives more stable training
)

# Agent configuration
alf.config('TrainerConfig', algorithm_ctor=Agent)
alf.config('Agent', rl_algorithm_cls=DiffusionFPOAlgorithm)

# Training configuration
alf.config(
    'TrainerConfig',
    num_updates_per_train_iter=2,
    unroll_length=512,
    mini_batch_size=4096,
    mini_batch_length=1,
    num_iterations=1000,
    evaluate=True,
    eval_interval=30,
    async_eval=False,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=10,
)

