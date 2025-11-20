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
from alf.networks.flow_matching_mlp_actor_network import FlowMatchingMLPActorNetwork, MLPTrajectoryHead
from alf.utils.losses import element_wise_squared_loss
import alf.nest.utils

# Import pybullet_envs to register HumanoidBulletEnv-v0
import pybullet_envs  # noqa: F401

# Environment configuration
NUM_PARALLEL_ENVIRONMENTS = 640
alf.config('create_environment', env_name="HumanoidBulletEnv-v0", num_parallel_environments=NUM_PARALLEL_ENVIRONMENTS)
alf.config('suite_gym.wrap_env', clip_action=False)

# Training configuration constants (defined early for use in network configs)
UNROLL_LENGTH = 512  # Number of time steps each environment proceeds per iteration

# Calculate max batch size for dropout mask pre-allocation
# max_batch_size = num_parallel_environments * unroll_length
MAX_BATCH_SIZE = NUM_PARALLEL_ENVIRONMENTS * UNROLL_LENGTH

# Observation: [44] (preprocessed state vector)
# Action: [17] (continuous torques for 17 joints)
# For FPO, we treat single-step actions as 1-step trajectories

# ============================================================================
# Previous FlowMatchingActorNetwork configuration (DiT-based, commented out)
# ============================================================================
# # FlowMatchingTrajectoryHead configuration
# # Treat single action [17] as 1-step trajectory [1, 17]
# head_ctor_old = partial(
#     FlowMatchingTrajectoryHead,
#     num_poses=1,  # Single step trajectory
#     d_ffn=256,
#     d_model=256,
#     action_dim=17,  # Humanoid action dimension
#     diffusion_steps=20,
#     dit_depth=3,
#     time_sample_alpha=0.0,
#     pretrained_checkpoint=None,  # No pretrained checkpoint for Bullet Humanoid
#     enable_trajectory_scaling=False,
# )
# 
# # FlowMatchingActorNetwork configuration
# # Note: encoder_input_dim should match the actual observation dimension after preprocessing
# # HumanoidBulletEnv-v0 outputs [376] but may be preprocessed to a different dimension
# # If observation is [376], set encoder_input_dim=376; if preprocessed to [44], use encoder_input_dim=44
# actor_network_ctor_old = partial(
#     FlowMatchingActorNetwork,
#     head_ctor=head_ctor_old,
#     flow_matching_steps=5,
#     encoder_input_dim=44,  # Match HumanoidBulletEnv-v0 observation dimension
#     encoder_output_dim=256,  # Encoded feature dimension for DiT
#     encoder_hidden_dims=(128,),  # Hidden layer dimensions
#     encoder_activation=torch.nn.ReLU,
# )
# ============================================================================

# MLPTrajectoryHead configuration (matching PHC architecture)
# For Bullet Humanoid: observation [44], action [17]
# NOTE: num_envs is used for pre-allocating condition dropout mask.
# Since condition_drop_ratio=0.0, the mask isn't used, but if enabled later,
# num_envs should be >= max batch size (num_parallel_environments * unroll_length)
head_ctor = partial(
    MLPTrajectoryHead,
    input_size=44,  # Observation dimension (HumanoidBulletEnv-v0 preprocessed)
    action_size=17,  # Action dimension (17 joints)
    hidden_size=512,  # Hidden dimension (matching PHC default)
    parameterization="velocity",  # Velocity parameterization (matching PHC)
    zero_action_input=False,  # Don't zero out action input
    prior_noise_std=1.0,  # Prior noise standard deviation
    solver_step_size=0.1,  # ODE solver step size (matching PHC)
    condition_drop_ratio=0.0,  # No condition dropout
    num_envs=MAX_BATCH_SIZE,  # Max batch size (num_parallel_environments * unroll_length) for dropout mask pre-allocation
)

# FlowMatchingMLPActorNetwork configuration (matching PHC FlowMatchingPolicy)
# Note: No encoder needed - MLP network uses observation directly
actor_network_ctor = partial(
    FlowMatchingMLPActorNetwork,
    input_size=44,  # Observation dimension (HumanoidBulletEnv-v0 preprocessed)
    action_size=17,  # Action dimension (17 joints)
    hidden_size=512,  # Hidden dimension (matching PHC default)
    parameterization="velocity",  # Velocity parameterization (matching PHC)
    zero_action_input=False,  # Don't zero out action input
    prior_noise_std=1.0,  # Prior noise standard deviation
    solver_step_size=0.1,  # ODE solver step size (matching PHC)
    condition_drop_ratio=0.0,  # No condition dropout
    num_envs=MAX_BATCH_SIZE,  # Max batch size (num_parallel_environments * unroll_length) for dropout mask pre-allocation
    head_ctor=head_ctor,  # Use MLPTrajectoryHead
)

# Value network configuration
# Note: preprocessing_combiner is required because _extract_value_observation_spec
# wraps the observation spec in a nested dict structure {'representation': {'context_info': ...}}
# even for simple tensor observations. NestConcat() combines the nested structure.
value_network_ctor = partial(
    ValueNetwork,
    preprocessing_combiner=alf.nest.utils.NestConcat(),  # Required for nested observation specs
    fc_layer_params=(32, 32, 32),
    activation=torch.tanh,
)

# Optimizer configuration
# Note: AdamTF is not configurable via alf.config, so we instantiate it directly
optimizer = AdamTF(
    lr=1e-4,  # Reduced from 3e-4 to prevent aggressive updates when returns are dropping
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
    # advantage_clip=10.0,  # CRITICAL: Clip advantages to prevent explosion
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
    clip_epsilon=0.02,  # Reduced from 0.05 to prevent aggressive updates when returns are dropping
    discretize_t_for_training=True,
    use_ppo_advantages=True,
    reward_mode="advantage",  # Use GAE advantages
    td_loss_weight=1.0,  # Increased from 0.5 to stabilize value network when returns are dropping
    fpo_loss_weight=1.0,  # Enable FPO loss
    imitation_loss_weight=0.0,  # Disable imitation loss (FPO only)
    warmup_value_network_iterations=0,
    cfm_loss_mode="eps_mse",
    use_noise_std_for_importance_ratio=True,  # Set to True for PHC-style importance ratio
    advantage_clip=None,  # Disable advantage clipping to match advantages_for_logging
    noise_std=1.0,  # Noise standard deviation for CFM loss (PHC uses 0.05), the larger gives more stable training
)

# Agent configuration
alf.config('TrainerConfig', algorithm_ctor=Agent)
alf.config('Agent', rl_algorithm_cls=DiffusionFPOAlgorithm)

# Training configuration
alf.config(
    'TrainerConfig',
    num_updates_per_train_iter=2,
    unroll_length=UNROLL_LENGTH,  # Use the constant defined above (384)
    mini_batch_size=40960,
    mini_batch_length=1,
    num_iterations=1000,
    evaluate=True,
    eval_interval=30,
    async_eval=False,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=10,
)

