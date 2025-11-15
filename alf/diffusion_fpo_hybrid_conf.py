"""
Configuration for HybridRLAgent with DiffusionFPOAlgorithm

This module provides configuration for using DiffusionFPOAlgorithm as rl_alg_ctor
in HybridRLAgent, mirroring the structure of diffusion_nft_hybrid_conf.py.
"""

import alf
import numpy as np
import warnings
from functools import partial
import torch
from torch import nn

from alf.optimizers import Adam
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.algorithms.entropy_target_algorithm import EntropyTargetAlgorithm
from alf.algorithms.lagrangian_reward_weight_algorithm import LagrangianRewardWeightAlgorithm
import alf.algorithms.data_transformer as dt
from alf.utils.averager import EMAverager
from alf.utils import schedulers

from OmniAD.model.hybrid_rl_agent import HybridRLAgent
from OmniAD.model.diffusion_fpo_algorithm import DiffusionFPOAlgorithm
from OmniAD.model.diffusion.diffusion_trajectory_model import DiffusionTrajectoryModel, DiffusionTrajectoryModelNFT, DiffusionTrajectoryModelFPO
from OmniAD.model.fiery.fiery_config import FieryConfig
from OmniAD.model.fiery.fiery_model import FieryModel, ValueHead
from OmniAD.model.fiery.flow_matching_bev_encoder import FlowMatchingBEVEncoder
from OmniAD.model.flow_matching_trajectory_head import FlowMatchingTrajectoryHead
from OmniAD.model.flow_matching_actor_network import FlowMatchingActorNetwork
from OmniAD.model.bev_value_network import BEVValueNetwork
from OmniAD.environment.control_wrapper import DummyControlWrapper
from OmniAD.model.representation_learner import RepresentationLearner, ReprsentationResNet
from OmniAD.model.losses import JointDistributionImitationLoss, GreedyTrajectoryImitationLoss, UntransformedActionLossWrapper, TrajectoryImitationLoss
from OmniAD.utils.utils import create_seed_sampler

# Enable strict NumPy error handling to surface sources of NaNs/Infs
np.seterr(divide='raise', invalid='raise', over='raise')
warnings.simplefilter('error', RuntimeWarning)

# Configuration parameters
rl_only = alf.define_config("rl_only", False)
with_repr_loss = alf.define_config("with_repr_loss", True)
freeze_repr_learner = alf.define_config("freeze_repr_learner", False)
pretrained_repr_checkpoint = alf.define_config("pretrained_repr_checkpoint", None)

# Pretrained checkpoint for DiffusionTrajectoryModel
pretrained_diffusion_checkpoint = alf.define_config(
    "pretrained_diffusion_checkpoint", 
    "/mnt/cwai/hpfs0/qiang.liu/e2e-rl/ckpts/fiery_b0_flowmatching_il_40000"
)

trajectory_steps = alf.define_config("trajectory_steps", 20)
trajectory_dim = alf.define_config("trajectory_dim", 3)
trajectory_time_horizon = alf.define_config("trajectory_time_horizon", 5.0)
imitation_learning = alf.define_config("imitation_learning", False)
open_loop_ratio = alf.define_config("open_loop_ratio", 0.1)
open_loop_render_ratio = alf.define_config("open_loop_render_ratio", 0.0)
open_loop_render_perturb_probability = alf.define_config("open_loop_render_perturb_probability", 0.0)
alf.import_config("fiery_common_conf.py")

# UntransformedTimeStep is needed for LagrangianRewardWeightAlgorithm
alf.config("UntransformedTimeStep", fields_to_keep=["reward"])
alf.config(
    "TrainerConfig",
    data_transformer_ctor=[dt.UntransformedTimeStep, dt.RewardClipping])

discount = alf.define_config("discount", 0.99)
alf.config(
    "AverageDiscountedReturnMetric",
    reward_transformer=dt.RewardClipping(),
    discount=discount)

fiery_config = FieryConfig()
fiery_config.time_horizon = trajectory_time_horizon
fiery_config.interval_length = trajectory_time_horizon / trajectory_steps
fiery_config.num_future_poses = trajectory_steps
fiery_config.encoder_cfg = "OmniAD/model/fiery/fiery_waymo_b0.yaml" 
dit_cfg = {
    'hidden_size': 256,
    'depth': 4,
    'num_heads': 8,
}

model_ctor = partial(
    # DiffusionTrajectoryModelNFT,
    DiffusionTrajectoryModelFPO,
    action_dim=trajectory_dim,
    trajectory_steps=trajectory_steps,
    diffusion_steps=20,
    inner_batch=-1,
    config=fiery_config,
    dit_cfg=dit_cfg,
    encoder='fiery',
    use_vae=False,
    time_sample_alpha=schedulers.LinearScheduler('percent', [(0, 0.5), (0.5, 1.0), (1.0, 1.0)]),
    dit_depth=3,
    repr_mode=True,
    pretrained_checkpoint=pretrained_diffusion_checkpoint if pretrained_diffusion_checkpoint else None,
)

# Representation learner
repr_loss_ctors = []
alf.config(
    "RepresentationLearner",
    model_ctor=model_ctor,
    loss_ctors=repr_loss_ctors,
    freeze_model=freeze_repr_learner,
    pretrained_checkpoint=pretrained_repr_checkpoint,
)

# Networks
fc_layers_params = (256, )
activation = torch.relu_
learning_rate = schedulers.StepScheduler("percent", [(0.1, 1e-4),
                                                     (0.50, 2e-5),
                                                     (0.75, 1e-5),
                                                     (1.00, 5e-6)])
optimizer = alf.optimizers.Adam(
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    clip_by_global_norm=True,
    gradient_clipping=10.0)

# Losses (optional IL losses)
imitate_transformed_action = alf.define_config("imitate_transformed_action", False)
imitate_untransformed_action = alf.define_config("imitate_untransformed_action", False)
loss_ctors = []
loss_weights = []
il_weight = 0.0 if rl_only else 0.2
control_wrapper_ctor = DummyControlWrapper

if not rl_only:
    rl_weight = 1.0
    if imitate_transformed_action:
        il_loss_ctor = GreedyTrajectoryImitationLoss
        alf.config(
            "GreedyTrajectoryImitationLoss",
            mask_loss_with_valid_target=True,
            loss_ctor=torch.nn.L1Loss)
        loss_ctors.append(il_loss_ctor)
        loss_weights.append(il_weight)
    if imitate_untransformed_action:
        il_loss_ctor = partial(
            UntransformedActionLossWrapper,
            control_wrapper_ctor=control_wrapper_ctor,
            loss_ctor=JointDistributionImitationLoss)
        loss_ctors.append(il_loss_ctor)
        loss_weights.append(il_weight)

# Flow Matching actor/value constructors
head_ctor = partial(
    FlowMatchingTrajectoryHead,
    num_poses=fiery_config.num_future_poses,
    d_ffn=fiery_config.tf_d_ffn,
    d_model=fiery_config.tf_d_model,
    config=fiery_config,
    action_dim=3,
    diffusion_steps=20,
    dit_depth=3,
    time_sample_alpha=0.0,
    pretrained_checkpoint=pretrained_diffusion_checkpoint if pretrained_diffusion_checkpoint else None,
)

actor_network_ctor = partial(
    FlowMatchingActorNetwork,
    head_ctor=head_ctor,
    flow_matching_steps=5,
)

value_network_ctor = partial(
    alf.networks.ValueNetwork,
    input_preprocessors=alf.layers.SummarizeGradient("value_input"),
    preprocessing_combiner=alf.nest.utils.NestConcat(),
    fc_layer_params=fc_layers_params,
    activation=activation,
    use_fc_ln=True)
# value_network_ctor = partial(BEVValueNetwork,
#                              bev_channels=256,
#                              use_conv=True,
#                              conv_layers=[(128, 3, 1), (64, 3, 1)],
#                              fc_layer_params=fc_layers_params,
#                              activation=activation,
#                              use_fc_ln=True)

# PPO configuration (advantages only)
alf.config(
    'PartialActionPPOLoss',
    action_dims=[0, trajectory_steps],
)

alf.config(
    'PPOLoss',
    entropy_regularization=0,
    normalize_scalar_advantages=True,
    gamma=discount,
    td_lambda=0.95,
    td_loss_weight=0.5)

alf.config(
    'PPOAlgorithm',
    loss_class=PPOLoss,
    actor_network_ctor=actor_network_ctor,
    value_network_ctor=value_network_ctor,
    optimizer=optimizer,
)

# DiffusionFPOAlgorithm configuration
alf.config(
    "DiffusionFPOAlgorithm",
    # PPO integration parameters
    use_ppo_advantages=True,
    reward_mode="mc_advantage",

    # Standard RL algorithm parameters
    actor_network_ctor=actor_network_ctor,
    value_network_ctor=value_network_ctor,
    optimizer=optimizer,

    # FPO-specific parameters
    n_mc_samples=50,
    average_losses_before_exp=True,
    clip_epsilon=0.05,  # default 0.05
    discretize_t_for_training=True,

    # Loss parameters
    td_loss_weight=1.0,
    fpo_loss_weight=1.0,
    imitation_loss_weight=100.0,
    warmup_value_network_iterations=200,
    cfm_loss_mode="eps_mse",
)

# Seed sampling configuration
sampler_ctor = None

# LagrangianRewardWeightAlgorithm configuration
alf.config(
    'LagrangianRewardWeightAlgorithm',
    reward_thresholds=[-1e-4, -1e-4, None, None],
    optimizer=alf.optimizers.AdamTF(lr=0.01))

# HybridRLAgent configuration with DiffusionFPOAlgorithm
alf.config(
    "HybridRLAgent",
    repr_alg_ctor=RepresentationLearner,
    rl_alg_ctor=DiffusionFPOAlgorithm,
    control_wrapper_ctor=control_wrapper_ctor,
    concat_action_for_rl=True,
    rl_weight=rl_weight,
    loss_ctors=loss_ctors,
    loss_weights=loss_weights,
    reward_weight_algorithm_ctor=LagrangianRewardWeightAlgorithm,
    entropy_target_algorithm_ctor=None,
    optimizer=optimizer,
    sampler_ctor=sampler_ctor,
    use_open_loop_data_for_rl=False,
)

# Trainer configuration
alf.config(
    "TrainerConfig",
    algorithm_ctor=HybridRLAgent,
    unroll_length=16,
    mini_batch_size=256,
    mini_batch_length=1,
    num_updates_per_train_iter=8,
    num_iterations=1000,
    num_checkpoints=4,
    sync_progress_to_envs=True,
    mask_out_loss_for_last_step=False,
    enable_amp=False,
    evaluate=False,
    async_eval=True,
    eval_interval=250,
    num_eval_episodes=1000,
    num_eval_environments=10,
    whole_replay_buffer_training=True,
    clear_replay_buffer=True,
    clear_replay_buffer_but_keep_one_step=False,
    replay_buffer_length=150,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summarize_action_distributions=True,
    confirm_checkpoint_upon_crash=False,
    no_thread_env_for_conf=True,
    empty_cache=False,
    use_root_inputs_for_after_train_iter=False,
    use_rollout_state=True,
    temporally_independent_train_step=True,
    summary_interval=5)

# Additional configurations
alf.config('summarize_gradients', with_histogram=False)
alf.config('summarize_variables', with_histogram=False)
alf.config('ReplayBuffer.gather_all', convert_to_default_device=False)
alf.config('Checkpointer.load', strict=True)


