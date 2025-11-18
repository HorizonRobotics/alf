import torch
import torch.nn.functional as F
from typing import Callable

import alf
from alf.data_structures import AlgStep, LossInfo, TimeStep
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm, ActorCriticInfo
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.utils import value_ops, tensor_utils, common
from alf.nest.utils import convert_device


# Extended info type for FPO algorithm
FPOInfo = alf.data_structures.namedtuple(
    "FPOInfo",
    ActorCriticInfo._fields + (
        "loss_eps",                 # Noise samples ε for FPO loss [T, B, Nmc, D]
        "loss_t",                   # Timestep samples τ for FPO loss [T, B, Nmc, 1]
        "initial_cfm_loss",         # CFM loss computed with old policy [T, B, Nmc]
        "initial_log_probs",        # Log probabilities computed with old policy [T, B, Nmc] (PHC-style)
        "ppo_advantages",           # PPO advantages from GAE
        "observation",              # Buffered observation for loss computation
        "returns",                  # Returns for value loss computation
        "valid_target",             # Valid target for FPO loss [B]
        "v_pred",                   # Predicted velocity from flow matching
        "v_target",                 # Target velocity for flow matching
        "ppo_value",                # PPO value network output for PPO value loss [T, B]
    ),
    default_value=()
)


class DiffusionFPOActorCriticLoss(ActorCriticLoss):
    """
    Combined loss for DiffusionFPO algorithm that includes:
    1. Actor loss (FPO loss with importance ratio clipping)
    2. Value loss (TD loss)
    3. Imitation loss (Flow Matching velocity loss)
    """
    
    def __init__(self,
                 compute_fpo_loss_fn=None,
                 gamma=0.99,
                 td_lambda=0.95,
                 entropy_regularization=0.0,
                 normalize_scalar_advantages=True,
                 advantage_clip=None,
                 td_loss_weight=1.0,
                 fpo_loss_weight=1.0,
                 imitation_loss_weight=1.0,
                 warmup_value_network_iterations=100,
                 **kwargs):
        super().__init__(
            gamma=gamma,
            td_lambda=td_lambda,
            entropy_regularization=entropy_regularization,
            normalize_scalar_advantages=normalize_scalar_advantages,
            advantage_clip=advantage_clip,
            td_loss_weight=td_loss_weight,
            **kwargs
        )
        self._compute_fpo_loss_fn = compute_fpo_loss_fn
        self._td_loss_weight = td_loss_weight
        self._fpo_loss_weight = fpo_loss_weight
        self._imitation_loss_weight = imitation_loss_weight
        self._warmup_value_network_iterations = warmup_value_network_iterations
        self._fpo_weight_schedule_fn = None  # Optional callable returning current fpo weight

    def set_fpo_weight_schedule(self, fn: Callable[[], float]):
        """Set a schedule function that returns the current FPO loss weight."""
        self._fpo_weight_schedule_fn = fn
    
    def calc_imitation_loss(self, info):
        """
        Compute imitation loss between v_pred and v_target using valid_target mask.
        
        Args:
            info: Rollout info containing v_pred, v_target, and valid_target
            
        Returns:
            LossInfo with imitation loss
        """
        # Extract velocities and valid_target mask from info
        v_pred = getattr(info, 'v_pred', None)
        v_target = getattr(info, 'v_target', None)
        valid_target = getattr(info, 'valid_target', ())
        
        # Assert that valid_target is available (not None and not empty tuple)
        assert valid_target is not None and valid_target != (), (
            "valid_target must be provided in info for imitation loss calculation. "
            "Make sure valid_target is extracted from observation and included in FPOInfo."
        )
        
        # If velocities are not available, return zero loss
        if v_pred is None or v_target is None:
            return LossInfo(
                loss=torch.tensor(0.0, device=info.value.device if hasattr(info, 'value') else 'cpu'),
                scalar_loss=torch.tensor(0.0, device=info.value.device if hasattr(info, 'value') else 'cpu'),
                extra={'imitation_loss': torch.tensor(0.0)}
            )
        
        # Apply valid_target mask (at this point we know it's a tensor from the assertion above)
        # Expand valid_target to match v_pred/v_target shape
        # valid_target shape: [B] or [T, B], v_pred/v_target shape: [B, D] or [T, B, D]
        if v_pred.ndim == 2 and valid_target.ndim == 1:
            # valid_target is [B], expand to [B, 1]
            mask = valid_target.unsqueeze(-1)
        elif v_pred.ndim == 3 and valid_target.ndim == 2:
            # valid_target is [T, B], expand to [T, B, 1]
            mask = valid_target.unsqueeze(-1)
        else:
            # Fallback: use all samples
            mask = torch.ones_like(v_pred)
        
        # Compute MSE loss between v_pred and v_target
        mse_loss = F.mse_loss(v_pred, v_target, reduction='none')
        
        # Apply mask: only compute loss for valid targets
        masked_loss = mse_loss * mask
        
        # Average over spatial dimensions (reduce to [B] or [T, B])
        loss_per_sample = masked_loss.mean(dim=-1)
        
        # Average over batch dimension
        total_loss = loss_per_sample.mean()

        # Dump debug summaries
        self._record_imitation_summaries(v_pred, v_target, valid_target, mask, mse_loss, masked_loss, loss_per_sample, total_loss)
        
        return LossInfo(
            loss=total_loss,
            scalar_loss=total_loss,
            extra={
                'imitation_loss': total_loss,
                'imitation_loss_per_sample': loss_per_sample.mean(dim=0) if v_pred.ndim == 3 else loss_per_sample
            }
        )

    def _record_imitation_summaries(self,
                                    v_pred: torch.Tensor,
                                    v_target: torch.Tensor,
                                    valid_target: torch.Tensor,
                                    mask: torch.Tensor,
                                    mse_loss: torch.Tensor,
                                    masked_loss: torch.Tensor,
                                    loss_per_sample: torch.Tensor,
                                    total_loss: torch.Tensor):
        """
        Record imitation-specific debug summaries.
        """
        if not alf.summary.should_record_summaries():
            return
        with alf.summary.scope("debug_imitation"):
            # Mask/valid statistics
            try:
                valid_ratio = mask.float().mean()
            except Exception:
                valid_ratio = torch.tensor(1.0, device=v_pred.device)
            alf.summary.scalar("valid_ratio", valid_ratio)

            # Pred/target stats (only valid entries)
            v_pred_valid = v_pred * mask
            v_target_valid = v_target * mask
            alf.summary.scalar("v_pred/norm", torch.norm(v_pred_valid))
            # Avoid NaN when mask is all zeros
            denom = mask.sum().clamp_min(1)
            alf.summary.scalar("v_pred/mean", v_pred_valid.sum() / denom)
            alf.summary.scalar("v_target/norm", torch.norm(v_target_valid))
            alf.summary.scalar("v_target/mean", v_target_valid.sum() / denom)

            # Loss stats
            alf.summary.scalar("mse/mean", mse_loss.mean())
            alf.summary.scalar("mse/max", mse_loss.max())
            alf.summary.scalar("masked_mse/mean", masked_loss.mean())
            alf.summary.scalar("loss_per_sample/mean", loss_per_sample.mean())
            alf.summary.scalar("loss/total", total_loss)
    
    def forward(self, info):
        """
        Compute FPO loss + Value network loss + Imitation loss
        
        Args:
            info: Rollout info containing advantages, values, v_pred, v_target, valid_target, etc.
            
        Returns:
            LossInfo with FPO loss + TD loss + Imitation loss
        """
        returns = info.returns
        td_loss = self._td_error_loss_fn(returns.detach(), info.value)
        
        # PPO uses the same value network as main algorithm, so no separate PPO value loss needed
        # The TD loss already trains the shared value network
        ppo_value_loss = torch.tensor(0.0, device=td_loss.device if isinstance(td_loss, torch.Tensor) else 'cpu')
        
        # Get current FPO weight based on schedule function
        current_fpo_weight = (
            self._fpo_weight_schedule_fn() if self._fpo_weight_schedule_fn is not None else self._fpo_loss_weight
        )
        
        # CRITICAL: Always compute FPO loss even during warmup (when weight=0) to ensure
        # encoder and trajectory_head.diffusion_model parameters receive gradients.
        # The weight multiplication (which can be 0) doesn't break the computation graph.
        if self._compute_fpo_loss_fn is not None:
            fpo_loss_info = self._compute_fpo_loss_fn(info)
        else:
            fpo_loss_info = LossInfo(
                loss=torch.tensor(0.0),
                scalar_loss=torch.tensor(0.0),
                extra={'fpo_loss': torch.tensor(0.0)}
            )
        
        # Compute imitation loss (Flow Matching velocity loss) only if weight > 0
        if self._imitation_loss_weight > 0.0:
            imitation_loss_info = self.calc_imitation_loss(info)
        else:
            imitation_loss_info = LossInfo(
                loss=torch.tensor(0.0),
                scalar_loss=torch.tensor(0.0),
                extra={'imitation_loss': torch.tensor(0.0)}
            )
        
        # Combine losses: FPO loss + TD loss + Imitation loss
        # Note: PPO uses the same value network, so TD loss trains it
        total_loss = (current_fpo_weight * fpo_loss_info.loss + 
                     self._td_loss_weight * td_loss.mean() +
                     self._imitation_loss_weight * imitation_loss_info.loss)
        
        return LossInfo(
            loss=total_loss,
            scalar_loss=total_loss,
            extra={
                'fpo_loss': fpo_loss_info.loss,
                'td_loss': td_loss.mean(),
                'value_loss': td_loss.mean(),
                'imitation_loss': imitation_loss_info.loss,
                **fpo_loss_info.extra,
                'imitation_loss_details': imitation_loss_info.extra if self._imitation_loss_weight > 0.0 else {}
            }
        )


@alf.configurable
class DiffusionFPOAlgorithm(ActorCriticAlgorithm):
    """
    DiffusionFPO algorithm that implements Flow Policy Optimization (FPO) for RL.
    
    This algorithm supports both GAE advantages and Monte Carlo returns-based advantages
    for computing the FPO loss. When use_returns_for_reward_signal=True, it computes
    MC returns and MC advantages which are more robust to value function errors.
    """
    
    @property
    def on_policy(self):
        return False
    
    @staticmethod
    def _extract_value_observation_spec(observation_spec, field_name='context_info'):
        """
        Extract observation spec for value network, retaining only specified field.
        
        This prevents dimension mismatch when NestConcat tries to concatenate
        fields with different dimensions (e.g., context_info is 2D, gt_trajectory is 3D,
        latent_bev_feature is 4D).
        
        Args:
            observation_spec: Full observation spec from representation learner
            field_name: Name of the field to extract (default: 'context_info')
                        Can be 'context_info' or 'latent_bev_feature'
            
        Returns:
            Filtered observation spec containing only the specified field
        """
        if isinstance(observation_spec, dict):
            if 'representation' in observation_spec:
                repr_spec = observation_spec['representation']
                if isinstance(repr_spec, dict):
                    # Extract specified field from nested representation
                    if field_name in repr_spec:
                        return {'representation': {field_name: repr_spec[field_name]}}
                    else:
                        # Fallback: return first field if specified field not found
                        first_key = next(iter(repr_spec.keys()))
                        return {'representation': {first_key: repr_spec[first_key]}}
                else:
                    # representation is not a dict, return as-is
                    return {'representation': repr_spec}
            elif field_name in observation_spec:
                # Flat structure with specified field
                return {'representation': {field_name: observation_spec[field_name]}}
            else:
                # Unknown structure, return as-is
                return observation_spec
        else:
            # observation_spec is not a dict, return wrapped
            return {'representation': {field_name: observation_spec}}
    
    @staticmethod
    def _extract_bev_observation_spec(observation_spec):
        """
        Extract observation spec for BEV value network, retaining only latent_bev_feature.
        
        This is a convenience method that calls _extract_value_observation_spec
        with field_name='latent_bev_feature'.
        
        Args:
            observation_spec: Full observation spec from representation learner
            
        Returns:
            Filtered observation spec containing only latent_bev_feature
        """
        return DiffusionFPOAlgorithm._extract_value_observation_spec(
            observation_spec, field_name='latent_bev_feature'
        )
    
    def __init__(self,
                 observation_spec: alf.NestedTensorSpec,
                 action_spec: alf.NestedBoundedTensorSpec,
                 reward_spec: alf.NestedTensorSpec = alf.TensorSpec(()),
                 n_mc_samples: int = 8,
                 average_losses_before_exp: bool = True,
                 clip_epsilon: float = 0.05,
                 discretize_t_for_training: bool = True,
                 use_ppo_advantages: bool = True,
                 advantage_normalization_factor: float = 5.0,  # Normalization factor for advantages
                 reward_mode: str = "advantage",  # "advantage" (GAE), "mc_advantage" (MC returns - value), or "mc_return" (MC returns)
                 actor_network_ctor: Callable = None,
                 value_network_ctor: Callable = None,
                 optimizer: alf.optimizers.Optimizer = None,
                 config: alf.algorithms.config.TrainerConfig = None,
                 debug_summaries: bool = False,
                 name: str = "DiffusionFPOAlgorithm",
                 td_loss_weight: float = 1.0,
                 fpo_loss_weight: float = 1.0,
                 imitation_loss_weight: float = 1.0,
                 warmup_value_network_iterations: int = 100,
                 cfm_loss_mode: str = "eps_mse",  # "eps_mse" (preferred) or "u_mse"
                 use_noise_schedule_weight: bool = False,  # Whether to apply noise schedule weighting to CFM loss
                 use_noise_std_for_importance_ratio: bool = False,  # Whether to use noise std (0.05) for importance ratio computation (PHC-style)
                 noise_std: float = 0.05,  # Noise standard deviation for CFM loss (PHC uses 0.05)
                 ):
        """
        Args:
            observation_spec: nested spec of the observations
            action_spec: nested spec of the actions
            reward_spec: a rank-1 or rank-0 tensor spec representing the reward(s)
            n_mc_samples: Number of Monte Carlo samples for CFM loss computation
            average_losses_before_exp: Whether to average losses before computing exp() for importance ratio
            clip_epsilon: Clipping epsilon for PPO-style importance ratio clipping
            discretize_t_for_training: Whether to discretize time steps for training
            use_ppo_advantages: Whether to use PPO for advantage calculation
            advantage_normalization_factor: Normalization factor for advantages (currently stored but not used directly)
            reward_mode: Reward signal mode for FPO loss:
                - "advantage": GAE advantages (default, lower variance)
                - "mc_advantage": Monte Carlo advantages (MC returns - value, higher variance but unbiased)
                - "mc_return": Monte Carlo returns directly (no value function subtraction)
            actor_network_ctor: Creator function for actor network
            value_network_ctor: Creator function for value network
            optimizer: Optimizer for training
            config: Training configuration
            debug_summaries: Whether to enable debug summaries
            name: Name of the algorithm
            td_loss_weight: Weight for TD loss
            fpo_loss_weight: Weight for FPO loss
            imitation_loss_weight: Weight for imitation loss
            warmup_value_network_iterations: Number of iterations to warmup value network
            cfm_loss_mode: CFM loss mode - "eps_mse" (preferred) or "u_mse"
            use_noise_schedule_weight: Whether to apply noise schedule weighting (1/2)/(τ(1-τ)) to CFM loss.
                If False, uses unweighted MSE loss (matches fpo.py reference implementation).
            use_noise_std_for_importance_ratio: If True, compute log_probs with noise std for importance ratio (PHC-style).
                If False, use CFM loss differences for importance ratio (ALF-style).
            noise_std: Noise standard deviation for CFM loss computation (default: 0.05, matches PHC).
        """
        # Create filtered observation spec for value networks (only context_info)
        # This prevents dimension mismatch when NestConcat tries to concatenate
        # fields with different dimensions (e.g., context_info is 2D, gt_trajectory is 3D,
        # latent_bev_feature is 4D)
        # Note: This is only used for networks that use NestConcat (e.g., alf.networks.ValueNetwork)
        value_observation_spec = DiffusionFPOAlgorithm._extract_value_observation_spec(observation_spec)
        
        # Create wrapper for value_network_ctor if needed
        # Only needed for networks that use NestConcat (e.g., alf.networks.ValueNetwork)
        # Networks like BEVValueNetwork don't need this since they manually extract features
        if value_network_ctor is not None:
            # Check if the value network is alf.networks.ValueNetwork (uses NestConcat)
            # We need to check the actual function/class, not just the name
            import inspect
            value_net_class = value_network_ctor
            if hasattr(value_network_ctor, 'func'):
                # It's a partial function, get the underlying class
                value_net_class = value_network_ctor.func
            elif hasattr(value_network_ctor, '__self__'):
                # It's a bound method
                value_net_class = value_network_ctor.__self__
            
            # Check if it's alf.networks.ValueNetwork (which uses NestConcat)
            is_alf_value_network = False
            is_bev_value_network = False
            if inspect.isclass(value_net_class):
                from alf.networks.value_networks import ValueNetwork
                is_alf_value_network = issubclass(value_net_class, ValueNetwork)
                # Check if it's BEVValueNetwork
                try:
                    from OmniAD.model.bev_value_network import BEVValueNetwork
                    is_bev_value_network = issubclass(value_net_class, BEVValueNetwork)
                except ImportError:
                    pass
            elif hasattr(value_net_class, '__name__'):
                # Check by name as fallback
                value_net_name = str(value_net_class)
                is_alf_value_network = 'ValueNetwork' in value_net_name and 'alf.networks' in value_net_name
                is_bev_value_network = 'BEVValueNetwork' in value_net_name
            
            if is_alf_value_network:
                # Use filtered spec for alf.networks.ValueNetwork (only context_info)
                def value_network_ctor_wrapper(input_tensor_spec):
                    # Ignore input_tensor_spec from parent, use filtered spec instead
                    return value_network_ctor(input_tensor_spec=value_observation_spec)
            elif is_bev_value_network:
                # Use BEV observation spec for BEVValueNetwork (only latent_bev_feature)
                bev_observation_spec = DiffusionFPOAlgorithm._extract_bev_observation_spec(observation_spec)
                def value_network_ctor_wrapper(input_tensor_spec):
                    # Ignore input_tensor_spec from parent, use BEV spec instead
                    return value_network_ctor(input_tensor_spec=bev_observation_spec)
            else:
                # For other custom networks, no wrapper needed
                # They manually extract features and don't use NestConcat
                value_network_ctor_wrapper = value_network_ctor
        else:
            value_network_ctor_wrapper = None
        
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_ctor=actor_network_ctor,
            value_network_ctor=value_network_ctor_wrapper,
            optimizer=optimizer,
            config=config,
            debug_summaries=debug_summaries,
            name=name
        )
        self._n_mc_samples = n_mc_samples
        self._average_losses_before_exp = average_losses_before_exp
        self._clip_epsilon = clip_epsilon
        self._discretize_t_for_training = discretize_t_for_training
        self._update_counter = 0
        self._old_trajectory_head = self._create_old_trajectory_head()
        self._use_ppo_advantages = use_ppo_advantages
        self._advantage_normalization_factor = advantage_normalization_factor
        # Validate reward_mode
        assert reward_mode in ["advantage", "mc_advantage", "mc_return"], (
            f"reward_mode must be 'advantage', 'mc_advantage', or 'mc_return', got '{reward_mode}'"
        )
        self._reward_mode = reward_mode
        self._td_loss_weight = td_loss_weight
        self._fpo_loss_weight = fpo_loss_weight
        self._imitation_loss_weight = imitation_loss_weight
        self._warmup_value_network_iterations = warmup_value_network_iterations
        assert cfm_loss_mode in ["eps_mse", "u_mse"], (
            f"cfm_loss_mode must be 'eps_mse' or 'u_mse', got '{cfm_loss_mode}'"
        )
        self._cfm_loss_mode = cfm_loss_mode
        self._use_noise_schedule_weight = use_noise_schedule_weight
        self._use_noise_std_for_importance_ratio = use_noise_std_for_importance_ratio
        self._noise_std = noise_std
        if use_ppo_advantages:
            # Share the same value network with PPOAlgorithm instead of creating a duplicate
            # Pass value_network_ctor=None to prevent PPOAlgorithm from creating its own
            self._ppo_algorithm = PPOAlgorithm(
                observation_spec=observation_spec,
                action_spec=action_spec,
                reward_spec=reward_spec,
                value_network_ctor=None,  # Don't create a separate value network
                config=config,
                debug_summaries=debug_summaries,
                name=f"{name}_PPO"
            )
            # Share the main value network with PPO algorithm
            # This ensures we only have ONE value network that's trained
            if self._value_network is not None:
                self._ppo_algorithm._value_network = self._value_network
            # PPO actor_network is not used (only preprocess_experience is called, which only uses value_network)
            # Set to None to prevent unused parameters from being registered with optimizer
            self._ppo_algorithm._actor_network = None
        self._loss = DiffusionFPOActorCriticLoss(
            compute_fpo_loss_fn=self._compute_fpo_loss_from_info,
            gamma=0.99,
            td_lambda=0.95,
            entropy_regularization=0.0,
            normalize_scalar_advantages=True,
            advantage_clip=10.0,  # Clip advantages to prevent explosion
            td_loss_weight=self._td_loss_weight,
            fpo_loss_weight=self._fpo_loss_weight,
            imitation_loss_weight=self._imitation_loss_weight,
            warmup_value_network_iterations=self._warmup_value_network_iterations
        )
        # Set FPO loss weight schedule: start at 0, then enable after a flat period
        self._loss.set_fpo_weight_schedule(lambda: self._scheduled_fpo_loss_weight(flat=warmup_value_network_iterations))
        
        # CRITICAL FIX: Ensure actor and value networks are in train mode and have gradients enabled
        # This fixes the bug where networks don't have gradients during training
        if self._actor_network is not None:
            self._actor_network.train()  # Set to training mode
            for param in self._actor_network.parameters():
                param.requires_grad = True  # Ensure gradients are enabled
        if self._value_network is not None:
            self._value_network.train()  # Set to training mode
            for param in self._value_network.parameters():
                param.requires_grad = True  # Ensure gradients are enabled
        
        # PPO's value network is now the same as main value network (shared)
        # No need to set it separately since it's already set above

    def calc_noise_schedule_weight(self, tau: torch.Tensor, eps: float = 1e-6, max_weight: float = 100.0, min_weight: float = -100.0) -> torch.Tensor:
        """
        Calculate the noise schedule weight for weighted Flow Matching loss.
        
        Implements the weight term from: ℓw_θ(τ, ϵ) = (1/2) · w(λ_τ) · (-dλ/dτ) · ||ε̂_θ(a_τ^t; λ_τ) - ϵ||²
        where:
            - λ_τ = log((1-τ)/τ) for linear interpolation
            - (-dλ/dτ) = 1/(τ(1-τ))
            - w(λ_τ) = 1 (constant weight)
        
        So the weight is: (1/2) / (τ(1-τ))
        
        Args:
            tau: Timestep tensor τ ∈ [0, 1], shape [..., 1] or [...,]
            eps: Small epsilon for numerical stability (default: 1e-6)
            max_weight: Maximum weight (default: 100.0)
            min_weight: Minimum weight (default: -100.0)
        Returns:
            Weight tensor with same shape as tau: (1/2) / (τ(1-τ))
        """
        # Clamp tau to avoid numerical instability at boundaries (0 and 1)
        tau_clamped = torch.clamp(tau, min=eps, max=1.0 - eps)
        # Compute weight: (1/2) / (τ(1-τ))
        weight = 0.5 / (tau_clamped * (1.0 - tau_clamped))
        # clamp weight to avoid numerical instability
        weight = torch.clamp(weight, min=min_weight, max=max_weight)
        return weight

    def _compute_cfm_loss(self,
                         action: torch.Tensor,
                         eps: torch.Tensor,
                         t: torch.Tensor,
                         context: torch.Tensor,
                         trajectory_head=None,
                         return_log_probs=False):
        '''
        Computes the CFM loss for a given action, eps, t, and context.
        Args:
            action: Action tensor, shape [..., action_dim]
            eps: Eps tensor, shape [..., action_dim]
            t: Timestep tensor, shape [..., 1]
            context: Context tensor, shape [..., context_dim] (raw observation, will be encoded)
            trajectory_head: Trajectory head model
            return_log_probs: If True, also return log_probs for importance ratio computation
        Returns:
            cfm_loss: CFM loss tensor
            log_probs: (optional) Log probabilities if return_log_probs=True
        '''
        if trajectory_head is None:
            trajectory_head = self._actor_network.trajectory_head
        
        # Encode context using actor network's encoder if it exists
        # The encoder transforms raw observation (e.g., [B, 44] or [B, 376]) to encoded feature (e.g., [B, 256])
        if hasattr(self._actor_network, 'encoder'):
            # Reshape context to [B, D] for encoding
            context_shape = context.shape
            context_flat_for_encoding = context.reshape(-1, context.shape[-1])  # [..., context_dim] -> [B, context_dim]
            context_encoded = self._actor_network.encoder(context_flat_for_encoding)  # [B, context_dim] -> [B, encoder_output_dim]
            # Reshape back to original batch dimensions with encoded dimension
            context_encoded = context_encoded.reshape(*context_shape[:-1], context_encoded.shape[-1])  # [..., encoder_output_dim]
            context = context_encoded
        
        *batch_dims, action_dim = action.shape
        if eps.ndim == len(batch_dims) + 2:
            samples_dim = eps.shape[-2]
            eps_expanded = eps
            t_expanded = t
            action_expanded = action.unsqueeze(-2)
            context_expanded = context.unsqueeze(-2)
            action_broadcast = action_expanded.expand(*batch_dims, samples_dim, action_dim)
            context_broadcast = context_expanded.expand(*batch_dims, samples_dim, context.shape[-1])
        else:
            samples_dim = 1
            eps_expanded = eps.unsqueeze(-2)
            t_expanded = t.unsqueeze(-2)
            action_expanded = action.unsqueeze(-2)
            context_broadcast = context.unsqueeze(-2)
            action_broadcast = action_expanded
        if t_expanded.ndim == len(batch_dims) + 1:
            t_expanded = t_expanded.unsqueeze(-1)
        total_samples = action_broadcast.shape[0] * samples_dim if len(batch_dims) == 1 else action_broadcast.numel() // action_dim
        action_flat = action_broadcast.reshape(total_samples, action_dim)
        eps_flat = eps_expanded.reshape(total_samples, action_dim)
        t_flat = t_expanded.reshape(total_samples, 1)
        context_flat = context_broadcast.reshape(total_samples, -1)
        x_t = (1.0 - t_flat) * action_flat + t_flat * eps_flat
        x_t_input = x_t.unsqueeze(1)
        # CRITICAL: This call to diffusion_model must maintain gradients
        # v_pred is part of the computation graph that flows back to diffusion_model parameters
        v_pred = trajectory_head.diffusion_model(
            x_t_input, context_flat, t_flat.squeeze(-1)
        ).squeeze(1)
        
        # Compute target velocity: u_t = x1 - x0 = eps - action
        v_target = eps_flat - action_flat  # Ground truth velocity: u = x1 - x0
        
        # Use PHC-style loss formula: log_probs = -((u_t - velocity)²) / (2 * noise_std²)
        # Then loss = -mean(log_probs) = mean(((u_t - velocity)²) / (2 * noise_std²))
        # This matches PHC implementation: -((u_t - velocity) ** 2) / (2 * 0.05 ** 2)
        squared_error = (v_target - v_pred) ** 2  # [total_samples, action_dim]
        
        # Compute log probabilities with noise std (PHC formula)
        log_probs = -squared_error / (2 * self._noise_std ** 2)  # [total_samples, action_dim]
        
        # Average over action dimension to get per-sample log_prob
        log_probs_per_sample = log_probs.mean(dim=-1)  # [total_samples]
        
        # CFM loss is negative mean of log_probs (PHC formula)
        # loss = -mean(log_probs) = mean(((u_t - velocity)²) / (2 * noise_std²))
        cfm_loss = -log_probs_per_sample  # [total_samples]
        
        # Optionally apply noise schedule weight (if enabled)
        if self._use_noise_schedule_weight:
            weight = self.calc_noise_schedule_weight(t_flat)  # [total_samples, 1]
            # Apply weight to CFM loss
            cfm_loss = weight.squeeze(-1) * cfm_loss  # [total_samples]
        # Reshape back robustly: works for [B, D] (batch_dims=[B]) and [T, B, D] (batch_dims=[T, B])
        if len(batch_dims) == 0:
            # Scalar batch, keep as [samples_dim]
            cfm_loss = cfm_loss.reshape(samples_dim)
            if return_log_probs:
                log_probs_per_sample = log_probs_per_sample.reshape(samples_dim)
        elif len(batch_dims) == 1:
            cfm_loss = cfm_loss.reshape(batch_dims[0], samples_dim)
            if return_log_probs:
                log_probs_per_sample = log_probs_per_sample.reshape(batch_dims[0], samples_dim)
        else:
            cfm_loss = cfm_loss.reshape(*batch_dims, samples_dim)
            if return_log_probs:
                log_probs_per_sample = log_probs_per_sample.reshape(*batch_dims, samples_dim)
        
        if return_log_probs:
            return cfm_loss, log_probs_per_sample
        return cfm_loss
    
    def _compute_fpo_loss_from_info(self, info):
        if not hasattr(info, 'action') or info.action == ():
            return LossInfo(
                loss=torch.tensor(0.0),
                scalar_loss=torch.tensor(0.0),
                extra={'fpo_loss': torch.tensor(0.0)}
            )
        action = info.action
        if not hasattr(info, 'observation') or info.observation == ():
            return LossInfo(
                loss=torch.tensor(0.0),
                scalar_loss=torch.tensor(0.0),
                extra={'fpo_loss': torch.tensor(0.0)}
            )
        # Extract context_info from observation (handles both dict and tensor cases)
        observation = info.observation
        if isinstance(observation, dict):
            if 'representation' in observation:
                repr_data = observation['representation']
                if isinstance(repr_data, dict) and 'context_info' in repr_data:
                    context = repr_data['context_info']
                else:
                    context = repr_data
            elif 'context_info' in observation:
                context = observation['context_info']
            else:
                context = next(iter(observation.values()))
        else:
            # observation is a tensor
            context = observation
        loss_eps = info.loss_eps
        loss_t = info.loss_t
        initial_cfm_loss = info.initial_cfm_loss
        initial_log_probs = getattr(info, 'initial_log_probs', ())
        advantages = info.ppo_advantages
        # Handle different action shapes: [T, B, D] or [T, B, num_poses, D]
        if action.ndim == 3:
            T, B, D = action.shape
            action_downscaled = action.clone()
        elif action.ndim == 4:
            # Action is [T, B, num_poses, D] - flatten poses dimension
            T, B, num_poses, D = action.shape
            action_downscaled = action.view(T, B, -1)  # [T, B, num_poses * D]
        else:
            raise ValueError(f"Unexpected action shape: {action.shape}, expected [T, B, D] or [T, B, num_poses, D]")
        Nmc = loss_eps.shape[2]
        # Only apply OmniAD-specific downscaling if action_dim >= 2
        # This is hardcoded for OmniAD (x, y coordinates) and should be disabled for other environments
        # Check if trajectory head has scaling enabled (OmniAD-specific)
        enable_scaling = getattr(self._actor_network.trajectory_head, '_enable_trajectory_scaling', False)
        if enable_scaling and action.ndim == 3 and D >= 2:
            # Only apply scaling if action is [T, B, D] (not already flattened)
            action_downscaled[..., 0] = (action_downscaled[..., 0] - 20) / 20
            action_downscaled[..., 1] = action_downscaled[..., 1] / 20
        # Ensure action_downscaled is [T, B, flattened_dim]
        if action_downscaled.ndim != 3:
            action_downscaled = action_downscaled.view(T, B, -1)
        action_flat = action_downscaled.reshape(T * B, -1)
        context_flat = context.reshape(T * B, -1)
        loss_eps_flat = loss_eps.reshape(T * B, Nmc, -1)
        loss_t_flat = loss_t.reshape(T * B, Nmc, 1)
        # CRITICAL: cfm_loss_current must maintain gradients to flow back to trajectory_head.diffusion_model
        # This is computed using the current (trainable) trajectory_head, not the frozen old one
        if self._use_noise_std_for_importance_ratio:
            # Compute CFM loss and log_probs for importance ratio (PHC-style)
            cfm_loss_current, log_probs_current = self._compute_cfm_loss(
                action=action_flat,
                eps=loss_eps_flat,
                t=loss_t_flat,
                context=context_flat,
                trajectory_head=self._actor_network.trajectory_head,  # Current trainable head
                return_log_probs=True
            )
        else:
            # Compute only CFM loss (ALF-style)
            cfm_loss_current = self._compute_cfm_loss(
                action=action_flat,
                eps=loss_eps_flat,
                t=loss_t_flat,
                context=context_flat,
                trajectory_head=self._actor_network.trajectory_head  # Current trainable head
            )
        # Ensure initial_cfm_loss has correct shape [T, B, Nmc] before reshaping
        # Handle both cases: if it's already [T, B, Nmc] or if it's [T*B, Nmc]
        if initial_cfm_loss.ndim == 2:
            # Check if shape matches expected [T*B, Nmc] or needs reshaping
            if initial_cfm_loss.shape[0] == T * B:
                initial_cfm_loss_flat = initial_cfm_loss
            elif initial_cfm_loss.shape[0] == T and initial_cfm_loss.shape[1] == B * Nmc:
                # Reshape from [T, B*Nmc] to [T, B, Nmc] then flatten
                initial_cfm_loss_flat = initial_cfm_loss.reshape(T, B, Nmc).reshape(T * B, Nmc)
            else:
                # Assume it's [T, B, Nmc] and reshape
                initial_cfm_loss_flat = initial_cfm_loss.reshape(T * B, Nmc)
        elif initial_cfm_loss.ndim == 3:
            # Already [T, B, Nmc]
            initial_cfm_loss_flat = initial_cfm_loss.reshape(T * B, Nmc)
        else:
            # Fallback: try reshaping directly
            initial_cfm_loss_flat = initial_cfm_loss.reshape(T * B, Nmc)
        if self._use_noise_std_for_importance_ratio:
            # PHC-style: Use log_probs difference for importance ratio
            # Ensure initial_log_probs has correct shape
            if initial_log_probs == ():
                raise ValueError("initial_log_probs must be provided when use_noise_std_for_importance_ratio=True")
            
            if initial_log_probs.ndim == 2:
                if initial_log_probs.shape[0] == T * B:
                    initial_log_probs_flat = initial_log_probs
                elif initial_log_probs.shape[0] == T and initial_log_probs.shape[1] == B * Nmc:
                    initial_log_probs_flat = initial_log_probs.reshape(T, B, Nmc).reshape(T * B, Nmc)
                else:
                    initial_log_probs_flat = initial_log_probs.reshape(T * B, Nmc)
            elif initial_log_probs.ndim == 3:
                initial_log_probs_flat = initial_log_probs.reshape(T * B, Nmc)
            else:
                initial_log_probs_flat = initial_log_probs.reshape(T * B, Nmc)
            
            # Compute importance ratio from log_probs difference
            # rho = exp(log_prob_current - log_prob_old)
            if self._average_losses_before_exp:
                mean_current_logprob = log_probs_current.mean(dim=-1, keepdim=True)
                mean_initial_logprob = initial_log_probs_flat.mean(dim=-1, keepdim=True)
                assert mean_current_logprob.shape == mean_initial_logprob.shape
                logprob_diff = mean_current_logprob - mean_initial_logprob
                logprob_diff_clipped = torch.clamp(logprob_diff, -3.0, 3.0)
                rho = torch.exp(logprob_diff_clipped).squeeze(-1)
            else:
                assert log_probs_current.shape == initial_log_probs_flat.shape
                logprob_diff = log_probs_current - initial_log_probs_flat
                logprob_diff_clipped = torch.clamp(logprob_diff, -3.0, 3.0)
                rho = torch.exp(logprob_diff_clipped)  # Shape: (T*B, Nmc)
        else:
            # ALF-style: Use CFM loss difference for importance ratio
            if self._average_losses_before_exp:
                mean_current_loss = cfm_loss_current.mean(dim=-1, keepdim=True)
                mean_initial_loss = initial_cfm_loss_flat.mean(dim=-1, keepdim=True)
                # Ensure shapes match for proper comparison
                assert mean_current_loss.shape == mean_initial_loss.shape, (
                    f"Shape mismatch: mean_current_loss {mean_current_loss.shape} vs "
                    f"mean_initial_loss {mean_initial_loss.shape}"
                )
                # Clip loss difference before exp() to prevent numerical instability
                # This prevents rho from exploding/vanishing when loss differences are large
                # NOTE: mean_initial_loss is already detached (computed in torch.no_grad() context)
                # But we need to ensure mean_current_loss maintains gradients
                loss_diff = mean_initial_loss - mean_current_loss
                loss_diff_clipped = torch.clamp(loss_diff, -3.0, 3.0)  # Todo: expose this as a parameter
                rho = torch.exp(loss_diff_clipped).squeeze(-1)
            else:
                # Ensure shapes match for proper comparison
                assert cfm_loss_current.shape == initial_cfm_loss_flat.shape, (
                    f"Shape mismatch: cfm_loss_current {cfm_loss_current.shape} vs "
                    f"initial_cfm_loss_flat {initial_cfm_loss_flat.shape}"
                )
                # NOTE: initial_cfm_loss_flat is already detached (computed in torch.no_grad() context)
                # But we need to ensure cfm_loss_current maintains gradients
                # BUG FIX: Do NOT average over MC samples here - keep them for proper broadcasting
                # This matches fpo.py behavior: rho_s = exp(clip(initial_cfm_loss - cfm_loss, -3, 3))
                # Shape: (T*B, Nmc) - one rho per MC sample
                loss_diff = initial_cfm_loss_flat - cfm_loss_current
                loss_diff_clipped = torch.clamp(loss_diff, -3.0, 3.0)  # Todo: expose this as a parameter
                rho = torch.exp(loss_diff_clipped)  # Shape: (T*B, Nmc) - keep MC dimension
        # NOTE: For GAE advantages, they're already detached (generalized_advantage_estimation returns .detach())
        # For MC advantages, we detach value in _compute_mc_advantage to ensure no gradients
        # Advantages are treated as constants (targets) in the policy loss, similar to standard PPO
        # Explicit detach here for safety and clarity (redundant for GAE but safe for MC)
        advantages_flat = advantages.reshape(T * B).detach()  # Shape: (T*B,)
        
        # CRITICAL: Clip advantages to prevent explosion and training failure
        # Large advantages multiplied by rho can cause loss to explode
        if self._loss._advantage_clip is not None and self._loss._advantage_clip > 0:
            advantages_flat = torch.clamp(advantages_flat, -self._loss._advantage_clip, self._loss._advantage_clip)
        
        # PPO-style clipping: use minimum for pessimistic clipping
        # This matches the reference implementation in fpo.py
        # CRITICAL: rho has gradients from cfm_loss_current, which flows back to trajectory_head.diffusion_model
        if self._average_losses_before_exp:
            # rho shape: (T*B,), advantages_flat shape: (T*B,)
            rho_clipped = torch.clamp(rho, 1.0 - self._clip_epsilon, 1.0 + self._clip_epsilon)
            surrogate_loss1 = rho * advantages_flat
            surrogate_loss2 = rho_clipped * advantages_flat
            fpo_loss = -torch.minimum(surrogate_loss1, surrogate_loss2).mean()
        else:
            # rho shape: (T*B, Nmc), need to broadcast advantages
            # Match fpo.py: surrogate_loss1 = rho_s * gae_advantages[..., None]
            advantages_expanded = advantages_flat.unsqueeze(-1)  # Shape: (T*B, 1)
            rho_clipped = torch.clamp(rho, 1.0 - self._clip_epsilon, 1.0 + self._clip_epsilon)  # Shape: (T*B, Nmc)
            surrogate_loss1 = rho * advantages_expanded  # Shape: (T*B, Nmc)
            surrogate_loss2 = rho_clipped * advantages_expanded  # Shape: (T*B, Nmc)
            # Average over all dimensions (batch and MC samples) - matches fpo.py: jnp.mean(jnp.minimum(...))
            fpo_loss = -torch.minimum(surrogate_loss1, surrogate_loss2).mean()
        self._record_debug_fpo_summaries(
            rho=rho,
            rho_clipped=rho_clipped,
            advantages_flat=advantages_flat,
            cfm_loss_current=cfm_loss_current,
            initial_cfm_loss_flat=initial_cfm_loss_flat,
            fpo_loss=fpo_loss,
            action_flat=action_flat,
            context_flat=context_flat,
        )
        return LossInfo(
            loss=fpo_loss,
            scalar_loss=fpo_loss,
            extra={
                'fpo_loss': fpo_loss,
                'policy_ratio_mean': rho.mean(),
                'policy_ratio_min': rho.min(),
                'policy_ratio_max': rho.max(),
                'clipped_ratio_mean': (torch.abs(rho - 1.0) > self._clip_epsilon).float().mean(),
            }
        )

    def _record_debug_fpo_summaries(self,
                                    rho: torch.Tensor,
                                    rho_clipped: torch.Tensor,
                                    advantages_flat: torch.Tensor,
                                    cfm_loss_current: torch.Tensor,
                                    initial_cfm_loss_flat: torch.Tensor,
                                    fpo_loss: torch.Tensor,
                                    action_flat: torch.Tensor = None,
                                    context_flat: torch.Tensor = None):
        """
        Record debug summaries for FPO training to monitor stability and values.
        
        Args:
            rho: Policy ratio (importance ratio) [T*B] or [T*B, Nmc] depending on average_losses_before_exp
            rho_clipped: Clipped policy ratio [T*B] or [T*B, Nmc] (same shape as rho)
            advantages_flat: Advantages [T*B]
            cfm_loss_current: Current CFM loss [T*B, Nmc] or [T*B]
            initial_cfm_loss_flat: Initial CFM loss [T*B, Nmc]
            fpo_loss: FPO loss scalar
            action_flat: Flattened action/trajectory [T*B, D] (optional)
            context_flat: Flattened context [T*B, C] (optional)
        """
        if alf.summary.should_record_summaries():
            with alf.summary.scope("debug_fpo"):
                # Policy ratio statistics
                alf.summary.scalar("policy_ratio/mean", rho.mean())
                alf.summary.scalar("policy_ratio/std", rho.std())
                alf.summary.scalar("policy_ratio/min", rho.min())
                alf.summary.scalar("policy_ratio/max", rho.max())
                
                # Track percentage of samples where rho was actually clipped
                # This is different from checking if rho is outside bounds - it checks if clipping changed the value
                was_clipped = (rho != rho_clipped).float()
                clipping_percentage = was_clipped.mean() * 100.0  # Convert to percentage
                alf.summary.scalar("clipping_percentage", clipping_percentage)
                
                # Also track the existing metric for samples outside clipping bounds
                clipped_mask = torch.abs(rho - 1.0) > self._clip_epsilon
                alf.summary.scalar("clipped_ratio/mean", clipped_mask.float().mean())
                
                # Advantages statistics
                alf.summary.scalar("advantages/mean", advantages_flat.mean())
                alf.summary.scalar("advantages/std", advantages_flat.std())
                alf.summary.scalar("advantages/min", advantages_flat.min())
                alf.summary.scalar("advantages/max", advantages_flat.max())
                
                # CFM loss statistics
                if cfm_loss_current.ndim > 1:
                    # If multi-dimensional, average over MC samples first
                    cfm_loss_avg = cfm_loss_current.mean(dim=-1)
                    alf.summary.scalar("cfm_loss_current/mean", cfm_loss_avg.mean())
                    alf.summary.scalar("cfm_loss_current/std", cfm_loss_avg.std())
                    alf.summary.scalar("cfm_loss_current/min", cfm_loss_avg.min())
                    alf.summary.scalar("cfm_loss_current/max", cfm_loss_avg.max())
                else:
                    alf.summary.scalar("cfm_loss_current/mean", cfm_loss_current.mean())
                    alf.summary.scalar("cfm_loss_current/std", cfm_loss_current.std())
                    alf.summary.scalar("cfm_loss_current/min", cfm_loss_current.min())
                    alf.summary.scalar("cfm_loss_current/max", cfm_loss_current.max())
                
                if initial_cfm_loss_flat.ndim > 1:
                    # If multi-dimensional, average over MC samples first
                    initial_cfm_loss_avg = initial_cfm_loss_flat.mean(dim=-1)
                    alf.summary.scalar("cfm_loss_initial/mean", initial_cfm_loss_avg.mean())
                    alf.summary.scalar("cfm_loss_initial/std", initial_cfm_loss_avg.std())
                    alf.summary.scalar("cfm_loss_initial/min", initial_cfm_loss_avg.min())
                    alf.summary.scalar("cfm_loss_initial/max", initial_cfm_loss_avg.max())
                else:
                    alf.summary.scalar("cfm_loss_initial/mean", initial_cfm_loss_flat.mean())
                    alf.summary.scalar("cfm_loss_initial/std", initial_cfm_loss_flat.std())
                    alf.summary.scalar("cfm_loss_initial/min", initial_cfm_loss_flat.min())
                    alf.summary.scalar("cfm_loss_initial/max", initial_cfm_loss_flat.max())
                
                # Critical diagnostic: CFM loss difference
                # Positive = current loss lower than old (CFM prediction improved)
                # Negative = current loss higher than old (may be OK if rewards improving)
                if cfm_loss_current.ndim > 1:
                    cfm_loss_diff = (initial_cfm_loss_avg - cfm_loss_avg).mean()
                else:
                    cfm_loss_diff = (initial_cfm_loss_flat - cfm_loss_current).mean()
                alf.summary.scalar("cfm_loss_diff/mean", cfm_loss_diff)  # Positive = CFM improvement
                # Note: In RL training, CFM loss may increase as policy optimizes for rewards.
                # Monitor rewards/advantages as primary signal; CFM loss is secondary.
                
                # Policy improvement indicator
                improvement_ratio = (rho > 1.0).float().mean()
                alf.summary.scalar("policy_improvement_ratio", improvement_ratio)  # Should be > 0.5
                
                # FPO loss diagnostics
                alf.summary.scalar("fpo_loss", fpo_loss)
                surrogate_loss1_mean = (rho * advantages_flat).mean()
                surrogate_loss2_mean = (rho_clipped * advantages_flat).mean()
                alf.summary.scalar("surrogate_loss1/mean", surrogate_loss1_mean)
                alf.summary.scalar("surrogate_loss2/mean", surrogate_loss2_mean)
                # Track which surrogate loss is being used (for debugging gradient flow)
                surrogate_used = (torch.minimum(rho * advantages_flat, rho_clipped * advantages_flat) == rho_clipped * advantages_flat).float().mean()
                alf.summary.scalar("surrogate_clipped_used_ratio", surrogate_used)  # Should be high when rho is outside bounds
                
                # Trajectory statistics (if available)
                if action_flat is not None:
                    alf.summary.scalar("trajectory/norm", torch.norm(action_flat))
                    alf.summary.scalar("trajectory/mean", action_flat.mean())
                    alf.summary.scalar("trajectory/std", action_flat.std())
                    alf.summary.scalar("trajectory/min", action_flat.min())
                    alf.summary.scalar("trajectory/max", action_flat.max())
                
                # Context statistics (if available)
                if context_flat is not None:
                    alf.summary.scalar("context/norm", torch.norm(context_flat))
                    alf.summary.scalar("context/mean", context_flat.mean())
                    alf.summary.scalar("context/std", context_flat.std())
                
                # Check for NaN/Inf
                if torch.isnan(rho).any():
                    alf.summary.scalar("nan_detected/rho", 1.0)
                if torch.isinf(rho).any():
                    alf.summary.scalar("inf_detected/rho", 1.0)
                if torch.isnan(advantages_flat).any():
                    alf.summary.scalar("nan_detected/advantages", 1.0)
                if torch.isinf(advantages_flat).any():
                    alf.summary.scalar("inf_detected/advantages", 1.0)
                if torch.isnan(fpo_loss):
                    alf.summary.scalar("nan_detected/fpo_loss", 1.0)
                if torch.isinf(fpo_loss):
                    alf.summary.scalar("inf_detected/fpo_loss", 1.0)

    def _create_old_trajectory_head(self):
        if not hasattr(self._actor_network, 'trajectory_head'):
            return None
        from copy import deepcopy
        current_head = self._actor_network.trajectory_head
        old_head = deepcopy(current_head)
        for param in old_head.parameters():
            param.requires_grad = False
            param.grad = None
        old_head.eval()
        return old_head
    
    def _prepare_fpo_loss_samples(self, generated_trajectory, observation, flow_steps=20):
        """
        Prepare FPO loss samples (loss_eps, loss_t) and compute initial CFM loss.
        
        Args:
            generated_trajectory: Generated trajectory from actor network [B, T, D]
            observation: Observation from TimeStep
            flow_steps: Number of flow matching steps [default: 20]
        Returns:
            tuple: (loss_eps, loss_t, initial_cfm_loss)
                - loss_eps: Noise samples for FPO loss [B, Nmc, D]
                - loss_t: Timestep samples for FPO loss [B, Nmc, 1]
                - initial_cfm_loss: Initial CFM loss computed with old policy [B, Nmc]
        """
        batch_size = generated_trajectory.shape[0]
        device = generated_trajectory.device
        
        # Downscale and flatten trajectory (only if scaling is enabled)
        # This is OmniAD-specific and should be disabled for other environments
        trajectory_downscaled = generated_trajectory.clone()
        # Check if trajectory head has scaling enabled (OmniAD-specific)
        enable_scaling = getattr(self._actor_network.trajectory_head, '_enable_trajectory_scaling', False)
        if enable_scaling and trajectory_downscaled.shape[-1] >= 2:
            trajectory_downscaled[..., 0] = (trajectory_downscaled[..., 0] - 20) / 20
            trajectory_downscaled[..., 1] = trajectory_downscaled[..., 1] / 20
        trajectory_flat = trajectory_downscaled.view(batch_size, -1)
        
        # Sample loss_eps and loss_t
        loss_eps = torch.randn(batch_size, self._n_mc_samples, trajectory_flat.shape[-1], device=device)
        if self._discretize_t_for_training:
            t_indices = torch.randint(0, flow_steps, (batch_size, self._n_mc_samples, 1), device=device)
            loss_t = (t_indices.float() / (flow_steps - 1)).to(device)
        else:
            loss_t = torch.rand(batch_size, self._n_mc_samples, 1, device=device)
        
        # Extract context_info from observation
        if isinstance(observation, dict):
            if 'representation' in observation:
                repr_data = observation['representation']
                if isinstance(repr_data, dict) and 'context_info' in repr_data:
                    context_info = repr_data['context_info']
                else:
                    context_info = repr_data
            elif 'context_info' in observation:
                context_info = observation['context_info']
            else:
                context_info = next(iter(observation.values()))
        else:
            context_info = observation
        if context_info.dim() > 2:
            context_info = context_info.view(context_info.shape[0], -1)
        
        # Compute initial CFM loss with old trajectory head
        with torch.no_grad():
            self._old_trajectory_head.eval()
            if self._use_noise_std_for_importance_ratio:
                # Compute CFM loss and log_probs for importance ratio (PHC-style)
                initial_cfm_loss, initial_log_probs = self._compute_cfm_loss(
                    action=trajectory_flat,
                    eps=loss_eps,
                    t=loss_t,
                    context=context_info,
                    trajectory_head=self._old_trajectory_head,
                    return_log_probs=True
                )
            else:
                # Compute only CFM loss (ALF-style)
                initial_cfm_loss = self._compute_cfm_loss(
                    action=trajectory_flat,
                    eps=loss_eps,
                    t=loss_t,
                    context=context_info,
                    trajectory_head=self._old_trajectory_head
                )
                initial_log_probs = None
        return loss_eps, loss_t, initial_cfm_loss, initial_log_probs
    
    def _update_old_model(self):
        """
        Update old model with hard copy: θ_old ← θ
        
        This is called after each training iteration to update the old policy.
        The old model is frozen (no gradients) and used for computing initial_cfm_loss
        during rollout and for importance ratio computation in FPO loss.
        """
        if self._old_trajectory_head is None:
            self._old_trajectory_head = self._create_old_trajectory_head()
            return
        if not hasattr(self._actor_network, 'trajectory_head'):
            return
        
        # Ensure old model is frozen before updating
        with torch.no_grad():
            current_params = dict(self._actor_network.trajectory_head.named_parameters())
            old_params = dict(self._old_trajectory_head.named_parameters())
            for name, old_param in old_params.items():
                if name in current_params:
                    # Hard copy: θ_old ← θ (not weighted average like NFT)
                    old_param.data.copy_(current_params[name].data)
        
        # Ensure old model stays in eval mode and frozen
        self._old_trajectory_head.eval()
        for param in self._old_trajectory_head.parameters():
            param.requires_grad = False
    
    def predict_step(self, inputs: TimeStep, state):
        """
        Predict for one step during evaluation.
        
        FlowMatchingActorNetwork returns a trajectory tensor, not an action distribution.
        This method extracts the trajectory and returns it as the action.
        For single-step actions, extracts the first step from the trajectory.
        """
        if self._predict_ema_id == -1:
            predict_model = self._actor_network
        else:
            predict_model = self._actor_emas[self._predict_ema_id]
        
        with torch.no_grad():
            actor_outputs, actor_state = predict_model(inputs.observation, state=state.actor)
        
        # Extract trajectory from actor output
        if isinstance(actor_outputs, dict) and 'trajectory' in actor_outputs:
            generated_trajectory = actor_outputs['trajectory']
        else:
            generated_trajectory = actor_outputs
        
        # For single-step actions, extract the first step from trajectory [B, num_poses, action_dim] -> [B, action_dim]
        # If trajectory is [B, num_poses, action_dim] and num_poses > 1, take first step
        if generated_trajectory.ndim == 3 and generated_trajectory.shape[1] > 1:
            action = generated_trajectory[:, 0, :]  # [B, action_dim]
        elif generated_trajectory.ndim == 3 and generated_trajectory.shape[1] == 1:
            action = generated_trajectory.squeeze(1)  # [B, 1, action_dim] -> [B, action_dim]
        else:
            # Already in correct shape [B, action_dim]
            action = generated_trajectory
        
        return AlgStep(
            output=action,
            state=state._replace(actor=actor_state),
            info=FPOInfo(
                action=common.detach(action),
                log_prob=(),
                value=(),
                step_type=inputs.step_type,
                reward=inputs.reward,
                discount=inputs.discount,
                action_distribution=actor_outputs,
                reward_weights=(),
                entropy=(),
                loss_eps=(),
                loss_t=(),
                initial_cfm_loss=(),
                initial_log_probs=(),
                ppo_advantages=(),
                observation=inputs.observation,
                returns=(),
                v_pred=(),
                v_target=(),
                valid_target=(),
                ppo_value=()
            )
        )
    
    def rollout_step(self, inputs: TimeStep, state):
        current_head = self._actor_network.trajectory_head
        try:
            self._actor_network.trajectory_head = self._old_trajectory_head
            with torch.no_grad():
                actor_outputs, actor_state = self._actor_network(inputs.observation)
        finally:
            # CRITICAL: Restore original trajectory_head and ensure it's trainable
            self._actor_network.trajectory_head = current_head
            # Ensure trajectory_head is in train mode and has gradients enabled
            if hasattr(self._actor_network, 'trajectory_head') and self._actor_network.trajectory_head is not None:
                self._actor_network.trajectory_head.train()
                for param in self._actor_network.trajectory_head.parameters():
                    param.requires_grad = True
        if isinstance(actor_outputs, dict) and 'trajectory' in actor_outputs:
            generated_trajectory = actor_outputs['trajectory']
        else:
            generated_trajectory = actor_outputs
        
        # Prepare observation for value network based on its input_tensor_spec
        # Check what field the value network expects from its input_tensor_spec
        value_network_spec = self._value_network.input_tensor_spec
        expected_field = None
        if isinstance(value_network_spec, dict):
            if 'representation' in value_network_spec:
                repr_spec = value_network_spec['representation']
                if isinstance(repr_spec, dict):
                    # Check which field the value network expects
                    if 'latent_bev_feature' in repr_spec:
                        # BEVValueNetwork expects latent_bev_feature
                        expected_field = 'latent_bev_feature'
                    elif 'context_info' in repr_spec:
                        # Standard ValueNetwork expects context_info
                        expected_field = 'context_info'
                    else:
                        # Fallback: use first available field
                        expected_field = next(iter(repr_spec.keys())) if repr_spec else None
        
        # Extract the appropriate field from observation
        if isinstance(inputs.observation, dict):
            if 'representation' in inputs.observation:
                repr_data = inputs.observation['representation']
                if isinstance(repr_data, dict):
                    if expected_field and expected_field in repr_data:
                        # Extract the specific field the value network expects
                        value_observation = {'representation': {expected_field: repr_data[expected_field]}}
                    elif 'context_info' in repr_data:
                        # Fallback to context_info
                        value_observation = {'representation': {'context_info': repr_data['context_info']}}
                    else:
                        # Use representation as-is
                        value_observation = {'representation': repr_data}
                else:
                    # representation is not a dict, wrap it
                    if expected_field == 'context_info':
                        value_observation = {'representation': {'context_info': repr_data}}
                    else:
                        value_observation = {'representation': repr_data}
            elif expected_field and expected_field in inputs.observation:
                # Extract the specific field
                value_observation = {'representation': {expected_field: inputs.observation[expected_field]}}
            elif 'context_info' in inputs.observation:
                # Fallback to context_info
                value_observation = {'representation': {'context_info': inputs.observation['context_info']}}
            else:
                # Use observation as-is
                value_observation = inputs.observation
        else:
            # inputs.observation is a tensor
            if expected_field == 'context_info':
                value_observation = {'representation': {'context_info': inputs.observation}}
            else:
                value_observation = {'representation': {'context_info': inputs.observation}}
        
        value_output, value_state = self._value_network(value_observation, state.value)
        
        # Prepare FPO loss samples and compute initial CFM loss
        result = self._prepare_fpo_loss_samples(
            generated_trajectory, inputs.observation
        )
        if len(result) == 4:
            loss_eps, loss_t, initial_cfm_loss, initial_log_probs = result
        else:
            # Backward compatibility: handle old return format
            loss_eps, loss_t, initial_cfm_loss = result
            initial_log_probs = None
        if self.has_multidim_reward():
            reward_weights = tensor_utils.tensor_extend_new_dim(
                self.reward_weights, dim=0, n=value_output.shape[0])
        else:
            reward_weights = ()
        
        # Extract single-step action from trajectory for environment
        # For single-step actions, extract the first step from trajectory [B, num_poses, action_dim] -> [B, action_dim]
        if generated_trajectory.ndim == 3 and generated_trajectory.shape[1] > 1:
            action = generated_trajectory[:, 0, :]  # [B, action_dim]
        elif generated_trajectory.ndim == 3 and generated_trajectory.shape[1] == 1:
            action = generated_trajectory.squeeze(1)  # [B, 1, action_dim] -> [B, action_dim]
        else:
            # Already in correct shape [B, action_dim]
            action = generated_trajectory
        
        return AlgStep(
            output=action,
            state=state._replace(
                actor=actor_state,
                value=value_state
            ),
            info=FPOInfo(
                action=common.detach(generated_trajectory),
                log_prob=(),
                value=value_output,
                step_type=inputs.step_type,
                reward=inputs.reward,
                discount=inputs.discount,
                action_distribution=actor_outputs,
                reward_weights=reward_weights,
                entropy=(),
                loss_eps=loss_eps,
                loss_t=loss_t,
                initial_cfm_loss=initial_cfm_loss,
                initial_log_probs=initial_log_probs if initial_log_probs is not None else (),
                ppo_advantages=(),
                observation=inputs.observation,
                returns=(),
                v_pred=(),  # Will be filled in train_step
                v_target=(),  # Will be filled in train_step
                valid_target=(),  # Will be filled in train_step (not available during rollout)
                ppo_value=()  # Will be filled in preprocess_experience
            )
        )
    
    def train_step(self, inputs: TimeStep, state, rollout_info):
        """
        FPO training step - follows PPO off-policy pattern
        
        Recomputes value network and actor network with targets to get v_pred and v_target.
        """
        # Prepare observation for value network based on its input_tensor_spec
        # Check what field the value network expects from its input_tensor_spec
        value_network_spec = self._value_network.input_tensor_spec
        expected_field = None
        if isinstance(value_network_spec, dict):
            if 'representation' in value_network_spec:
                repr_spec = value_network_spec['representation']
                if isinstance(repr_spec, dict):
                    # Check which field the value network expects
                    if 'latent_bev_feature' in repr_spec:
                        # BEVValueNetwork expects latent_bev_feature
                        expected_field = 'latent_bev_feature'
                    elif 'context_info' in repr_spec:
                        # Standard ValueNetwork expects context_info
                        expected_field = 'context_info'
                    else:
                        # Fallback: use first available field
                        expected_field = next(iter(repr_spec.keys())) if repr_spec else None
        
        # Extract the appropriate field from observation
        if isinstance(inputs.observation, dict):
            if 'representation' in inputs.observation:
                repr_data = inputs.observation['representation']
                if isinstance(repr_data, dict):
                    if expected_field and expected_field in repr_data:
                        # Extract the specific field the value network expects
                        value_observation = {'representation': {expected_field: repr_data[expected_field]}}
                    elif 'context_info' in repr_data:
                        # Fallback to context_info
                        value_observation = {'representation': {'context_info': repr_data['context_info']}}
                    else:
                        # Use representation as-is
                        value_observation = {'representation': repr_data}
                else:
                    # representation is not a dict, wrap it
                    if expected_field == 'context_info':
                        value_observation = {'representation': {'context_info': repr_data}}
                    else:
                        value_observation = {'representation': repr_data}
            elif expected_field and expected_field in inputs.observation:
                # Extract the specific field
                value_observation = {'representation': {expected_field: inputs.observation[expected_field]}}
            elif 'context_info' in inputs.observation:
                # Fallback to context_info
                value_observation = {'representation': {'context_info': inputs.observation['context_info']}}
            else:
                # Use observation as-is
                value_observation = inputs.observation
        else:
            # inputs.observation is a tensor
            if expected_field == 'context_info':
                value_observation = {'representation': {'context_info': inputs.observation}}
            else:
                value_observation = {'representation': {'context_info': inputs.observation}}
        
        value_output, value_state = self._value_network(value_observation, state.value)
        
        # PPO uses the same value network as main algorithm (shared)
        # So value_output is used for both TD loss and PPO advantages
        
        # Extract gt_trajectory from inputs.observation (stored by repr_learner when _repr_mode=True)
        # The structure is: inputs.observation['representation']['gt_trajectory']
        gt_trajectory = None
        if isinstance(inputs.observation, dict) and 'representation' in inputs.observation:
            repr_data = inputs.observation['representation']
            if isinstance(repr_data, dict) and 'gt_trajectory' in repr_data:
                gt_trajectory = repr_data['gt_trajectory']
        
        # Extract valid_target from rollout_info (set by hybrid_rl_agent during rollout)
        # rollout_info.valid_target is populated in hybrid_rl_agent._step()
        valid_target = getattr(rollout_info, 'valid_target', ())
        
        # If we have gt_trajectory, recompute actor_step with targets to get v_pred and v_target
        if gt_trajectory is not None:
            targets = {'trajectory': gt_trajectory}
            
            # Call actor network with targets to compute velocities
            # Note: actor_inputs is the processed representation from rollout_step
            actor_inputs = inputs.observation
            if isinstance(actor_inputs, dict):
                actor_inputs = actor_inputs.copy()
                actor_inputs['targets'] = targets
            
            # Call actor network forward (returns tuple: (output, state))
            actor_output, _ = self._actor_network(actor_inputs)
            
            # Extract v_pred and v_target from actor output
            if isinstance(actor_output, dict):
                v_pred = actor_output.get('v_pred', actor_output.get('representation', {}).get('v_pred'))
                v_target = actor_output.get('v_target', actor_output.get('representation', {}).get('v_target'))
                
                # Store in rollout_info for loss calculation
                rollout_info = rollout_info._replace(v_pred=v_pred, v_target=v_target)

        # Return rollout info for loss computation
        return AlgStep(
            output=rollout_info.action,
            state=state._replace(value=value_state),
            info=rollout_info._replace(
                step_type=inputs.step_type,
                reward=inputs.reward,
                discount=inputs.discount,
                value=value_output,
                reward_weights=getattr(rollout_info, 'reward_weights', ()),
                v_pred=getattr(rollout_info, 'v_pred', ()),
                v_target=getattr(rollout_info, 'v_target', ()),
                valid_target=valid_target,  # Extracted from observation in train_step
                ppo_value=()  # Not needed since PPO uses same value network
            )
        )
    
    def _compute_advantage(self, ppo_info):
        """
        Compute GAE advantages from PPO info.
        
        Returns:
            tuple: (ppo_advantages_for_fpo, advantages_for_logging)
                - ppo_advantages_for_fpo: Normalized advantages for FPO loss
                - advantages_for_logging: Unnormalized advantages for logging
        """
        norm_ppo_advantages = ppo_info.normalized_advantages
        if norm_ppo_advantages == ():
            # Fallback to unnormalized advantages
            return ppo_info.advantages, ppo_info.advantages
        else:
            # Use normalized GAE advantages
            return norm_ppo_advantages, ppo_info.advantages
    
    def _compute_mc_return(self, rollout_info, ppo_info):
        """
        Compute Monte Carlo returns from rewards and values.
        
        Returns:
            tuple: (mc_returns_for_fpo, mc_returns_for_logging)
                - mc_returns_for_fpo: Scalar MC returns for FPO loss [B, T]
                - mc_returns_for_logging: Scalar MC returns for logging [B, T]
        """
        # Get necessary info for computing MC returns (ensure all on same device)
        value = convert_device(rollout_info.value)  # [T, B] or [B, T] - original value from rollout
        step_type = convert_device(rollout_info.step_type) if hasattr(rollout_info, 'step_type') else None
        discount = convert_device(rollout_info.discount) if hasattr(rollout_info, 'discount') else None
        reward = convert_device(rollout_info.reward) if hasattr(rollout_info, 'reward') else None
        
        if step_type is not None and discount is not None and reward is not None:
            # Compute Monte Carlo returns (discounted_return uses value only at termination)
            # Ensure all tensors are on the same device
            if isinstance(reward, torch.Tensor):
                target_device = reward.device
                if isinstance(value, torch.Tensor):
                    value = value.to(target_device)
                if isinstance(discount, torch.Tensor):
                    discount = discount.to(target_device)
                if isinstance(step_type, torch.Tensor):
                    step_type = step_type.to(target_device)
            
            gamma = self._loss._gamma if hasattr(self._loss, '_gamma') else 0.99
            if reward.ndim == 3:
                discounts = discount.unsqueeze(-1) * gamma
            else:
                discounts = discount * gamma
            
            mc_returns = value_ops.discounted_return(
                rewards=reward,
                values=value,
                step_types=step_type,
                discounts=discounts,
                time_major=False  # PPO uses batch-major: inputs [B, T], outputs [B, T-1] or [B, T-1, D]
            )
            
            # Pad mc_returns to match value's T timesteps: [B, T-1] -> [B, T]
            if mc_returns.ndim == 2:
                mc_returns_padded = torch.cat([mc_returns, value[:, -1:]], dim=1)  # [B, T]
            elif mc_returns.ndim == 3:
                mc_returns_padded = torch.cat([mc_returns, value[:, -1:, :]], dim=1)  # [B, T, D]
            else:
                mc_returns_padded = mc_returns
            
            # Reduce multidimensional returns to scalar if needed
            reward_weights = getattr(rollout_info, 'reward_weights', ())
            if reward_weights == ():
                reward_weights = getattr(ppo_info, 'reward_weights', ())
            
            if mc_returns_padded.ndim == 3 and reward_weights != ():
                if isinstance(reward_weights, torch.Tensor):
                    reward_weights = reward_weights.to(mc_returns_padded.device)
                if reward_weights.ndim == 1:
                    mc_returns_scalar = (mc_returns_padded * reward_weights.view(1, 1, -1)).sum(-1)
                elif reward_weights.ndim == 2:
                    mc_returns_scalar = (mc_returns_padded * reward_weights.unsqueeze(1)).sum(-1)
                elif reward_weights.ndim == 3:
                    mc_returns_scalar = (mc_returns_padded * reward_weights).sum(-1)
                else:
                    mc_returns_scalar = mc_returns_padded.mean(-1)
            elif mc_returns_padded.ndim == 3:
                mc_returns_scalar = mc_returns_padded.mean(-1)
            else:
                mc_returns_scalar = mc_returns_padded
            
            return mc_returns_scalar, mc_returns_scalar
        else:
            # Fallback: use returns from PPO info
            returns = ppo_info.returns
            return returns, returns
    
    def _compute_mc_advantage(self, rollout_info, ppo_info):
        """
        Compute Monte Carlo advantages: A = MC_returns - value.
        
        Returns:
            tuple: (mc_advantages_for_fpo, mc_advantages_for_logging)
                - mc_advantages_for_fpo: Normalized MC advantages for FPO loss [B, T]
                - mc_advantages_for_logging: Unnormalized MC advantages for logging [B, T]
        """
        # Get necessary info for computing MC returns (ensure all on same device)
        value = convert_device(rollout_info.value)  # [T, B] or [B, T] - original value from rollout
        step_type = convert_device(rollout_info.step_type) if hasattr(rollout_info, 'step_type') else None
        discount = convert_device(rollout_info.discount) if hasattr(rollout_info, 'discount') else None
        reward = convert_device(rollout_info.reward) if hasattr(rollout_info, 'reward') else None
        
        if step_type is not None and discount is not None and reward is not None:
            # Compute Monte Carlo returns (discounted_return uses value only at termination)
            # Ensure all tensors are on the same device
            if isinstance(reward, torch.Tensor):
                target_device = reward.device
                if isinstance(value, torch.Tensor):
                    value = value.to(target_device)
                if isinstance(discount, torch.Tensor):
                    discount = discount.to(target_device)
                if isinstance(step_type, torch.Tensor):
                    step_type = step_type.to(target_device)
            
            gamma = self._loss._gamma if hasattr(self._loss, '_gamma') else 0.99
            if reward.ndim == 3:
                discounts = discount.unsqueeze(-1) * gamma
            else:
                discounts = discount * gamma
            
            mc_returns = value_ops.discounted_return(
                rewards=reward,
                values=value,
                step_types=step_type,
                discounts=discounts,
                time_major=False  # PPO uses batch-major: inputs [B, T], outputs [B, T-1] or [B, T-1, D]
            )
            
            # Compute MC advantages: A = MC_returns - value (first T-1 timesteps)
            # mc_returns: [B, T-1, D] or [B, T-1] (already detached from discounted_return)
            # value: [B, T, D] or [B, T] (from rollout, should be detached to prevent gradients)
            # Detach value to ensure advantages don't have gradients from value network
            value_detached = value.detach()
            if mc_returns.ndim == 3 and value_detached.ndim == 3:
                # Both are multidimensional: [B, T-1, D] and [B, T, D]
                # Compute advantages for first T-1 timesteps
                mc_advantages = mc_returns - value_detached[:, :-1, :]  # [B, T-1, D]
            elif mc_returns.ndim == 2 and value_detached.ndim == 2:
                # Both are scalar: [B, T-1] and [B, T]
                mc_advantages = mc_returns - value_detached[:, :-1]  # [B, T-1]
            else:
                # Shape mismatch: align timesteps
                if mc_returns.ndim == 3 and value_detached.ndim == 3:
                    mc_advantages = mc_returns - value_detached[:, :mc_returns.shape[1], :]
                else:
                    mc_advantages = mc_returns - value_detached[:, :mc_returns.shape[1]]
            
            # Reduce multidimensional advantages to scalar (similar to normalized_advantages)
            # Get reward_weights if available (from rollout_info or ppo_info)
            reward_weights = getattr(rollout_info, 'reward_weights', ())
            if reward_weights == ():
                reward_weights = getattr(ppo_info, 'reward_weights', ())
            
            # Ensure reward_weights is on the same device as mc_advantages
            if reward_weights != () and isinstance(reward_weights, torch.Tensor):
                reward_weights = reward_weights.to(mc_advantages.device)
            
            if mc_advantages.ndim == 3 and reward_weights != ():
                # Multidimensional advantages: reduce by weighted sum (same as normalized_advantages)
                # reward_weights: [D] or [B, D] or [B, T, D]
                if reward_weights.ndim == 1:
                    # [D] -> broadcast to [B, T-1, D]
                    mc_advantages_scalar = (mc_advantages * reward_weights.view(1, 1, -1)).sum(-1)  # [B, T-1]
                elif reward_weights.ndim == 2:
                    # [B, D] -> broadcast to [B, T-1, D]
                    mc_advantages_scalar = (mc_advantages * reward_weights.unsqueeze(1)).sum(-1)  # [B, T-1]
                elif reward_weights.ndim == 3:
                    # [B, T, D] -> use first T-1 timesteps
                    mc_advantages_scalar = (mc_advantages * reward_weights[:, :-1, :]).sum(-1)  # [B, T-1]
                else:
                    # Fallback: simple mean over reward dimension
                    mc_advantages_scalar = mc_advantages.mean(-1)  # [B, T-1]
            elif mc_advantages.ndim == 3:
                # Multidimensional but no reward_weights: use mean (fallback)
                mc_advantages_scalar = mc_advantages.mean(-1)  # [B, T-1]
            else:
                # Already scalar: [B, T-1]
                mc_advantages_scalar = mc_advantages  # [B, T-1]
            
            # Extend scalar advantages with zero along time dimension (same as normalized_advantages)
            # Pad with zero to match value's T timesteps: [B, T-1] -> [B, T]
            if mc_advantages_scalar.ndim == 2:
                # Batch-major: pad along dim=1 (time dimension)
                simple_advantage = tensor_utils.tensor_extend_zero(mc_advantages_scalar, dim=1)  # [B, T]
            else:
                # Already correct shape or need different handling
                simple_advantage = mc_advantages_scalar
            
            # Normalize simple advantage (similar to GAE normalization)
            # If normalizing, use batch normalization
            if self._loss.normalizing_scalar_advantages or self._loss.normalizing_advantages:
                # Use the same normalization as GAE advantages
                if hasattr(ppo_info, 'normalized_advantages') and ppo_info.normalized_advantages != ():
                    # Scale simple advantage to match GAE's normalization
                    # This approximates normalization by normalizing simple_advantage
                    adv_mean = simple_advantage.mean()
                    adv_std = simple_advantage.std() + 1e-8
                    normalized_simple_advantage = (simple_advantage - adv_mean) / adv_std
                else:
                    normalized_simple_advantage = simple_advantage
            else:
                normalized_simple_advantage = simple_advantage
            
            return normalized_simple_advantage, simple_advantage
        else:
            # Fallback: use returns - value (which equals advantages if returns = value + advantages)
            returns = ppo_info.returns
            if returns.ndim != value.ndim or returns.shape != value.shape:
                value = tensor_utils.tensor_extend_zero(value, dim=1) if value.ndim == 2 else value
            simple_advantage = returns - value
            return simple_advantage, simple_advantage
    
    def preprocess_experience(self, root_inputs: TimeStep, rollout_info, batch_info):
        """
        Compute reward signal for FPO loss based on reward_mode:
        - "advantage": GAE advantages (default, lower variance)
        - "mc_advantage": Monte Carlo advantages (MC returns - value, higher variance but unbiased)
        - "mc_return": Monte Carlo returns directly (no value function subtraction)
        """
        if self._use_ppo_advantages and self._ppo_algorithm is not None:
            # Use PPO's advantage calculation
            root_inputs, ppo_info = self._ppo_algorithm.preprocess_experience(
                root_inputs, rollout_info, batch_info
            )
            
            # Choose reward signal based on reward_mode
            if self._reward_mode == "advantage":
                ppo_advantages_for_fpo, advantages_for_logging = self._compute_advantage(ppo_info)
            elif self._reward_mode == "mc_advantage":
                ppo_advantages_for_fpo, advantages_for_logging = self._compute_mc_advantage(rollout_info, ppo_info)
            elif self._reward_mode == "mc_return":
                ppo_advantages_for_fpo, advantages_for_logging = self._compute_mc_return(rollout_info, ppo_info)
            else:
                # Fallback to GAE advantages if reward_mode is invalid (shouldn't happen due to assertion)
                ppo_advantages_for_fpo, advantages_for_logging = self._compute_advantage(ppo_info)
            
            # Monitor advantages for debugging divergence
            if alf.summary.should_record_summaries():
                with alf.summary.scope("debug_fpo"):
                    method = self._reward_mode
                    if isinstance(advantages_for_logging, torch.Tensor):
                        alf.summary.scalar(f"advantages_{method}/mean", advantages_for_logging.mean())
                        alf.summary.scalar(f"advantages_{method}/std", advantages_for_logging.std())
                        alf.summary.scalar(f"advantages_{method}/min", advantages_for_logging.min())
                        alf.summary.scalar(f"advantages_{method}/max", advantages_for_logging.max())
                    
                    # Returns statistics
                    if isinstance(ppo_info.returns, torch.Tensor):
                        alf.summary.scalar("returns/mean", ppo_info.returns.mean())
                        alf.summary.scalar("returns/std", ppo_info.returns.std())
                        alf.summary.scalar("returns/min", ppo_info.returns.min())
                        alf.summary.scalar("returns/max", ppo_info.returns.max())
            
            # PPO now uses the same value network as main algorithm (shared)
            # Store returns for TD loss computation
            rollout_info = rollout_info._replace(
                ppo_advantages=ppo_advantages_for_fpo,
                returns=ppo_info.returns,
                ppo_value=()  # Not needed since PPO uses same value network
            )
        return root_inputs, rollout_info
    
    def _scheduled_fpo_loss_weight(self, flat: int = 100) -> float:
        """
        Return FPO loss weight with a flat warmup: 0 for the first `flat` iterations,
        then jump to the configured `self._fpo_loss_weight`.

        Uses the algorithm's update counter to determine the current iteration.
        """
        if self._update_counter < flat:
            return 0.0
        return self._fpo_loss_weight

    def after_train_iter(self, root_inputs, rollout_info):
        super().after_train_iter(root_inputs, rollout_info)
        self._update_old_model()
        self._update_counter += 1


