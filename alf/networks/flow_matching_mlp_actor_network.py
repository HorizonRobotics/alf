"""
Flow Matching MLP Actor Network

This module implements an actor network that matches the PHC FlowMatchingPolicy architecture,
using MLP + adaLN instead of DiT for velocity prediction.

Architecture:
- FlowMatchingMLPActorNetwork: Provides ALF interface, calls trajectory_head in forward()
- MLPTrajectoryHead: Contains flow matching network (actor_mlp, actor_norm, noise_emb, mu, obs_norm)
                     and implements sample_actions functionality from PHC's FlowMatchingPolicy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable
import math

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. ODE solver will use simple Euler step.")


class TimestepEmbedder(nn.Module):
    """Timestep embedding for flow matching, matching PHC implementation."""
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings (matching PHC)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] tensor of timesteps in [0, 1]
        Returns:
            embeddings: [B, hidden_size] tensor of timestep embeddings
        """
        t_freq = self.timestep_embedding(timesteps, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization (adaLN) matching PHC implementation."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Learnable modulation parameters (matching PHC: 2 * hidden_size for shift and scale)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # Zero-initialize last layer (matching PHC)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x: torch.Tensor, timestep_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, hidden_size] input features
            timestep_emb: [B, hidden_size] or [B, 1, hidden_size] timestep embeddings
        Returns:
            normalized: [B, hidden_size] normalized and modulated features
        """
        # Handle timestep_emb shape
        if timestep_emb.dim() == 3:
            timestep_emb = timestep_emb.squeeze(1)
        
        # Get modulation parameters from timestep embedding
        shift, scale = self.adaLN_modulation(timestep_emb).chunk(2, dim=1)
        
        # Apply layer norm with adaptive shift and scale
        x_norm = self.norm1(x)
        x = (1 + scale) * x_norm + shift
        
        return x


def layer_init(layer, std: float = None):
    """Initialize layer weights, matching PHC's layer_init."""
    if std is None:
        # Default initialization
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
    else:
        nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0)
    return layer


class RunningNorm(nn.Module):
    """Running normalization matching PHC's observation normalization."""
    
    def __init__(self, num_features: int, momentum: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('count', torch.tensor(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input using running statistics.
        
        Args:
            x: [B, num_features] input tensor
        Returns:
            normalized: [B, num_features] normalized tensor
        """
        if self.training:
            # Update running statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            if self.count == 0:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
            else:
                self.running_mean.mul_(self.momentum).add_(batch_mean * (1 - self.momentum))
                self.running_var.mul_(self.momentum).add_(batch_var * (1 - self.momentum))
            
            self.count += x.shape[0]
        
        # Normalize
        normalized = (x - self.running_mean) / (torch.sqrt(self.running_var) + self.eps)
        return normalized


class CondOTProbPath:
    """
    Conditional Optimal Transport Probability Path matching PHC implementation.
    Used for computing target velocity u_t = actions - noise.
    
    Uses linear scheduler: alpha_t = t, sigma_t = 1 - t
    This gives: x_t = t * x_1 + (1 - t) * x_0
    and: dx_t = x_1 - x_0
    """
    
    def __init__(self):
        pass
    
    def sample(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor):
        """
        Sample from the conditional OT path.
        
        Args:
            t: [B] timesteps in [0, 1]
            x_0: [B, action_dim] source (noise)
            x_1: [B, action_dim] target (actions)
        
        Returns:
            x_t: [B, action_dim] interpolated sample
            dx_t: [B, action_dim] target velocity (x_1 - x_0)
        """
        # Linear interpolation: x_t = t * x_1 + (1 - t) * x_0
        t_expanded = t.view(-1, 1) if t.dim() == 1 else t
        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        # Target velocity: dx_t = x_1 - x_0
        dx_t = x_1 - x_0
        return type('PathSample', (), {'x_t': x_t, 'dx_t': dx_t})()
    
    def target_to_velocity(self, x_1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convert from target x_1 to velocity (matching PHC).
        
        For linear scheduler: alpha_t = t, sigma_t = 1 - t
        dx_t = x_1 - x_0 = x_1 - (x_t - t * x_1) / (1 - t)
        Simplifying: dx_t = (x_1 - x_t) / (1 - t) when t != 1
        
        Args:
            x_1: [B, action_dim] target data point
            x_t: [B, action_dim] path sample at time t
            t: [B, 1] time in [0, 1]
        
        Returns:
            velocity: [B, action_dim] velocity
        """
        # For linear scheduler: dx_t = x_1 - x_0
        # From x_t = t * x_1 + (1 - t) * x_0, we get:
        # x_0 = (x_t - t * x_1) / (1 - t)
        # So: dx_t = x_1 - (x_t - t * x_1) / (1 - t)
        # Simplifying: dx_t = (x_1 - x_t) / (1 - t)
        t_expanded = t.view(-1, 1) if t.dim() == 1 else t
        # Avoid division by zero when t = 1
        eps = 1e-8
        velocity = (x_1 - x_t) / (1 - t_expanded + eps)
        return velocity


class ODESolver:
    """
    ODE Solver for flow matching, matching PHC implementation.
    Uses torchdiffeq if available, otherwise falls back to simple Euler step.
    """
    
    def __init__(self):
        pass
    
    def sample(self,
               velocity_fn: Callable,
               time_grid: torch.Tensor,
               x_init: torch.Tensor,
               method: str = "euler",
               step_size: float = 0.1,
               atol: float = 1e-5,
               rtol: float = 1e-5,
               return_intermediates: bool = False,
               **model_extras) -> torch.Tensor:
        """
        Solve ODE using velocity function.
        
        Args:
            velocity_fn: Function (x, t, **extras) -> velocity
            time_grid: [2] tensor with [t_start, t_end], typically [0.0, 1.0]
            x_init: [B, action_dim] initial condition
            method: ODE solver method
            step_size: Step size for Euler method
            atol: Absolute tolerance
            rtol: Relative tolerance
            return_intermediates: Whether to return intermediate steps
            **model_extras: Additional arguments for velocity_fn
        
        Returns:
            x_final: [B, action_dim] final solution
        """
        if TORCHDIFFEQ_AVAILABLE:
            # Use torchdiffeq for accurate ODE solving
            def ode_func(t, x):
                # t is a scalar, x is [B, action_dim]
                t_batch = torch.ones(x.shape[0], device=x.device) * t.item()
                return velocity_fn(x, t_batch, **model_extras)
            
            ode_opts = {"step_size": step_size} if step_size is not None else {}
            
            with torch.no_grad():
                sol = odeint(
                    ode_func,
                    x_init,
                    time_grid,
                    method=method,
                    options=ode_opts,
                    atol=atol,
                    rtol=rtol,
                )
            
            if return_intermediates:
                return sol
            else:
                return sol[-1]
        else:
            # Fallback: Simple Euler step
            x = x_init
            t_start, t_end = time_grid[0].item(), time_grid[1].item()
            num_steps = max(1, int((t_end - t_start) / step_size))
            dt = (t_end - t_start) / num_steps
            
            for i in range(num_steps):
                t = t_start + i * dt
                t_batch = torch.ones(x.shape[0], device=x.device) * t
                velocity = velocity_fn(x, t_batch, **model_extras)
                x = x + dt * velocity
            
            return x


class MLPTrajectoryHead(nn.Module):
    """
    MLP-based Flow Matching Trajectory Head matching PHC's FlowMatchingPolicy.
    
    This contains all the flow matching network layers and implements:
    - diffusion_model(): Predicts velocity given (x_t, context, t)
    - forward(): Training/inference forward pass
    - sample_actions(): Action sampling using ODE solver (matching PHC's sample_actions)
    """
    
    def __init__(self,
                 input_size: int,
                 action_size: int,
                 hidden_size: int = 512,
                 parameterization: str = "velocity",
                 zero_action_input: bool = False,
                 prior_noise_std: float = 1.0,
                 solver_step_size: float = 0.1,
                 condition_drop_ratio: float = 0.0,
                 num_envs: int = 4096,
                 **kwargs):
        """
        Initialize MLP trajectory head.
        
        Args:
            input_size: Observation dimension
            action_size: Action dimension
            hidden_size: Hidden dimension for MLP (default: 512)
            parameterization: "velocity" or "data" (default: "velocity")
            zero_action_input: If True, zero out action input (default: False)
            prior_noise_std: Standard deviation for prior noise (default: 1.0)
            solver_step_size: Step size for ODE solver (default: 0.1)
            condition_drop_ratio: Condition dropout ratio (default: 0.0)
            num_envs: Number of environments for dropout mask (default: 4096)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__()
        
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.parameterization = parameterization
        self.zero_action_input = zero_action_input
        self.prior_noise_std = prior_noise_std
        self.solver_step_size = solver_step_size
        self.condition_drop_ratio = condition_drop_ratio
        
        # Observation normalization (matching PHC)
        self.obs_norm = RunningNorm(input_size)
        
        # Actor MLP (matching PHC architecture)
        # Input: [x_t (or zeros), obs] concatenated -> [input_size + action_size]
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(input_size + action_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
        )
        
        # Adaptive Layer Normalization
        self.actor_norm = AdaptiveLayerNorm(hidden_size)
        self.post_adaln_non_linearity = nn.SiLU()
        
        # Noise embedder for timestep conditioning
        self.noise_emb = TimestepEmbedder(hidden_size)
        
        # Final output layer (velocity or data prediction)
        self.mu = nn.Sequential(
            layer_init(nn.Linear(hidden_size, action_size), std=0.01),
        )
        
        # ODE solver and path for action sampling
        self.solver = ODESolver()
        self.path = CondOTProbPath()
        
        # Condition dropout mask (if needed)
        if condition_drop_ratio > 0:
            self.sample_mask = torch.bernoulli(torch.ones(num_envs, 1) * condition_drop_ratio) * torch.ones(num_envs, input_size + action_size)
        else:
            self.sample_mask = None
    
    def _compute_velocity(self, 
                         x_t: torch.Tensor, 
                         context: torch.Tensor, 
                         t: torch.Tensor,
                         condition_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Core velocity computation function used by both diffusion_model() and sample_actions().
        
        This is the shared implementation that computes velocity from (x_t, context, t).
        
        Args:
            x_t: [B, action_dim] - interpolated action at timestep t
            context: [B, input_size] - observation/context
            t: [B] - timestep in [0, 1]
            condition_mask: Optional [B, input_size + action_size] condition dropout mask
        
        Returns:
            velocity: [B, action_dim] - predicted velocity
        """
        # Normalize observation
        obs_pointer = self.obs_norm(context)
        
        # Zero out action input if configured
        x_t_eff = torch.zeros_like(x_t) if self.zero_action_input else x_t
        
        # Concatenate action and observation
        x_inp = torch.cat([x_t_eff, obs_pointer], dim=-1)  # [B, action_size + input_size]
        
        # Apply condition dropout mask if provided
        if condition_mask is not None:
            x_inp = x_inp * condition_mask
        
        # Get timestep embedding
        # Scale timestep by 0.0 if zero_action_input, else 1.0 (matching PHC)
        t_scaled = t * (0.0 if self.zero_action_input else 1.0)
        noise_emb = self.noise_emb(t_scaled)  # [B, hidden_size]
        
        # Forward through actor MLP
        hidden = self.actor_mlp(x_inp)  # [B, hidden_size]
        
        # Apply adaptive layer normalization
        hidden = self.actor_norm(hidden, noise_emb)  # [B, hidden_size]
        hidden = self.post_adaln_non_linearity(hidden)  # [B, hidden_size]
        
        # Predict velocity or data
        if self.parameterization == "velocity":
            velocity = self.mu(hidden)  # [B, action_size]
        elif self.parameterization == "data":
            x1 = self.mu(hidden)  # [B, action_size]
            # Convert data prediction to velocity using path
            velocity = self.path.target_to_velocity(x_1=x1, x_t=x_t, t=t.unsqueeze(-1))  # [B, action_size]
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
        
        return velocity
    
    def diffusion_model(self, x_t: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity using MLP architecture (matching PHC).
        
        This is called by the algorithm to compute CFM loss.
        Uses the shared _compute_velocity() method.
        
        Args:
            x_t: [B, 1, action_dim] or [B, action_dim] - interpolated action at timestep t
            context: [B, input_size] - observation/context
            t: [B] - timestep in [0, 1]
            
        Returns:
            velocity: [B, 1, action_dim] or [B, action_dim] - predicted velocity
        """
        # Handle input shapes
        original_shape = x_t.shape
        if x_t.dim() == 3:
            # [B, 1, action_dim] -> [B, action_dim]
            x_t = x_t.squeeze(1)
        
        # Use shared velocity computation (no condition_mask for training)
        velocity = self._compute_velocity(x_t, context, t, condition_mask=None)
        
        # Return in same shape as input
        if len(original_shape) == 3:
            return velocity.unsqueeze(1)  # [B, 1, action_size]
        else:
            return velocity  # [B, action_size]
    
    def sample_noise(self, noise_shape, device):
        """Sample noise for flow matching (matching PHC)."""
        noise = torch.randn(noise_shape, dtype=torch.float32, device=device)
        noise = noise * self.prior_noise_std
        return noise
    
    def sample_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample actions using flow matching ODE solver (matching PHC's sample_actions).
        
        Args:
            obs: [B, input_size] observations
        
        Returns:
            actions: [B, action_size] sampled actions
        """
        assert not torch.is_grad_enabled(), "Autograd should not be enabled during the sampling chain!"
        
        B = obs.shape[0]
        device = obs.device
        
        # Sample initial noise
        x_0 = self.sample_noise([B, self.action_size], device)
        time_grid = torch.tensor([0.0, 1.0], device=device)
        
        # Condition dropout mask (if enabled)
        active_condition_mask = None
        if self.condition_drop_ratio > 0 and self.sample_mask is not None:
            # Use first B rows of sample_mask, or broadcast if needed
            if self.sample_mask.shape[0] >= B:
                active_condition_mask = self.sample_mask[:B].to(device)
            else:
                # Broadcast to batch size
                active_condition_mask = self.sample_mask[0:1].expand(B, -1).to(device)
        
        def velocity_fn(x, t, obs, condition_mask=None):
            """
            Velocity function for ODE solver (matching PHC).
            
            Uses the shared _compute_velocity() method to ensure consistency
            between training (diffusion_model) and inference (sample_actions).
            """
            # Convert scalar t to batch tensor [B]
            t_batch = torch.ones([B], device=device) * t
            
            # Use shared velocity computation
            velocity = self._compute_velocity(x, obs, t_batch, condition_mask=condition_mask)
            return velocity
        
        # Solve ODE to get actions
        x_1 = self.solver.sample(
            velocity_fn,
            time_grid=time_grid,
            x_init=x_0,
            method="euler",
            return_intermediates=False,
            atol=1e-5,
            rtol=1e-5,
            step_size=self.solver_step_size,
            obs=obs,
            condition_mask=active_condition_mask,
        )
        
        return x_1
    
    def forward(self, context: torch.Tensor, targets: Optional[Dict[str, torch.Tensor]] = None):
        """
        Forward pass for training/inference.
        
        Args:
            context: [B, input_size] observation/context
            targets: Optional dict with 'trajectory' key for training
        
        Returns:
            dict with 'trajectory' key containing [B, action_size] actions
        """
        if self.training and targets is not None and 'trajectory' in targets:
            # Training mode: return trajectory for loss computation
            # The algorithm will call diffusion_model() separately for CFM loss
            trajectory = targets['trajectory']
            # If trajectory is [B, num_poses, action_dim], take first pose or flatten
            if trajectory.dim() == 3:
                trajectory = trajectory[:, 0, :]  # Take first pose
            return {'trajectory': trajectory}
        else:
            # Inference mode: sample actions
            actions = self.sample_actions(context)
            return {'trajectory': actions}


class FlowMatchingMLPActorNetwork(nn.Module):
    """
    MLP-based Flow Matching Actor Network matching PHC architecture.
    
    This network provides the ALF interface and calls trajectory_head in forward().
    The trajectory_head contains the flow matching network that computes actions and velocities.
    """
    
    def __init__(self,
                 input_size: int,
                 action_size: int,
                 hidden_size: int = 512,
                 parameterization: str = "velocity",
                 zero_action_input: bool = False,
                 prior_noise_std: float = 1.0,
                 solver_step_size: float = 0.1,
                 condition_drop_ratio: float = 0.0,
                 num_envs: int = 4096,
                 head_ctor: Optional[Callable] = None,
                 **kwargs):
        """
        Initialize Flow Matching MLP actor network.
        
        Args:
            input_size: Observation dimension
            action_size: Action dimension
            hidden_size: Hidden dimension for MLP (default: 512)
            parameterization: "velocity" or "data" (default: "velocity")
            zero_action_input: If True, zero out action input (default: False)
            prior_noise_std: Standard deviation for prior noise (default: 1.0)
            solver_step_size: Step size for ODE solver (default: 0.1)
            condition_drop_ratio: Condition dropout ratio (default: 0.0)
            num_envs: Number of environments for dropout mask (default: 4096)
            head_ctor: Optional constructor for trajectory_head (for compatibility)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__()
        
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.parameterization = parameterization
        self.zero_action_input = zero_action_input
        self.prior_noise_std = prior_noise_std
        
        # Create trajectory_head (contains flow matching network)
        if head_ctor is not None:
            # Use provided constructor (for compatibility with ALF patterns)
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                              if k not in ['input_tensor_spec', 'action_spec', 'reward_spec']}
            trajectory_head = head_ctor(**filtered_kwargs)
        else:
            # Create default MLPTrajectoryHead
            trajectory_head = MLPTrajectoryHead(
                input_size=input_size,
                action_size=action_size,
                hidden_size=hidden_size,
                parameterization=parameterization,
                zero_action_input=zero_action_input,
                prior_noise_std=prior_noise_std,
                solver_step_size=solver_step_size,
                condition_drop_ratio=condition_drop_ratio,
                num_envs=num_envs,
                **kwargs
            )
        
        # Register as submodule
        self.add_module('trajectory_head', trajectory_head)
    
    @property
    def state_spec(self):
        """Return the state spec of the actor network. This is stateless."""
        return ()  # stateless
    
    def forward(self, inputs: Dict[str, torch.Tensor], state=()):
        """
        Forward pass for action generation.
        
        This calls trajectory_head.forward() similar to FlowMatchingActorNetwork.
        
        Args:
            inputs: Dictionary containing observation or representation
            state: Optional state (not used)
            
        Returns:
            Dictionary containing generated trajectory and additional info
        """
        # Extract observation/context
        if isinstance(inputs, torch.Tensor):
            context = inputs
        elif isinstance(inputs, dict):
            if 'representation' in inputs:
                repr_data = inputs['representation']
                if isinstance(repr_data, dict):
                    # Handle nested dictionary representation
                    if 'context_info' in repr_data:
                        context = repr_data['context_info']
                    elif 'trajectory_query' in repr_data:
                        context = repr_data['trajectory_query']
                    else:
                        # Fallback: use the first tensor in the nested dict
                        context = next(iter(v for v in repr_data.values() if torch.is_tensor(v)))
                else:
                    # Handle tensor representation
                    context = repr_data
            elif 'context' in inputs:
                context = inputs['context']
            elif 'trajectory_query' in inputs:
                context = inputs['trajectory_query']
            elif 'observation' in inputs:
                context = inputs['observation']
            else:
                # Fallback: use the first tensor in inputs
                context = next(iter(inputs.values()))
        else:
            raise ValueError(f"Unexpected input type: {type(inputs)}")
        
        # Ensure context has correct shape [B, input_size]
        if context.dim() > 2:
            context = context.view(context.shape[0], -1)
        if context.shape[-1] != self.input_size:
            # If context is larger, take first input_size dimensions
            context = context[..., :self.input_size]
        
        # Extract targets if available (for training mode)
        targets = None
        if self.training:
            if hasattr(inputs, 'targets'):
                targets = inputs.targets
            elif isinstance(inputs, dict):
                if 'targets' in inputs:
                    targets = inputs['targets']
                elif 'representation' in inputs:
                    repr_data = inputs['representation']
                    if isinstance(repr_data, dict) and 'gt_trajectory' in repr_data:
                        gt_traj = repr_data['gt_trajectory']
                        if hasattr(gt_traj, 'data'):
                            targets = {'trajectory': gt_traj.data}
                        else:
                            targets = {'trajectory': gt_traj}
        
        # Call trajectory_head.forward() (matching FlowMatchingActorNetwork pattern)
        outputs = self.trajectory_head(context, targets)
        
        return outputs, state
    
    def rollout_step(self, inputs: Dict[str, torch.Tensor], eta: float = 0.0, steps: Optional[int] = None):
        """
        Generate action for RL rollout.
        
        Args:
            inputs: Dictionary containing observation
            eta: Sampling parameter (not used)
            steps: Number of sampling steps (not used)
            
        Returns:
            Generated action [B, action_size]
        """
        # Extract observation
        if isinstance(inputs, dict):
            obs = inputs.get('observation', inputs.get('context', next(iter(inputs.values()))))
        else:
            obs = inputs
        
        # Ensure correct shape
        if obs.dim() > 2:
            obs = obs.view(obs.shape[0], -1)
        if obs.shape[-1] != self.input_size:
            obs = obs[..., :self.input_size]
        
        # Use trajectory_head to sample actions
        outputs = self.trajectory_head(obs)
        trajectory = outputs['trajectory']  # [B, action_size]
        
        return trajectory
    
    def sample(self, inputs: Dict[str, torch.Tensor], eta: float = 0.0, steps: Optional[int] = None):
        """Sample action from the Flow Matching model."""
        return self.rollout_step(inputs, eta=eta, steps=steps)
    
    def get_log_prob(self, action: torch.Tensor, inputs: Dict[str, torch.Tensor]):
        """
        Get log probability of action (placeholder for Flow Matching).
        
        Args:
            action: Action tensor [B, action_size]
            inputs: Input context
            
        Returns:
            Log probability tensor [B]
        """
        # For Flow Matching, we don't have a direct log probability
        # This is a placeholder that returns zeros
        batch_size = action.shape[0]
        device = action.device
        return torch.zeros(batch_size, device=device)
