"""
Flow Matching Trajectory Head

This module implements a trajectory prediction head using Flow Matching instead of DDIM.
It's designed to work with FlowMatchingBEVEncoder as the representation learner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np

from alf.networks.DiT import DiT
from diffusers import FlowMatchEulerDiscreteScheduler


class FlowMatchingTrajectoryHead(nn.Module):
    """Flow Matching trajectory prediction head."""
    
    def __init__(self, 
                 num_poses: int, 
                 d_ffn: int, 
                 d_model: int, 
                 action_dim: int = 3,
                 diffusion_steps: int = 20,
                 dit_depth: int = 3,
                 time_sample_alpha: float = 0.0,
                 pretrained_checkpoint: str = None,
                 enable_trajectory_scaling: bool = False):
        super().__init__()
        
        self._num_poses = num_poses
        self._action_dim = action_dim
        self._diffusion_steps = diffusion_steps
        self._time_sample_alpha = time_sample_alpha
        self._enable_trajectory_scaling = enable_trajectory_scaling
        
        # Flow Matching DiT (similar to DiffusionTrajectoryModel)
        self.diffusion_model = DiT(
            in_channels=action_dim,
            hidden_size=256,
            depth=dit_depth,
            num_heads=8,
            num_frames=num_poses
        )
        
        # Flow Matching Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=diffusion_steps
        )
        
        # Trajectory scaling (similar to DiffusionTrajectoryModel)
        self._setup_trajectory_scaling()
        
        # Load pretrained DiT weights if checkpoint is provided
        if pretrained_checkpoint is not None:
            self.load_pretrained_dit_weights(pretrained_checkpoint)
    
    def _setup_trajectory_scaling(self):
        """Setup trajectory scaling parameters"""
        self.traj_scale = 1.0
        self.traj_offset = 0.0
    
    def load_pretrained_dit_weights(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained DiT weights from a DiffusionTrajectoryModel checkpoint.
        
        Args:
            checkpoint_path: Path to the pretrained DiffusionTrajectoryModel checkpoint
            strict: Whether to strictly enforce that the keys match
        """
        import torch
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle ALF checkpoint format (flat dict with dotted keys)
            if isinstance(checkpoint, dict):
                if 'algorithm' in checkpoint:
                    # ALF checkpoint: keys like '_model.encoder.weight'
                    state_dict = checkpoint['algorithm']
                    # Remove '_model.' prefix from keys to get 'encoder.weight', etc.
                    cleaned_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('_model.'):
                            new_key = key.replace('_model.', '', 1)
                            cleaned_state_dict[new_key] = value
                    state_dict = cleaned_state_dict
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    # Assume the dict itself is the state dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Filter to only get 'diffusion_model.*' keys (DiT weights)
            dit_state_dict = {}
            encoder_keys = []
            traj_head_keys = []
            for key, value in state_dict.items():
                if key.startswith('diffusion_model.'):
                    # Keep 'diffusion_model.' prefix for FlowMatchingTrajectoryHead
                    dit_state_dict[key] = value
                elif key.startswith('encoder.'):
                    encoder_keys.append(key)
                elif key.startswith('traj_head.'):
                    traj_head_keys.append(key)
            
            # Check if checkpoint has traj_head instead of diffusion_model
            if len(traj_head_keys) > 0 and len(dit_state_dict) == 0:
                print(f"⚠ WARNING: Checkpoint has 'traj_head' ({len(traj_head_keys)} keys) but no 'diffusion_model'")
                print(f"⚠ This checkpoint was trained with repr_mode=True (simple MLP head)")
                print(f"⚠ FlowMatchingTrajectoryHead requires a checkpoint with 'diffusion_model' (DiT)")
                print(f"⚠ Cannot load traj_head weights into diffusion_model - incompatible architectures")
                print(f"⚠ Skipping weight loading. DiT will be initialized randomly.")
                return
            
            print(f"Found {len(dit_state_dict)} DiT parameter keys to load")
            print(f"Skipped {len(encoder_keys)} encoder parameter keys")
            
            if len(dit_state_dict) == 0:
                print(f"⚠ WARNING: No diffusion_model keys found in checkpoint")
                print(f"⚠ DiT will remain with random initialization")
                return
            
            # Load the filtered state dict
            missing_keys, unexpected_keys = self.load_state_dict(dit_state_dict, strict=strict)
            
            if missing_keys:
                print(f"Missing keys when loading pretrained DiT weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained DiT weights: {unexpected_keys}")
            
            print(f"✓ Loaded pretrained DiT weights from {checkpoint_path}")
            
        except Exception as e:
            print(f"✗ Failed to load pretrained DiT weights from {checkpoint_path}: {e}")
            raise
    
    def _downscale_traj(self, traj: torch.Tensor) -> torch.Tensor:
        """Downscale trajectory for training (OmniAD-specific: x, y coordinates)"""
        if not self._enable_trajectory_scaling:
            return traj
        traj = traj.clone()
        if traj.shape[-1] >= 2:
            traj[..., 0] = (traj[..., 0] - 20) / 20  # x axis
            traj[..., 1] = traj[..., 1] / 20  # y axis
        return traj
    
    def _upscale_traj(self, traj: torch.Tensor) -> torch.Tensor:
        """Upscale trajectory for inference (OmniAD-specific: x, y coordinates)"""
        if not self._enable_trajectory_scaling:
            return traj
        traj = traj.clone()
        if traj.shape[-1] >= 2:
            traj[..., 0] = traj[..., 0] * 20 + 20  # x axis
            traj[..., 1] = traj[..., 1] * 20  # y axis
        if traj.shape[-1] >= 3:
            # Wrap heading to [-π, π] range
            traj[..., 2] = ((traj[..., 2] + torch.pi) % (2 * torch.pi)) - torch.pi
        return traj
    
    def _prepare_model_input(self, x_target: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Prepare model input for flow matching"""
        # Flow matching: x_t = (1-t) * x_0 + t * noise
        t = timesteps.unsqueeze(-1)  # [B, 1]
        x_t = (1 - t) * x_target + t * noise
        return x_t
    
    def forward(self, context_info: torch.Tensor, targets: Optional[Dict] = None):
        """Forward pass for Flow Matching trajectory generation"""
        
        if self.training and targets is not None:
            return self._forward_train(context_info, targets)
        else:
            return self._forward_inference(context_info)
    
    def _forward_train(self, context_info: torch.Tensor, targets: Dict[str, torch.Tensor]):
        """Flow Matching training forward pass"""
        
        # Get target trajectory
        x_target = targets["trajectory"]
        # Reshape to [B, num_poses, action_dim] if needed
        if x_target.ndim == 2:
            # Assume [B, action_dim] -> reshape to [B, num_poses, action_dim]
            batch_size = x_target.shape[0]
            if x_target.shape[1] == self._num_poses * self._action_dim:
                x_target = x_target.view(batch_size, self._num_poses, self._action_dim)
            elif x_target.shape[1] == self._action_dim:
                x_target = x_target.unsqueeze(1)  # [B, 1, action_dim]
        x_target = self._downscale_traj(x_target)
        batch_size = x_target.shape[0]
        x_target = x_target.view(batch_size, -1)  # [B, trajectory_steps * action_dim]
        
        # Flow Matching forward process
        noise = torch.randn_like(x_target)  # ε ~ N(0, 1)
        timesteps = torch.rand(noise.shape[0], device=noise.device).pow(self._time_sample_alpha)  # Random time steps
        
        # Prepare model input: x_t = (1-t) * x_0 + t * ε
        model_input = self._prepare_model_input(x_target, noise, timesteps)
        
        # Target velocity: v = ε - x_0 (flow matching)
        v_target = noise - x_target
        
        # Predict velocity using DiT
        v_pred = self.diffusion_model(
            model_input.unsqueeze(1),  # [B, 1, trajectory_steps * action_dim]
            context_info,
            timesteps
        )
        v_pred = v_pred.squeeze(1)  # [B, trajectory_steps * action_dim]
        
        # Generate zero trajectory for environment compatibility
        traj_0 = torch.zeros(
            batch_size, self._num_poses, self._action_dim, 
            device=v_pred.device
        )
        
        return {
            'v_pred': v_pred,
            'v_target': v_target,
            'trajectory': traj_0
        }
    
    def _forward_inference(self, context_info: torch.Tensor):
        """Flow Matching inference forward pass"""
        
        batch_size = context_info.shape[0]
        device = context_info.device
        
        # Initialize with noise
        x_input = torch.randn(
            batch_size, self._num_poses * self._action_dim,
            device=device
        )
        
        # Flow Matching sampling
        trajectory = self.p_sample_loop(x_input, context_info)
        
        # Reshape to [B, num_poses, action_dim]
        trajectory = trajectory.view(batch_size, self._num_poses, self._action_dim)
        
        # Upscale trajectory
        trajectory = self._upscale_traj(trajectory)
        
        return {'trajectory': trajectory}
    
    def p_sample_loop(self, x_input: torch.Tensor, context_info: torch.Tensor) -> torch.Tensor:
        """Flow Matching sampling loop"""
        
        batch_size = x_input.shape[0]
        
        # Initialize diffusion timesteps
        self.scheduler.set_timesteps(self._diffusion_steps, device=x_input.device)
        
        # Iteratively denoise the input
        ts_reverse = torch.linspace(1.0, 0.0, self._diffusion_steps, device=x_input.device)
        for i in range(self._diffusion_steps - 1):
            time_step = ts_reverse[i].expand(batch_size)
            v = self.diffusion_model(x_input.unsqueeze(1), 
                                     context_info.clone().detach(), 
                                     time_step)
            v = v.squeeze(1)
            
            x_input = self.scheduler.step(
                model_output=v, 
                timestep=self.scheduler.timesteps[i], 
                sample=x_input).prev_sample
        
        return x_input
 