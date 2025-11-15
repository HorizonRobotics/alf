"""
Flow Matching Actor Network

This module implements an actor network that uses FlowMatchingTrajectoryHead
for trajectory generation instead of DDIM-based approaches.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
from functools import partial


class FlowMatchingActorNetwork(nn.Module):
    """
    Actor network that uses FlowMatchingTrajectoryHead for trajectory generation.
    
    This network takes context from FlowMatchingBEVEncoder and generates
    trajectories using Flow Matching instead of DDIM.
    """
    
    def __init__(self,
                 head_ctor: Callable,
                 flow_matching_steps: int = 5,
                 encoder_input_dim: int = 44,
                 encoder_output_dim: int = 256,
                 encoder_hidden_dims: tuple = (128,),
                 encoder_activation: nn.Module = nn.ReLU,
                 **kwargs):
        """
        Initialize Flow Matching actor network.
        
        Args:
            head_ctor: Constructor for FlowMatchingTrajectoryHead
            flow_matching_steps: Number of Flow Matching steps for inference
            encoder_input_dim: Input dimension for the encoder MLP (default: 44)
            encoder_output_dim: Output dimension for the encoder MLP (default: 256)
            encoder_hidden_dims: Hidden layer dimensions for the encoder MLP (default: (128,))
            encoder_activation: Activation function for encoder layers (default: ReLU)
            **kwargs: Additional arguments passed to head_ctor
        """
        super().__init__()
        
        self._flow_matching_steps = flow_matching_steps
        
        # Filter out ALF-specific arguments that FlowMatchingTrajectoryHead doesn't expect
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['input_tensor_spec', 'action_spec', 'reward_spec']}
        
        # Create the Flow Matching trajectory head
        self.trajectory_head = head_ctor(**filtered_kwargs)
        
        # Create MLP encoder to map context_info to latent feature
        encoder_layers = []
        input_dim = encoder_input_dim
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(encoder_activation())
            input_dim = hidden_dim
        # Final layer to output dimension
        encoder_layers.append(nn.Linear(input_dim, encoder_output_dim))
        self.encoder = nn.Sequential(*encoder_layers)
    
    @property
    def state_spec(self):
        """Return the state spec of the actor network. This is stateless."""
        return ()  # stateless
        
    def forward(self, inputs: Dict[str, torch.Tensor], state=()):
        """
        Forward pass for trajectory generation.
        
        Args:
            inputs: Dictionary containing representation from RepresentationLearner
            state: Optional state (not used)
            
        Returns:
            Dictionary containing generated trajectory and additional info
        """
        # Extract representation from inputs
        # inputs can be either:
        # 1. A tensor (from RepresentationLearner output)
        # 2. A dictionary with 'representation' key containing tensor
        # 3. A dictionary with 'representation' key containing nested dict
        # 4. LogSimObservation (from E2ESimEnvironment)
        if isinstance(inputs, torch.Tensor):
            # Direct tensor input from RepresentationLearner
            context_info = inputs
        elif hasattr(inputs, 'targets') and hasattr(inputs, 'features'):
            # This is a LogSimObservation (has features and targets attributes)
            # Extract context_info from features dict
            context_info = inputs.features.get('representation', None)
            if context_info is None:
                # Try to get context from first tensor in features dict
                context_info = next(iter(v for v in inputs.features.values() if torch.is_tensor(v)))
        elif isinstance(inputs, dict):
            # Dictionary input with various possible keys
            if 'representation' in inputs:
                repr_data = inputs['representation']
                if isinstance(repr_data, dict):
                    # Handle nested dictionary representation (from DiffusionNFTAlgorithm)
                    if 'context_info' in repr_data:
                        context_info = repr_data['context_info']
                    elif 'trajectory_query' in repr_data:
                        context_info = repr_data['trajectory_query']
                    else:
                        # Fallback: use the first tensor in the nested dict
                        context_info = next(iter(v for v in repr_data.values() if torch.is_tensor(v)))
                else:
                    # Handle tensor representation (backward compatibility)
                    context_info = repr_data
            elif 'context' in inputs:
                context_info = inputs['context']
            elif 'trajectory_query' in inputs:
                context_info = inputs['trajectory_query']
            else:
                # Fallback: use the first tensor in inputs
                context_info = next(iter(inputs.values()))
        else:
            raise ValueError(f"Unexpected input type: {type(inputs)}")
        
        # Ensure context has correct shape [B, D]
        if context_info.dim() > 2:
            context_info = context_info.view(context_info.shape[0], -1)
        
        # Encode context_info to latent feature using MLP encoder
        context_info = self.encoder(context_info)  # [B, encoder_input_dim] -> [B, encoder_output_dim]
        
        # Extract targets if available (for training mode)
        targets = None
        if self.training:
            if hasattr(inputs, 'targets'):
                # Extract targets from LogSimObservation
                targets = {'trajectory': inputs.targets.get('trajectory') if hasattr(inputs.targets, 'get') else inputs.targets.get('trajectory')}
            elif isinstance(inputs, dict):
                if 'targets' in inputs:
                    targets = inputs['targets']
                elif 'representation' in inputs:
                    repr_data = inputs['representation']
                    if isinstance(repr_data, dict) and 'gt_trajectory' in repr_data:
                        # Extract ground truth trajectory from nested representation
                        gt_traj = repr_data['gt_trajectory']
                        # Convert Trajectory object to tensor if needed
                        if hasattr(gt_traj, 'data'):
                            targets = {'trajectory': gt_traj.data}
                        else:
                            targets = {'trajectory': gt_traj}
        
        # Generate trajectory using Flow Matching
        if self.training and targets is not None and 'trajectory' in targets:
            # Training mode: return velocity predictions and targets
            outputs = self.trajectory_head(context_info, targets)
            # Ensure v_pred and v_target are in the output
            if isinstance(outputs, dict):
                if 'v_pred' in outputs and 'v_target' in outputs:
                    # Store them in representation dict
                    outputs['representation'] = outputs.get('representation', {})
                    if not isinstance(outputs['representation'], dict):
                        outputs['representation'] = {}
                    outputs['representation']['v_pred'] = outputs['v_pred']
                    outputs['representation']['v_target'] = outputs['v_target']
        else:
            # Inference mode: return generated trajectory
            outputs = self.trajectory_head(context_info)
        
        return outputs, state
    
    def rollout_step(self, inputs: Dict[str, torch.Tensor], eta: float = 0.0, steps: Optional[int] = None):
        """
        Generate trajectory for RL rollout.
        
        Args:
            inputs: Dictionary containing context information
            eta: Sampling parameter (not used in Flow Matching)
            steps: Number of samplin gsteps (uses default if None)
            
        Returns:
            Generated trajectory [B, num_poses, action_dim]
        """
        
        # Extract context
        if 'context' in inputs:
            context_info = inputs['context']
        elif 'trajectory_query' in inputs:
            context_info = inputs['trajectory_query']
        else:
            context_info = next(iter(inputs.values()))
        
        # Ensure correct shape
        if context_info.dim() > 2:
            context_info = context_info.view(context_info.shape[0], -1)
        
        # Encode context_info to latent feature using MLP encoder
        context_info = self.encoder(context_info)  # [B, encoder_input_dim] -> [B, encoder_output_dim]
        
        # Generate trajectory using forward inference
        # Note: steps parameter is not used as FlowMatchingTrajectoryHead uses fixed diffusion_steps
        outputs = self.trajectory_head(context_info)
        trajectory = outputs['trajectory']  # [B, num_poses, action_dim]
        
        return trajectory
    
    def get_log_prob(self, action: torch.Tensor, inputs: Dict[str, torch.Tensor]):
        """
        Get log probability of action (placeholder for Flow Matching).
        
        Args:
            action: Action tensor [B, num_poses, action_dim]
            inputs: Input context
            
        Returns:
            Log probability tensor [B]
        """
        # For Flow Matching, we don't have a direct log probability
        # This is a placeholder that returns zeros
        batch_size = action.shape[0]
        device = action.device
        return torch.zeros(batch_size, device=device)
    
    def sample(self, inputs: Dict[str, torch.Tensor], eta: float = 0.0, steps: Optional[int] = None):
        """
        Sample trajectory from the Flow Matching model.
        
        Args:
            inputs: Dictionary containing context information
            eta: Sampling parameter (not used in Flow Matching)
            steps: Number of sampling steps
            
        Returns:
            Generated trajectory [B, num_poses, action_dim]
        """
        return self.rollout_step(inputs, eta=eta, steps=steps)
