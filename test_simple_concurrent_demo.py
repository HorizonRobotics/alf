#!/usr/bin/env python3
"""Demo for SimpleConcurrentAlgorithm with HandcraftedAlgorithm."""

import os
import sys

import gym
import numpy as np
import torch
from gym import spaces

# Add ALF to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import alf
import alf.nest
from alf.algorithms.handcrafted_algorithm import HandcraftedAlgorithm
from alf.algorithms.simple_concurrent_algorithm import SimpleConcurrentAlgorithm
from alf.data_structures import TimeStep
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


class RandomHandcraftedAlgorithm(HandcraftedAlgorithm):
    """Simple random algorithm for testing."""
    
    def _policy_func(self, observation):
        """Random action selection."""
        batch_size = alf.nest.get_nest_batch_size(observation)
        return torch.randint(0, 3, (batch_size,))

class SimpleGridWorld(gym.Env):
    """Simple 5x1 gridworld for testing."""
    
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # LEFT, STAY, RIGHT
        self.reset()
    
    def reset(self):
        self.position = 0
        obs = np.zeros(5, dtype=np.float32)
        obs[self.position] = 1.0
        return obs
    
    def step(self, action):
        # Move: 0=LEFT, 1=STAY, 2=RIGHT
        if action == 0:  # LEFT
            self.position = max(0, self.position - 1)
        elif action == 2:  # RIGHT
            self.position = min(4, self.position + 1)
        # action == 1 stays in place
        
        obs = np.zeros(5, dtype=np.float32)
        obs[self.position] = 1.0
        
        # Reward: reach position 4
        reward = 1.0 if self.position == 4 else 0.0
        done = self.position == 4
        
        return obs, reward, done, {}

def test_simple_concurrent():
    """Test SimpleConcurrentAlgorithm with HandcraftedAlgorithm."""
    print("Testing SimpleConcurrentAlgorithm with HandcraftedAlgorithm...")
    
    # Create environment
    env = SimpleGridWorld()
    
    # Create specs
    observation_spec = TensorSpec(shape=(5,), dtype=torch.float32)
    action_spec = BoundedTensorSpec(shape=(), dtype=torch.int64, minimum=0, maximum=2)
    
    print(f"Observation spec: {observation_spec}")
    print(f"Action spec: {action_spec}")
    
    # Create algorithm constructor
    def alg_ctor(observation_spec, action_spec, **kwargs):
        return RandomHandcraftedAlgorithm(
            observation_spec=observation_spec,
            action_spec=action_spec,
            **kwargs)
    
    # Create SimpleConcurrentAlgorithm
    algorithm = SimpleConcurrentAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        algorithm_ctor=alg_ctor,
        num_copies=2)
    
    print(f"Algorithm created with {algorithm._num_copies} copies")
    
    # Test rollout step
    obs = env.reset()
    time_step = TimeStep(
        observation=torch.tensor(obs).unsqueeze(0),  # Add batch dimension
        reward=torch.tensor([0.0]),
        discount=torch.tensor([1.0]),
        step_type=torch.tensor([0]))  # 0 = FIRST
    
    state = algorithm.get_initial_rollout_state(1)  # batch_size = 1
    
    print("Testing rollout_step...")
    alg_step = algorithm.rollout_step(time_step, state)
    print(f"Rollout output shape: {alf.nest.get_nest_batch_size(alg_step.output)}")
    print(f"Rollout output: {alg_step.output}")
    print(f"New state length: {len(alg_step.state)}")
    
    # Test predict step
    print("Testing predict_step...")
    alg_step = algorithm.predict_step(time_step, state)
    print(f"Predict output shape: {alf.nest.get_nest_batch_size(alg_step.output)}")
    print(f"Predict output: {alg_step.output}")
    
    # Test with larger batch size (must be multiple of num_copies)
    print("\nTesting with batch size 4 (multiple of 2)...")
    batch_obs = torch.tensor([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0]])
    batch_time_step = TimeStep(
        observation=batch_obs,
        reward=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        discount=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        step_type=torch.tensor([0, 0, 0, 0]))
    
    batch_state = algorithm.get_initial_rollout_state(4)
    batch_alg_step = algorithm.rollout_step(batch_time_step, batch_state)
    print(f"Batch rollout output shape: {alf.nest.get_nest_batch_size(batch_alg_step.output)}")
    print(f"Batch rollout output: {batch_alg_step.output}")
    
    print("✅ SimpleConcurrentAlgorithm demo passed!")

if __name__ == '__main__':
    test_simple_concurrent()
