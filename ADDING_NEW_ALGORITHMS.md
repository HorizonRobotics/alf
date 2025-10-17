# Adding New Algorithms to ALF

This guide explains how to add a new algorithm to the ALF (Agent Learning Framework) using RLPD (Reinforcement Learning with Prior Data) as a concrete example. RLPD extends SAC with high update-to-data (UTD) ratios, bootstrapped critics, and alternating actor-critic updates.

## Table of Contents

1. [Understanding the Algorithm Hierarchy](#understanding-the-algorithm-hierarchy)
2. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
3. [Key Implementation Patterns](#key-implementation-patterns)
4. [Testing Your Algorithm](#testing-your-algorithm)
5. [Creating Configuration Files](#creating-configuration-files)

---

## Understanding the Algorithm Hierarchy

ALF algorithms follow a clear inheritance hierarchy:

```
Algorithm (base class)
├── RLAlgorithm (for RL algorithms)
│   ├── OnPolicyAlgorithm (for A2C, PPO, etc.)
│   └── OffPolicyAlgorithm (for SAC, DDPG, etc.)
│       ├── SacAlgorithm
│       │   ├── RlpdAlgorithm
│       │   └── QrsacAlgorithm
│       └── IqlAlgorithm
└── Other specialized base classes
```

**Key Decision**: Choose the appropriate base class:
- Extend `OffPolicyAlgorithm` or `OnPolicyAlgorithm` for new RL algorithms
- Extend existing algorithms (like `SacAlgorithm`) when making modifications to them
- Extend `Algorithm` directly for non-RL algorithms

---

## Step-by-Step Implementation Guide

### Step 1: Create the Algorithm File

Create `alf/algorithms/your_algorithm.py`. Start with the standard imports and class definition:

```python
"""Your Algorithm description."""

from enum import Enum
import torch
from typing import Callable, Optional, Union

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.sac_algorithm import SacAlgorithm  # or appropriate base
from alf.data_structures import TimeStep, AlgStep, LossInfo, namedtuple
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, math_ops

@alf.configurable
class YourAlgorithm(SacAlgorithm):  # Choose appropriate base class
    r"""Brief description of the algorithm.

    Reference paper:

    ::

        Author et al. "Paper Title", arXiv:XXXX.XXXXX

    Key differences from base algorithm:
    1. First major difference
    2. Second major difference
    ...
    """
```

**RLPD Example**:
```python
@alf.configurable
class RlpdAlgorithm(SacAlgorithm):
    r"""RLPD algorithm, described in:

    ::

        Ball et al "Efficient Online Reinforcement Learning with Offline Data",
        arXiv:2302.02948

    Currently, only continuous action spaces are supported. There are two differences
    versus the above RLPD algorithm:

    1. Add an option of using bootstrapped critics...
    2. Besides critics UTD, the actor UTD is also configurable...
    """
```

### Step 2: Define Custom Data Structures

Create `namedtuple` structures for algorithm-specific information. These are used to pass data between different methods.

```python
# Information structure returned by train_step
YourAlgoInfo = namedtuple("YourAlgoInfo", [
    "reward",
    "step_type",
    "discount",
    "action",
    "actor",  # Actor-specific info
    "critic",  # Critic-specific info
    "custom_field",  # Your custom fields
], default_value=())

# If you have custom critic information
YourCriticInfo = namedtuple("YourCriticInfo", [
    "critics",
    "target_critic",
    "additional_info"
], default_value=())
```

**RLPD Example**:
```python
RlpdInfo = namedtuple("RlpdInfo", [
    "reward", "step_type", "discount", "action", "action_distribution",
    "actor", "critic", "alpha", "log_pi", "discounted_return", "repr",
    "bootstrap_mask"  # RLPD-specific: mask for bootstrapped critics
], default_value=())

RlpdCriticInfo = namedtuple("RlpdCriticInfo",
    ["critics", "target_critic"],
    default_value=())
```

### Step 3: Implement `__init__()`

The constructor should:
1. Accept all necessary hyperparameters
2. Call the parent class constructor
3. Initialize algorithm-specific parameters
4. Validate configurations

```python
def __init__(self,
             observation_spec,
             action_spec: BoundedTensorSpec,
             reward_spec=TensorSpec(()),
             # Base class parameters
             actor_network_cls=ActorDistributionNetwork,
             critic_network_cls=CriticNetwork,
             # Your custom parameters
             custom_param1=default_value1,
             custom_param2=default_value2,
             # Common parameters
             env=None,
             config: TrainerConfig = None,
             actor_optimizer=None,
             critic_optimizer=None,
             checkpoint=None,
             debug_summaries=False,
             name="YourAlgorithm"):
    """
    Args:
        observation_spec: representing the observations
        action_spec: representing the actions
        reward_spec: representing the reward(s)
        custom_param1: description of your custom parameter
        custom_param2: description of your custom parameter
        ... (document all other parameters)
    """
    # Call parent constructor with all required parameters
    super().__init__(
        observation_spec=observation_spec,
        action_spec=action_spec,
        reward_spec=reward_spec,
        actor_network_cls=actor_network_cls,
        critic_network_cls=critic_network_cls,
        env=env,
        config=config,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        checkpoint=checkpoint,
        debug_summaries=debug_summaries,
        name=name)

    # Validate configurations
    assert some_condition, "Helpful error message"

    # Initialize your algorithm-specific attributes
    self._custom_param1 = custom_param1
    self._custom_param2 = custom_param2
    self._custom_state = None
```

**RLPD Example**:
```python
def __init__(self,
             observation_spec,
             action_spec: BoundedTensorSpec,
             # ... base parameters ...
             num_sampled_critic_targets=2,
             use_bootstrap_critics=True,
             bootstrap_mask_prob=0.8,
             actor_utd: Optional[int] = None,
             critic_utd: Optional[int] = None,
             # ... other parameters ...
             name="RlpdAlgorithm"):

    super().__init__(...)  # Pass all base class parameters

    # Validate RLPD-specific constraints
    assert self._act_type == ActionType.Continuous, (
        "RLPD algorithm only supports continuous action spaces.")
    assert num_sampled_critic_targets <= num_critic_replicas, (...)

    # Initialize RLPD-specific state
    if actor_utd is None and critic_utd is None:
        self._train_mode = TrainMode.standard
    else:
        # Configure alternating actor-critic updates
        total_utd = alf.config_util.get_config_value("num_updates_per_train_iter")
        if critic_utd is not None:
            actor_utd = total_utd - critic_utd
        else:
            critic_utd = total_utd - actor_utd
        self._train_mode = TrainMode.critic
        self._actor_utd = actor_utd
        self._critic_utd = critic_utd

    self._num_sampled_critic_targets = num_sampled_critic_targets
    self._use_bootstrap_critics = use_bootstrap_critics
    self._bootstrap_mask_prob = bootstrap_mask_prob
```

### Step 4: Implement Core Step Functions

There are three main step functions you may need to override:

#### 4.1 `predict_step()` - For Deployment

Use this when the algorithm needs special inference behavior:

```python
def predict_step(self, inputs: TimeStep, state):
    """Generate actions for deployment (no exploration).

    Args:
        inputs: Current time step from environment
        state: Algorithm state

    Returns:
        AlgStep with action, new state, and info
    """
    # Usually just call parent or implement custom inference logic
    return super().predict_step(inputs, state)
```

#### 4.2 `rollout_step()` - For Data Collection

Override this to modify behavior during environment interaction:

```python
def rollout_step(self, inputs: TimeStep, state):
    """Generate actions during training data collection.

    This is where you add exploration noise, maintain rollout-specific
    state, or collect additional information to store in replay buffer.
    """
    # Call parent to get base behavior
    alg_step = super().rollout_step(inputs, state)

    # Add your custom logic
    if self._needs_custom_processing:
        custom_info = self._compute_custom_info(inputs, state)
        alg_step = alg_step._replace(
            info=alg_step.info._replace(custom_field=custom_info))

    return alg_step
```

**RLPD Example** - Adding bootstrap masks:
```python
def rollout_step(self, inputs: TimeStep, state=None):
    alg_step = super().rollout_step(inputs, state)
    if not self._use_bootstrap_critics:
        return alg_step

    # Generate bootstrap masks at episode start (step_type == 0)
    update_mask = (inputs.step_type == 0)
    if update_mask.any():
        # Sample mask for each critic replica independently
        prob_t = torch.full(
            (inputs.step_type.shape[0], self._num_critic_replicas),
            self._bootstrap_mask_prob)
        mask = torch.bernoulli(prob_t)

        if self._bootstrap_mask is None:
            self._bootstrap_mask = mask
        else:
            self._bootstrap_mask[update_mask] = mask[update_mask]

    # Attach mask to info for storage in replay buffer
    info = alg_step.info._replace(bootstrap_mask=self._bootstrap_mask)
    return alg_step._replace(info=info)
```

#### 4.3 `train_step()` - For Training

This is the most important method. It computes forward passes needed for training:

```python
def train_step(self, inputs: TimeStep, state, rollout_info):
    """Perform one training step.

    Args:
        inputs: Current time step
        state: Training state (potentially different from rollout state)
        rollout_info: Information collected during rollout_step

    Returns:
        AlgStep with:
        - output: typically the action
        - state: updated training state
        - info: information needed for calc_loss()
    """
    # Option 1: Completely custom implementation
    observation = inputs.observation

    # Forward pass through your networks
    action_dist, action_state = self._actor_network(observation, state.actor)
    critics, critic_state = self._critic_networks((observation, action))

    # Compute quantities needed for loss
    log_prob = action_dist.log_prob(action)

    # Package info for calc_loss
    info = YourAlgoInfo(
        reward=inputs.reward,
        action=action,
        actor=actor_info,
        critic=critic_info,
        custom_field=custom_data)

    new_state = YourState(actor=action_state, critic=critic_state)
    return AlgStep(action, new_state, info)
```

**RLPD Example** - Alternating actor/critic updates:
```python
def train_step(self, inputs: TimeStep, state: SacState, rollout_info: RlpdInfo):
    # First call initializes the info spec - always do standard training
    if self._train_mode == TrainMode.standard or (
            self._critic_update_counter == 0 and self._actor_update_counter == 0):
        alg_step = super().train_step(inputs, state, rollout_info)
        self._critic_update_counter += 1
        info = alg_step.info._replace(bootstrap_mask=rollout_info.bootstrap_mask)
        return alg_step._replace(info=info)

    # Alternating mode: only update actor OR critic per train_step
    assert not self._is_eval
    self._training_started = True

    # Compute target observation (shared by both modes)
    if self._target_repr_alg is not None:
        with torch.no_grad():
            tgt_repr_step = self._target_repr_alg.predict_step(inputs, state.target_repr)
            target_observation = tgt_repr_step.output
            target_repr_state = tgt_repr_step.state
    else:
        target_observation = inputs.observation
        target_repr_state = ()

    observation, new_state, info = self._repr_step("train", inputs, state,
                                                     rollout_info.repr)
    (action_distribution, action, critics,
     action_state) = self._predict_action(observation, state=state.action)

    new_state = new_state._replace(action=action_state, ...)

    # Compute log probability
    log_pi = nest.map_structure(lambda dist, a: dist.log_prob(a),
                                action_distribution, action)
    log_pi = sum(nest.flatten(log_pi))

    # Branch based on training mode
    if self._train_mode == TrainMode.actor:
        # Only update actor and alpha
        actor_state, actor_info = self._actor_train_step(
            observation, state.actor, action, critics, log_pi, action_distribution)
        alpha_loss = self._alpha_train_step(log_pi)
        critic_info = RlpdCriticInfo()  # Empty
        new_state = new_state._replace(actor=actor_state)
        self._actor_update_counter += 1
    else:
        # Only update critic
        critic_state, critic_info = self._critic_train_step(
            observation, target_observation, state.critic, rollout_info,
            action, action_distribution)
        alpha_loss = ()
        actor_info = LossInfo(extra=SacActorInfo())  # Empty
        new_state = new_state._replace(critic=critic_state)
        self._critic_update_counter += 1

    # Package everything for calc_loss
    info = info._replace(
        reward=inputs.reward,
        actor=actor_info,
        critic=critic_info,
        alpha=alpha_loss,
        bootstrap_mask=rollout_info.bootstrap_mask)

    return AlgStep(action, new_state, info)
```

### Step 5: Implement `calc_loss()`

This method computes the actual loss from the info returned by `train_step()`:

```python
def calc_loss(self, info: YourAlgoInfo):
    """Calculate loss for gradient updates.

    Args:
        info: Batched information from train_step, shape [T, B, ...]

    Returns:
        LossInfo with loss tensor of shape [T, B]
    """
    # Compute individual loss components
    actor_loss = self._calc_actor_loss(info)
    critic_loss = self._calc_critic_loss(info)
    auxiliary_loss = self._calc_auxiliary_loss(info)

    # Combine losses
    total_loss = math_ops.add_ignore_empty(
        actor_loss.loss, critic_loss.loss)
    total_loss = math_ops.add_ignore_empty(
        total_loss, auxiliary_loss)

    return LossInfo(
        loss=total_loss,
        priority=critic_loss.priority,  # For prioritized replay
        extra=YourLossInfo(actor=actor_loss, critic=critic_loss))
```

**RLPD Example** - Applying bootstrap masks to critic loss:
```python
def _calc_critic_loss(self, info: RlpdInfo):
    """Compute critic loss with optional bootstrap masking."""

    # Skip critic loss if in actor training mode
    if self._train_mode == TrainMode.actor:
        return LossInfo()

    # Add entropy reward to target (standard SAC approach)
    if self._use_entropy_reward:
        with torch.no_grad():
            log_pi = info.log_pi
            entropy_reward = -torch.exp(self._log_alpha) * log_pi
            discount = self._critic_losses[0].gamma * info.discount
            info = info._replace(
                reward=(info.reward +
                       common.expand_dims_as(entropy_reward * discount, info.reward)))

    critic_info = info.critic
    critic_losses = []

    # Compute loss for each critic replica
    for i, loss_fn in enumerate(self._critic_losses):
        critic_loss = loss_fn(
            info=info,
            value=critic_info.critics[:, :, i, ...],
            target_value=critic_info.target_critic).loss

        # RLPD-specific: Apply bootstrap mask
        if self._use_bootstrap_critics:
            bootstrap_mask = info.bootstrap_mask[:, :, i] / self._bootstrap_mask_prob
            critic_loss = critic_loss * bootstrap_mask

        critic_losses.append(critic_loss)

    critic_loss = math_ops.add_n(critic_losses)

    # Optional: Calculate priority for prioritized replay
    if self._calculate_priority:
        valid_masks = (info.step_type != StepType.LAST).to(torch.float32)
        valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
        priority = ((critic_loss * valid_masks).sum(dim=0) / valid_n).sqrt()
    else:
        priority = ()

    return LossInfo(
        loss=critic_loss,
        priority=priority,
        extra=critic_loss / float(self._num_critic_replicas))
```

### Step 6: Override Helper Methods (As Needed)

You may need to override various helper methods:

```python
def _compute_critics(self, critic_net, observation, action,
                     critics_state, **kwargs):
    """Customize how critic values are computed."""
    # Add custom logic, then call parent or implement from scratch
    return super()._compute_critics(
        critic_net, observation, action, critics_state, **kwargs)

def after_update(self, root_inputs, info):
    """Called after each gradient update.

    Use this for:
    - Updating target networks
    - Adjusting hyperparameters
    - Logging custom metrics
    """
    # Your custom post-update logic
    self._update_custom_state()

    # Call parent to handle standard updates (e.g., target network updates)
    super().after_update(root_inputs, info)

def after_train_iter(self, inputs, info):
    """Called after each training iteration.

    Use this for:
    - Periodic parameter resets
    - Logging iteration-level metrics
    """
    super().after_train_iter(inputs, info)
```

**RLPD Example** - Custom critic computation with subsampling:
```python
def _compute_critics(self,
                     critic_net,
                     observation,
                     action,
                     critics_state,
                     replica_consensus='mean',  # RLPD: configurable consensus
                     sample_subset=False,        # RLPD: target critic subsampling
                     apply_reward_weights=True):
    """Compute critics with optional subsampling of target critics."""

    observation = (observation, action)
    critics, critics_state = critic_net(observation, state=critics_state)

    # Reshape for multi-dim reward
    if self.has_multidim_reward():
        remaining_shape = critics.shape[2:]
        critics = critics.reshape(-1, self._num_critic_replicas,
                                  *self._reward_spec.shape,
                                  *remaining_shape)

    # RLPD-specific: Subsample target critics for lower variance
    if sample_subset and self._num_sampled_critic_targets < self._num_critic_replicas:
        critics = critics[:,
                          torch.randperm(self._num_critic_replicas)
                          [:self._num_sampled_critic_targets], ...]

    # Apply consensus strategy
    if replica_consensus == 'min':
        if self.has_multidim_reward():
            sign = self.reward_weights.sign()
            critics = (critics * sign).min(dim=1)[0] * sign
        else:
            critics = critics.min(dim=1)[0]
    elif replica_consensus == 'mean':
        critics = critics.mean(dim=1)

    if apply_reward_weights and self.has_multidim_reward():
        critics = self._apply_reward_weights(critics)

    return critics, critics_state

def after_update(self, root_inputs, info: RlpdInfo):
    """Update training mode after each gradient update."""
    self._update_train_mode()  # Switch between actor/critic modes
    super().after_update(root_inputs, info)

def _update_train_mode(self):
    """Alternate between actor and critic training modes."""
    if self._train_mode == TrainMode.actor:
        if self._actor_update_counter % self._actor_utd == 0:
            self._train_mode = TrainMode.critic
    elif self._train_mode == TrainMode.critic:
        if self._critic_update_counter % self._critic_utd == 0:
            self._train_mode = TrainMode.actor
```

---

## Key Implementation Patterns

### Pattern 1: Using `namedtuple` for Information Flow

ALF uses `namedtuple` extensively for passing structured data:

```python
# Define at module level
YourInfo = namedtuple("YourInfo", ["field1", "field2", "field3"],
                      default_value=())

# Use in methods
def train_step(self, inputs, state, rollout_info):
    # Create info
    info = YourInfo(field1=value1, field2=value2, field3=value3)
    return AlgStep(action, state, info)

def calc_loss(self, info: YourInfo):
    # Access fields
    loss = self._compute_loss(info.field1, info.field2)
    return LossInfo(loss=loss)
```

**Benefits**:
- Type hints for better IDE support
- Immutable (use `_replace()` to create modified copies)
- Default values avoid errors when fields are optional

### Pattern 2: State Management

Algorithms maintain hierarchical state structures:

```python
# Define state spec in __init__
YourState = namedtuple("YourState", ["actor", "critic", "custom"])

train_state_spec = YourState(
    actor=actor_network.state_spec,
    critic=critic_network.state_spec,
    custom=custom_component.state_spec)

# In train_step, maintain and update state
def train_step(self, inputs, state: YourState, rollout_info):
    actor_output, actor_state = self._actor_network(obs, state.actor)
    critic_output, critic_state = self._critic_network(obs, state.critic)

    new_state = YourState(
        actor=actor_state,
        critic=critic_state,
        custom=updated_custom_state)

    return AlgStep(output, new_state, info)
```

### Pattern 3: Conditional Updates

For algorithms that don't always update all components:

```python
def train_step(self, inputs, state, rollout_info):
    if self._should_update_actor:
        actor_info = self._actor_train_step(...)
    else:
        actor_info = LossInfo()  # Empty loss

    if self._should_update_critic:
        critic_info = self._critic_train_step(...)
    else:
        critic_info = LossInfo()  # Empty loss

    # Both can coexist in info even if one is empty
    info = YourInfo(actor=actor_info, critic=critic_info)
    return AlgStep(output, state, info)
```

### Pattern 4: Masking and Weighting Losses

Apply masks or weights to losses element-wise:

```python
def _calc_critic_loss(self, info):
    # Base loss computation
    critic_loss = self._critic_loss_fn(
        value=info.critic.critics,
        target_value=info.critic.target_critic).loss

    # Apply custom mask (shape [T, B])
    if self._use_custom_mask:
        mask = self._compute_mask(info)
        critic_loss = critic_loss * mask

    # Apply importance weights from prioritized replay
    if self._use_importance_weights:
        critic_loss = critic_loss * info.importance_weights

    return LossInfo(loss=critic_loss)
```

### Pattern 5: Using `@alf.configurable`

Make your algorithm configurable:

```python
@alf.configurable
class YourAlgorithm(BaseAlgorithm):
    def __init__(self, param1=default1, param2=default2, ...):
        ...

# In config file:
alf.config('YourAlgorithm',
           param1=value1,
           param2=value2)
```

---

## Testing Your Algorithm

### Create a Test File

Create `alf/algorithms/your_algorithm_test.py`:

```python
from absl import logging
from absl.testing import parameterized
from functools import partial
import torch
import unittest

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.your_algorithm import YourAlgorithm
from alf.environments.suite_unittest import PolicyUnittestEnv, ActionType
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


class YourAlgorithmTest(parameterized.TestCase, alf.test.TestCase):

    @parameterized.parameters((1,), (2,))
    def test_your_algorithm(self, reward_dim):
        """Test basic training loop."""
        num_env = 4
        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=1,
            mini_batch_length=2,
            mini_batch_size=64,
            initial_collect_steps=100,
            num_updates_per_train_iter=5)

        steps_per_episode = 13
        env = PolicyUnittestEnv(
            num_env,
            steps_per_episode,
            action_type=ActionType.Continuous,
            reward_dim=reward_dim)

        eval_env = PolicyUnittestEnv(
            100,
            steps_per_episode,
            action_type=ActionType.Continuous,
            reward_dim=reward_dim)

        obs_spec = env._observation_spec
        action_spec = env._action_spec
        reward_spec = env._reward_spec

        # Define networks
        actor_network = partial(
            ActorDistributionNetwork,
            fc_layer_params=(10, 10))

        critic_network = partial(
            CriticNetwork,
            joint_fc_layer_params=(10, 10))

        # Create algorithm
        alg = YourAlgorithm(
            observation_spec=obs_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network,
            critic_network_cls=critic_network,
            # Your custom parameters
            custom_param=custom_value,
            env=env,
            config=config,
            actor_optimizer=alf.optimizers.Adam(lr=1e-2),
            critic_optimizer=alf.optimizers.Adam(lr=1e-2),
            debug_summaries=False,
            name="TestAlgorithm")

        # Train and evaluate
        for i in range(200):
            alg.train_iter()
            if i < config.initial_collect_steps:
                continue

            eval_env.reset()
            eval_time_step = unroll(eval_env, alg, steps_per_episode - 1)
            logging.log_every_n_seconds(
                logging.INFO,
                "%d reward=%f" % (i, float(eval_time_step.reward.mean())),
                n_seconds=1)

        # Assert performance threshold
        self.assertAlmostEqual(
            1.0, float(eval_time_step.reward.mean()), delta=0.3)


if __name__ == '__main__':
    alf.test.main()
```

**RLPD Test Pattern**:
```python
@parameterized.parameters(
    (True, 1, 1),              # Standard mode
    (False, 3, 2),             # With bootstrap
    (True, 1, 1, 1),           # Alternating updates
    (True, 2, 1, 2, True))     # Full RLPD features
def test_rlpd_algorithm(self,
                        use_naive_parallel_network,
                        reward_dim,
                        num_sampled_critic_targets,
                        actor_utd=None,
                        critic_utd=None,
                        use_bootstrap_critics=False):
    # Test with different configurations
    ...
```

### Run Tests

```bash
# Run your specific test
python -m unittest alf.algorithms.your_algorithm_test

# Run with pytest (if available)
pytest alf/algorithms/your_algorithm_test.py -v

# Run all algorithm tests
pytest alf/algorithms/ -k "test" -v
```

---

## Creating Configuration Files

### Basic Configuration File

Create `alf/examples/your_algo_env_conf.py`:

```python
"""Configuration for YourAlgorithm on EnvironmentName."""

import alf
from functools import partial
from alf.algorithms.agent import Agent
from alf.algorithms.your_algorithm import YourAlgorithm

# Import base configs if extending existing setups
# from alf.examples import some_base_conf

# Configure environment
alf.config('create_environment',
           env_name='YourEnv-v0',
           num_parallel_environments=32)

# Configure networks
actor_network_cls = partial(
    alf.networks.ActorDistributionNetwork,
    fc_layer_params=(256, 256),
    continuous_projection_net_ctor=partial(
        alf.networks.NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=alf.math.clipped_exp))

critic_network_cls = partial(
    alf.networks.CriticNetwork,
    joint_fc_layer_params=(256, 256))

# Configure algorithm
alf.config('YourAlgorithm',
           actor_network_cls=actor_network_cls,
           critic_network_cls=critic_network_cls,
           # Your algorithm-specific parameters
           custom_param1=value1,
           custom_param2=value2,
           # Common parameters
           target_update_tau=0.005,
           target_update_period=1)

# Configure agent wrapper
alf.config('Agent',
           optimizer=alf.optimizers.Adam(lr=3e-4),
           rl_algorithm_cls=YourAlgorithm)

# Configure trainer
alf.config('TrainerConfig',
           algorithm_ctor=Agent,
           num_iterations=0,
           num_env_steps=1000000,
           unroll_length=1,
           mini_batch_length=2,
           mini_batch_size=256,
           num_updates_per_train_iter=1,
           evaluate=True,
           eval_interval=10000,
           debug_summaries=True,
           summarize_grads_and_vars=True,
           summary_interval=100)
```

**RLPD Configuration Pattern**:
```python
from functools import partial
import alf
from alf.algorithms.agent import Agent
from alf.algorithms.rlpd_algorithm import RlpdAlgorithm
from alf.examples.benchmarks.dm_control import dmc_conf

# Reuse base configuration
actor_network_cls = dmc_conf.actor_distribution_network_cls

# CRUCIAL: Enable LayerNorm for high UTD training
critic_network_cls = partial(
    alf.networks.CriticNetwork,
    joint_fc_layer_params=dmc_conf.hidden_layers,
    use_fc_ln=True)  # Essential for RLPD stability

alf.config('Agent',
           optimizer=dmc_conf.optimizer,
           rl_algorithm_cls=RlpdAlgorithm)

alf.config('RlpdAlgorithm',
           actor_network_cls=actor_network_cls,
           critic_network_cls=critic_network_cls,
           # RLPD-specific settings
           num_critic_replicas=10,
           num_sampled_critic_targets=1,    # Lower variance targets
           use_bootstrap_critics=False,      # Optional variance reduction
           bootstrap_mask_prob=0.8,
           actor_utd=3,                     # Actor updates per iteration
           critic_utd=10,                   # Critic updates per iteration
           use_entropy_reward=True,
           target_update_tau=0.005)

# Custom entropy target
alf.config('calc_default_target_entropy', min_prob=0.184)

alf.config('TrainerConfig',
           algorithm_ctor=Agent,
           whole_replay_buffer_training=False,
           clear_replay_buffer=False,
           # Total UTD must equal actor_utd + critic_utd
           num_updates_per_train_iter=13,
           summarize_gradient_noise_scale=False,
           summarize_action_distributions=False,
           random_seed=0)
```

### Training with Your Configuration

```bash
# Train with your config
python -m alf.bin.train \
    --conf=alf/examples/your_algo_env_conf.py \
    --root_dir=~/tmp/your_algo_runs/run1

# Override parameters
python -m alf.bin.train \
    --conf=alf/examples/your_algo_env_conf.py \
    --root_dir=~/tmp/your_algo_runs/run2 \
    --conf_param='YourAlgorithm.custom_param1=new_value' \
    --conf_param='TrainerConfig.num_env_steps=5000000'

# Multi-GPU training
python -m alf.bin.train \
    --conf=alf/examples/your_algo_env_conf.py \
    --root_dir=~/tmp/your_algo_runs/run3 \
    --distributed multi-gpu
```

---

## Summary Checklist

When implementing a new algorithm, ensure you:

- [ ] Choose the appropriate base class (`Algorithm`, `RLAlgorithm`, `OffPolicyAlgorithm`, etc.)
- [ ] Define custom `namedtuple` structures for information flow
- [ ] Implement `__init__()` with proper parameter passing to parent
- [ ] Override `rollout_step()` if you need custom data collection behavior
- [ ] Override `train_step()` with your training logic
- [ ] Implement `calc_loss()` to compute losses from train_info
- [ ] Override helper methods as needed (`after_update()`, `after_train_iter()`, etc.)
- [ ] Add `@alf.configurable` decorator
- [ ] Create comprehensive unit tests
- [ ] Write a configuration file for typical use cases
- [ ] Document all parameters and special behaviors in docstrings
- [ ] Add algorithm to README.md if it's a published algorithm

---

## Additional Resources

- **Algorithm Base Classes**: `alf/algorithms/algorithm.py`, `alf/algorithms/rl_algorithm.py`
- **Example Algorithms**: Check `alf/algorithms/sac_algorithm.py`, `alf/algorithms/ppo_algorithm.py`
- **Network Building**: See `alf/networks/` for available network architectures
- **Loss Functions**: `alf/algorithms/one_step_loss.py`, `alf/algorithms/multi_step_loss.py`
- **Testing Utilities**: `alf/test.py`, `alf/environments/suite_unittest.py`

For questions or contributions, refer to the [ALF documentation](https://alf.readthedocs.io/) and the [contributing guide](https://alf.readthedocs.io/en/latest/contributing.html).
