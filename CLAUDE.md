# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ALF (Agent Learning Framework) is a PyTorch-based reinforcement learning framework designed for implementing complex algorithms with many components. The framework emphasizes flexibility and ease of implementation, supporting both on-policy and off-policy algorithms, as well as hybrid training modes combining online RL with offline data.

## Build, Test, and Run Commands

### Training
```bash
# Train with ALF Python config file (preferred)
python -m alf.bin.train --conf=CONF_FILE --root_dir=LOG_DIR

# Train with legacy Gin config file
python -m alf.bin.train --gin_file=GIN_FILE --root_dir=LOG_DIR

# Single-node multi-GPU training
python -m alf.bin.train --conf=CONF_FILE --root_dir=LOG_DIR --distributed multi-gpu

# Multi-node multi-GPU training (use on each node)
torchrun \
    --nproc_per_node=NGPU_ON_NODE \
    --nnodes=NUMBER_OF_NODES \
    --node_rank=NODE_RANK \
    --master_addr=HOST_IP \
    --master_port=12345 \
    ./alf/bin/train.py \
    --conf=CONF_FILE \
    --root_dir=LOG_DIR \
    --distributed multi-node-multi-gpu
```

### Evaluation and Visualization
```bash
# Evaluate trained model with visualization
python -m alf.bin.play --root_dir=LOG_DIR

# Monitor training with TensorBoard
tensorboard --logdir=LOG_DIR
```

### Configuration
```bash
# Override config parameters via command line
python -m alf.bin.train --conf=CONF_FILE --root_dir=LOG_DIR \
    --conf_param='TrainerConfig.num_iterations=10000' \
    --conf_param='create_environment.num_parallel_environments=32'
```

### Testing
```bash
# Run tests (based on CI configuration)
pytest alf/
```

## Architecture and Key Concepts

### Core Algorithm Structure

The framework is built around a hierarchical algorithm design:

1. **Algorithm Base Class** (`alf/algorithms/algorithm.py`):
   - Central abstraction for all learning algorithms
   - Manages three distinct execution modes with separate state specifications:
     - `predict_step()`: for deployment/inference
     - `rollout_step()`: for data collection during training
     - `train_step()`: for computing gradients and updating parameters
   - Handles optimizer management, including support for multiple optimizers per algorithm
   - Supports nested sub-algorithms with automatic optimizer assignment
   - Implements custom `state_dict()` and `load_state_dict()` to handle cycles, parameter sharing, and optimizer states

2. **Training Phases**:
   - **On-policy training**: Uses experiences directly from `rollout_step()` for training
   - **Off-policy training**: Stores experiences in replay buffer, samples batches for `train_step()`
   - **Hybrid training**: Combines online RL replay buffer with offline demonstration data

3. **State Management**:
   - `train_state_spec`: RNN states for training
   - `rollout_state_spec`: RNN states for rollout (can differ from train states)
   - `predict_state_spec`: RNN states for prediction/deployment
   - `use_rollout_state` flag controls whether training uses rollout states from replay buffer or zeros

### Configuration System

ALF uses a custom Python-based configuration system (`alf.config()`), replacing the older Gin-config system:

- **Config Files**: Located in `alf/examples/*_conf.py`
- **Config Helpers** (`alf/config_helpers.py`):
  - `get_observation_spec()`: Get observation spec after data transformers
  - `get_raw_observation_spec()`: Get raw environment observation spec
  - `get_action_spec()`: Get action spec from environment
  - `parse_config()`: Parse config file and initialize environment
- **TrainerConfig** (`alf/algorithms/config.py`): Central configuration for training parameters
- **Multi-process adjustments**: Config automatically adjusts for distributed training (DDP)

### Data Flow Architecture

1. **Environment Interaction**:
   ```
   Environment → TimeStep → Algorithm.rollout_step() → AlgStep → Action
   ```

2. **Training Pipeline**:
   ```
   Experience (from rollout or replay buffer)
   → DataTransformer.transform_experience()
   → Algorithm.train_step()
   → LossInfo
   → Algorithm.calc_loss()
   → Algorithm.update_with_gradient()
   ```

3. **Replay Buffer** (`alf/experience_replayers/replay_buffer.py`):
   - Stores experiences for off-policy training
   - Supports prioritized sampling
   - Handles frame stacking and multi-step returns
   - Shared memory support for multi-process environments

### Key Components

- **Networks** (`alf/networks/`):
  - Modular network architectures (encoding, actor, critic, value networks)
  - Support for recurrent models (LSTM, GRU)
  - Specialized networks like normalizing flows, spatial broadcast decoders

- **Algorithms** (`alf/algorithms/`):
  - Each algorithm inherits from `Algorithm` base class
  - Implements three core methods: `predict_step()`, `rollout_step()`, `train_step()`
  - Examples: A2C, PPO, SAC, DDPG, MuZero, IQL, Flow Matching
  - Algorithms can be composed hierarchically (e.g., safe RL with editor policies)

- **Trainers** (`alf/trainers/policy_trainer.py`):
  - `RLTrainer`: Main training loop for RL algorithms
  - `SLTrainer`: For supervised learning tasks
  - Handles unrolling, training iterations, checkpointing, evaluation

- **Data Transformers** (`alf/algorithms/data_transformer.py`):
  - Transform observations (e.g., frame stacking, reward normalization)
  - Applied both during rollout and when retrieving from replay buffer
  - State-preserving transformations

### Distributed Training

- **Single-node multi-GPU**: Uses PyTorch DDP with process spawning
- **Multi-node multi-GPU**: Uses `torchrun` launcher
- **Key considerations**:
  - Config parameters automatically adjusted per process (e.g., `mini_batch_size`, `num_parallel_environments`)
  - Evaluation only runs on rank 0
  - Gradients synchronized via DDP hooks

## Important Implementation Details

### Optimizer Management

- Algorithms can have a default optimizer and additional optimizers for specific sub-modules
- Use `add_optimizer()` to assign an optimizer to specific modules/parameters
- `_setup_optimizers()` automatically assigns unhandled parameters to default optimizer
- Override `_trainable_attributes_to_ignore()` to exclude certain attributes from optimization

### Checkpoint Loading

- Algorithms support pre-loading checkpoints via `checkpoint` parameter: `"prefix@path"`
- Example: `checkpoint="alg._sub_alg1@/path/to/ckpt-100"` loads checkpoint subset
- Prefix defaults to "alg" if omitted
- Pre-loaded checkpoints prevent re-loading during training initialization

### RNN State Handling

- For off-policy algorithms with RNNs:
  - Set `TrainerConfig.use_rollout_state=True` to use states from replay buffer
  - Set `mini_batch_length > 1` for proper temporal correlation
  - Warning: `mini_batch_length=1` with RNNs means no gradient flow through recurrent connections

### Hybrid Training (Online RL + Offline Data)

- Specify `offline_buffer_dir` in TrainerConfig to enable hybrid training
- Control mixing with `rl_train_after_update_steps` and `rl_train_every_update_steps`
- Implement `train_step_offline()` and `calc_loss_offline()` if default behavior insufficient
- Offline loss weighted by `offline_loss_weight()`

### Memory Management

- Set `TrainerConfig.empty_cache=True` to clear CUDA cache between updates (reduces memory at cost of speed)
- Algorithms track CPU and GPU memory usage in summaries
- Experience specs cached to avoid repeated computation

## Common Development Patterns

### Creating a New Algorithm

1. Inherit from `Algorithm` or a more specific base (e.g., `RLAlgorithm`, `OnPolicyAlgorithm`, `OffPolicyAlgorithm`)
2. Implement required methods:
   - `__init__()`: Initialize networks, optimizers, state specs
   - `predict_step()`: Inference-only forward pass
   - `rollout_step()`: Data collection with exploration
   - `train_step()`: Training forward pass
   - `calc_loss()`: Compute loss from train_info
3. Register optimizers for sub-networks if needed
4. Define state specs if using recurrent models

### Writing Config Files

Place configs in `alf/examples/` with pattern `<algorithm>_<env>_conf.py`:

```python
import alf
from alf.algorithms.sac_algorithm import SacAlgorithm

alf.config('create_environment',
           env_name='Pendulum-v0',
           num_parallel_environments=8)

alf.config('SacAlgorithm',
           actor_network_ctor=...,
           critic_network_ctor=...,
           optimizer=alf.optimizers.Adam(lr=3e-4))

alf.config('TrainerConfig',
           algorithm_ctor=SacAlgorithm,
           num_iterations=10000,
           unroll_length=1,
           mini_batch_length=2,
           mini_batch_size=256)
```

### Adding Custom Networks

- Place custom networks in `alf/networks/`
- Inherit from `alf.networks.Network` or specific base classes
- Use `alf.config()` to register and configure networks
- Leverage existing building blocks from `alf.layers` and `alf.networks`

## File Organization

```
alf/
├── algorithms/          # All RL and learning algorithms
│   ├── algorithm.py     # Base Algorithm class
│   ├── config.py        # TrainerConfig and algorithm configs
│   └── *_algorithm.py   # Specific algorithm implementations
├── bin/                 # Entry points for training, playing, evaluation
│   ├── train.py         # Main training script
│   └── play.py          # Evaluation and visualization script
├── environments/        # Environment wrappers and utilities
├── examples/            # Training configs for various tasks
│   └── *_conf.py        # Python config files (preferred)
├── experience_replayers/ # Replay buffer implementations
├── networks/            # Neural network architectures
├── trainers/            # Training loop implementations
├── utils/               # Common utilities
├── config_helpers.py    # Config system helpers
├── config_util.py       # Core config system implementation
└── tensor_specs.py      # Tensor specification utilities
```
