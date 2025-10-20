# ALF Algorithm Framework - Interface Specifications

This document provides detailed interface documentation for the core algorithm classes in ALF, including method signatures, parameter types, return types, and important implementation invariants.

---

## Algorithm

**Location:** `alf/algorithms/algorithm.py`

**Purpose:** Base class for all learning algorithms (both RL and non-RL). Provides core infrastructure for state management, optimizer handling, checkpointing, and gradient updates.

### Constructor

```python
Algorithm(
    train_state_spec=(),
    rollout_state_spec=None,
    predict_state_spec=None,
    is_on_policy=None,
    optimizer=None,
    checkpoint=None,
    config: TrainerConfig = None,
    debug_summaries=False,
    name="Algorithm"
)
```

**Parameters:**
- `train_state_spec` (nested TensorSpec): RNN state spec for `train_step()`
- `rollout_state_spec` (nested TensorSpec | None): RNN state spec for `rollout_step()`. If None, defaults to `train_state_spec`
- `predict_state_spec` (nested TensorSpec | None): RNN state spec for `predict_step()`. If None, defaults to `rollout_state_spec`
- `is_on_policy` (None | bool): Algorithm mode (None means undetermined)
- `optimizer` (torch.optim.Optimizer | None): Default optimizer for all parameters
- `checkpoint` (str | None): Format: `"prefix@path"` for pre-loading checkpoints (e.g., `"alg._sub_alg@/path/to/ckpt-100"`)
- `config` (TrainerConfig | None): Training configuration
- `debug_summaries` (bool): Enable debug summaries
- `name` (str): Algorithm name

### Core Execution Methods (to be implemented by subclasses)

#### predict_step
```python
def predict_step(inputs: TimeStep, state) -> AlgStep
```

**Purpose:** Inference-only forward pass for deployment/evaluation.

**Parameters:**
- `inputs` (TimeStep): Current observation and auxiliary data
  - Shape: `[B, ...]` where B is batch size
  - Fields: `step_type`, `reward`, `discount`, `observation`, `prev_action`, `env_id`
- `state` (nested Tensor): Must match `predict_state_spec`
  - For RNNs: typically `[B, state_dim]`
  - For feedforward: `()` (empty)

**Returns:** `AlgStep` namedtuple:
- `output` (nested Tensor): Action or policy output, matches `action_spec`
  - Shape: `[B, action_dim]` for continuous actions
  - Type: sampled action (not distribution)
- `state` (nested Tensor): Updated state, matches `predict_state_spec`
- `info` (nested Tensor): Auxiliary info (usually empty `()` for predict)

**Implementation Notes:**
- Default implementation calls `rollout_step()` and strips info
- Should not accumulate gradients
- Typically runs in `torch.no_grad()` context

---

#### rollout_step
```python
def rollout_step(inputs: TimeStep, state) -> AlgStep
```

**Purpose:** Data collection step during training. Generates actions with exploration and collects training info.

**Parameters:**
- `inputs` (TimeStep): Same as `predict_step`
  - Batch size: B (number of parallel environments)
  - Shape: `[B, ...]`
- `state` (nested Tensor): Must match `rollout_state_spec`
  - For RNNs: `[B, ...]`

**Returns:** `AlgStep` namedtuple:
- `output` (nested Tensor | Distribution): Action or action distribution
  - For stochastic policies: distribution object (e.g., Normal, Categorical)
  - For deterministic policies: action tensor `[B, action_dim]`
  - This output is what gets executed in the environment
- `state` (nested Tensor): Updated state, matches `rollout_state_spec`
- `info` (nested Tensor): Training info collected for this step
  - Will be temporally batched with other steps into `[T, B, ...]`
  - Used by `calc_loss()` for training
  - Common fields: `action_distribution`, `value`, `log_prob`, etc.

**Batching Behavior:**
- On-policy: All B environments run the **same policy** with the same algorithm instance
- Off-policy: Still same policy, but experiences stored in replay buffer for later training

**Implementation Notes:**
- Called every environment step during rollout
- Must generate valid outputs for ALL B environments simultaneously
- Cannot run different algorithm instances for different environments in a batch
- Info is collected and later stacked as: `[unroll_length, B, ...]`

---

#### train_step
```python
def train_step(inputs: TimeStep, state, rollout_info) -> AlgStep
```

**Purpose:** Training forward pass. Generates training outputs given experience (used for off-policy only).

**Parameters:**
- `inputs` (TimeStep): Experience retrieved from replay buffer
  - Shape: `[B, ...]` (potentially `[mini_batch_size, mini_batch_length, ...]` after reshaping)
  - Contains observations, rewards, discounts from past environment interactions
- `state` (nested Tensor): Must match `train_state_spec`
  - For on-policy: typically empty `()`
  - For off-policy with RNNs: `[B, ...]` from replay buffer or zeros
  - Can be different structure from `rollout_state_spec`
- `rollout_info` (nested Tensor): Info collected from corresponding `rollout_step()`
  - Retrieved from replay buffer for off-policy
  - Empty for on-policy training

**Returns:** `AlgStep` namedtuple:
- `output` (nested Tensor): Usually not used in training (gradient computed from `info`)
- `state` (nested Tensor): Updated state, matches `train_state_spec`
- `info` (nested Tensor): Training info for loss computation
  - Shape: `[B, ...]` or `[mini_batch_size, mini_batch_length, ...]`
  - Must be convertible to LossInfo by `calc_loss()`

**For OnPolicyAlgorithm:**
- Default implementation delegates to `rollout_step()`
- This allows on-policy algorithms to be used in off-policy contexts

**Implementation Notes:**
- For off-policy algorithms: can have completely different logic from `rollout_step()`
- For RNN algorithms: use `train_state_spec` which may differ from `rollout_state_spec`
- Called on mini-batches during training phase

---

### Loss Computation and Optimization

#### calc_loss
```python
def calc_loss(info: nested Tensor) -> LossInfo
```

**Purpose:** Compute loss from training info.

**Parameters:**
- `info` (nested Tensor): Training info from `rollout_step()` or `train_step()`
  - Shape: `[T, B, ...]` (time-major)
  - Typically contains: action, value estimates, target values, etc.

**Returns:** `LossInfo` namedtuple:
- `loss` (Tensor | ()): Scalar or `[T, B]` tensor for aggregation
- `scalar_loss` (Tensor | ()): Pre-computed scalar loss (optional)
- `priority` (Tensor | ()): For prioritized experience replay, `[B]` shape
- `extra` (nested Tensor): Additional info for summaries

**Shape Invariants:**
- Output `.loss` should have shape `[T, B]` for proper masking and weighting
- Output `.scalar_loss` must be scalar (rank-0)

---

#### update_with_gradient
```python
def update_with_gradient(
    loss_info: LossInfo,
    valid_masks: Tensor = None,
    weight: float = 1.0,
    batch_info: BatchInfo = None
) -> tuple[LossInfo, list[Parameter]]
```

**Purpose:** Perform one gradient update step.

**Parameters:**
- `loss_info` (LossInfo): Loss computed from `calc_loss()`
- `valid_masks` (Tensor | None):
  - Shape: `[T, B]`, dtype: float32
  - Values in [0, 1], indicates which samples are valid
  - Used to mask out invalid transitions (e.g., last steps in episode)
- `weight` (float): Multiplier for loss before backward pass
- `batch_info` (BatchInfo | None): Metadata from replay buffer
  - Contains `importance_weights` for prioritized sampling

**Returns:** tuple of:
- `loss_info` (LossInfo): Aggregated and averaged loss
- `params` (list[(name: str, param: Parameter)]): All updated parameters

**Implementation Details:**
- Aggregates loss across time and batch dimensions using masks and weights
- Applies importance weights from prioritized sampling if available
- Calls optimizer.zero_grad(), backward(), and optimizer.step()

---

### State Management

#### State Spec Properties

```python
@property
def train_state_spec(self) -> nested TensorSpec
def rollout_state_spec(self) -> nested TensorSpec
def predict_state_spec(self) -> nested TensorSpec
```

**Purpose:** Retrieve RNN state specifications for each execution mode.

**Invariants:**
- All three are nested TensorSpecs (can be empty `()` for feedforward models)
- By default: if not specified, all three are the same
- For algorithms with RNNs: typically all three are identical LSTMStateTensorSpec
- For algorithms with different training vs. rollout: `train_state_spec ⊆ rollout_state_spec` (train is subset)

---

#### get_initial_*_state
```python
def get_initial_train_state(batch_size: int) -> nested Tensor
def get_initial_rollout_state(batch_size: int) -> nested Tensor
def get_initial_predict_state(batch_size: int) -> nested Tensor
```

**Purpose:** Create zero-initialized states for a given batch size.

**Parameters:**
- `batch_size` (int): Number of parallel environments/samples

**Returns:** nested Tensor:
- Matches corresponding state spec
- Shape: `[batch_size, ...]`
- Values: zeros
- Cached internally for performance

---

### Replay Buffer Integration

#### set_replay_buffer
```python
def set_replay_buffer(
    num_envs: int,
    max_length: int,
    prioritized_sampling: bool = False
) -> None
```

**Purpose:** Configure replay buffer parameters (lazy initialization).

**Parameters:**
- `num_envs` (int): Total number of parallel environments
- `max_length` (int): Maximum number of steps stored per environment
- `prioritized_sampling` (bool): Enable prioritized experience replay

**Notes:**
- Must be called before `observe_for_replay()`
- Actual buffer created lazily when first experience is observed
- For on-policy algorithms: not needed

---

#### observe_for_replay
```python
def observe_for_replay(exp: Experience) -> None
```

**Purpose:** Store experience to replay buffer (off-policy only).

**Parameters:**
- `exp` (Experience): Single step experience
  - Shape: `[B, ...]`
  - Fields: `time_step`, `action`, `state`, `rollout_info`

**Implementation Details:**
- Converts distributions to parameters for storage
- Prunes rollout state if `use_rollout_state=False`
- Lazily initializes replay buffer on first call
- Atomic operation

---

#### transform_timestep & transform_experience
```python
def transform_timestep(time_step: TimeStep, state) -> tuple[TimeStep, nested Tensor]
def transform_experience(experience: Experience) -> Experience
```

**Purpose:** Apply data transformations (frame stacking, normalization, etc.).

**Parameters:**
- `time_step` (TimeStep): Raw observation from environment
- `experience` (Experience): Experience from replay buffer

**Returns:**
- Transformed time step/experience with applied transformations
- Transformer state for stateful transformers

**Notes:**
- Called on rollout data before `rollout_step()`
- Called on replayed data before `train_step()`
- Configured via `config.data_transformer`

---

### Optimizer Management

#### add_optimizer
```python
def add_optimizer(
    optimizer: torch.optim.Optimizer,
    modules_and_params: list[Module | Parameter]
) -> None
```

**Purpose:** Assign a specific optimizer to a set of modules/parameters.

**Parameters:**
- `optimizer` (Optimizer): PyTorch optimizer instance
- `modules_and_params` (list): Modules (nn.Module, nn.ModuleList, nn.ParameterList) or individual Parameters

**Notes:**
- Called during algorithm initialization
- Parameters not assigned to any optimizer get default optimizer
- Multiple optimizers supported for different components

---

#### optimizers
```python
def optimizers(recurse: bool = True, include_ignored_attributes: bool = False) -> list[Optimizer]
```

**Purpose:** Get all optimizers in this algorithm and sub-algorithms.

---

### Checkpointing

#### Checkpoint Loading
- Format: `"prefix@path"` (prefix optional, defaults to `"alg"`)
- Example: `"alg._actor@/path/to/ckpt-100"` loads only the actor
- Pre-loaded before training starts in `_post_init()`
- Supports hierarchical loading via dot-notation

---

---

## RLAlgorithm

**Location:** `alf/algorithms/rl_algorithm.py`

**Parent:** `Algorithm`

**Purpose:** Abstract base class for all RL algorithms. Adds RL-specific functionality including environment interaction, unrolling, metrics, and offline training support.

### Constructor

```python
RLAlgorithm(
    observation_spec: nested TensorSpec,
    action_spec: BoundedTensorSpec,
    train_state_spec: nested TensorSpec,
    reward_spec: TensorSpec = TensorSpec(()),
    predict_state_spec: TensorSpec | None = None,
    rollout_state_spec: TensorSpec | None = None,
    is_on_policy: bool | None = None,
    reward_weights: list[float] | None = None,
    env: Environment | None = None,
    config: TrainerConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    checkpoint: str | None = None,
    is_eval: bool = False,
    overwrite_policy_output: bool = False,
    debug_summaries: bool = False,
    name: str = "RLAlgorithm"
)
```

**New Parameters:**
- `observation_spec` (nested TensorSpec): Specification of observations from environment
  - Shape: typically `[obs_dim]` for vectors, `[C, H, W]` for images
  - Can be nested for multi-modal observations
- `action_spec` (BoundedTensorSpec): Action space specification
  - Must be BoundedTensorSpec with `.minimum` and `.maximum`
  - Shape: `[action_dim]`
  - Example: `BoundedTensorSpec(shape=[2], minimum=[-1.0, -1.0], maximum=[1.0, 1.0])`
- `reward_spec` (TensorSpec): Reward specification
  - Default: scalar `TensorSpec(())`
  - Can be multi-dimensional for multi-task learning
- `reward_weights` (list[float] | None): Weights for multi-dimensional rewards
  - Only used if `reward_spec.numel > 1`
  - Must match reward dimensions
- `env` (Environment): Batched environment for interaction
  - Expected: runs B parallel environments
  - B accessed via `env.batch_size`
  - Only required for root algorithm
- `is_eval` (bool): Evaluation-only mode
  - If True, skip creating components not needed for inference (e.g., critic networks)
- `overwrite_policy_output` (bool): Whether to overwrite action with `next_step.prev_action`
  - Useful for data collection from expert policies

### Environment Interaction

#### unroll
```python
def unroll(unroll_length: int) -> Experience | None
```

**Purpose:** Collect experiences from the environment for `unroll_length` steps.

**Parameters:**
- `unroll_length` (int): Number of environment steps to collect per environment

**Returns:** `Experience` namedtuple:
- `time_step` (TimeStep): Observations and rewards, shape `[T, B, ...]`
  - Where T = unroll_length, B = batch_size
  - Contains: `step_type`, `reward`, `discount`, `observation`, `prev_action`, `env_id`
- `action` (Tensor): Actions executed, shape `[T, B, action_dim]`
- `state` (nested Tensor): RNN states at each step, shape `[T, B, ...]`
- `rollout_info` (nested Tensor): Info from `rollout_step()`, shape `[T, B, ...]`
- `discount` (Tensor): Episode discount, shape `[T, B]`

**Batching Details:**
- All B environments pass through the SAME algorithm instance's `rollout_step()`
- All B environments proceed synchronously (same forward pass)
- All B environments get different outputs based on different observations/states
- **Conditional routing is supported:** You can selectively invoke different sub-algorithms for different batch elements using `conditional_update()` or masking
- If any environment resets (step_type=FIRST), state is reset via `reset_state_if_necessary()`

**Implementation Notes:**
- Calls `rollout_step()` T times
- After each rollout, calls `env.step(action)` to get next observation
- Transforms observations via `transform_timestep()`
- Stores experiences to replay buffer if off-policy (via `observe_for_replay()`)
- Returns None if no data collected (can happen with async_unroll)

---

#### _sync_unroll (implementation detail)
```python
def _sync_unroll(unroll_length: int) -> Experience
```

**Purpose:** Synchronous implementation of `unroll()`.

**Key Implementation:**
```python
for step in range(unroll_length):
    # Reset state if necessary (episode boundary)
    state = reset_state_if_necessary(state, initial_state, time_step.is_first())

    # Transform raw observation
    transformed_time_step, trans_state = transform_timestep(time_step, trans_state)

    # Get action from policy
    policy_step = rollout_step(transformed_time_step, state)

    # Execute in environment
    next_time_step = env.step(policy_step.output)

    # Store experience if off-policy
    if not on_policy:
        observe_for_replay(experience)

    # Collect training info
    experience_list.append(experience)
```

**Batching Semantics (detailed):**
- Example with B=4 environments running PPO:
  ```
  t=0: All 4 envs get observations
       All 4 envs pass through same policy network
       All 4 envs get same policy but different observations
       → 4 different actions sampled
       → 4 different rewards obtained

  t=1: All 4 envs continue with previous state
       (No reset since no episode ends)
  ...

  Result: experience.time_step.shape = [T, 4, obs_dim]
          experience.action.shape = [T, 4, action_dim]
  ```

---

### Training Iteration

#### train_iter
```python
def train_iter() -> int
```

**Purpose:** Perform one complete training iteration (rollout + training).

**Returns:** int - number of training samples processed

**Routing:**
- On-policy: calls `_train_iter_on_policy()`
- Off-policy: calls `_train_iter_off_policy()`

**On-Policy Flow:**
```python
experience = unroll(unroll_length)  # [T, B, ...]
train_info = experience.rollout_info
loss_info = calc_loss(train_info)
loss_info, params = update_with_gradient(loss_info, valid_masks)
after_update(time_step, train_info)
return T * B  # steps processed
```

**Off-Policy Flow:**
```python
# Periodically collect experiences
if should_unroll():
    experience = unroll(unroll_length)  # [T, B, ...]
    observe_for_replay(experience)      # Store for later training

# Sample and train from replay buffer
for num_updates:
    experience, batch_info = replay_buffer.get_batch(batch_size, mini_batch_length)
    for mini_batch:
        train_info = collect_train_info(mini_batch)  # Call train_step()
        loss_info = calc_loss(train_info)
        loss_info, params = update_with_gradient(loss_info, ..., batch_info)
```

---

### Metrics

#### get_metrics & get_step_metrics
```python
def get_metrics() -> list[Metric]
def get_step_metrics() -> list[Metric]
```

**Purpose:** Return metrics tracked during training.

**Returns:**
- `get_step_metrics()`: Step metrics only (EnvironmentSteps, NumberOfEpisodes)
- `get_metrics()`: All metrics including AverageReturn, AverageEpisodeLength, etc.

---

### Offline Training Support

#### load_offline_replay_buffer
```python
def load_offline_replay_buffer(
    untransformed_observation_spec: nested TensorSpec,
    ddp_rank: int
) -> None
```

**Purpose:** Load offline replay buffer from checkpoint for hybrid training.

**Parameters:**
- `untransformed_observation_spec`: Observation spec of offline data
- `ddp_rank` (int): DDP worker rank (-1 if not using DDP)

**Implementation Details:**
- Loads replay buffer checkpoint from `config.offline_buffer_dir`
- Supports directory of buffers (one per DDP worker)
- Creates `self._offline_replay_buffer` internally

---

#### train_step_offline & calc_loss_offline
```python
def train_step_offline(
    inputs: TimeStep,
    state: nested Tensor,
    rollout_info: nested Tensor,
    pre_train: bool = False
) -> AlgStep

def calc_loss_offline(
    info_offline: nested Tensor,
    pre_train: bool = False
) -> LossInfo
```

**Purpose:** Training steps specialized for offline data.

**Default Behavior:**
- `train_step_offline()` delegates to `train_step()`
- `calc_loss_offline()` delegates to `calc_loss()`

**Parameters:**
- `pre_train` (bool): If True, algorithm is in pre-training phase (offline-only)
  - Useful for different loss weighting or training procedures

**Hybrid Training Flow:**
- Pre-train phase: train on offline data only
- RL phase: train on both online and offline data with weighting
- Final phase: train on online data only (offline buffer released when weight=0)

---

### Summarization

#### summarize_rollout, summarize_train, summarize_play
```python
def summarize_rollout(
    experience: Experience,
    custom_summary: Callable[[Experience], None] | None = None
) -> None

def summarize_train(
    experience: Experience,
    train_info: nested Tensor,
    loss_info: LossInfo,
    params: list[(str, Parameter)]
) -> None

def summarize_play(
    experience: Experience,
    custom_summary: Callable[[Experience], None] | None = None
) -> None
```

**Purpose:** Generate TensorBoard summaries.

**Parameters:**
- `experience` (Experience): Collected during rollout/training
- `train_info` (nested Tensor): From `rollout_step()` or `train_step()`
- `loss_info` (LossInfo): Computed loss
- `params` (list): Parameters with gradients
- `custom_summary` (Callable | None): User hook for additional summaries

---

## OnPolicyAlgorithm

**Location:** `alf/algorithms/on_policy_algorithm.py`

**Parent:** `OffPolicyAlgorithm` (inherits from it but overrides key behaviors)

**Purpose:** Base class for on-policy algorithms (e.g., PPO, A2C, A3C).

### Key Property

```python
@property
def on_policy(self) -> bool
    return True
```

### Core Method Override

#### train_step (on-policy override)
```python
def train_step(
    inputs: TimeStep,
    state: nested Tensor,
    rollout_info: nested Tensor
) -> AlgStep
```

**Purpose:** For on-policy algorithms, `train_step()` simply delegates to `rollout_step()`.

**Implementation:**
```python
def train_step(self, inputs, state, rollout_info):
    return self.rollout_step(inputs, state)
```

**Why This Works:**
- On-policy algorithms use `rollout_step()` output immediately for training
- No separate "training distribution" - training on collected data
- The `rollout_info` parameter is ignored (collected from same `rollout_step()`)
- Allows on-policy algorithms to be trained in off-policy contexts

---

## OffPolicyAlgorithm

**Location:** `alf/algorithms/off_policy_algorithm.py`

**Parent:** `RLAlgorithm`

**Purpose:** Base class for off-policy algorithms (e.g., SAC, DDPG, DQN, TD3).

### Key Property

```python
@property
def on_policy(self) -> bool
    return False
```

### Training Flow

**Pseudo-code (from docstring):**

```python
# (1) Collection stage - collect experiences during rollout
for step in range(steps_per_collection):
    policy_step = rollout_step(time_step, policy_state)  # Exploration policy
    experience = make_experience(time_step, policy_step)
    store_experience(experience)  # → replay buffer
    action = sample(policy_step.action)  # Sample action
    time_step = env.step(action)

# (2) Training stage - train on sampled experiences from buffer
for train_step in range(training_steps_per_collection):
    experiences = replay_buffer.get_batch()  # Sample batch
    batched_train_info = []
    for experience in experiences:
        policy_step = train_step(experience, state)  # Training policy
        batched_train_info.append(policy_step.info)
    loss = calc_loss(batched_train_info)
    update_with_gradient(loss)
```

**Key Distinctions:**
- `rollout_step()`: Used for data collection with exploration
- `train_step()`: Used for training, may differ from rollout (e.g., different network, no exploration)
- Experiences stored in replay buffer between collection and training

---

## ReplayBuffer

**Location:** `alf/experience_replayers/replay_buffer.py`

**Parent:** `RingBuffer` (circular buffer implementation)

**Purpose:** Stores experiences from environment for off-policy training with support for prioritization and episodic information tracking.

### Constructor

```python
ReplayBuffer(
    data_spec: nested TensorSpec,
    num_environments: int,
    max_length: int = 1024,
    num_earliest_frames_ignored: int = 0,
    prioritized_sampling: bool = False,
    initial_priority: float = 1.0,
    recent_data_steps: int = 1,
    recent_data_ratio: float = 0.0,
    with_replacement: bool = False,
    device: str = "cpu",
    mp_context: multiprocessing.context = None,
    keep_episodic_info: bool | None = None,
    record_episodic_return: bool = False,
    default_return: float = -1000.0,
    gamma: float = 0.99,
    reward_clip: tuple[float, float] | None = None,
    enable_checkpoint: bool = False,
    name: str = "ReplayBuffer"
)
```

**Key Parameters:**
- `data_spec` (nested TensorSpec): Structure of stored experiences
  - Typically: Experience spec with `time_step`, `action`, `state`, etc.
  - Can contain nested structures
- `num_environments` (int): Separate buffer per environment
  - Each environment has independent ring buffer
  - Important for maintaining episode boundaries
- `max_length` (int): Length of ring buffer per environment
  - Total storage: `num_environments * max_length`
- `num_earliest_frames_ignored` (int): Frames to ignore (frame stacking)
  - If > 0, automatically sets `keep_episodic_info=True`
  - Sampling avoids early frames of stacked sequences
- `prioritized_sampling` (bool): Use prioritized experience replay
  - Requires calling `update_priority()` after training
  - Uses segment trees for efficient sampling
- `recent_data_ratio` (float): Fraction of batch from recent experiences
  - Example: 0.2 means 20% from last `recent_data_steps` steps
- `keep_episodic_info` (bool | None): Track episode boundaries
  - If True, uses `_indexed_pos` to track episode starts/ends
  - Needed for frame stacking or episodic returns
- `record_episodic_return` (bool): Compute discounted return per episode
  - Returns stored in `_episodic_discounted_return`
  - Requires `keep_episodic_info=True`
- `gamma` (float): Discount factor for episodic returns
  - Must match `TDLoss.gamma` for consistency

### BatchInfo Named Tuple

```python
BatchInfo(
    env_ids: Tensor,              # [B] int64 - environment IDs
    positions: Tensor,            # [B] int64 - starting positions in ring buffer
    importance_weights: Tensor,   # [B] float - priority weights (if prioritized)
    replay_buffer: ReplayBuffer,  # reference to this buffer
    discounted_return: Tensor     # [B] float - episodic returns (if recorded)
)
```

**Usage:**
- Returned by `get_batch()` and `gather_all()`
- Passed to `update_priority()` after training
- Passed to `Algorithm.update_with_gradient()` for priority-weighted loss

### Core Methods

#### add_batch
```python
@atomic
@torch.no_grad()
def add_batch(
    batch: nested Tensor,
    env_ids: Tensor | None = None,
    blocking: bool = False
) -> None
```

**Purpose:** Add a batch of experiences to the buffer.

**Parameters:**
- `batch` (nested Tensor): Shape `[B, ...]` or `[num_environments, ...]`
  - Must be convertible to device
  - Contains `time_step`, `action`, `state`, etc.
- `env_ids` (Tensor | None): Environment IDs for each sample
  - If None, assumes batch_size == num_environments
  - Allows partial updates (e.g., only env 0 and 2)
  - Shape: `[B]` int64
- `blocking` (bool): Block if buffer is full
  - If False: overwrites oldest data

**Side Effects:**
- Updates episode boundary tracking (if `keep_episodic_info`)
- Initializes priorities (if `prioritized_sampling`)
- Computes episodic returns (if `record_episodic_return`)

**Internal Details:**
```
Position tracking:
- _current_pos[env]: always-incrementing position (never wraps)
- circular(_current_pos[env]): actual index in buffer (0 to max_length-1)
- _current_size[env]: how many steps stored (up to max_length)

Episode tracking (_keep_episodic_info):
- _indexed_pos[env, idx]: stores episode start position for each step
- _headless_indexed_pos[env]: backup for overwritten episode starts
```

---

#### get_batch
```python
@atomic
@torch.no_grad()
def get_batch(
    batch_size: int,
    batch_length: int
) -> tuple[nested Tensor, BatchInfo]
```

**Purpose:** Sample a random batch of trajectories.

**Parameters:**
- `batch_size` (int): Number of trajectories to sample
- `batch_length` (int): Length of each trajectory
  - For RNN training: typically > 1 for temporal correlation
  - For feedforward: can be 1

**Returns:**
- `nested Tensor` - Trajectories:
  - Shape: `[batch_size, batch_length, ...]`
  - Sampled from buffer accounting for episode boundaries
  - Transformed data (observations, etc.)
- `BatchInfo` - Metadata:
  - `env_ids`: Which environments sampled from
  - `positions`: Starting position in each environment
  - `importance_weights`: Priority weights (if prioritized)
  - `discounted_return`: Episode returns (if recorded)

**Sampling Modes:**
1. **Uniform sampling**: Random trajectory selection
2. **Prioritized sampling**: Weighted by TD error or importance
3. **Recent data**: Fraction from most recent steps

**Implementation Details:**
```python
# Sampling without replacement (default):
# - Each environment contributes roughly equally
# - r values are evenly spaced in [0, 1)

# Prioritized sampling uses segment trees:
# - SumSegmentTree: sum of all priorities (for sampling)
# - MaxSegmentTree: max priority (for initial priority)
# - Importance weights: priority / avg_priority
```

**Invariants:**
- Will never sample incomplete trajectories (shorter than batch_length)
- Respects episode boundaries if available
- Respects `num_earliest_frames_ignored`

---

#### gather_all
```python
@atomic
def gather_all(
    ignore_earliest_frames: bool = False,
    convert_to_default_device: bool = True
) -> tuple[nested Tensor, BatchInfo]
```

**Purpose:** Get ALL data from buffer (used for "whole replay buffer training" algorithms like PPO).

**Parameters:**
- `ignore_earliest_frames` (bool): Skip first N frames from frame stacking
- `convert_to_default_device` (bool): Move to default compute device

**Returns:**
- All buffer data: shape `[num_environments, buffer_size, ...]`
- BatchInfo with env_ids = [0, 1, ..., num_envs-1]

**Preconditions:**
- All environments must have same size
- All environments must have same ending position

---

#### update_priority
```python
@torch.no_grad()
def update_priority(
    env_ids: Tensor,
    positions: Tensor,
    priorities: Tensor
) -> None
```

**Purpose:** Update TD error priorities for prioritized experience replay.

**Parameters:**
- `env_ids` (Tensor): `[B]` int64 - which environments
- `positions` (Tensor): `[B]` int64 - positions (from BatchInfo)
- `priorities` (Tensor): `[B]` float32 - new priorities

**Usage Pattern:**
```python
# After training and computing TD error
loss_info = calc_loss(...)  # Should contain loss_info.priority
batch_info = replay_buffer.get_batch(...)
priority = (loss_info.priority + eps) ** alpha
replay_buffer.update_priority(
    batch_info.env_ids,
    batch_info.positions,
    priority
)
```

**Implementation:**
- Uses segment trees to efficiently update
- Only updates if positions still valid (not overwritten)

---

### Episodic Information Tracking

#### Episode Tracking (internal)

```
_indexed_pos[env, idx]:
  - For FIRST step: stores position of last step in episode
  - For non-FIRST step: stores position of episode's FIRST step

_headless_indexed_pos[env]:
  - Backup for FIRST steps overwritten by new data
  - Handles episodes longer than max_length

_episodic_discounted_return[env, idx]:
  - Discounted return at each step
  - Computed when episode ends (discount == 0)
  - Default value for incomplete episodes
```

#### get_episode_begin_position & steps_to_episode_end
```python
def get_episode_begin_position(pos: Tensor, env_ids: Tensor) -> Tensor
def steps_to_episode_end(pos: Tensor, env_ids: Tensor) -> Tensor
```

**Purpose:** Query episode boundary information.

**Returns:**
- `get_episode_begin_position()`: Position of episode's first step
  - May be outside buffer if episode > max_length
- `steps_to_episode_end()`: Distance to episode end

---

### Memory Layout

**Position Terminology:**
- `pos` (always-increasing): Conceptual infinite buffer position
- `idx` (wrapped): Actual index in finite buffer (0 to max_length-1)
- Relation: `idx == pos % max_length`

**Example (max_length=5):**
```
Step 0: pos=0, idx=0, _current_pos=[0]
Step 1: pos=1, idx=1, _current_pos=[1]
...
Step 5: pos=5, idx=0 (wraps!), _current_pos=[5]
Step 6: pos=6, idx=1, _current_pos=[6]
```

---

## Summary of Key Invariants

### Batching

| Aspect | On-Policy | Off-Policy |
|--------|-----------|-----------|
| Algorithm instance | Single shared instance for all B envs | Single shared instance for all B envs |
| Forward pass | One vectorized call with batch [B, ...] | One vectorized call with batch [B, ...] |
| Observations | Different per environment | Different per environment |
| Actions | Sampled from outputs; can differ per env | Sampled from outputs; can differ per env |
| Conditional routing | Supported via `conditional_update()` or masking | Supported via `conditional_update()` or masking |
| Different sub-algorithms | Can route different batch elements through different sub-algorithms | Can route different batch elements through different sub-algorithms |
| Result | All B envs processed simultaneously | All B envs processed simultaneously |

### States

| Spec | Train | Rollout | Predict | Notes |
|------|-------|---------|---------|-------|
| Relation | Subset | Full | Subset | train_spec ⊆ rollout_spec ⊆ predict_spec |
| RNN models | Identity | Identity | Identity | For non-recurrent: all empty () |
| Off-policy | Can differ | Full | Can differ | train can be smaller |
| Use case | Training | Collection | Deployment | - |

---

## Conditional Batch Routing: Multiple Sub-Algorithms per Batch

The framework supports selective invocation of different sub-algorithms for different batch elements within a single forward pass. This is a key flexibility feature.

### Pattern 1: conditional_update()

**Location:** `alf/utils/conditional_ops.py`

```python
from alf.utils.conditional_ops import conditional_update

def conditional_update(target, cond, func, *args, **kwargs):
    """Selectively apply func to batch elements where cond is True

    Args:
        target (nested Tensor): [B, ...] - default values
        cond (Tensor): [B] bool - which batch elements to update
        func (Callable): function to apply (receives sliced batch of selected elements)
        *args, **kwargs: arguments to func

    Returns:
        nested Tensor: [B, ...] with func applied only where cond=True
    """
```

**Example: DynamicActionRepeatAgent**

```python
def rollout_step(self, time_step, state):
    # Determine which batch elements need new actions
    switch_action = (state.steps == 0) | time_step.is_first()

    def _generate_new_action(time_step, state):
        # This function receives ONLY the sliced inputs (batch elements where condition is True)
        rl_step = self._rl.rollout_step(time_step, state.rl)
        return ActionRepeatState(action=rl_step.output, steps=0, ...)

    # conditional_update applies _generate_new_action only where switch_action=True
    # Other batch elements keep their previous state
    new_state = conditional_update(
        target=state,
        cond=switch_action,
        func=_generate_new_action,
        time_step=time_step,
        state=state
    )
    return AlgStep(output=new_state.action, state=new_state)
```

**How it works:**
1. Boolean mask `cond` determines which batch elements to process
2. `func` is called only with sliced inputs where `cond=True`
3. Outputs from `func` are scattered back to full batch shape
4. Batch elements where `cond=False` retain original `target` values

**Performance:** Only processes relevant batch elements, efficient for sparse conditions.

---

### Pattern 2: Masking-based Routing

Direct multiplication by mask tensor for per-batch-element weighting.

**Example: RlpdAlgorithm (Bootstrap Critic Selection)**

```python
def _calc_critic_loss(self, info):
    critic_losses = []

    for i, critic_loss_fn in enumerate(self._critic_losses):
        critic_loss = critic_loss_fn(...)  # [B] or [T, B]

        if self._use_bootstrap_critics:
            # bootstrap_mask: [B, num_replicas] per-batch, per-critic Bernoulli mask
            mask = info.bootstrap_mask[:, i]  # [B] - which batch elements use critic i
            # Weight loss: zero out loss for batch elements not using this critic
            critic_loss = critic_loss * mask

        critic_losses.append(critic_loss)

    return LossInfo(loss=sum(critic_losses), ...)
```

**How it works:**
1. Mask tensor has shape matching batch dimension
2. Multiply loss element-wise by mask: loss * mask
3. Zeroed losses don't contribute to gradient updates
4. Different batch elements can use different sub-algorithms

**Use cases:**
- Ensemble methods (different ensemble members per batch element)
- Probabilistic routing (each batch element randomly selects sub-algorithm)
- Curriculum learning (different learning stages for different batch elements)

---

### Pattern 3: Sequential Composition with Flexible Routing

Apply multiple sub-algorithms sequentially, with custom input routing.

**Example: SequentialAlg**

```python
def predict_step(self, inputs, state):
    """Sequential application of N sub-algorithms/networks"""
    x = inputs
    new_state = []
    info = {}

    for i, alg in enumerate(self._algorithms):
        # Custom input specification: use input from previous stage or from dict
        if self._input_keys[i]:
            x = get_nested_field(var_dict, self._input_keys[i])

        # Apply sub-algorithm
        alg_step = alg.predict_step(x, state[i])
        x = alg_step.output
        new_state.append(alg_step.state)
        info[alg.name] = alg_step.info
        var_dict[self._output_keys[i]] = x

    return AlgStep(output=x, state=new_state, info=info)
```

**Features:**
- Chain sub-algorithms together: output of one → input of next
- Custom routing: specify which intermediate outputs feed into which sub-algorithms
- Supports branching: same output can be fed to multiple sub-algorithms
- Loss accumulation: losses from all sub-algorithms aggregated

---

### Pattern 4: Hierarchical State Management

When using multiple sub-algorithms, maintain separate state for each.

```python
@dataclass
class HierarchicalState:
    """State for algorithm with multiple sub-algorithms"""
    alg1_state: nested Tensor      # State for first sub-algorithm
    alg2_state: nested Tensor      # State for second sub-algorithm
    meta_state: nested Tensor      # State for routing/selection logic

def rollout_step(self, time_step, state):
    # Apply first sub-algorithm
    step1 = self._alg1.rollout_step(time_step, state.alg1_state)

    # Conditionally apply second sub-algorithm
    condition = self._select_alg2(step1.output)  # [B] bool mask

    def _apply_alg2(time_step, alg2_state):
        return self._alg2.rollout_step(time_step, alg2_state)

    alg2_state = conditional_update(
        target=state.alg2_state,
        cond=condition,
        func=_apply_alg2,
        time_step=time_step,
        alg2_state=state.alg2_state
    )

    # Combine outputs
    output = self._combine(step1.output, alg2_output)

    return AlgStep(
        output=output,
        state=HierarchicalState(
            alg1_state=step1.state,
            alg2_state=alg2_state,
            meta_state=new_meta_state
        ),
        info=info
    )
```

---

## Extending the Framework

### Creating a New Algorithm

1. **Inherit from appropriate base:**
   - On-policy: `OnPolicyAlgorithm`
   - Off-policy: `OffPolicyAlgorithm`
   - Non-RL: `Algorithm`

2. **Implement required methods:**
   - `rollout_step(inputs, state)` - REQUIRED (data collection)
   - `calc_loss(info)` - REQUIRED (loss computation)
   - `train_step(inputs, state, rollout_info)` - For off-policy only
   - Other methods can be overridden for custom behavior

3. **Register optimizers:**
   - Call `add_optimizer()` for custom components needing different learning rates

4. **Define state specs:**
   - Set `train_state_spec`, `rollout_state_spec` in `__init__` if using RNNs

### Key Design Principles

1. **Separation of Concerns:**
   - `rollout_step`: Exploration & collection
   - `train_step`: Training & exploitation
   - `calc_loss`: Loss computation (independent of execution mode)

2. **Batching Semantics:**
   - All B environments share algorithm instance
   - All operations vectorized across B
   - Conditional routing allows different sub-algorithms per batch element

3. **State Management:**
   - States are algorithm's only temporal memory
   - Reset at episode boundaries automatically
   - Can differ across execution modes
   - Each nested algorithm maintains independent state

4. **Flexibility:**
   - Algorithm can be nested (hierarchical)
   - Parameters can be shared across components
   - Multiple optimizers supported
   - Each nested algorithm can have independent replay buffer

---

## Nested Algorithms and Concurrent RL Architecture

ALF's framework is specifically designed to support complex hierarchical algorithms where multiple sub-algorithms are composed together, each with independent state, optimizers, and replay buffers. This enables sophisticated multi-algorithm architectures.

### Overview: What's Supported

You **CAN** build a concurrent RL abstraction that:
- Takes a base Algorithm (e.g., DQN, SAC, PPO) and creates K independent copies
- Routes batch element `i` to copy `i % K`
- Each copy maintains its own:
  - Network parameters
  - Optimizer and learning rate
  - Replay buffer (for off-policy algorithms)
  - Training state
  - RNN state (if applicable)

### 1. Nested Algorithm Storage

**Instance-specific Storage Pattern:**

Nested algorithms are stored as direct PyTorch submodules:

```python
class ConcurrentRLAlgorithm(OffPolicyAlgorithm):
    def __init__(self, base_algorithm_cls, num_copies, ...):
        super().__init__(...)

        # Store K independent algorithm copies
        # Option A: Use nn.ModuleList for indexing
        self._algorithms = nn.ModuleList([
            base_algorithm_cls(...) for _ in range(num_copies)
        ])

        # Option B: Use nn.ModuleDict for naming
        self._algorithms = nn.ModuleDict({
            f'alg_{i}': base_algorithm_cls(...)
            for i in range(num_copies)
        })
```

**Why this works:**
- PyTorch's `nn.Module` framework automatically registers all nested Algorithms as submodules
- Each nested algorithm gets its own independent instance with separate parameters
- State dict/load automatically handles the hierarchy (confirmed by cycle detection code)

**Real-world examples:**
- `Agent` algorithm (alf/algorithms/agent.py): Contains `_rl_algorithm`, `_irm`, `_goal_generator`, etc.
- `PPGAlgorithm` (alf/algorithms/ppg_algorithm.py): Contains `_aux_algorithm` with own optimizer
- All examples maintain independent state and optimizer configurations

### 2. Independent Replay Buffers Per Algorithm

**Critical Finding: YES - Each nested algorithm automatically gets its own**

**How it works:**

File: `alf/algorithms/algorithm.py` (lines 151-153, 375-456)

```python
class Algorithm:
    def __init__(self, ...):
        # Each instance gets its own buffer
        self._replay_buffer = None
        self._replay_buffer_num_envs = None
        self._replay_buffer_max_length = None
```

Each algorithm instance maintains:
- `self._replay_buffer`: Instance-specific (not shared)
- `self._observers`: Instance-specific list of callbacks
- `observe_for_replay()`: Adds to THIS algorithm's buffer only

**Setting up independent buffers per copy:**

```python
class ConcurrentRLAlgorithm(OffPolicyAlgorithm):
    def __init__(self, base_algorithm_cls, num_copies, config, ...):
        for i in range(num_copies):
            alg = base_algorithm_cls(...)
            # Each algorithm gets its own buffer configuration
            alg.set_replay_buffer(
                num_envs=batch_size,  # Can be different per algorithm
                max_length=config.replay_buffer_length,
                prioritized_sampling=config.priority_replay
            )
            self._algorithms.append(alg)
```

**Real-world verification:**

`PPGAuxAlgorithm` (alf/algorithms/ppg/ppg_aux_algorithm.py, lines 61-173) demonstrates this:

```python
class PPGAuxAlgorithm(OffPolicyAlgorithm):
    """An algorithm used as sub-algorithm of PPGAlgorithm.
    Auxiliary phase updates does not require new rollouts.
    Instead it will collect all of the experiences in ITS OWN replay buffer.
    """

    def observe_for_aux_replay(self, exp):
        if self._replay_buffer is None:
            # Create OWN replay buffer (completely separate from parent)
            self._replay_buffer = ReplayBuffer(
                data_spec=exp_spec,
                num_environments=exp.env_id.shape[0],
                max_length=max_length,
                prioritized_sampling=False,
                name='ppg_aux_replay_buffer')  # Unique name per instance

        self._replay_buffer.add_batch(exp, exp.env_id)
```

**Why this matters for concurrent RL:**
- Parent algorithm can have ZERO replay buffer (if not off-policy)
- Each of K child algorithms gets its own buffer when `observe_for_replay()` is called on it
- Buffers are lazy-initialized on first experience (efficient)
- Each buffer can have independent settings (size, prioritization, etc.)

### 3. Independent Optimizer Management

**Each nested algorithm can have independent optimizers:**

File: `alf/algorithms/algorithm.py` (lines 671-748) - `_setup_optimizers_()` recurses through children

```python
def _setup_optimizers_(self, param_to_name):
    """Setup optimizers recursively for all nested algorithms"""

    for child in self._get_children():
        if isinstance(child, Algorithm):
            # RECURSIVE call - child sets up its own optimizers
            params, child_handled = child._setup_optimizers_(param_to_name)
```

**Pattern for concurrent RL with independent optimizers:**

```python
class ConcurrentRLAlgorithm(OffPolicyAlgorithm):
    def __init__(self, base_algorithm_cls, num_copies, config, ...):
        for i in range(num_copies):
            # Each copy gets its own optimizer instance
            optimizer = torch.optim.Adam(lr=config.learning_rate)
            alg = base_algorithm_cls(
                ...,
                optimizer=optimizer  # Independent per copy
            )
            self._algorithms.append(alg)
```

**Real-world example:**

PPG (alf/algorithms/ppg_algorithm.py, lines 107-121):

```python
# Parent has one optimizer
super().__init__(
    optimizer=policy_optimizer,
    ...)

# Child has completely different optimizer
self._aux_algorithm = PPGAuxAlgorithm(
    optimizer=aux_optimizer,  # Different from parent
    ...)
```

Both optimizers coexist peacefully, managed separately.

### 4. State Management for Concurrent Algorithms

**Each nested algorithm maintains independent state:**

State is organized in composite structures (namedtuples or dataclasses):

```python
class ConcurrentRLState(NamedTuple):
    """State for concurrent RL algorithm"""
    algorithm_states: List[nested Tensor]  # One per algorithm copy
    routing_state: nested Tensor           # For routing logic
```

During forward pass:

```python
def rollout_step(self, time_step, state):
    new_state = ConcurrentRLState(
        algorithm_states=[],
        routing_state=state.routing_state
    )

    batch_size = time_step.observation.shape[0]

    for i in range(batch_size):
        alg_idx = i % len(self._algorithms)  # Routing: i % K

        # Call specific algorithm with its specific state
        alg_step = self._algorithms[alg_idx].rollout_step(
            time_step[i:i+1],  # Single batch element
            state.algorithm_states[alg_idx]
        )

        new_state.algorithm_states[alg_idx] = alg_step.state
        # Collect outputs for reconstruction
```

**Key properties:**
- Each algorithm's state is independent
- States are reset separately at episode boundaries
- For RNN algorithms, each copy maintains independent hidden state

### 5. Checkpointing and State Dict

**Nested algorithms checkpoint automatically:**

File: `alf/algorithms/algorithm.py` (lines 590-720)

```python
def state_dict(self, destination=None, prefix='', visited=None, **kwargs):
    """Recursively saves state for all nested algorithms"""

    if visited is None:
        visited = {self}  # Cycle detection

    # Recursively traverse all nested modules
    for name, child in self._modules.items():
        if child is not None and child not in visited:
            visited.add(child)
            child.state_dict(destination, prefix + name + '.', visited=visited)

    # Save optimizers for this algorithm
    if isinstance(self, Algorithm):
        for i, opt in enumerate(self._optimizers):
            opts_dict[prefix + '_optimizers.%d' % i] = opt.state_dict()
```

**For concurrent RL:**
```
checkpoint structure:
  alg/
    _algorithms/
      0/  # Copy 0
        _parameters/
        _optimizers/0/  # Its optimizer state
      1/  # Copy 1
        _parameters/
        _optimizers/0/  # Its optimizer state
      ...
      K-1/
    _optimizers/0/  # Parent's optimizer (if any)
```

Each copy's parameters and optimizer states are saved/loaded independently.

### 6. Cycle Detection and Parameter Sharing

**Cycle detection is built-in:**

File: `alf/algorithms/algorithm.py` (lines 606-617)

```python
def _assert_no_cycle_or_duplicate(self):
    """Ensure no cycles in algorithm hierarchy"""
    visited = set()
    to_be_visited = [self]
    while to_be_visited:
        node = to_be_visited.pop(0)
        visited.add(node)
        for child in node._get_children():
            assert child not in visited, "Cycle detected"
            if isinstance(child, Algorithm):
                to_be_visited.append(child)
```

**For concurrent RL:** This is automatically checked - if you create K copies of an algorithm class (not K references to the same instance), there will be no cycles.

### 7. Gotchas and Constraints

#### 7.1 Algorithm Nesting Requirement

**From algorithm.py docstring (lines 78-87):**
```
A requirement for this optimizer structure to work is that there is no
algorithm which is a submodule of a non-algorithm module. Currently,
this is not checked by the framework. It's up to the user to make sure
this is true.
```

**For concurrent RL:** Keep all algorithm copies in `nn.ModuleList` or `nn.ModuleDict`. Don't wrap them in another non-Algorithm module.

#### 7.2 Parameter Visibility Control

```python
@property
def force_params_visible_to_parent(self) -> bool:
    """Whether nested algorithm's parameters visible to parent optimizer"""
```

**Default:** Child algorithm's parameters are hidden from parent optimizer (good for independent optimization).

**For concurrent RL:** Leave this as default - you want each copy's parameters handled independently.

#### 7.3 Trainable Attributes to Ignore

```python
def _trainable_attributes_to_ignore(self):
    """Which attributes should not get assigned to default optimizer"""
    return []  # Override to prevent certain attributes
```

**For concurrent RL:** If storing `_algorithms` directly, you may need to override this to prevent the list from being optimized as a whole.

#### 7.4 Single Offline Buffer Per Algorithm

```python
# In Algorithm class
self._offline_replay_buffer = None  # Only ONE per algorithm
```

**For concurrent RL:** If using offline/hybrid training, only the parent algorithm can have an offline buffer (or each child, but not both). Design the learning pipeline accordingly.

### 8. Architecture Design for Concurrent RL

**Recommended structure:**

```python
class ConcurrentRLAlgorithm(OffPolicyAlgorithm):
    """K concurrent instances of base algorithm, routing batch elements"""

    def __init__(self, base_algorithm_cls, num_copies, ...):
        super().__init__(...)

        # Store K independent copies
        self._num_copies = num_copies
        self._algorithms = nn.ModuleList([
            base_algorithm_cls(
                observation_spec=observation_spec,
                action_spec=action_spec,
                train_state_spec=train_state_spec,
                optimizer=torch.optim.Adam(lr=learning_rate),
                config=config,
                name=f'alg_{i}'
            )
            for i in range(num_copies)
        ])

        # Initialize replay buffers for each
        for alg in self._algorithms:
            alg.set_replay_buffer(batch_size, buffer_length)

    def rollout_step(self, time_step, state):
        """Route batch elements to appropriate algorithm copy"""
        outputs = []
        new_state = []

        for i, alg in enumerate(self._algorithms):
            # Find batch elements assigned to this algorithm
            batch_indices = [j for j in range(batch_size) if j % self._num_copies == i]

            if not batch_indices:
                continue

            # Slice batch for this algorithm
            sliced_time_step = alf.nest.map_structure(
                lambda x: x[batch_indices], time_step)
            sliced_alg_state = state[i]

            # Apply algorithm
            alg_step = alg.rollout_step(sliced_time_step, sliced_alg_state)

            # Scatter outputs back to full batch
            for j, out in zip(batch_indices, alg_step.output):
                outputs[j] = out

            new_state[i] = alg_step.state

        return AlgStep(output=torch.cat(outputs), state=new_state, info=...)

    def train_step(self, inputs, state, rollout_info):
        """Train each algorithm copy independently"""
        # Similar routing pattern
        pass

    def calc_loss(self, info):
        """Combine losses from all algorithm copies"""
        losses = []
        for i, alg in enumerate(self._algorithms):
            loss = alg.calc_loss(info[i])
            losses.append(loss.loss)

        return LossInfo(loss=sum(losses))
```

### 9. Advantages of This Approach

1. **True Independence:** Each algorithm copy has:
   - Separate parameters (no weight sharing)
   - Separate optimizer state
   - Separate replay buffer
   - Separate training trajectory

2. **Batch-wise Load Balancing:** Naturally distributes workload if batch elements represent different tasks

3. **Ensemble Diversity:** Multiple independently trained agents can provide uncertainty estimates

4. **Fault Isolation:** One algorithm's instability doesn't affect others (can reset individually)

5. **Modular Switching:** Can swap algorithm implementations without changing routing logic

### 10. Implementation Considerations

- **Batch slicing/reconstruction:** Use `alf.nest.map_structure()` for clean tensor manipulation
- **State aggregation:** Use namedtuples or dataclasses to organize K state objects
- **Loss combination:** Simple summation (trains all K equally) or weighted if desired
- **Inference:** Route through appropriate algorithm copy, or ensemble predictions from all

---

## AlgorithmContainer

**Location:** `alf/algorithms/containers.py`

**Parent:** `Algorithm`

**Purpose:** Base class for algorithms containing multiple sub-algorithms. Provides sensible default implementations of common interface methods that delegate to sub-algorithms and aggregate their results. This is a key building block for hierarchical algorithm architectures.

### Overview

`AlgorithmContainer` is useful as a step toward implementing concurrent RL algorithms or other multi-algorithm architectures. It handles:
- Aggregating losses from multiple sub-algorithms
- Routing lifecycle methods (`preprocess_experience`, `after_update`, `after_train_iter`) to all sub-algorithms
- Managing on-policy/off-policy properties consistently across sub-algorithms

### Constructor

```python
AlgorithmContainer(
    algs: dict[Algorithm],
    train_state_spec: nested TensorSpec,
    rollout_state_spec: nested TensorSpec,
    predict_state_spec: nested TensorSpec,
    is_on_policy: None | bool,
    debug_summaries: bool,
    name: str
)
```

**Parameters:**
- `algs` (dict[Algorithm]): Dictionary mapping names to Algorithm instances
  - **Must be a dict** (not list) - keys become part of state/info dictionaries
  - Example: `{'actor': actor_alg, 'critic': critic_alg}`
  - Each value must be an Algorithm instance
- `train_state_spec` (nested TensorSpec): Combined state spec for all sub-algorithms
  - Typically: composite of all sub-algorithm train_state_specs
- `rollout_state_spec` (nested TensorSpec): Combined state spec for rollout
- `predict_state_spec` (nested TensorSpec): Combined state spec for prediction
- `is_on_policy` (None | bool): Overall on-policy mode
  - If None: automatically determined from sub-algorithms
  - If all sub-algs are on-policy → True
  - If all sub-algs are off-policy → False
  - If mixed: raises error
- `debug_summaries` (bool): Enable debug summaries
- `name` (str): Algorithm name

**Invariant Checking:**
- If `is_on_policy` specified: all sub-algorithms must match
- If `is_on_policy` not specified: automatically infers from sub-algorithms
- If sub-algorithms have mixed on/off-policy modes: raises ValueError

### Key Methods

#### calc_loss (aggregation)

```python
def calc_loss(self, info: dict[str, nested Tensor]) -> LossInfo
```

**Purpose:** Combine losses from all sub-algorithms into a single LossInfo.

**Parameters:**
- `info` (dict[str, nested Tensor]): Dictionary where:
  - Keys match sub-algorithm names from `self._algs`
  - Values are training info specific to each sub-algorithm
  - Shape: depends on each sub-algorithm

**Returns:** `LossInfo`:
- `loss`: Sum of all sub-algorithm losses
- `scalar_loss`: Sum of all sub-algorithm scalar_losses
- `priority`: Combined priority (used for prioritized sampling)
- `extra` (dict): Dictionary mapping sub-algorithm names to their extra info

**Implementation:**
```python
for name, alg in self._algs.items():
    loss_info = alg.calc_loss(info[name])
    # Uses add_ignore_empty to handle empty () tensors
    loss = add_ignore_empty(loss_info.loss, accumulated_loss)
    scalar_loss = add_ignore_empty(loss_info.scalar_loss, accumulated_scalar_loss)
    priority = add_ignore_empty(loss_info.priority, accumulated_priority)
```

---

#### set_on_policy & set_path

```python
def set_on_policy(self, is_on_policy: bool) -> None
def set_path(self, path: str) -> None
```

**Purpose:** Propagate configuration to all sub-algorithms.

**Implementation:**
- `set_on_policy()`: Calls `alg.set_on_policy(is_on_policy)` on each sub-algorithm
- `set_path()`: Sets path for each sub-algorithm, using naming scheme `path.alg_name`

---

#### preprocess_experience

```python
def preprocess_experience(
    self,
    root_inputs: nested Tensor,
    rollout_info: dict[str, nested Tensor],
    batch_info: BatchInfo
) -> tuple[nested Tensor, dict[str, nested Tensor]]
```

**Purpose:** Preprocess experiences through all sub-algorithms (e.g., data transformation).

**Parameters:**
- `root_inputs` (nested Tensor): Original inputs
- `rollout_info` (dict[str, nested Tensor]): Per-algorithm rollout info
- `batch_info` (BatchInfo): Metadata from replay buffer

**Returns:**
- Modified `root_inputs` (threaded through all sub-algorithms)
- Dictionary of preprocessed infos per sub-algorithm

**Flow:**
```
root_inputs
  → alg_1.preprocess_experience(info['alg_1'])
  → root_inputs_1, info_1['alg_1']
  → alg_2.preprocess_experience(info_1['alg_2'])
  → root_inputs_2, info_2['alg_2']
  → ...
```

---

#### after_update & after_train_iter

```python
def after_update(self, root_inputs: nested Tensor, info: dict[str, nested Tensor]) -> None
def after_train_iter(self, root_inputs: nested Tensor, rollout_info: dict[str, nested Tensor]) -> None
```

**Purpose:** Lifecycle hooks called after gradient updates or training iterations.

**Behavior:**
- Calls the corresponding method on each sub-algorithm with its slice of info
- Useful for updating learning rates, resetting statistics, etc.

---

### Related Container Classes

#### SequentialAlg

**Purpose:** Chain multiple Algorithms/Networks sequentially with flexible input routing.

```python
alg = SequentialAlg(
    value=('input.observation', value_network),
    action_dist=('input.observation', actor_network),
    action=sample_action_fn,
    loss=(loss_info_template, loss_fn),
    output='action'
)
```

**Key Features:**
- Input specs: `('nested.str.path', module)` routes specific inputs
- Automatic state stacking: combines states from all sub-modules
- Info aggregation: returns dict of info from each Algorithm sub-component
- Output selection: can return any computed value via `output` parameter

**Returns `AlgStep` with:**
- `output`: Last module's output (or specified via `output` param)
- `state`: List of states from all modules
- `info`: Dict mapping sub-algorithm names to their info

---

#### EchoAlg

**Purpose:** Algorithm with feedback loop - output feeds back as input in next step.

```python
class EchoAlg(Algorithm):
    def __init__(self, alg: Algorithm, echo_spec: nested TensorSpec)
```

**Constraints:**
- Inner algorithm must expect input dict: `{'input': ..., 'echo': ...}`
- Inner algorithm must return output dict: `{'output': ..., 'echo': ...}`
- `echo` output becomes `echo` input for next timestep

**State:** `(alg_state, echo_output)` - maintains algorithm state and echo feedback

---

#### RLAlgWrapper

**Purpose:** Wrap a non-RL Algorithm as an RLAlgorithm for use with RLTrainer.

```python
rl_wrapper = RLAlgWrapper(
    observation_spec=obs_spec,
    action_spec=action_spec,
    algorithm=non_rl_algorithm,
    env=environment,
    config=trainer_config
)
```

**Use Case:** When you have a complex Algorithm (possibly built with SequentialAlg or AlgorithmContainer) that already implements the core logic, but you need it to work with RLTrainer which expects RLAlgorithm interface.

**Delegation:**
- Wraps algorithm and delegates all core methods
- Adds RL-specific features from RLAlgorithm (environment interaction, metrics)
- Transparent wrapper - no behavior changes

---

### Implementation Pattern for Concurrent RL

`AlgorithmContainer` provides an excellent foundation for building concurrent RL algorithms:

```python
class ConcurrentRLAlgorithmV2(OffPolicyAlgorithm):
    """Concurrent RL using AlgorithmContainer for loss aggregation"""

    def __init__(self, base_algorithm_cls, num_copies, config, ...):
        super().__init__(...)

        # Create K copies as a dict
        algs_dict = {
            f'alg_{i}': base_algorithm_cls(...)
            for i in range(num_copies)
        }

        # Create container to handle loss aggregation
        self._container = AlgorithmContainer(
            algs=algs_dict,
            train_state_spec=...,
            is_on_policy=self.on_policy,
            ...
        )

    def rollout_step(self, time_step, state):
        """Route each batch element to appropriate algorithm"""
        outputs = {}
        new_state = {}

        for name, alg in self._container._algs.items():
            # Route batch elements for this algorithm
            batch_indices = [i for i in range(B) if route(i) == name]
            sliced_time_step = nest.map_structure(lambda x: x[batch_indices], time_step)

            alg_step = alg.rollout_step(sliced_time_step, state[name])

            # Scatter back
            for j, idx in enumerate(batch_indices):
                outputs[idx] = alg_step.output[j]
            new_state[name] = alg_step.state

        return AlgStep(output=torch.cat([outputs[i] for i in range(B)]), ...)

    def calc_loss(self, info):
        # Container automatically aggregates!
        return self._container.calc_loss(info)
```

**Advantages over manual implementation:**
- Loss aggregation handled automatically
- Lifecycle methods routed automatically
- On-policy/off-policy consistency enforced
- Less boilerplate code

---

### Why Use AlgorithmContainer?

1. **Separation of Concerns:** Container handles "multi-algorithm bookkeeping", you implement routing
2. **Consistency Enforcement:** Ensures all sub-algorithms have compatible modes
3. **Extensibility:** Built-in support for preprocessing, post-update hooks
4. **Reusability:** Standard interface that other components expect
5. **Cleaner Code:** Less manual aggregation logic needed

---

## Appendix: Understanding "Unroll"

The term **"unroll"** in ALF refers to collecting a trajectory (sequence of timesteps) from the environment. It's **not specific to RNNs** - the name comes from "unrolling" the environment interaction loop for a fixed number of steps.

### What Does Unroll Do?

The `unroll(unroll_length)` method:
- Collects experiences from the environment for exactly `unroll_length` timesteps
- Interacts with all B parallel environments simultaneously
- Calls `rollout_step()` exactly `unroll_length` times
- Returns stacked experiences with shape `[T, B, ...]` where T = unroll_length

### Pseudocode

```python
def unroll(unroll_length):
    """Collect unroll_length timesteps of data"""
    experiences = []

    for t in range(unroll_length):
        # Get action from policy
        policy_step = rollout_step(time_step, state)

        # Execute action in environment
        time_step = env.step(policy_step.output)

        # Store experience
        experience = make_experience(time_step, policy_step)
        experiences.append(experience)

    # Return shape [T, B, ...]
    return stack_experiences(experiences)
```

### On-Policy vs Off-Policy Unrolling

**On-policy** algorithms:
- Call `unroll()` once
- Immediately train on the entire unrolled trajectory
- Then start a new unroll
- Flow: `unroll() → train_from_unroll() → next unroll()`

**Off-policy** algorithms:
- Call `unroll()` periodically
- Store all experiences in replay buffer
- Train repeatedly by sampling from replay buffer
- Flow: `unroll() → observe_for_replay() → train_from_replay_buffer()`

### Why "Unroll"?

The name comes from viewing the environment interaction as "unrolling" the recurrent dynamics:

```
Input → [RNN cell] → Output → Input → [RNN cell] → Output → ...
  t=0                           t=1

"Unrolling" for T steps:
Input → Cell → Output ──┐
            ↑           │
        ┌───────────────┘
        │
Input → Cell → Output ──┐
            ↑           │
        ┌───────────────┘
        │
Input → Cell → Output
            ↓
        Output sequence [T steps]
```

For environment interaction, we "unroll" the policy-environment loop:

```
time_step=t₀ → rollout_step() → action → env.step() → time_step=t₁
time_step=t₁ → rollout_step() → action → env.step() → time_step=t₂
time_step=t₂ → rollout_step() → action → env.step() → time_step=t₃
                                                       ...
Result: trajectory of length T
```

### Key Points

- **Not just for RNNs:** Works identically for feedforward and recurrent models
- **Temporal structure:** Preserves the order and dependency between steps
- **Batched:** All B environments unroll in parallel
- **Configurable length:** Set via `config.unroll_length` in TrainerConfig
- **Output shape:** Always `[T, B, ...]` (time-major by default)

### Related Methods

- `unroll()`: Collect experiences from environment (in RLAlgorithm)
- `train_from_unroll()`: Train on collected experiences (in Algorithm)
- `train_from_replay_buffer()`: Train on replayed experiences from unrolled data (in Algorithm)

The "unroll" terminology is borrowed from ML/RNN literature where you typically unroll a recurrent network over T timesteps to compute gradients. In ALF, it naturally applies to collecting sequential data from any environment.

