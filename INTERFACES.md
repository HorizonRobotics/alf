# ALF Algorithm Framework - Interface Specifications

This document provides interface documentation focused on building concurrent RL architectures.

---

## Algorithm

**Location:** `alf/algorithms/algorithm.py`

**Purpose:** Base class for all learning algorithms. Provides core infrastructure for state management, optimizer handling, and gradient updates.

### Core Execution Methods

#### rollout_step
```python
def rollout_step(inputs: TimeStep, state) -> AlgStep
```

**Purpose:** Data collection during training. Generates actions and collects training info.

**Parameters:**
- `inputs` (TimeStep): Shape `[B, ...]` - observations from B parallel environments
- `state` (nested Tensor): Must match `rollout_state_spec`

**Returns:** `AlgStep`:
- `output`: Action or distribution (executed in environment)
- `state`: Updated state
- `info`: Training info for this step (stacked into `[T, B, ...]`)

**Key Point:** All B environments pass through same algorithm instance, but you can use `conditional_update()` or masking to route different batch elements through different sub-algorithms.

---

#### train_step
```python
def train_step(inputs: TimeStep, state, rollout_info) -> AlgStep
```

**Purpose:** Training forward pass (off-policy only).

**Parameters:**
- `inputs` (TimeStep): Experience from replay buffer, shape `[B, ...]`
- `state` (nested Tensor): Must match `train_state_spec` (can differ from `rollout_state_spec`)
- `rollout_info`: Info from corresponding `rollout_step()`

**Returns:** `AlgStep` with training info for loss computation

**Note:** On-policy algorithms delegate to `rollout_step()`.

---

#### calc_loss
```python
def calc_loss(info: nested Tensor) -> LossInfo
```

**Purpose:** Compute loss from training info.

**Returns:** `LossInfo`:
- `loss`: Shape `[T, B]` or scalar
- `priority`: For prioritized replay (shape `[B]`)
- `extra`: Additional info for summaries

---

#### update_with_gradient
```python
def update_with_gradient(
    loss_info: LossInfo,
    valid_masks: Tensor = None,
    batch_info: BatchInfo = None
) -> tuple[LossInfo, list[Parameter]]
```

**Purpose:** Perform one gradient update.

**Key Parameters:**
- `valid_masks`: Shape `[T, B]` - which samples are valid
- `batch_info`: Contains `importance_weights` for prioritized sampling

---

### State Management

```python
@property
def train_state_spec(self) -> nested TensorSpec
def rollout_state_spec(self) -> nested TensorSpec
def predict_state_spec(self) -> nested TensorSpec
```

**Invariant:** `train_state_spec ⊆ rollout_state_spec`

Each nested algorithm maintains independent state.

---

### Replay Buffer Integration

#### set_replay_buffer
```python
def set_replay_buffer(num_envs: int, max_length: int, prioritized_sampling: bool = False)
```

**Key Point:** Each algorithm instance has `self._replay_buffer` (instance-specific, not shared).

#### observe_for_replay
```python
def observe_for_replay(exp: Experience) -> None
```

Stores experience to THIS algorithm's replay buffer.

---

### Optimizer Management

#### add_optimizer
```python
def add_optimizer(optimizer: torch.optim.Optimizer, modules_and_params: list[Module | Parameter])
```

Assign specific optimizer to modules/parameters. Multiple optimizers supported.

**Key Architecture Point:** `_setup_optimizers_()` recurses through child algorithms - each nested algorithm can have independent optimizers.

---

### Nested Algorithm Properties

```python
@property
def force_params_visible_to_parent(self) -> bool
```

Default: False (child parameters hidden from parent optimizer - good for independent optimization)

---

## RLAlgorithm

**Location:** `alf/algorithms/rl_algorithm.py`

**Parent:** `Algorithm`

**Purpose:** Adds RL-specific functionality: environment interaction, unrolling.

### Constructor Key Parameters

```python
RLAlgorithm(
    observation_spec: nested TensorSpec,
    action_spec: BoundedTensorSpec,
    train_state_spec: nested TensorSpec,
    is_on_policy: bool | None = None,
    env: Environment | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    ...
)
```

---

### Environment Interaction

#### unroll
```python
def unroll(unroll_length: int) -> Experience | None
```

**Purpose:** Collect `unroll_length` timesteps from environment.

**Returns:** `Experience` with shape `[T, B, ...]` where T=unroll_length, B=batch_size

**Flow:**
```python
for step in range(unroll_length):
    state = reset_state_if_necessary(state, time_step.is_first())
    policy_step = rollout_step(transformed_time_step, state)
    next_time_step = env.step(policy_step.output)
    if not on_policy:
        observe_for_replay(experience)
```

---

## OnPolicyAlgorithm

**Location:** `alf/algorithms/on_policy_algorithm.py`

```python
@property
def on_policy(self) -> bool:
    return True

def train_step(self, inputs, state, rollout_info):
    return self.rollout_step(inputs, state)  # Delegates to rollout
```

---

## OffPolicyAlgorithm

**Location:** `alf/algorithms/off_policy_algorithm.py`

```python
@property
def on_policy(self) -> bool:
    return False
```

**Training Flow:**
```python
# (1) Collection stage
for step in range(steps_per_collection):
    policy_step = rollout_step(time_step, state)
    store_experience(experience)  # → replay buffer
    time_step = env.step(action)

# (2) Training stage
for train_step in range(training_steps):
    experiences = replay_buffer.get_batch()
    policy_step = train_step(experience, state)
    loss = calc_loss(policy_step.info)
    update_with_gradient(loss)
```

---

## ReplayBuffer

**Location:** `alf/experience_replayers/replay_buffer.py`

**Purpose:** Stores experiences for off-policy training.

### Core Methods

#### add_batch
```python
def add_batch(batch: nested Tensor, env_ids: Tensor | None = None) -> None
```

**Parameters:**
- `batch`: Shape `[B, ...]` or `[num_environments, ...]`
- `env_ids`: Shape `[B]` - which environments (allows partial updates)

---

#### get_batch
```python
def get_batch(batch_size: int, batch_length: int) -> tuple[nested Tensor, BatchInfo]
```

**Returns:**
- Trajectories: Shape `[batch_size, batch_length, ...]`
- `BatchInfo`: Contains `env_ids`, `positions`, `importance_weights`

---

## Nested Algorithms and Concurrent RL Architecture

### Key Findings

**You CAN build concurrent RL with K independent copies:**

1. **Storage:** Use `nn.ModuleList` or `nn.ModuleDict`
   ```python
   self._algorithms = nn.ModuleList([
       base_algorithm_cls(...) for _ in range(K)
   ])
   ```

2. **Independent Replay Buffers:** Each algorithm instance has `self._replay_buffer` (instance-specific)
   - Call `alg.set_replay_buffer()` on each copy
   - Verified by PPGAuxAlgorithm example

3. **Independent Optimizers:** `_setup_optimizers_()` recurses through children
   ```python
   for i in range(K):
       optimizer = torch.optim.Adam(lr=learning_rate)
       alg = base_algorithm_cls(..., optimizer=optimizer)
   ```

4. **State Management:** Use composite structures
   ```python
   class ConcurrentRLState(NamedTuple):
       algorithm_states: List[nested Tensor]  # One per copy
       routing_state: nested Tensor
   ```

5. **Checkpointing:** Automatic hierarchy preservation
   ```
   alg/_algorithms/0/_parameters/
   alg/_algorithms/0/_optimizers/0/
   alg/_algorithms/1/_parameters/
   ...
   ```

---

### Gotchas

1. **Algorithm Nesting Requirement:** Keep all algorithm copies in `nn.ModuleList`/`nn.ModuleDict`. Don't wrap in non-Algorithm module.

2. **Parameter Visibility:** Default `force_params_visible_to_parent=False` is correct for independent optimization.

3. **Offline Buffer:** Only ONE offline buffer per algorithm (parent OR children, not both).

---

### Recommended Architecture

```python
class ConcurrentRLAlgorithm(OffPolicyAlgorithm):
    def __init__(self, base_algorithm_cls, num_copies, ...):
        super().__init__(...)

        self._num_copies = num_copies
        self._algorithms = nn.ModuleList([
            base_algorithm_cls(
                observation_spec=observation_spec,
                action_spec=action_spec,
                optimizer=torch.optim.Adam(lr=learning_rate),
                name=f'alg_{i}'
            )
            for i in range(num_copies)
        ])

        for alg in self._algorithms:
            alg.set_replay_buffer(num_envs, max_length)

    def rollout_step(self, time_step, state):
        # Route batch element i to algorithm i % K
        outputs = []
        new_state = []

        for i, alg in enumerate(self._algorithms):
            batch_indices = [j for j in range(B) if j % self._num_copies == i]
            sliced_time_step = alf.nest.map_structure(
                lambda x: x[batch_indices], time_step)

            alg_step = alg.rollout_step(sliced_time_step, state[i])

            # Scatter outputs back to full batch
            for j, idx in enumerate(batch_indices):
                outputs[idx] = alg_step.output[j]
            new_state[i] = alg_step.state

        return AlgStep(output=torch.cat(outputs), state=new_state, ...)
```

---

## AlgorithmContainer

**Location:** `alf/algorithms/containers.py`

**Purpose:** Base class for managing multiple sub-algorithms. Handles loss aggregation and lifecycle routing.

### Constructor

```python
AlgorithmContainer(
    algs: dict[Algorithm],  # MUST be dict, keys become part of info dict
    train_state_spec: nested TensorSpec,
    rollout_state_spec: nested TensorSpec,
    predict_state_spec: nested TensorSpec,
    is_on_policy: None | bool,  # Auto-inferred if None
    ...
)
```

**Invariant:** All sub-algorithms must have compatible `is_on_policy` values.

---

### Key Methods

#### calc_loss (automatic aggregation)
```python
def calc_loss(self, info: dict[str, nested Tensor]) -> LossInfo
```

Automatically sums losses from all sub-algorithms using `add_ignore_empty()`.

#### Lifecycle Methods
```python
def preprocess_experience(root_inputs, rollout_info: dict, batch_info)
def after_update(root_inputs, info: dict)
def after_train_iter(root_inputs, rollout_info: dict)
```

Routes to each sub-algorithm with its slice of info dict.

---

### Using AlgorithmContainer for Concurrent RL

```python
class ConcurrentRLAlgorithmV2(OffPolicyAlgorithm):
    def __init__(self, base_algorithm_cls, num_copies, ...):
        super().__init__(...)

        algs_dict = {
            f'alg_{i}': base_algorithm_cls(...)
            for i in range(num_copies)
        }

        self._container = AlgorithmContainer(
            algs=algs_dict,
            train_state_spec=...,
            is_on_policy=self.on_policy,
            ...
        )

    def calc_loss(self, info):
        return self._container.calc_loss(info)  # Automatic aggregation!
```

**Advantages:**
- Loss aggregation handled automatically
- Lifecycle methods routed automatically
- On-policy/off-policy consistency enforced

---

### Related Containers

#### SequentialAlg
Chains algorithms/networks sequentially with flexible input routing.

#### RLAlgWrapper
Wraps non-RL Algorithm as RLAlgorithm for use with RLTrainer.

---

## Conditional Batch Routing Patterns

### Pattern 1: conditional_update()

```python
from alf.utils.conditional_ops import conditional_update

def rollout_step(self, time_step, state):
    condition = compute_condition(time_step)  # [B] bool

    new_state = conditional_update(
        target=state,
        cond=condition,
        func=self._sub_alg.rollout_step,
        time_step=time_step,
        state=state
    )
```

Only processes batch elements where `condition=True`.

---

### Pattern 2: Masking

```python
def calc_loss(self, info):
    for i, alg in enumerate(self._algorithms):
        loss = alg.calc_loss(info[i])
        mask = info.routing_mask[:, i]  # [B] - which batch elements use alg i
        masked_loss = loss * mask  # Zero out loss for other batch elements
```

---

## Appendix: Key Terminology

### Unroll

**What:** `unroll(unroll_length)` collects `unroll_length` timesteps from environment.

**Returns:** Shape `[T, B, ...]` where T=unroll_length, B=batch_size

**Not RNN-specific:** Works identically for feedforward and recurrent models.

**Flow:**
- On-policy: `unroll() → train_from_unroll() → next unroll()`
- Off-policy: `unroll() → observe_for_replay() → train_from_replay_buffer()`

---

### Batching Semantics

| Aspect | Behavior |
|--------|----------|
| Algorithm instance | Single shared instance for all B envs |
| Forward pass | One vectorized call `[B, ...]` |
| Conditional routing | Supported via `conditional_update()` or masking |
| Different sub-algorithms | Can route batch elements to different sub-algorithms |

---

### State Specs

| Spec | Purpose |
|------|---------|
| `train_state_spec` | RNN state for training |
| `rollout_state_spec` | RNN state for data collection |
| `predict_state_spec` | RNN state for deployment |

**Invariant:** `train_state_spec ⊆ rollout_state_spec`

---

## Coding Practices and Helper Functions

### Nested Structures

**What are nested structures?**

A "nest" or "nested structure" is ALF's fundamental data organization pattern. It can be:
- A `Tensor` (leaf)
- A `list` or unnamed `tuple` of nests
- A `dict` of nests
- A `namedtuple` of nests

**Type annotation:**
```python
from alf.nest import NestedTensor
state: NestedTensor  # Can be Tensor, list, tuple, dict, or namedtuple
```

---

### Core nest Functions

**Location:** `alf/nest/nest.py`

#### map_structure
```python
alf.nest.map_structure(func, *nests)
```

Apply function to corresponding elements across nests.

**Example:**
```python
state = {'a': torch.tensor([1, 2]), 'b': torch.tensor([3, 4])}
doubled = alf.nest.map_structure(lambda x: x * 2, state)
# Result: {'a': tensor([2, 4]), 'b': tensor([6, 8])}

# Multiple nests:
sum_state = alf.nest.map_structure(lambda x, y: x + y, state1, state2)
```

**Key use case for concurrent RL:** Slicing batches
```python
# Route batch element i to algorithm i % K
sliced_time_step = alf.nest.map_structure(
    lambda x: x[batch_indices], time_step)
```

---

#### flatten & pack_sequence_as
```python
flat_list = alf.nest.flatten(nest)
reconstructed = alf.nest.pack_sequence_as(nest, flat_list)
```

Convert nest to flat list and back.

**Example:**
```python
state = {'a': torch.tensor([1]), 'b': torch.tensor([2])}
flat = alf.nest.flatten(state)  # [tensor([1]), tensor([2])]
reconstructed = alf.nest.pack_sequence_as(state, flat)  # {'a': tensor([1]), 'b': tensor([2])}
```

---

#### get_field & transform_nest
```python
value = alf.nest.get_field(nest, 'a.b.c')  # Navigate nested paths
new_nest = alf.nest.transform_nest(nest, 'a.b', lambda x: x * 2)
```

Navigate and modify nested structures by path.

**Example:**
```python
info = dict(actor=dict(loss=0.5, entropy=0.1), critic=dict(loss=1.0))

actor_loss = alf.nest.get_field(info, 'actor.loss')  # 0.5

# Update nested field
info = alf.nest.transform_nest(info, 'actor.loss', lambda x: x * 0.5)
# Result: {'actor': {'loss': 0.25, 'entropy': 0.1}, 'critic': {'loss': 1.0}}
```

**Why this matters:** Much cleaner than `info._replace(actor=info.actor._replace(loss=...))` for namedtuples.

---

### Batch Operations on Nests

#### conditional_update
```python
alf.utils.conditional_ops.conditional_update(target, cond, func, *args, **kwargs)
```

**Location:** `alf/utils/conditional_ops.py`

Selectively apply function to batch elements where condition is True.

**How it works:**
1. Slice inputs where `cond=True` (gather operation)
2. Call `func` only on sliced inputs (efficient!)
3. Scatter results back to full batch shape
4. Elements where `cond=False` retain `target` values

**Example:**
```python
# Only generate new actions for environments that just started
switch_action = time_step.is_first()  # [B] bool

new_state = conditional_update(
    target=old_state,
    cond=switch_action,
    func=self._algorithm.rollout_step,
    time_step=time_step,
    state=old_state
)
# Only environments with switch_action=True get new actions computed
```

**Performance:** If only 2 out of 32 environments need updates, only 2 forward passes are done.

---

#### select_from_mask
```python
selected = alf.utils.conditional_ops.select_from_mask(data, mask)
```

Extract batch elements where mask is True.

**Example:**
```python
mask = torch.tensor([True, False, True, False])
data = {'obs': torch.randn(4, 10), 'action': torch.randn(4, 2)}
selected = select_from_mask(data, mask)
# Result: {'obs': shape [2, 10], 'action': shape [2, 2]}
```

---

### Nest Combiners

**Location:** `alf/nest/utils.py`

Helper classes for combining nested structures (useful for network inputs).

#### NestConcat
```python
combiner = alf.nest.utils.NestConcat(dim=-1)
combined = combiner({'obs': tensor1, 'goal': tensor2})
# Concatenates all tensors along last dimension
```

#### NestSum
```python
combiner = alf.nest.utils.NestSum(average=True)
combined = combiner([tensor1, tensor2, tensor3])
# Averages all tensors element-wise
```

#### stack_nests
```python
stacked = alf.nest.utils.stack_nests([nest1, nest2, nest3], dim=0)
# Stacks list of nests along new dimension -> shape [3, original_shape...]
```

**Use case:** Converting list of timesteps into temporal batch `[T, B, ...]`.

---

### Common Patterns in ALF

#### Pattern 1: State Management
```python
class MyAlgorithmState(NamedTuple):
    rnn_state: NestedTensor
    step_counter: torch.Tensor
    last_action: torch.Tensor

# Access
current_count = state.step_counter

# Update specific field
new_state = state._replace(step_counter=state.step_counter + 1)

# Or with nest.transform_nest
new_state = alf.nest.transform_nest(state, 'step_counter', lambda x: x + 1)
```

---

#### Pattern 2: Routing Batch Elements
```python
def rollout_step(self, time_step, state):
    for i, alg in enumerate(self._algorithms):
        # Find which batch elements go to this algorithm
        mask = (torch.arange(B) % K == i)  # [B] bool

        # Slice
        sliced_time_step = alf.nest.map_structure(
            lambda x: x[mask], time_step)

        # Process
        alg_step = alg.rollout_step(sliced_time_step, state[i])

        # Scatter back (manual indexing or use advanced indexing)
        outputs[mask] = alg_step.output
```

---

#### Pattern 3: Loss Aggregation with add_ignore_empty
```python
from alf.utils.math_ops import add_ignore_empty

# Accumulate losses that may be empty ()
total_loss = ()
for alg in self._algorithms:
    loss_info = alg.calc_loss(info[alg.name])
    total_loss = add_ignore_empty(total_loss, loss_info.loss)

return LossInfo(loss=total_loss)
```

**Why needed:** Some algorithms may return `()` (empty tuple) for loss, and `() + tensor` would fail.

---

#### Pattern 4: Temporal Batching
```python
# Collect unroll_length steps
experiences = []
for t in range(unroll_length):
    policy_step = rollout_step(time_step, state)
    experiences.append(make_experience(time_step, policy_step))

# Stack into [T, B, ...]
batched_exp = alf.nest.utils.stack_nests(experiences, dim=0)
```

---

### Key Helper Functions

#### get_nest_batch_size
```python
batch_size = alf.nest.get_nest_batch_size(nested)
# Returns size of dim 0, assuming all leaves have same batch size
```

#### convert_device
```python
cuda_nest = alf.nest.utils.convert_device(nest, device='cuda')
```

#### zeros_like
```python
zero_state = alf.nest.utils.zeros_like(state)
# Creates nested structure of zeros with same shapes
```

---

### Design Philosophy

1. **Nests are first-class citizens:** Most ALF functions accept/return nested structures, not just tensors.

2. **Structural invariants:** Functions like `map_structure` require all nests to have **same structure** (verified automatically).

3. **Composability:** Combine `map_structure`, `flatten`, `pack_sequence_as` to build complex transformations.

4. **Performance:** `conditional_update` and masking avoid unnecessary computation for sparse operations.

5. **Type safety:** Use `NestedTensor` type annotation for clarity, though Python doesn't enforce it at runtime.
