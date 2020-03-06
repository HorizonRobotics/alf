# Knowlege Base

## Debugging

By default, most of the code is running in graph mode (i.e. decoreated by
tf.function directly or indirectly), which makes debugging difficult. When
 debugging the unittests, you can include the following line to make the code
 run in eager mode, which makes debugging much easier:
 ```python
 tf.config.experimental_run_functions_eagerly(True)
 ```

If you are debugging model running using alf.bin.train (i.e. with command `python -m alf.bin.train ...`),
you can set the following gin configuration to run in eager mode:
```
TrainerConfig.use_tf_functions=False
```

### Debugging in graph mode

Sometime, an error may only appear in graph mode. And tensowflow usually does
not stop at where the error occurs (which is typically in the transformed code)
because it captures the exception raised from the transformed code. Here is the
[code segment](https://github.com/tensorflow/tensorflow/blob/0f57d4f0b3eb4278ea1127f6fcd9fcafa58dd59c/tensorflow/python/framework/func_graph.py#L956-L970). You may find it is easier
to debug by commenting out the exception handling as the following so that you
can debugging into the transformed code:
```python
          # try:
          return autograph.converted_call(
              original_func,
              args,
              kwargs,
              options=autograph.ConversionOptions(
                  recursive=True,
                  optional_features=autograph_options,
                  user_requested=True,
              ))
          # except Exception as e:  # pylint:disable=broad-except
          #   if hasattr(e, "ag_error_metadata"):
          #     raise e.ag_error_metadata.to_exception(e)
          #   else:
          #     raise
```
### Debugging using VisualStudio Code

Currently, ALF uses separate processes to launch multiple environments. Because
vscode does not support debug for multiprocessing (https://github.com/microsoft/ptvsd/issues/1706),
in order to debug in vscode, you need to make ALF not to start separate processes
by setting the following gin config:
```
create_environment.num_parallel_environments=1
Trainer._create_environment.nonparallel=True
```

## Algorithm

Algorithm is the most important concept in ALF. TODO: more description about the design.

## ActionTimeStep

`ActionTimeStep` is a data structure extended from `TimeStep` of [TF-Agents](https://github.com/tensorflow/agents)
for storing the information from the result of each environment step. It contains
five fields:
* `observation`: observation from the environment. It can be a nest of Tensors.
  It is obtained after the environment execute `prev_action`.
* `step_type`: type of this step. It has three possible values:
  - StepType.FIRST is the first step of an episode, which is typically the step generated from env.reset().
  - StepType.LAST is the last step of an episode.
  - StepType.MID is for all the other steps in an episode.
* `reward`: reward from the previous action. In most RL literature, the reward
  for an action a_t at time t is usually written as r_t. However, in ALF, reward_t
  will always represent the reward for the previous action at time t-1.
* `discount`: discount value for discounting future reward. When calculating
  the cumulative discounted return, `discount` is used to discount the future
  reward. There is some subtleties on how this value is set which we will describe later.
* `prev_action`: this field is not in the TimeStep structure of TF-Agents.
  However, because of it is frequently used, ALF includes it in ActionTimeStep
  structure.

### About ActionTimeStep.discount

When a [gym](https://https://gym.openai.com/) environment is registered, there is an optional parameter named
 `max_episode_steps` which has default value of `None`. For example, the
 following is the registration for MountainCar environment:
 ```python
 register(
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)
 ```
Gym creates an `EnvSpec` object for each registered environment. `EnvSpec` has
an attribute `timestep_limit` which returns the value of `max_episode_steps`.

 A gym environment can be loaded by using `gym.make()` defined in
 `gym.envs.registration`. If `timestep_limit` of the spec of this environment is not
 `None`, this function will wrap the environment using
 `gym.wrappers.time_limit.TimeLimit`. This wrapper will end an episode by
 returning `done=True` if the number of steps exceeds `max_episode_steps`.

In TF-Agents, each TimeStep is associated with a `discount` value. In general,
if an episode ends, `TimeStep.step_type` is set to `StepType.LAST` and
`TimeStep.discount` is set to 0 to prevent using the value estimation at the
last step. However, if an episode ends because the `max_episode_steps` is
reached, it wants to use the original `discount` instead of 0 so that the value
estimation at the last step can be properly used to estimate the value of
previous steps. In order to achieve this, instead of using `gym.make` TF-Agents
creates an environemnt in the following way to avoid
`gym.wrappers.time_limit.TimeLimit`:
```python
  gym_spec = gym.spec(environment_name)
  gym_env = gym_spec.make()
```

Then TF-Agents uses its own wrapper `tf_agents.environments.wrappers.TimeLimit`
to wrap the environment to limit the steps so that it does not change the
discount when `max_episode_steps` is reached.

The following table summarizes how step type and discount affect the learning.

| Step type       | Discount           | Value used for bootstrapping the previous value?  | Value to be learned? | Note|
| ------------- |:-------------:| -----:|-----:|------:|
| FIRST      | 1 | No | Yes | First step of an episode|
| MID     | 1   |Yes  |  Yes | Any step other than FIRST and LAST |
| LAST | 0      |   No | No | Last step because of a normal game end |
| LAST | 1 | Yes | No | Last step because of time limit |

### Missing one TimeStep of losses every rollout length for on-policy training
<<<<<<< HEAD
TrainingInfo fields are populated in `OnPolicyDriver:_train_loop_body` for
on-policy training, and `SyncOffPolicyDriver:_rollout_loop_body` for
off-policy.
=======
TrainingInfo fields are populated in sync_off_policy_driver around line 143.
>>>>>>> adding documentation for TrainingInfo, and the case of missing one TimeStep in training.

For every `T` (unroll_length) number of transitions, only the ending `TimeStep`
of each transition is stored into `TrainingInfo`.  `TimeStep` `0` is not stored,
only `1` through `T` `TimeStep`s are.  Also, according to
how `TrainingInfo` is populated (sync_off_policy_driver.py around line 143)
the reward in `TimeStep` `1` comes from executing the action of `TimeStep` `0`,
and is not used in loss calculation.  This leaves only the rewards of steps `2`
to `T`, a total of `T - 1` steps, for calculating losses.  Would this cause
any significant problem with training?  Most likely, it wouldn't.

Case 1) When episode length varies or when it is fixed but is not divisible by unroll
length, then the time step that's being dropped could appear randomly
throughout the episode, no serious bias in training.

Case 2) When rewards are only given at episode end,
and each episode is of the same length that's divisible by the unroll length,
for example, unroll 2 steps for episodes of equal length 4, step `0`
is discarded, steps `1`, `2` make into training, and `3`, `4` make into
training (step `4` being the first step of the next episode).  The reward for
the first episode (steps `0` to `3`) is stored at step `4` and is used in
training.  No reward is dropped.  It's not randomly dropping `TimeStep`s,
so there is still some bias, but at least all the rewards are taken into
account during training.

Case 3) This actually becomes a problem when rewards are given at the second
last step of each episode, and when episode lengths are fixed and divisible
by unroll length, in which case, rewards are dropped consistently out of
training.

Case 3 is probably very rare, but documenting here just in case.

In ALF, some RL algorithm which only process rollout experience once (e.g. PPO,
IMPALA) is implemented as an off policy training which retrieves the experience
from a replay buffer.  In order to make sure the first step from the rollout is
used, the length of the replay buffer should be `unroll_length + 1`.
