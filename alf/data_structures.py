# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Various data structures.
Converted to PyTorch from the TF version.
"""
import collections
import numpy as np
import torch

import alf.nest as nest
import alf.tensor_specs as ts
from alf.utils.tensor_utils import to_tensor


def namedtuple(typename, field_names, default_value=None, default_values=()):
    """namedtuple with default value.

    Args:
        typename (str): type name of this namedtuple
        field_names (list[str]): name of each field
        default_value (Any): the default value for all fields
        default_values (list|dict): default value for each field
    Returns:
        the type for the namedtuple
    """
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (default_value, ) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


class StepType(object):
    """Defines the status of a `TimeStep` within a sequence."""
    # Denotes the first `TimeStep` in a sequence.
    FIRST = np.int32(0)
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = np.int32(1)
    # Denotes the last `TimeStep` in a sequence.
    LAST = np.int32(2)

    def __new__(cls, value):
        """Add ability to create StepType constants from a value."""
        if value == cls.FIRST:
            return cls.FIRST
        if value == cls.MID:
            return cls.MID
        if value == cls.LAST:
            return cls.LAST

        raise ValueError(
            'No known conversion for `%r` into a StepType' % value)


class TimeStep(
        namedtuple('TimeStep', [
            'step_type',
            'reward',
            'discount',
            'observation',
            'prev_action',
            'env_id',
        ])):
    """TimeStep with action.

    A `TimeStep` contains the data emitted by an environment at each step of
    interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
    NumPy array or a dict or list of arrays), and an associated `reward` and
    `discount`.

    The first `TimeStep` in a sequence will equal `StepType.FIRST`. The final
    `TimeStep` will equal `StepType.LAST`. All other `TimeStep`s in a sequence
    will equal `StepType.MID.

    Attributes:
        step_type: a `Tensor` or numpy int of `StepType` enum values.
        reward: a `Tensor` of reward values from executing 'prev_action'.
        discount: A discount value in the range `[0, 1]`.
        observation: A (nested) 'Tensor' for observation
        prev_action: A (nested) 'Tensor' for action from previous time step
        env_id: A scalar 'Tensor' of the environment ID of the time step
    """

    def is_first(self):
        return self.step_type == StepType.FIRST

    def is_mid(self):
        return self.step_type == StepType.MID

    def is_last(self):
        return self.step_type == StepType.LAST


def _create_timestep(observation, prev_action, reward, discount, env_id,
                     step_type):
    discount = to_tensor(discount)
    step_type = to_tensor(step_type)

    def make_tensors(struct):
        return nest.map_structure(to_tensor, struct)

    return TimeStep(
        step_type=step_type.view(discount.shape),
        reward=to_tensor(reward, dtype=torch.float32),
        discount=discount,
        observation=make_tensors(observation),
        prev_action=make_tensors(prev_action),
        env_id=to_tensor(env_id, dtype=torch.int32))


def timestep_first(observation, prev_action, reward, discount, env_id):
    return _create_timestep(observation, prev_action, reward, discount, env_id,
                            torch.tensor(StepType.FIRST, dtype=torch.int32))


def timestep_mid(observation, prev_action, reward, discount, env_id):
    return _create_timestep(observation, prev_action, reward, discount, env_id,
                            torch.tensor(StepType.MID, dtype=torch.int32))


def timestep_last(observation, prev_action, reward, discount, env_id):
    return _create_timestep(observation, prev_action, reward, discount, env_id,
                            torch.tensor(StepType.LAST, dtype=torch.int32))


AlgStep = namedtuple('AlgStep', ['output', 'state', 'info'], default_value=())


def restart(observation, action_spec, env_id=None, batched=False):
    """Returns a `TimeStep` with `step_type` set equal to `StepType.FIRST`.

    Called by env.reset().

    Args:
        observation (nested tensors): observations of the env
        action_spec (nested TensorSpec): tensor spec of actions
        env_id (batched or scalar torch.int32): (optional) ID of the env
        batched (bool): (optional) whether batched envs or not

    Returns:
        A `TimeStep`.
    """
    first_observation = nest.flatten(observation)
    assert all(
        map(torch.is_tensor,
            first_observation)), ("Elements in observation must be Tensor")

    # TODO: Check leading dimension of first_observation
    # against batch_size if all are known statically.
    if batched:
        batch_size = first_observation[0].shape[0]
        step_type = torch.full((batch_size, ),
                               StepType.FIRST,
                               dtype=torch.int32)
        reward = torch.full((batch_size, ), 0.0, dtype=torch.float32)
        discount = torch.full((batch_size, ), 1.0, dtype=torch.float32)
        prev_action = nest.map_structure(
            lambda spec: spec.zeros(outer_dims=(batch_size, )), action_spec)
        env_id = torch.arange(batch_size, dtype=torch.int32)
    else:
        step_type = torch.full((), StepType.FIRST, dtype=torch.int32)
        reward = torch.tensor(0.0, dtype=torch.float32)
        discount = torch.tensor(1.0, dtype=torch.float32)
        prev_action = nest.map_structure(lambda spec: spec.zeros(),
                                         action_spec)
        if env_id is None:
            env_id = torch.tensor(0, dtype=torch.int32)
    return TimeStep(step_type, reward, discount, observation, prev_action,
                    env_id)


def transition(observation, prev_action, reward, discount=1.0, env_id=None):
    """Returns a `TimeStep` with `step_type` set equal to `StepType.MID`.

    Called by env.step() if not 'Done'.

    The batch size is inferred from the shape of `reward`.

    If `discount` is a scalar, and `observation` contains Tensors,
    then `discount` will be broadcasted to match `reward.shape`.

    Args:
        observation (nested tensors): current observations of the env.
        prev_action (nested tensors): previous actions to the the env.
        reward (float): A scalar, or 1D NumPy array, or tensor.
        discount (float): (optional) A scalar, or 1D NumPy array, or tensor.
        env_id (torch.int32): (optional) A scalar or 1D tensor of
            the environment ID(s).

    Returns:
        A `TimeStep`.

    Raises:
        ValueError: If observations are tensors but reward's rank
            is not `0` or `1`.
    """
    flat_observation = nest.flatten(observation)
    assert all(
        map(torch.is_tensor,
            flat_observation)), ("Elements in observation must be Tensor")

    # TODO: If reward.shape.rank == 2, and static
    # batch sizes are available for both flat_observation and reward,
    # check that these match.
    reward = to_tensor(reward, dtype=torch.float32)
    assert reward.dim() <= 1, "Expected reward to be a scalar or vector."
    if reward.dim() == 0:
        shape = []
        if env_id is None:
            env_id = torch.tensor(0, dtype=torch.int32)
    else:
        flat_action = nest.flatten(prev_action)
        assert flat_observation[0].shape[:1] == reward.shape
        assert flat_action[0].shape[:1] == reward.shape
        shape = reward.shape
        env_id = torch.arange(shape[0], dtype=torch.int32)
    step_type = torch.full(shape, StepType.MID, dtype=torch.int32)
    discount = to_tensor(discount, dtype=torch.float32)

    if discount.dim() == 0:
        discount = torch.full(shape, discount, dtype=torch.float32)
    else:
        assert reward.shape == discount.shape
    return TimeStep(step_type, reward, discount, observation, prev_action,
                    env_id)


def termination(observation, prev_action, reward, env_id=None):
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

    Called by env.step() if 'Done'. 'discount' should not be sent in and
    will be set as 0.

    Args:
        observation (nested tensors): current observations of the env.
        prev_action (nested tensors): previous actions to the the env.
        reward (float): A scalar, or 1D NumPy array, or tensor.
        env_id (torch.int32): (optional) A scalar or 1D tensor of
            the environment ID(s).

    Returns:
        A `TimeStep`.

    Raises:
        ValueError: If observations are tensors but reward's statically known rank
            is not `0` or `1`.
    """
    flat_observation = nest.flatten(observation)
    assert all(
        map(torch.is_tensor,
            flat_observation)), ("Elements in observation must be Tensor")

    reward = to_tensor(reward, dtype=torch.float32)
    assert reward.dim() <= 1, "Expected reward to be a scalar or vector."
    if reward.dim() == 0:
        shape = []
        if env_id is None:
            env_id = torch.tensor(0, dtype=torch.int32)
    else:
        flat_action = nest.flatten(prev_action)
        assert flat_observation[0].shape[:1] == reward.shape
        assert flat_action[0].shape[:1] == reward.shape
        shape = reward.shape
        env_id = torch.arange(shape[0], dtype=torch.int32)
    step_type = torch.full(shape, StepType.LAST, dtype=torch.int32)
    discount = torch.full(shape, 0.0, dtype=torch.float32)
    return TimeStep(step_type, reward, discount, observation, prev_action,
                    env_id)


def time_step_spec(observation_spec, action_spec):
    """Returns a `TimeStep` spec given the observation_spec and the action_spec.
    """

    def is_valid_tensor_spec(spec):
        return isinstance(spec, ts.TensorSpec)

    assert all(map(is_valid_tensor_spec, nest.flatten(observation_spec)))
    assert all(map(is_valid_tensor_spec, nest.flatten(action_spec)))
    return TimeStep(
        step_type=ts.TensorSpec([], torch.int32),
        reward=ts.TensorSpec([], torch.float32),
        discount=ts.BoundedTensorSpec([],
                                      torch.float32,
                                      minimum=0.0,
                                      maximum=1.0),
        observation=observation_spec,
        prev_action=action_spec,
        env_id=ts.TensorSpec([], torch.int32))


TrainingInfo = namedtuple(
    "TrainingInfo",
    [
        "action",
        "step_type",
        "reward",
        "discount",

        # For on-policy training, it's the AlgStep.info from rollout
        # For off-policy training, it's the AlgStep.info from train_step
        "info",

        # Only used for off-policy training. It's the AlgStep.info from rollout
        "rollout_info",
        "env_id"
    ],
    default_value=())


class Experience(
        namedtuple(
            "Experience",
            [
                'step_type',
                'reward',
                'discount',
                'observation',
                'prev_action',
                'env_id',
                'action',
                'rollout_info',  # AlgStep.info from rollout()
                'state'  # state passed to rollout() to generate `action`
            ])):
    def is_first(self):
        if torch.is_tensor(self.step_type):
            return torch.eq(self.step_type, StepType.FIRST)
        return np.equal(self.step_type, StepType.FIRST)

    def is_mid(self):
        if torch.is_tensor(self.step_type):
            return torch.eq(self.step_type, StepType.MID)
        return np.equal(self.step_type, StepType.MID)

    def is_last(self):
        if torch.is_tensor(self.step_type):
            return torch.eq(self.step_type, StepType.LAST)
        return np.equal(self.step_type, StepType.LAST)


def make_experience(time_step: TimeStep, alg_step: AlgStep, state):
    """Make an instance of Experience from TimeStep and AlgStep.

    Args:
        time_step (TimeStep): time step from the environment
        alg_step (AlgStep): policy step returned from rollout()
        state (nested Tensor): state used for calling rollout() to get the
            `policy_step`
    Returns:
        Experience
    """
    return Experience(
        step_type=time_step.step_type,
        reward=time_step.reward,
        discount=time_step.discount,
        observation=time_step.observation,
        prev_action=time_step.prev_action,
        env_id=time_step.env_id,
        action=alg_step.output,
        rollout_info=alg_step.info,
        state=state)


def experience_to_time_step(exp: Experience):
    """Make TimeStep from Experience."""
    return TimeStep(
        step_type=exp.step_type,
        reward=exp.reward,
        discount=exp.discount,
        observation=exp.observation,
        prev_action=exp.prev_action,
        env_id=exp.env_id)


LossInfo = namedtuple(
    "LossInfo",
    [
        "loss",  # batch loss shape should be (T, B) or (B,)
        "scalar_loss",  # shape is ()
        "extra"  # nested batch and/or scalar losses, for summary only
    ],
    default_value=())
