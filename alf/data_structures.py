# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
from alf.nest import map_structure
import collections
import numpy as np
import torch


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
    FIRST = torch.tensor(0, dtype=torch.int32)
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = torch.tensor(1, dtype=torch.int32)
    # Denotes the last `TimeStep` in a sequence.
    LAST = torch.tensor(2, dtype=torch.int32)

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

    *TODO*
    Returned with every call to `step` and `reset` on an environment.

    A `TimeStep` contains the data emitted by an environment at each step of
    interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
    NumPy array or a dict or list of arrays), and an associated `reward` and
    `discount`.

    The first `TimeStep` in a sequence will equal `StepType.FIRST`. The final
    `TimeStep` will equal `StepType.LAST`. All other `TimeStep`s in a sequence
    will equal `StepType.MID.

    Attributes:
      step_type: a `Tensor` of `StepType` enum values.
      reward: a `Tensor` of reward values from executing 'prev_action'.
      discount: A discount value in the range `[0, 1]`.
      observation: A (nested) 'Tensor' for observation
      prev_action: A 'Tensor' for action from previous time step
      env_id: the ID of the environment from which this time_step is
    """

    def is_first(self):
        if torch.is_tensor(self.step_type):
            return torch.eq(self.step_type, StepType.FIRST)
        raise ValueError('step_type is not a Torch Tensor')

    def is_mid(self):
        if torch.is_tensor(self.step_type):
            return torch.eq(self.step_type, StepType.MID)
        raise ValueError('step_type is not a Torch Tensor')

    def is_last(self):
        if torch.is_tensor(self.step_type):
            return torch.eq(self.step_type, StepType.LAST)
        raise ValueError('step_type is not a Torch Tensor')


def _create_timestep(observation, prev_action, reward, discount, env_id,
                     step_type):
    discount = torch.as_tensor(discount)
    # as_tensor reuses the underlying data store of numpy array if possible.
    create_tensor = lambda t: torch.as_tensor(t).detach()
    make_tensors = lambda struct: map_structure(create_tensor, struct)
    return TimeStep(
        step_type=torch.reshape(step_type, discount.shape),
        reward=make_tensors(reward),
        discount=discount,
        observation=make_tensors(observation),
        prev_action=make_tensors(prev_action),
        env_id=torch.as_tensor(env_id, dtype=torch.int64))


def timestep_first(observation, prev_action, reward, discount, env_id):
    return _create_timestep(observation, prev_action, reward, discount, env_id,
                            StepType.FIRST)


def timestep_mid(observation, prev_action, reward, discount, env_id):
    return _create_timestep(observation, prev_action, reward, discount, env_id,
                            StepType.MID)


def timestep_last(observation, prev_action, reward, discount, env_id):
    return _create_timestep(observation, prev_action, reward, discount, env_id,
                            StepType.LAST)


AlgStep = namedtuple('AlgStep', ['output', 'state', 'info'], default_value=())

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
        raise ValueError('step_type is not a Torch Tensor')

    def is_mid(self):
        if torch.is_tensor(self.step_type):
            return torch.eq(self.step_type, StepType.MID)
        raise ValueError('step_type is not a Torch Tensor')

    def is_last(self):
        if torch.is_tensor(self.step_type):
            return torch.eq(self.step_type, StepType.LAST)
        raise ValueError('step_type is not a Torch Tensor')


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


LossInfo = namedtuple(
    "LossInfo",
    [
        "loss",  # batch loss shape should be (T, B) or (B,)
        "scalar_loss",  # shape is ()
        "extra"  # nested batch and/or scalar losses, for summary only
    ],
    default_value=())
