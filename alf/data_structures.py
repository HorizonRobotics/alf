# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.utils import common
from alf.utils.tensor_utils import to_tensor


def namedtuple(typename, field_names, default_value=None, default_values=()):
    """namedtuple with default value.

    Args:
        typename (str): type name of this namedtuple.
        field_names (list[str]): name of each field.
        default_value (Any): the default value for all fields.
        default_values (list|dict): default value for each field.
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
    """Defines the status of a ``TimeStep`` within a sequence."""
    # Denotes the first `TimeStep` in a sequence.
    FIRST = np.int32(0)
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = np.int32(1)
    # Denotes the last `TimeStep` in a sequence.
    LAST = np.int32(2)

    def __new__(cls, value):
        """Add ability to create ``StepType`` constants from a value."""
        if value == cls.FIRST:
            return cls.FIRST
        if value == cls.MID:
            return cls.MID
        if value == cls.LAST:
            return cls.LAST

        raise ValueError(
            'No known conversion for `%r` into a StepType' % value)


class TimeStep(
        namedtuple(
            'TimeStep', [
                'step_type', 'reward', 'discount', 'observation',
                'prev_action', 'env_id', 'untransformed', "env_info"
            ],
            default_value=())):
    """A ``TimeStep`` contains the data emitted by an environment at each step of
    interaction. A ``TimeStep`` holds a ``step_type``, an ``observation`` (typically a
    NumPy array or a dict or list of arrays), and an associated ``reward`` and
    ``discount``.

    The first ``TimeStep`` in a sequence will equal ``StepType.FIRST``. The final
    ``TimeStep`` will equal ``StepType.LAST``. All other ``TimeStep``s in a sequence
    will equal to ``StepType.MID``.

    It has eight attributes:

    - step_type: a ``Tensor`` or numpy int of ``StepType`` enum values.
    - reward: a ``Tensor`` of reward values from executing 'prev_action'.
    - discount: A discount value in the range :math:`[0, 1]`.
    - observation: A (nested) ``Tensor`` for observation.
    - prev_action: A (nested) ``Tensor`` for action from previous time step.
    - env_id: A scalar ``Tensor`` of the environment ID of the time step.
    - untransformed: a nest that represents the entire time step itself *before*
      any transformation (e.g., observation or reward transformation); used for
      experience replay observing by subalgorithms.
    - env_info: A dictionary containing information returned by Gym environments'
      ``info``.
    """

    def is_first(self):
        return self.step_type == StepType.FIRST

    def is_mid(self):
        return self.step_type == StepType.MID

    def is_last(self):
        return self.step_type == StepType.LAST

    def cuda(self):
        """Get the cuda version of this data structure."""
        r = getattr(self, "_cuda", None)
        if r is None:
            r = nest.map_structure(
                lambda x: x.cuda() if isinstance(x, torch.Tensor) else x, self)
            self._cuda = r
        return r

    def cpu(self):
        """Get the cpu version of this data structure."""
        r = getattr(self, "_cpu", None)
        if r is None:
            r = nest.map_structure(
                lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, self)
            self._cpu = r
        return r


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
                'state',  # state passed to rollout() to generate `action`
                'batch_info',
                'replay_buffer',
                'rollout_info_field',
            ],
            default_value=())):
    """An ``Experience`` is a ``TimeStep`` in the context of training an RL algorithm.
    For the training purpose, it's augmented with several new attributes:

    - action: A (nested) ``Tensor`` for action taken for the current time step.
    - rollout_info: ``AlgStep.info`` from ``rollout_step()``.
    - state: State passed to ``rollout_step()`` to generate ``action``.
    - batch_info: Its type is ``alf.experience_replays.replay_buffer.BatchInfo``.
        This is only used when experiece is passed as an argument for ``Algorithm.calc_loss()``.
        Different from other members, the shape of the tensors in ``batch_info``
        is [B], where B is the batch size.
    - replay_buffer: The replay buffer where the batch_info generated from.
        Currently, this field is available when experience is passed to
        ``Algorithm.calc_loss()``, ``Algorithm.preprocess_experience()`` or
        ``DataTransformer.transform_experience()``
    - rollout_info_field: The name of the rollout_info field in replay buffer.
        This is useful when an algorithm needs to access its rollout_info in
        the replay buffer.
    """

    def is_first(self):
        return self.step_type == StepType.FIRST

    def is_mid(self):
        return self.step_type == StepType.MID

    def is_last(self):
        return self.step_type == StepType.LAST


AlgStep = namedtuple('AlgStep', ['output', 'state', 'info'], default_value=())


def _is_numpy_array(x):
    return isinstance(x, (np.number, np.ndarray))


def _generate_time_step(batched,
                        observation,
                        step_type,
                        discount,
                        prev_action=None,
                        action_spec=None,
                        reward=None,
                        reward_spec=ts.TensorSpec(()),
                        env_id=None,
                        env_info={}):

    flat_observation = nest.flatten(observation)

    if all(map(_is_numpy_array, flat_observation)):
        md = np
        if reward is not None:
            reward = np.float32(reward)
        discount = np.float32(discount)
    else:
        assert all(
            map(torch.is_tensor,
                flat_observation)), ("Elements in observation must be Tensor")
        md = torch
        if reward is not None:
            reward = to_tensor(reward, dtype=torch.float32)
        discount = to_tensor(discount, dtype=torch.float32)

    if batched:
        batch_size = flat_observation[0].shape[0]
        outer_dims = (batch_size, )
        if env_id is None:
            env_id = md.arange(batch_size, dtype=md.int32)
        if reward is not None:
            assert reward.shape[:1] == outer_dims
        if prev_action is not None:
            flat_action = nest.flatten(prev_action)
            assert flat_action[0].shape[:1] == outer_dims
    else:
        outer_dims = ()
        if env_id is None:
            env_id = md.zeros((), dtype=md.int32)

    step_type = md.full(outer_dims, step_type, dtype=md.int32)
    if reward is None:
        reward = md.zeros(outer_dims + reward_spec.shape, dtype=md.float32)
    discount = md.ones(outer_dims, dtype=md.float32) * discount
    if prev_action is None:
        prev_action = nest.map_structure(
            lambda spec: md.zeros(
                outer_dims + spec.shape,
                dtype=getattr(md, ts.torch_dtype_to_str(spec.dtype))),
            action_spec)

    return TimeStep(
        step_type,
        reward,
        discount,
        observation,
        prev_action,
        env_id,
        env_info=env_info)


def restart(observation,
            action_spec,
            reward_spec=ts.TensorSpec(()),
            env_id=None,
            env_info={},
            batched=False):
    """Returns a ``TimeStep`` with ``step_type`` set equal to ``StepType.FIRST``.

    Called by ``env.reset()``.

    Args:
        observation (nested tensors): observations of the env.
        action_spec (nested TensorSpec): tensor spec of actions.
        reward_spec (TensorSpec): a rank-1 or rank-0 (default) tensor spec
        env_id (batched or scalar torch.int32): (optional) ID of the env.
        env_info (dict): extra info returned by the environment.
        batched (bool): (optional) whether batched envs or not.

    Returns:
        TimeStep:
    """
    return _generate_time_step(
        batched=batched,
        observation=observation,
        step_type=StepType.FIRST,
        discount=1.,
        action_spec=action_spec,
        reward_spec=reward_spec,
        env_id=env_id,
        env_info=env_info)


def transition(observation,
               prev_action,
               reward,
               reward_spec=ts.TensorSpec(()),
               discount=1.0,
               env_id=None,
               env_info={}):
    """Returns a ``TimeStep`` with ``step_type`` set equal to ``StepType.MID``.

    Called by ``env.step()`` if not 'Done'.

    The batch size is inferred from the shape of ``reward``.

    If ``discount`` is a scalar, and ``observation`` contains tensors,
    then ``discount`` will be broadcasted to match ``reward.shape``.

    Args:
        observation (nested tensors): current observations of the env.
        prev_action (nested tensors): previous actions to the the env.
        reward (float): A scalar, or 1D NumPy array, or tensor.
        reward_spec (TensorSpec): a rank-1 or rank-0 (default) tensor spec. Used
            to tell if the transition is batched or not.
        discount (float): (optional) A scalar, or 1D NumPy array, or tensor.
        env_id (torch.int32): (optional) A scalar or 1D tensor of the environment
            ID(s).
        env_info (dict): extra info returned by the environment.

    Returns:
        TimeStep:

    Raises:
        ValueError: If observations are tensors but reward's rank
        is not 0 or 1.
    """
    return _generate_time_step(
        batched=torch.as_tensor(reward).ndim > len(reward_spec.shape),
        observation=observation,
        step_type=StepType.MID,
        discount=discount,
        prev_action=prev_action,
        reward=reward,
        reward_spec=reward_spec,
        env_id=env_id,
        env_info=env_info)


def termination(observation,
                prev_action,
                reward,
                reward_spec=ts.TensorSpec(()),
                env_id=None,
                env_info={}):
    """Returns a ``TimeStep`` with ``step_type`` set to ``StepType.LAST``.

    Called by ``env.step()`` if 'Done'. ``discount`` should not be sent in and
    will be set as 0.

    Args:
        observation (nested tensors): current observations of the env.
        prev_action (nested tensors): previous actions to the the env.
        reward (float): A scalar, or 1D NumPy array, or tensor.
        reward_spec (TensorSpec): a rank-1 or rank-0 (default) tensor spec. Used
            to tell if the termination is batched or not.
        env_id (torch.int32): (optional) A scalar or 1D tensor of the environment
            ID(s).
        env_info (dict): extra info returned by the environment.

    Returns:
        TimeStep:

    Raises:
        ValueError: If observations are tensors but reward's statically known rank
            is not 0 or 1.
    """
    return _generate_time_step(
        batched=torch.as_tensor(reward).ndim > len(reward_spec.shape),
        observation=observation,
        step_type=StepType.LAST,
        discount=0.,
        prev_action=prev_action,
        reward=reward,
        reward_spec=reward_spec,
        env_id=env_id,
        env_info=env_info)


def time_step_spec(observation_spec, action_spec, reward_spec):
    """Returns a ``TimeStep`` spec given the ``observation_spec`` and the
    ``action_spec``.
    """

    def is_valid_tensor_spec(spec):
        return isinstance(spec, ts.TensorSpec)

    assert all(map(is_valid_tensor_spec, nest.flatten(observation_spec)))
    assert all(map(is_valid_tensor_spec, nest.flatten(action_spec)))
    return TimeStep(
        step_type=ts.TensorSpec([], torch.int32),
        reward=reward_spec,
        discount=ts.BoundedTensorSpec([],
                                      torch.float32,
                                      minimum=0.0,
                                      maximum=1.0),
        observation=observation_spec,
        prev_action=action_spec,
        env_id=ts.TensorSpec([], torch.int32))


def make_experience(time_step: TimeStep, alg_step: AlgStep, state):
    """Make an instance of ``Experience`` from ``TimeStep`` and ``AlgStep``.

    Args:
        time_step (TimeStep): time step from the environment.
        alg_step (AlgStep): policy step returned from ``rollout()``.
        state (nested Tensor): state used for calling ``rollout()`` to get the
            ``policy_step``.
    Returns:
        Experience:
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
    """Make ``TimeStep`` from ``Experience``."""
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
        "extra",  # nested batch and/or scalar losses, for summary only

        # Priority for each sample. This will be used to update the priority in
        # the replay buffer so that in the future, this sample will be sampled
        # with probability proportional to this weight powered to
        # config.priority_replay_alpha.  If not empty, its shape should be (B,).
        "priority",
    ],
    default_value=())
