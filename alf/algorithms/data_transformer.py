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
"""Data transformers for transforming data from environment or replay buffer."""
from absl import logging
import copy
from functools import partial
import numpy as np
import torch
from torch import nn
from typing import Iterable, Optional

import alf
from alf.data_structures import AlgStep, Experience, namedtuple, StepType, TimeStep
from alf.experience_replayers.replay_buffer import ReplayBuffer, BatchInfo
from alf.nest.utils import convert_device
from alf.utils.normalizers import WindowNormalizer, EMNormalizer, AdaptiveNormalizer
from alf.utils import common
from alf.utils.normalizers import ScalarAdaptiveNormalizer

FrameStackState = namedtuple('FrameStackState', ['steps', 'prev_frames'])


class DataTransformer(nn.Module):
    """Base class for data transformers.

    DataTransformer is used for transforming raw data from environment before
    passing to actual algorithms.

    Most data transformers can subclass from ``SimpleDataTransformer``, which
    provides a simpler interface.
    """

    def __init__(self, transformed_observation_spec, state_spec):
        """
        Args:
            transformed_observation_spec (nested TensorSpec): describing the
                transformed observation
            state_spec (nested TensorSpec): describing the state of the
                transformer when it is used to transform ``TimeStep``
        """
        self._transformed_observation_spec = transformed_observation_spec
        self._state_spec = state_spec
        super().__init__()

    @property
    def stack_size(self):
        """The number of frames being stacked as one observation."""
        return 1

    @property
    def transformed_observation_spec(self):
        """Get the transformed observation_spec."""
        assert self._transformed_observation_spec is not None, (
            "transformed_observation_spec is not set. This might be caused by "
            "not setting observation_spec when constructing IdentityDataTransformer."
        )
        return self._transformed_observation_spec

    @property
    def state_spec(self):
        """Get the state spec of this transformer."""
        return self._state_spec

    def transform_timestep(self, timestep: TimeStep, state):
        """Transform a TimeStep structure.

        This is used during unroll or predict.

        Args:
            timestep (TimeStep): the TimeStep needs to be transformed
            state (nested Tensor): the state of the transformer running over the
                timestep sequence. It should be the returned state from the previous
                call to transform_timestep. For the initial call to ``transform_timestep``
                an zero state following the ``state_spec`` can be used.
        Returns:
            tuple:
            - transformed TimeStep
            - state of the transformer
        """
        raise NotImplementedError()

    def transform_experience(self, experience: Experience):
        """Transform an Experience structure.

        This is used on the experience data retrieved from replay buffer.

        Args:
            experience (Experience): the experience retrieved from replay buffer.
                Note that ``experience.batch_info``, ``experience.replay_buffer``
                need to be set.
        Returns:
            Experience: transformed experience
        """
        raise NotImplementedError()


class SequentialDataTransformer(DataTransformer):
    """A data transformer consisting of a sequence of data transformers."""

    def __init__(self, data_transformer_ctors, observation_spec):
        """
        Args:
            data_transformer_ctor (list[Callable]): Functions for creating data
                transformers. Each of them will be called as
                ``data_transformer_ctors[i](observation_spec)`` to create a data
                transformer.
            observation_spec (nested TensorSpec): describing the raw observation
                in timestep. It is the observation passed to the first data
                transformer.
        """
        data_transformers = nn.ModuleList()
        state_spec = []
        max_stack_size = 1
        for i, ctor in enumerate(data_transformer_ctors):
            obs_trans = ctor(observation_spec=observation_spec)
            if isinstance(obs_trans, FrameStacker):
                max_stack_size = max(max_stack_size, obs_trans.stack_size)
            observation_spec = obs_trans.transformed_observation_spec
            data_transformers.append(obs_trans)
            state_spec.append(obs_trans.state_spec)
        SequentialDataTransformer._validate_order(data_transformers)

        super().__init__(observation_spec, state_spec)
        self._stack_size = max_stack_size
        self._data_transformers = data_transformers

    @staticmethod
    def _validate_order(data_transformers):
        def _tier_of(data_transformer):
            if isinstance(data_transformer, UntransformedTimeStep):
                return 1
            if isinstance(data_transformer,
                          (HindsightExperienceTransformer, FrameStacker)):
                return 2
            return 3

        prev_tier = 0
        for i in range(len(data_transformers)):
            tier = _tier_of(data_transformers[i])
            assert tier >= prev_tier, (
                f'{type(data_transformers[i]).__name__} must be placed before '
                f'{type(data_transformers[i - 1]).__name__}. Please check '
                'docs/notes/knowledge_base.rst for details.')
            prev_tier = tier

    @property
    def stack_size(self):
        return self._stack_size

    def members(self):
        return self._data_transformers

    def transform_timestep(self, timestep: TimeStep, state):
        new_state = []
        for trans, state in zip(self._data_transformers, state):
            timestep, s = trans.transform_timestep(timestep, state)
            new_state.append(s)
        return timestep, new_state

    def transform_experience(self, experience):
        for trans in self._data_transformers:
            experience = trans.transform_experience(experience)
        return experience


@alf.configurable
class FrameStacker(DataTransformer):
    def __init__(self,
                 observation_spec,
                 stack_size=4,
                 stack_axis=0,
                 fields=None):
        """Create a FrameStacker object.

        Args:
            observation_spec (nested TensorSpec): describing the observation in timestep
            stack_size (int): stack so many frames
            stack_axis (int): the dimension to stack the observation.
            fields (list[str]): fields to be stacked, A field str is a multi-level
                path denoted by "A.B.C". If None, then non-nested observation is stacked.
        """
        assert stack_size >= 1, (
            "stack_size should be an integer greater than "
            "or equal to 1")
        self._stack_axis = stack_axis
        self._stack_size = stack_size
        self._frames = dict()
        self._fields = fields if (fields is not None) else [None]
        self._exp_fields = []
        prev_frames_spec = []
        stacked_observation_spec = observation_spec
        for field in self._fields:
            if field is not None:
                exp_field = 'observation.' + field
            else:
                exp_field = 'observation'
            self._exp_fields.append(exp_field)

            spec = alf.nest.get_field(observation_spec, field)
            prev_frames_spec.append([spec] * (self._stack_size - 1))
            stacked_observation_spec = alf.nest.transform_nest(
                stacked_observation_spec, field, self._make_stacked_spec)

        super().__init__(
            transformed_observation_spec=stacked_observation_spec,
            state_spec=FrameStackState(
                steps=alf.TensorSpec((), dtype=torch.int64),
                prev_frames=prev_frames_spec))

    @property
    def stack_size(self):
        """Get stack_size."""
        return self._stack_size

    def _make_stacked_spec(self, spec):
        assert isinstance(
            spec, alf.TensorSpec), (str(type(spec)) + "is not a TensorSpec")
        if spec.ndim > 0:
            stacked_shape = list(copy.copy(spec.shape))
            stacked_shape[self._stack_axis] = stacked_shape[
                self._stack_axis] * self._stack_size
            stacked_shape = tuple(stacked_shape)
        else:
            stacked_shape = (self._stack_size, )
        if not spec.is_bounded():
            return alf.TensorSpec(stacked_shape, spec.dtype)
        else:
            if spec.minimum.shape != ():
                assert spec.minimum.shape == spec.shape
                rep = [1] * spec.minimum.ndim
                rep[self._stack_axis] = self._stack_size
                minimum = np.tile(spec.minimum, rep)
            else:
                minimum = spec.minimum
            if spec.maximum.shape != ():
                assert spec.maximum.shape == spec.shape
                rep = [1] * spec.maximum.ndim
                rep[self._stack_axis] = self._stack_size
                maximum = np.tile(spec.maximum, rep)
            else:
                maximum = spec.maximum
            return alf.BoundedTensorSpec(
                stacked_shape,
                minimum=minimum,
                maximum=maximum,
                dtype=spec.dtype)

    def _make_state(self, spec):
        stacked_shape = list(copy.copy(spec.shape))
        stacked_shape[self._stack_axis] = stacked_shape[self._stack_axis] * (
            self._stack_size - 1)
        stacked_shape = tuple(stacked_shape)
        return alf.TensorSpec(stacked_shape, spec.dtype)

    def transform_timestep(self, time_step, state):
        if self._stack_size == 1:
            return time_step, state

        is_first = time_step.step_type == StepType.FIRST
        steps = state.steps + 1
        steps[is_first] = 0
        stack_axis = self._stack_axis
        if stack_axis >= 0:
            stack_axis += 1

        prev_frames = copy.copy(state.prev_frames)

        def _stack_frame(obs, i):
            prev_frames[i] = copy.copy(prev_frames[i])
            # repeat the first frame if needed
            for t in range(self._stack_size - 1):
                first_samples = alf.utils.common.expand_dims_as(is_first, obs)
                prev_frames[i][t] = torch.where(first_samples, obs,
                                                prev_frames[i][t])

            if obs.ndim > 1:
                stacked = torch.cat(prev_frames[i] + [obs], dim=stack_axis)
            else:
                stacked = torch.stack(prev_frames[i] + [obs], dim=1)
            prev_frames[i].pop(0)
            prev_frames[i].append(obs)
            return stacked

        observation = time_step.observation
        for i, field in enumerate(self._fields):
            observation = alf.nest.transform_nest(observation, field,
                                                  partial(_stack_frame, i=i))

        return (time_step._replace(observation=observation),
                FrameStackState(steps=steps, prev_frames=prev_frames))

    def transform_experience(self, experience: Experience):
        if self._stack_size == 1:
            return experience

        assert experience.batch_info != ()
        batch_info: BatchInfo = experience.batch_info
        replay_buffer: ReplayBuffer = experience.replay_buffer

        with alf.device(replay_buffer.device):
            # [B]
            env_ids = convert_device(batch_info.env_ids)
            # [B]
            positions = convert_device(batch_info.positions)

            prev_positions = torch.arange(self._stack_size -
                                          1) - self._stack_size + 1

            # [B, stack_size - 1]
            prev_positions = positions.unsqueeze(
                -1) + prev_positions.unsqueeze(0)
            episode_begin_positions = replay_buffer.get_episode_begin_position(
                positions, env_ids)
            # [B, 1]
            episode_begin_positions = episode_begin_positions.unsqueeze(-1)
            # [B, stack_size - 1]
            prev_positions = torch.max(prev_positions, episode_begin_positions)
            # [B]
            valid_prev = prev_positions[:,
                                        0] >= replay_buffer.get_earliest_position(
                                            env_ids)
            assert torch.all(valid_prev), (
                "Some previous posisions are no longer in the replay buffer: "
                f"{prev_positions[:, 0][~valid_prev]}, "
                f"{replay_buffer.get_earliest_position(env_ids)[~valid_prev]}")
            # [B, 1]
            env_ids = env_ids.unsqueeze(-1)

        batch_size, mini_batch_length = experience.step_type.shape

        # [[0, 1, ..., stack_size-1],
        #  [1, 2, ..., stack_size],
        #  ...
        #  [mini_batch_length - 1, ...]]
        #
        # [mini_batch_length, stack_size]
        obs_index = (torch.arange(self._stack_size).unsqueeze(0) +
                     torch.arange(mini_batch_length).unsqueeze(1))
        B = torch.arange(batch_size)
        obs_index = (B.unsqueeze(-1).unsqueeze(-1), obs_index.unsqueeze(0))

        def _stack_frame(obs, i):
            prev_obs = replay_buffer.get_field(self._exp_fields[i], env_ids,
                                               prev_positions)
            stacked_shape = alf.nest.get_field(
                self._transformed_observation_spec, self._fields[i]).shape
            # [batch_size, mini_batch_length + stack_size - 1, ...]
            stacked_obs = torch.cat((prev_obs, obs), dim=1)
            # [batch_size, mini_batch_length, stack_size, ...]
            stacked_obs = stacked_obs[obs_index]
            if self._stack_axis != 0 and obs.ndim > 3:
                stack_axis = self._stack_axis
                if stack_axis < 0:
                    stack_axis += stacked_obs.ndim
                else:
                    stack_axis += 3
                stacked_obs = stacked_obs.unsqueeze(stack_axis)
                stacked_obs = stacked_obs.transpose(2, stack_axis)
                stacked_obs = stacked_obs.squeeze(2)
            stacked_obs = stacked_obs.reshape(batch_size, mini_batch_length,
                                              *stacked_shape)
            return stacked_obs

        observation = experience.observation
        for i, field in enumerate(self._fields):
            observation = alf.nest.transform_nest(observation, field,
                                                  partial(_stack_frame, i=i))
        return experience._replace(
            time_step=experience.time_step._replace(observation=observation))


class SimpleDataTransformer(DataTransformer):
    """Base class for simple data transformers.

    For simple data transformers, there is no state for ``transform_timestep`` and
    ``transform_experience``. And ``transform_experience`` use the same function
    ``_transform`` to do the transformation of the ``time_step`` field of the
    experience.
    """

    def __init__(self, transformed_observation_spec):
        super().__init__(transformed_observation_spec, state_spec=())

    def transform_timestep(self, timestep: TimeStep, state):
        """Transform TimeStep.
        Note that for TimeStep, the shapes are [B, ...].

        Args:
            timestep: data to be transformed
        Returns:
            transformed TimeStep
        """
        return self._transform(timestep), ()

    def transform_experience(self, experience: Experience):
        """Transform Experience.

        For Experience, the shapes are [B, T, ...]

        Args:
            experience: data to be transformed
        Returns:
            transformed Experience
        """
        transformed_time_step = self._transform(experience.time_step)
        return experience._replace(time_step=transformed_time_step)

    def _transform(self, timestep):
        """Transform TimeStep.
        Note that this function is used by both ``transform_timestep``
        and ``transform_experience``.
        Args:
            timestep (TimeStep): data to be transformed. The shape is
            [B, ...] or [B, T, ...].
        Returns:
            transformed TimeStep
        """
        raise NotImplementedError()


class IdentityDataTransformer(SimpleDataTransformer):
    """A data transformer that keeps the data unchanged."""

    def __init__(self, observation_spec=None):
        """
        observation_spec (nested TensorSpec): describing the observation. This
            should be provided when ``transformed_observation_spec`` property
            needs to be accessed.
        """
        super().__init__(observation_spec)

    def _transform(self, timestep):
        return timestep


@alf.configurable
class ImageScaleTransformer(SimpleDataTransformer):
    def __init__(self, observation_spec, min=-1.0, max=1.0, fields=None):
        """Scale image to min and max (0->min, 255->max).

        Args:
            observation_spec (nested TensorSpec): describing the observation in timestep
            fields (list[str]): the fields to be applied with the transformation. If
                None, then ``observation`` must be an int ``Tensor`` in [0,255].
                A field str can be a multi-step path denoted by "A.B.C".
            min (float): normalize minimum to this value
            max (float): normalize maximum to this value
        """
        self._fields = fields if (fields is not None) else [None]
        self._scale = (max - min) / 255.
        self._min = min
        new_observation_spec = observation_spec

        def _transform_spec(spec):
            assert isinstance(
                spec,
                alf.TensorSpec), (str(type(spec)) + "is not a TensorSpec")
            assert ImageScaleTransformer._check_img_type(spec), (
                "Image must have int dtype in [0, 255]!")
            return alf.BoundedTensorSpec(
                spec.shape, dtype=torch.float32, minimum=min, maximum=max)

        for field in self._fields:
            new_observation_spec = alf.nest.transform_nest(
                new_observation_spec, field, _transform_spec)

        super().__init__(new_observation_spec)

    @staticmethod
    def _check_img_type(img):
        if 'int' not in alf.tensor_specs.torch_dtype_to_str(img.dtype):
            return False
        if isinstance(img, torch.Tensor):
            return (img.min() >= 0 and img.max() <= 255)
        elif isinstance(img, alf.BoundedTensorSpec):
            return (img.minimum >= 0 and img.maximum <= 255)
        return False

    def _transform(self, timestep):
        def _transform_image(obs):
            assert isinstance(obs,
                              torch.Tensor), str(type(obs)) + ' is not Tensor'
            assert ImageScaleTransformer._check_img_type(obs), (
                "Image must have int dtype in [0, 255]!")
            obs = self._scale * obs
            if self._min != 0:
                obs.add_(self._min)
            return obs

        observation = timestep.observation
        for field in self._fields:
            observation = alf.nest.transform_nest(observation, field,
                                                  _transform_image)
        return timestep._replace(observation=observation)


@alf.configurable
class ObservationNormalizer(SimpleDataTransformer):
    def __init__(self,
                 observation_spec,
                 fields=None,
                 clipping=0.,
                 window_size=10000,
                 update_rate=1e-4,
                 speed=8.0,
                 zero_mean=True,
                 update_mode="replay",
                 mode="adaptive"):
        """Create an observation normalizer with optional value clipping to be
        used as the ``data_transformer`` of an algorithm. It will be called
        before both ``rollout_step()`` and ``train_step()``.

        The normalizer by default doesn't automatically update the mean and std.
        Instead, it will check when ``self.forward()`` is called, whether an
        algorithm is unrolling or training. It only updates the mean and std
        during unroll. This is the suggested way of using an observation
        normalizer (i.e., update the stats when encountering new data for the
        first time). This same strategy has been used by OpenAI's baselines for
        training their Robotics environments.

        Args:
            observation_spec (nested TensorSpec): describing the observation in timestep
            fields (None|list[str]): If None, normalize all fields. Otherwise,
                only normalized the specified fields. Each string in ``fields``
                is a a multi-step path denoted by "A.B.C".
            clipping (float): a floating value for clipping the normalized
                observation into ``[-clipping, clipping]``. Only valid if it's
                greater than 0.
            window_size (int): the window size of ``WindowNormalizer``.
            update_rate (float): the update rate of ``EMNormalizer``.
            speed (float): the speed of updating for ``AdaptiveNormalizer``.
            zero_mean (bool): whether to make the normalized value be zero-mean
            update_mode (str): update stats during specified mode in ["replay",
                "rollout", "pretrain"].
            mode (str): a value in ["adaptive", "window", "em"] indicates which
                normalizer to use.
        """
        super().__init__(observation_spec)
        self._update_mode = update_mode
        self._clipping = float(clipping)
        self._fields = fields
        if fields is not None:
            observation_spec = dict([(field,
                                      alf.nest.get_field(
                                          observation_spec, field))
                                     for field in fields])
        if mode == "adaptive":
            self._normalizer = AdaptiveNormalizer(
                tensor_spec=observation_spec,
                speed=float(speed),
                auto_update=False,
                zero_mean=zero_mean,
                name="observations/adaptive_normalizer")
        elif mode == "window":
            self._normalzier = WindowNormalizer(
                tensor_spec=observation_spec,
                window_size=int(window_size),
                zero_mean=zero_mean,
                auto_update=False)
        elif mode == "em":
            self._normalizer = EMNormalizer(
                tensor_spec=observation_spec,
                update_rate=float(update_rate),
                zero_mean=zero_mean,
                auto_update=False)
        else:
            raise ValueError("Unsupported mode: " + mode)

    def _transform(self, timestep):
        """Normalize a given observation. If during unroll, then first update
        the normalizer. The normalizer won't be updated in other circumstances.
        """
        observation = timestep.observation
        if self._fields is None:
            obs = observation
        else:
            obs = dict([(field, alf.nest.get_field(observation, field))
                        for field in self._fields])
        if ((self._update_mode == "replay" and common.is_replay())
                or (self._update_mode == "rollout" and common.is_rollout())
                or (self._update_mode == "pretrain" and common.is_pretrain())):
            self._normalizer.update(obs)
        obs = self._normalizer.normalize(obs, self._clipping)
        if self._fields is None:
            observation = obs
        else:
            for f, o in obs.items():
                observation = alf.nest.set_field(observation, f, o)
        return timestep._replace(observation=observation)


class RewardTransformer(SimpleDataTransformer):
    """Base class for transforming reward.
    """

    def __init__(self, observation_spec):
        """
        Args:
            observation_spec (nested TensorSpec): describing the observation in timestep
        """
        super().__init__(observation_spec)

    def _transform(self, timestep):
        return timestep._replace(reward=self.forward(timestep.reward))


@alf.configurable
class RewardClipping(RewardTransformer):
    """Clamp immediate rewards to the range :math:`[min, max]`.

    Can be used as a reward shaping function passed to an algorithm
    (e.g. ``ActorCriticAlgorithm``).

    Note that if the reward is multi-dimensional, the clipping is applied to all
    the dimensions. If per-dimension operation is desired,
    """

    def __init__(self, observation_spec=(), minmax=(-1, 1)):
        """
        Args:
            observation_spec (nested TensorSpec): describing the observation in timestep
            minmax (tuple[float]): clip this range
        """
        super().__init__(observation_spec)
        assert minmax[0] <= minmax[1], "range error"
        self._minmax = minmax

    def forward(self, reward):
        return reward.clamp(*self._minmax)


@alf.configurable
class RewardNormalizer(RewardTransformer):
    """Transform reward to be zero-mean and unit-variance."""

    def __init__(self,
                 observation_spec=(),
                 normalizer=None,
                 update_max_calls=0,
                 clip_value=-1.0,
                 update_mode="replay"):
        """
        Args:
            observation_spec (nested TensorSpec): describing the observation in
                timestep
            normalizer (Normalizer): the normalizer to be used to normalizer the
                reward. If None, will use ``AdaptiveNormalizer`` according to
                env reward spec.
            update_max_calls (int): If >0, then the normalier's statistics will
                only be updated so many first calls of ``_transform()``.
            clip_value (float): if > 0, will clip the normalized reward within
                [-clip_value, clip_value]. Do not clip if ``clip_value`` < 0
            update_mode (str): update stats during either "replay" or "rollout".
        """
        super().__init__(observation_spec)
        if normalizer is None:
            normalizer = AdaptiveNormalizer(
                # ``get_reward_spec()`` is only a tmp solution. In some rare cases,
                # reward spec might have been changed by data transformers before
                # this one.
                # TODO: we should pass a ``time_step`` spec to the constructor.
                tensor_spec=alf.get_reward_spec(),
                auto_update=False,
                debug_summaries=True)
        self._normalizer = normalizer
        self._clip_value = clip_value
        self._update_mode = update_mode
        self._max_calls = update_max_calls
        self._calls = 0

    @property
    def normalizer(self):
        return self._normalizer

    @property
    def clip_value(self):
        return self._clip_value

    def forward(self, reward):
        norm = self._normalizer
        if ((self._update_mode == "replay" and common.is_replay())
                or (self._update_mode == "rollout" and common.is_rollout())):
            if self._max_calls == 0 or self._calls < self._max_calls:
                norm.update(reward)
            self._calls += 1

        return norm.normalize(reward, clip_value=self._clip_value)


@alf.configurable
class RewardScaling(RewardTransformer):
    """Scale immediate rewards by a factor of ``scale``.

    Can be used as a reward shaping function passed to an algorithm
    (e.g. ``ActorCriticAlgorithm``).

    Note that if the reward is multi-dimensional, the scaling is applied to all
    the dimensions. If per-dimension operation is desired,
    ``FunctionalRewardTransformer`` can be used.
    """

    def __init__(self, scale, observation_spec=()):
        """
        Args:
            scale (float): scale factor
            observation_spec (nested TensorSpec): describing the observation in timestep
        """
        super().__init__(observation_spec)
        self._scale = scale

    def forward(self, reward):
        return reward * self._scale


@alf.configurable
class RewardShifting(RewardTransformer):
    """Shift immediate rewards by a displacement of ``bias``.

    Note that if the reward is multi-dimensional, the shifting is applied to all
    the dimensions. If per-dimension operation is desired,
    ``FunctionalRewardTransformer`` can be used.
    """

    def __init__(self, bias, observation_spec=()):
        """
        Args:
            bias (float): displacement amount
            observation_spec (nested TensorSpec): describing the observation in timestep
        """
        super().__init__(observation_spec)
        self._bias = bias

    def forward(self, reward):
        return reward + self._bias


@alf.configurable
class FunctionalRewardTransformer(RewardTransformer):
    """Transform reward according to a provided function.

    Can be used as a reward shaping function passed to an algorithm
    (e.g. ``ActorCriticAlgorithm``).
    """

    def __init__(self, func, observation_spec=()):
        """
        Args:
            func (Callable): the transformation function to be applied to the
                reward. It takes reward as input and outputs a transformed reward.
            observation_spec (nested TensorSpec): describing the observation in timestep
        """
        super().__init__(observation_spec)
        self._trans_func = func

    def forward(self, reward):
        return self._trans_func(reward)


@alf.configurable
def l2_dist_close_reward_fn(achieved_goal, goal, threshold=.05):
    """Giving -1/0 reward based on how close the achieved state is to the goal state.

    Args:
        achieved_goal (Tensor): achieved state, of shape ``[batch_size, batch_length, ...]``
        goal (Tensor): goal state, of shape ``[batch_size, batch_length, ...]``
        threshold (float): L2 distance threshold for the reward.

    Returns:
        Tensor for -1/0 reward of shape ``[batch_size, batch_length]``.
    """

    if goal.dim() == 2:  # when goals are 1-dimensional
        assert achieved_goal.dim() == goal.dim()
        achieved_goal = achieved_goal.unsqueeze(2)
        goal = goal.unsqueeze(2)
    return -(torch.norm(achieved_goal - goal, dim=2) >= threshold).to(
        torch.float32)


@alf.configurable
class HindsightExperienceTransformer(DataTransformer):
    """Randomly transform her_proportion of `batch_size` trajectories with hindsight relabel.

        This transformer assumes that input observation is a dict of at least two fields:
        1) an ``achieved_goal`` field, indicating the current state of the environment, and
        2) a ``desired_goal`` field, indicating the desired state of the environment.
        The achieved_goal from a future timestep will be used to relabel the desired_goal
        of the current timestep.
        The exact field names can be provided via arguments to the class ``__init__``.

        To use this class, add it to any existing data transformers, e.g. use this config if
        ``ObservationNormalizer`` is an existing data transformer:

        .. code-block:: python

            ReplayBuffer.keep_episodic_info=True
            HindsightExperienceTransformer.her_proportion=0.8
            TrainerConfig.data_transformer_ctor=[@HindsightExperienceTransformer, @ObservationNormalizer]

        See unit test for more details on behavior.
    """

    def __init__(self,
                 observation_spec,
                 her_proportion=0.8,
                 achieved_goal_field="time_step.observation.achieved_goal",
                 desired_goal_field="time_step.observation.desired_goal",
                 reward_fn=l2_dist_close_reward_fn):
        """
        Args:
            her_proportion (float): proportion of hindsight relabeled experience.
            achieved_goal_field (str): path to the achieved_goal field in the
                exp nest.
            desired_goal_field (str): path to the desired_goal field in the
                exp nest.
            reward_fn (Callable): function to recompute reward based on
                achieve_goal and desired_goal.  Default gives reward 0 when
                L2 distance less than 0.05 and -1 otherwise, same as is done in
                suite_robotics environments.
        """
        super().__init__(
            transformed_observation_spec=observation_spec, state_spec=())
        self._her_proportion = her_proportion
        self._achieved_goal_field = achieved_goal_field
        self._desired_goal_field = desired_goal_field
        self._reward_fn = reward_fn

    def transform_timestep(self, timestep: TimeStep, state):
        return timestep, state

    def transform_experience(self, experience: Experience):
        """Hindsight relabel experience
        Note: The environments where the samples are from are ordered in the
            returned batch.

        Args:
            experience (Experience): experience sampled from replay buffer with batch_info
                and batch_info.replay_buffer both populated.

        Returns:
            Experience: the relabeled experience, with batch_info potentially changed.
        """
        her_proportion = self._her_proportion
        if her_proportion == 0:
            return experience
        info = experience.batch_info
        assert info != (), "Hindsight requires batch_info to be populated"
        # buffer (ReplayBuffer) is needed for access to future achieved goals.
        buffer = info.replay_buffer
        assert buffer != (), "Hindsight requires replay_buffer to be populated"
        accessed_fields = [
            "batch_info", "time_step.reward", "time_step.step_type",
            self._desired_goal_field, self._achieved_goal_field
        ]
        with alf.device(buffer.device):
            experience = alf.nest.transform_nest(
                experience, "batch_info.replay_buffer", lambda _: ())
            for f in accessed_fields:
                experience = alf.nest.transform_nest(
                    experience, f, lambda t: convert_device(t))
            result = experience
            info = experience.batch_info

            env_ids = info.env_ids
            start_pos = info.positions
            shape = result.reward.shape
            batch_size, batch_length = shape[:2]
            # TODO: add support for batch_length > 2.
            assert batch_length == 2, shape

            # relabel only these sampled indices
            her_cond = torch.rand(batch_size) < her_proportion
            (her_indices, ) = torch.where(her_cond)

            last_step_pos = start_pos[her_indices] + batch_length - 1
            last_env_ids = env_ids[her_indices]
            # Get x, y indices of LAST steps
            dist = buffer.steps_to_episode_end(last_step_pos, last_env_ids)
            if alf.summary.should_record_summaries():
                alf.summary.scalar(
                    "replayer/" + buffer._name + ".mean_steps_to_episode_end",
                    torch.mean(dist.type(torch.float32)))

            # get random future state
            future_idx = last_step_pos + (torch.rand(*dist.shape) *
                                          (dist + 1)).to(torch.int64)
            future_ag = buffer.get_field(self._achieved_goal_field,
                                         last_env_ids, future_idx).unsqueeze(1)

            # relabel desired goal
            result_desired_goal = alf.nest.get_field(result,
                                                     self._desired_goal_field)
            relabed_goal = result_desired_goal.clone()
            her_batch_index_tuple = (her_indices.unsqueeze(1),
                                     torch.arange(batch_length).unsqueeze(0))
            relabed_goal[her_batch_index_tuple] = future_ag

            # recompute rewards
            result_ag = alf.nest.get_field(result, self._achieved_goal_field)
            relabeled_rewards = self._reward_fn(result_ag, relabed_goal)

            non_her_or_fst = ~her_cond.unsqueeze(1) & (result.step_type !=
                                                       StepType.FIRST)
            # assert reward function is the same as used by the environment.
            if not torch.allclose(relabeled_rewards[non_her_or_fst],
                                  result.reward[non_her_or_fst]):
                not_close = torch.abs(relabeled_rewards[non_her_or_fst] -
                                      result.reward[non_her_or_fst]) > 0.01
                msg = (
                    "hindsight_relabel:\nrelabeled_reward\n{}\n!=\n" +
                    "env_reward\n{}\nag:\n{}\ndg:\n{}\nenv_ids:\n{}\nstart_pos:"
                    + "\n{}").format(
                        relabeled_rewards[non_her_or_fst][not_close],
                        result.reward[non_her_or_fst][not_close],
                        result_ag[non_her_or_fst][not_close],
                        result_desired_goal[non_her_or_fst][not_close],
                        env_ids.unsqueeze(1).expand(
                            shape[:2])[non_her_or_fst][not_close],
                        start_pos.unsqueeze(1).expand(
                            shape[:2])[non_her_or_fst][not_close])
                logging.warning(msg)
                # assert False, msg
                relabeled_rewards[non_her_or_fst] = result.reward[
                    non_her_or_fst]

        if alf.summary.should_record_summaries():
            alf.summary.scalar(
                "replayer/" + buffer._name + ".reward_mean_before_relabel",
                torch.mean(result.reward[her_indices][:-1]))
            alf.summary.scalar(
                "replayer/" + buffer._name + ".reward_mean_after_relabel",
                torch.mean(relabeled_rewards[her_indices][:-1]))

        result = alf.nest.transform_nest(
            result, self._desired_goal_field, lambda _: relabed_goal)

        result = result.update_time_step_field('reward', relabeled_rewards)

        if alf.get_default_device() != buffer.device:
            for f in accessed_fields:
                result = alf.nest.transform_nest(
                    result, f, lambda t: convert_device(t))
        result = alf.nest.transform_nest(
            result, "batch_info.replay_buffer", lambda _: buffer)
        return result


@alf.configurable
class UntransformedTimeStep(SimpleDataTransformer):
    """Put the time step itself to its field "untransformed". Note that this
    data transformer must be applied first, before any other data transformer.
    """

    def __init__(self,
                 observation_spec=None,
                 fields_to_keep: Optional[Iterable[str]] = None):
        """
        observation_spec (nested TensorSpec): describing the observation. This
            should be provided when ``transformed_observation_spec`` property
            needs to be accessed.
        fields_to_keep (list[str]): fields to be kept in ``untransformed``. This
            is useful if memory usage is a concern so that you only keep what
            you need.
        """
        super().__init__(observation_spec)
        self._fields_to_keep = fields_to_keep

    def _transform(self, timestep):
        if self._fields_to_keep is not None:
            return timestep._replace(
                untransformed=TimeStep(
                    **{f: getattr(timestep, f)
                       for f in self._fields_to_keep}))
        return timestep._replace(untransformed=timestep)


@alf.configurable
def create_data_transformer(data_transformer_ctor,
                            observation_spec,
                            device: Optional[str] = None):
    """Create a data transformer.

    Args:
        data_transformer_ctor (Callable|list[Callable]): Function(s)
            for creating data transformer(s). Each of them will be called
            as ``data_transformer_ctor(observation_spec)`` to create a data
            transformer. Available transformers are in ``algorithms.data_transformer``.
        observation_spec (nested TensorSpec): the spec of the raw observation.
        device: If not None, the data transformer(s) will be created on the
            specified device.
    Returns:
        DataTransformer
    """
    if data_transformer_ctor is None:
        return IdentityDataTransformer(observation_spec=observation_spec)
    elif not isinstance(data_transformer_ctor, Iterable):
        data_transformer_ctor = [data_transformer_ctor]

    with alf.device(device or alf.get_default_device()):
        if len(data_transformer_ctor) == 1:
            return data_transformer_ctor[0](observation_spec=observation_spec)

        return SequentialDataTransformer(data_transformer_ctor,
                                         observation_spec)
