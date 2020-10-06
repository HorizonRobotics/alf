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
"""Data transformers for transforming data from environment or replay buffer."""

import copy
from functools import partial
import gin
import numpy as np
import torch
from torch import nn
from typing import Iterable

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

    DataTransformer is used for tranforming raw data from environment before
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
                transfomer.
        """
        data_transformers = nn.ModuleList()
        has_non_frame_stacker = False
        state_spec = []
        max_stack_size = 1
        for ctor in data_transformer_ctors:
            obs_trans = ctor(observation_spec)
            if isinstance(obs_trans, FrameStacker):
                max_stack_size = max(max_stack_size, obs_trans.stack_size)
                assert not has_non_frame_stacker, (
                    "FrameStacker need to be the "
                    "first data transformers if it is used.")
            else:
                has_non_frame_stacker = True
            observation_spec = obs_trans.transformed_observation_spec
            data_transformers.append(obs_trans)
            state_spec.append(obs_trans.state_spec)

        super().__init__(observation_spec, state_spec)
        self._stack_size = max_stack_size
        self._data_transformers = data_transformers

    @property
    def stack_size(self):
        return self._stack_size

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


@gin.configurable
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
        self._fields = fields or [None]
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
                minimum = np.repeat(
                    spec.minimum,
                    repeats=self._stack_size,
                    axis=self._stack_axis)
            else:
                minimum = spec.minimum
            if spec.maximum.shape != ():
                assert spec.maximum.shape == spec.shape
                maximum = np.repeat(
                    spec.maximum,
                    repeats=self._stack_size,
                    axis=self._stack_axis)
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
            return time_step

        is_first = time_step.step_type == StepType.FIRST
        steps = state.steps + 1
        steps[is_first] = 0
        stack_axis = self._stack_axis
        if stack_axis >= 0:
            stack_axis += 1
        first_samples = is_first.nonzero()

        prev_frames = copy.copy(state.prev_frames)

        def _stack_frame(obs, i):
            prev_frames[i] = copy.copy(prev_frames[i])
            # repeat the first frame
            if first_samples.numel() > 0:
                for t in range(self._stack_size - 1):
                    # prev_frames[i][t] might be used somewhere else, we should
                    # not directly modify it.
                    prev_frames[i][t] = prev_frames[i][t].clone()
                    prev_frames[i][t][first_samples] = obs[first_samples]
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
            # [B, 1]
            env_ids = env_ids.unsqueeze(-1)
            assert torch.all(
                prev_positions[:, 0] >= replay_buffer.get_earliest_position(
                    env_ids)
            ), ("Some previous posisions are no longer in the replay buffer")

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
            prev_obs = convert_device(prev_obs)
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
        return experience._replace(observation=observation)


class SimpleDataTransformer(DataTransformer):
    """Base class for simple data transformers.

    For simple data transformers, there is no state for ``transform_timestep`` and
    ``transform_timestep`` and ``transform_experience`` use same function
    ``_transform`` to do the trasformation
    """

    def __init__(self, transformed_observation_spec):
        super().__init__(transformed_observation_spec, state_spec=())

    def transform_timestep(self, timestep: TimeStep, state):
        return self._transform(timestep), ()

    def transform_experience(self, experience: Experience):
        return self._transform(experience)

    def _transform(self, timestep_or_exp):
        """Transform TimeStep or Experience.

        Note that for TimeStep, the shapes are [B, ...].
        For Experience, the shapes are [B, T, ...]

        Args:
            timestep_or_exp (TimeStep|Experience): data to be transformed
        Returns:
            transformed TimeStep of Experience
        """
        raise NotImplementedError()


class IdentityDataTransformer(SimpleDataTransformer):
    """A data transformer that keeps the data unchanged."""

    def __init__(self, observation_spec=None):
        """
        observation_spec (nested TensorSpec): describing the observation. This
            should be provided when ``transformed_observation_spec`` propery
            needs to be accessed.
        """
        super().__init__(observation_spec)

    def _transform(self, timestep_or_exp):
        return timestep_or_exp


@gin.configurable
class ImageScaleTransformer(SimpleDataTransformer):
    def __init__(self, observation_spec, min=-1.0, max=1.0, fields=None):
        """Scale image to min and max (0->min, 255->max).

        Args:
            observation_spec (nested TensorSpec): describing the observation in timestep
            fields (list[str]): the fields to be applied with the transformation. If
                None, then ``observation`` must be a ``Tensor`` with dtype ``uint8``.
                A field str can be a multi-step path denoted by "A.B.C".
            min (float): normalize minimum to this value
            max (float): normalize maximum to this value
        """
        self._fields = fields or [None]
        self._scale = (max - min) / 255.
        self._min = min
        new_observation_spec = observation_spec

        def _transform_spec(spec):
            assert isinstance(
                spec,
                alf.TensorSpec), (str(type(spec)) + "is not a TensorSpec")
            assert spec.dtype == torch.uint8, "Image must have dtype uint8!"
            return alf.BoundedTensorSpec(
                spec.shape, dtype=torch.float32, minimum=min, maximum=max)

        for field in self._fields:
            new_observation_spec = alf.nest.transform_nest(
                new_observation_spec, field, _transform_spec)

        super().__init__(new_observation_spec)

    def _transform(self, timestep_or_exp):
        def _transform_image(obs):
            assert isinstance(obs,
                              torch.Tensor), str(type(obs)) + ' is not Tensor'
            assert obs.dtype == torch.uint8, "Image must have dtype uint8!"
            obs = obs.type(torch.float32)
            return self._scale * obs + self._min

        observation = timestep_or_exp.observation
        for field in self._fields:
            observation = alf.nest.transform_nest(observation, field,
                                                  _transform_image)
        return timestep_or_exp._replace(observation=observation)


@gin.configurable
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
            update_mode (str): update stats during either "replay" or "rollout".
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

    def _transform(self, timestep_or_exp):
        """Normalize a given observation. If during unroll, then first update
        the normalizer. The normalizer won't be updated in other circumstances.
        """
        observation = timestep_or_exp.observation
        if self._fields is None:
            obs = observation
        else:
            obs = dict([(field, alf.nest.get_field(observation, field))
                        for field in self._fields])
        if ((self._update_mode == "replay" and common.is_replay())
                or (self._update_mode == "rollout" and common.is_rollout())):
            self._normalizer.update(obs)
        obs = self._normalizer.normalize(obs, self._clipping)
        if self._fields is None:
            observation = obs
        else:
            for f, o in obs.items():
                observation = alf.nest.set_field(observation, f, o)
        return timestep_or_exp._replace(observation=observation)


@gin.configurable
class RewardClipping(SimpleDataTransformer):
    """Clamp immediate rewards to the range :math:`[min, max]`.

    Can be used as a reward shaping function passed to an algorithm
    (e.g. ``ActorCriticAlgorithm``).
    """

    def __init__(self, observation_spec, minmax=(-1, 1)):
        """
        Args:
            observation_spec (nested TensorSpec): describing the observation in timestep
            minmax (tuple[float]): clip this range
        """
        super().__init__(observation_spec)
        assert minmax[0] <= minmax[1], "range error"
        self._minmax = minmax

    def _transform(self, timestep_or_exp):
        return timestep_or_exp._replace(
            reward=timestep_or_exp.reward.clamp(*self._minmax))


@gin.configurable
class RewardNormalizer(SimpleDataTransformer):
    """Transform reward to be zero-mean and unit-variance."""

    def __init__(self,
                 observation_spec,
                 normalizer=None,
                 clip_value=-1.0,
                 update_mode="replay"):
        """
        Args:
            observation_spec (nested TensorSpec): describing the observation in
                timestep
            normalizer (Normalizer): the normalizer to be used to normalizer the
                reward. If None, will use ``ScalarAdaptiveNormalizer``.
            clip_value (float): if > 0, will clip the normalized reward within
                [-clip_value, clip_value]. Do not clip if ``clip_value`` < 0
            update_mode (str): update stats during either "replay" or "rollout".
        """
        super().__init__(observation_spec)
        if normalizer is None:
            normalizer = ScalarAdaptiveNormalizer(auto_update=False)
        self._normalizer = normalizer
        self._clip_value = clip_value
        self._update_mode = update_mode

    def _transform(self, timestep_or_exp):
        if ((self._update_mode == "replay" and common.is_replay())
                or (self._update_mode == "rollout" and common.is_rollout())):
            self._normalizer.update(timestep_or_exp.reward)
        return timestep_or_exp._replace(
            reward=self._normalizer.normalize(
                timestep_or_exp.reward, clip_value=self._clip_value))


@gin.configurable
class RewardScaling(SimpleDataTransformer):
    """Scale immediate rewards by a factor of ``scale``.

    Can be used as a reward shaping function passed to an algorithm
    (e.g. ``ActorCriticAlgorithm``).
    """

    def __init__(self, observation_spec, scale):
        """
        Args:
            observation_spec (nested TensorSpec): describing the observation in timestep
            scale (float): scale factor
        """
        super().__init__(observation_spec)
        self._scale = scale

    def _transform(self, timestep_or_exp):
        return timestep_or_exp._replace(
            reward=timestep_or_exp.reward * self._scale)


def create_data_transformer(data_transformer_ctor, observation_spec):
    """Create a data transformer.

    Args:
        data_transformer_ctor (Callable|list[Callable]): Function(s)
            for creating data transformer(s). Each of them will be called
            as ``data_transformer_ctor(observation_spec)`` to create a data
            transformer. Available transformers are in ``algorithms.data_transformer``.
        observation_spec (nested TensorSpec): the spec of the raw observation.
    Returns:
        DataTransformer
    """
    if data_transformer_ctor is None:
        return IdentityDataTransformer(observation_spec)
    elif not isinstance(data_transformer_ctor, Iterable):
        data_transformer_ctor = [data_transformer_ctor]

    if len(data_transformer_ctor) == 1:
        return data_transformer_ctor[0](observation_spec)

    return SequentialDataTransformer(data_transformer_ctor, observation_spec)
