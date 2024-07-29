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
"""Wrappers for gym (numpy) environments. """

from functools import partial
from absl import logging
from typing import List
from collections import deque, OrderedDict
import copy
import cv2
import gym
import numpy as np
import random

import alf
from alf.nest import transform_nest


def transform_space(observation_space, field, func):
    """Transform the child space in observation_space indicated by field using func

    Args:
        observation_space (gym.Space): space to be transformed
        field (str): field of the space to be transformed, multi-level path denoted by "A.B.C"
            If None, then non-nested observation_space is transformed
        func (Callable): transform function. The function will be called as
            func(observation_space, level) and should return new observation_space.
    Returns:
        transformed space
    """

    def _traverse_transform(space, levels):
        if not levels:
            return func(space)

        assert isinstance(space, gym.spaces.Dict)
        level = levels[0]

        new_val = copy.deepcopy(space)
        new_val.spaces[level] = _traverse_transform(
            space=space.spaces[level], levels=levels[1:])
        return new_val

    return _traverse_transform(
        space=observation_space, levels=field.split('.') if field else [])


@alf.configurable
class BaseObservationWrapper(gym.ObservationWrapper):
    """Base observation Wrapper

    BaseObservationWrapper provide basic functions and generic interface for transformation.

    The key interface functions are:
    1. transform_space(): transform space.
    2. transform_observation(): transform observation.
    """

    def __init__(self, env, fields=None):
        """
        Args:
            env (gym.Env): the gym environment
            fields (list[str]): fields to be applied transformation, A field str is a multi-level
                path denoted by "A.B.C". If None, then non-nested observation is transformed
        """
        super().__init__(env)

        self._fields = fields if (fields is not None) else [None]
        assert isinstance(self._fields, list), f"{fields} is not a list!"
        observation_space = env.observation_space
        for field in self._fields:
            observation_space = transform_space(
                observation_space=observation_space,
                field=field,
                func=self.transform_space)
        self.observation_space = observation_space

    def observation(self, observation):
        for field in self._fields:
            observation = transform_nest(
                nested=observation,
                field=field,
                func=self.transform_observation)
        return observation

    def transform_space(self, observation_space):
        """Transform space

        Subclass should implement this to perform transformation

        Args:
             observation_space (gym.Space): space to be transformed
        Returns:
            transformed space
        """
        raise NotImplementedError("transform_space is not implemented")

    def transform_observation(self, observation):
        """Transform observation

        Subclass should implement this to perform transformation

        Args:
             observation (ndarray): observation to be transformed
        Returns:
            transformed space
        """
        raise NotImplementedError("transform_observation is not implemented")


@alf.configurable
class ImageChannelFirst(BaseObservationWrapper):
    """Make images in observations channel_first. """

    def __init__(self, env, fields=None):
        super().__init__(env, fields=fields)

    def transform_space(self, observation_space):
        if isinstance(observation_space, gym.spaces.Box):
            if self._need_channel_transpose(observation_space.shape):
                low = observation_space.low
                high = observation_space.high
                if np.isscalar(low) and np.isscalar(high):
                    shape = (observation_space.shape[-1:] +
                             observation_space.shape[1:])
                    return gym.spaces.Box(
                        low=low,
                        high=high,
                        shape=shape,
                        dtype=observation_space.dtype)
                else:
                    low = self._make_channel_first(
                        observation_space.low, transpose=True)
                    high = self._make_channel_first(
                        observation_space.high, transpose=True)
                    return gym.spaces.Box(
                        low=low, high=high, dtype=observation_space.dtype)
        return observation_space

    def transform_observation(self, observation):
        transpose = self._need_channel_transpose(observation.shape)
        return self._make_channel_first(observation, transpose)

    def _need_channel_transpose(self, shape):
        if len(shape) == 3:
            return True
        return False

    def _make_channel_first(self, np_array, transpose=False):
        """
        Note that the extra copy() after np.transpose is crucial to pickle dump speed
        when called by subprocesses. An explanation of the non-contiguous memory caused
        by numpy transpose can be found in the following:

        https://stackoverflow.com/questions/26998223/
        """

        if transpose:
            rank = np_array.ndim
            np_array = np.transpose(np_array,
                                    (rank - 1, ) + tuple(range(rank - 1)))
        # TODO: do a generic memory contiguous check at ProcessEnvironment
        # using np_array.flags.continuous or np_array.flags.f_continuous and copy
        # if not contiguous
        return np_array.copy()


@alf.configurable
class FrameStack(BaseObservationWrapper):
    """Stack previous `stack_size` frames, applied to Gym env.

    This is deprecated. Please use ``alf.algorithms.data_transformer.FrameStacker``,
    which is more memory-efficient.
    """

    def __init__(self,
                 env,
                 stack_size=4,
                 channel_order='channels_last',
                 fields=None):
        """Create a FrameStack object.

        Args:
            env (gym.Space): gym environment.
            stack_size (int): stack so many frames
            channel_order (str): The ordering of the dimensions in the input images
                from the env, should be one of `channels_last` or `channels_first`.
            fields (list[str]): fields to be stacked, A field str is a multi-level
                path denoted by "A.B.C". If None, then non-nested observation is stacked.
        """
        logging.warning(
            "FrameStack is deprecated. Please use data_transformer.FrameStacker "
            "instead, which is more memory-efficient")
        self._channel_order = channel_order
        assert channel_order in ['channels_last', 'channels_first']
        if self._channel_order == 'channels_last':
            stack_axis = -1
        else:
            stack_axis = 0
        self._stack_axis = stack_axis
        self._stack_size = stack_size
        self._frames = dict()
        super().__init__(env, fields=fields)

    def transform_space(self, observation_space):
        if isinstance(observation_space, gym.spaces.Box):
            low = np.repeat(
                observation_space.low,
                repeats=self._stack_size,
                axis=self._stack_axis)
            high = np.repeat(
                observation_space.high,
                repeats=self._stack_size,
                axis=self._stack_axis)
            return gym.spaces.Box(
                low=low, high=high, dtype=observation_space.dtype)
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            return gym.spaces.MultiDiscrete(
                [observation_space.nvec] * self._stack_size)
        else:
            raise ValueError("Unsupported space:%s" % observation_space)

    def observation(self, observation):
        for field in self._fields:
            observation = transform_nest(
                nested=observation,
                field=field,
                func=lambda obs: self.transform_observation(obs, field))
        return observation

    def transform_observation(self, observation, field):
        queue = self._frames.get(field, None)
        if not queue:
            queue = deque(maxlen=self._stack_size)
            for _ in range(self._stack_size):
                queue.append(observation)
            self._frames[field] = queue
        else:
            queue.append(observation)
        return np.concatenate(queue, axis=self._stack_axis)

    def reset(self):
        self._frames = dict()
        return super().reset()


@alf.configurable
class FrameSkip(gym.Wrapper):
    """
    Repeat same action n times and return the last observation
     and accumulated reward
    """

    def __init__(self, env, skip):
        """Create a FrameSkip object

        Args:
            env (gym.Env): the gym environment
            skip (int): skip `skip` frames (skip=1 means no skip)
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        obs = None
        accumulated_reward = 0
        done = False
        info = {}
        num_env_frames = 0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            accumulated_reward += reward
            if 'num_env_frames' in info:
                # in case FrameSkip wrapper is being nested:
                n_steps = info['num_env_frames']
            else:
                n_steps = 1
            num_env_frames += n_steps
            if done:
                break
        info['num_env_frames'] = num_env_frames
        return obs, accumulated_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self.env, name)


@alf.configurable
class FrameResize(BaseObservationWrapper):
    def __init__(self,
                 env,
                 width=84,
                 height=84,
                 fields=None,
                 interpolation=cv2.INTER_AREA):
        """Create a FrameResize instance

        Args:
             env (gym.Env): the gym environment
             width (int): resize width
             height (int): resize height
             fields (list[str]):  fields to be resized, A field str is a multi-level
                path denoted by "A.B.C". If None, then non-nested observation is resized
             interpolation (int): cv2 interpolation type
        """
        self._width = width
        self._height = height
        self._interpolation = interpolation
        super().__init__(env, fields=fields)

    def transform_space(self, observation_space):
        obs_shape = observation_space.shape
        assert len(obs_shape) == 3, "observation shape should be (H,W,C)"
        return gym.spaces.Box(
            low=observation_space.low.min(),
            high=observation_space.high.max(),
            shape=[self._height, self._width] + list(obs_shape[2:]),
            dtype=observation_space.dtype)

    def transform_observation(self, observation):
        obs = cv2.resize(
            observation, (self._width, self._height),
            interpolation=self._interpolation)
        if len(obs.shape) != 3:
            obs = obs[:, :, np.newaxis]
        return obs


@alf.configurable
class EpisodicRandomFrameCrop(BaseObservationWrapper):
    def __init__(self,
                 env: gym.Env,
                 cropping_fraction=0.8,
                 channel_order: str = "channels_last",
                 share_cropping: bool = True,
                 fields: List[str] = None):
        """Create a frame cropping wrapper that augments the data distribution
        by randomly crops the image frame according to the specified fraction.
        Each episode has a randomized cropping location which is *consistent*
        over the episode.

        Args:
            env: the gym environment
            cropping_fraction: the portion of the original image to crop (keep)
            channel_order: The ordering of the dimensions in the input images
                from the env, should either "channels_last" or "channels_first".
            share_cropping: if there are multiple image fields, whether they
                share the same cropping position at each time step. This might
                be useful if there are multiple images with the same camera intrinsics,
                e.g., RGB + depth.
            fields: fields to be cropped. A field str is a multi-level
                path denoted by "A.B.C". If None, then non-nested observation is
                cropped.
        """
        assert 0 < cropping_fraction < 1

        def _dict_space(space):
            if isinstance(space, gym.spaces.Dict):
                return space.spaces
            return space

        self._original_observation_space = _dict_space(env.observation_space)
        self._cropping_fraction = cropping_fraction
        self._channel_order = channel_order
        self._share_cropping = share_cropping
        assert channel_order in ['channels_last', 'channels_first']
        super().__init__(env, fields=fields)
        self._observation_space = _dict_space(self.observation_space)
        self._syx = alf.nest.map_structure(lambda _: None,
                                           self._observation_space)

    def observation(self, observation):
        for field in self._fields:
            syx = alf.nest.get_field(self._syx, field)
            space = alf.nest.get_field(self._observation_space, field)
            observation = transform_nest(
                nested=observation,
                field=field,
                func=partial(
                    self.transform_observation,
                    sy=syx[0],
                    sx=syx[1],
                    space=space))
        return observation

    def _get_hwc(self, space):
        assert len(
            space.shape) == 3, "observation shape should be (H,W,C) or (C,H,W)"
        if self._channel_order == "channels_last":
            return space.shape
        return space.shape[1:] + space.shape[:1]

    def reset(self, **kwargs):
        """Randomly select cropping start positions.
        """
        sy, sx = None, None
        for field in self._fields:
            if not self._share_cropping or sy is None:
                ori_space = alf.nest.get_field(
                    self._original_observation_space, field)
                space = alf.nest.get_field(self._observation_space, field)
                H, W, _ = self._get_hwc(ori_space)
                h, w, _ = self._get_hwc(space)
                sy, sx = np.random.randint(H - h), np.random.randint(W - w)
            self._syx = alf.nest.set_field(
                nested=self._syx, field=field, new_value=(sy, sx))
        return super().reset(**kwargs)

    def transform_space(self, observation_space):
        H, W, C = self._get_hwc(observation_space)
        h, w = int(self._cropping_fraction * H), int(
            self._cropping_fraction * W)
        if self._channel_order == "channels_last":
            new_shape = (h, w, C)
        else:
            new_shape = (C, h, w)
        return gym.spaces.Box(
            low=observation_space.low.min(),
            high=observation_space.high.max(),
            shape=new_shape,
            dtype=observation_space.dtype)

    def transform_observation(self, observation, sy, sx, space):
        h, w, _ = self._get_hwc(space)
        if self._channel_order == "channels_last":
            obs = observation[sy:sy + h, sx:sx + w, :]
        else:
            obs = observation[:, sy:sy + h, sx:sx + w]
        return obs


@alf.configurable
class FrameFlip(BaseObservationWrapper):
    def __init__(self,
                 env: gym.Env,
                 ud_flip_prob: float = 0.5,
                 lr_flip_prob: float = 0.5,
                 channel_order: str = 'channels_last',
                 fields: List[str] = None):
        """Create a frame flipping wrapper that randomly flips the image fields
        either vertically or horizontally. For each episode, all fields will
        have the SAME flipping operation.

        The prob for each flipping result::

            identical: (1 - udp) * (1 - lrp)
            ud_flip: udp * (1 - lrp)
            lr_flip: (1 - udp) * lrp
            rotate180: udp * lrp

        This wrapper is usually used for data augmentation.

        Args:
            env: the gym environment
            ud_flip_prob: the prob of flipping up-down on the original image.
            lr_flip_prob: the prob of flipping left-right, *after* the testing of
                up-down flipping.
            channel_order: The ordering of the dimensions in the input images
                from the env, should either "channels_last" or "channels_first".
            fields: fields to be cropped. A field str is a multi-level
                path denoted by "A.B.C". If None, then non-nested observation is
                cropped.
        """
        assert 1 >= ud_flip_prob >= 0
        assert 1 >= lr_flip_prob >= 0
        super().__init__(env, fields=fields)
        self._channel_order = channel_order
        assert channel_order in ['channels_last', 'channels_first']
        self._ud_flip_prob = ud_flip_prob
        self._lr_flip_prob = lr_flip_prob
        self._flip_mode = 0

    def reset(self, **kargs):
        self._flip_mode = 0
        if np.random.rand() < self._ud_flip_prob:
            self._flip_mode |= 1
        if np.random.rand() < self._lr_flip_prob:
            self._flip_mode |= 2
        # flip_mode: 0 - no flippng; 1 - up-down; 2 - left-right; 3 - rotate
        return super().reset(**kargs)

    def transform_space(self, observation_space):
        assert len(observation_space.shape) == 3, (
            "The observation space must be (C,H,W) or (H,W,C)!")
        return observation_space

    def transform_observation(self, observation):
        if self._channel_order == 'channels_last':
            if self._flip_mode & 1:
                observation = observation[::-1, ...]
            if self._flip_mode & 2:
                observation = observation[:, ::-1, ...]
        else:
            if self._flip_mode & 1:
                observation = observation[:, ::-1, :]
            if self._flip_mode & 2:
                observation = observation[..., ::-1]
        return observation


@alf.configurable
class FrameCrop(BaseObservationWrapper):
    def __init__(self,
                 env,
                 sx=0,
                 sy=0,
                 width=84,
                 height=84,
                 channel_order='channels_last',
                 fields=None):
        """Create a FrameCrop instance

        Args:
             env (gym.Env): the gym environment
             sx (int): start position along the horizontal direction (x-axis)
             sy (int): start position along the vertical direction (y-axis)
             width (int): crop width
             height (int): crop height
            channel_order (str): The ordering of the dimensions in the input images
                from the env, should be one of `channels_last` or `channels_first`.
             fields (list[str]):  fields to be cropped, A field str is a multi-level
                path denoted by "A.B.C". If None, then non-nested observation is
                cropped.
        """
        assert sx >= 0 and sy >= 0, (
            "The start positions should be non-negative",
            "Got ({}, {}).".format(sx, sy))
        self._sx = sx
        self._sy = sy
        self._width = width
        self._height = height
        self._channel_order = channel_order
        assert channel_order in ['channels_last', 'channels_first']
        super().__init__(env, fields=fields)

    def transform_space(self, observation_space):
        obs_shape = observation_space.shape
        assert len(
            obs_shape) == 3, "observation shape should be (H,W,C) or (C,H,W)"

        if self._channel_order == "channels_last":
            new_shape = [self._height, self._width] + list(obs_shape[2:])
            height_axis = 0  # (H,W,C)
        else:
            new_shape = list(obs_shape[:1]) + [self._height, self._width]
            height_axis = 1  # (C,H,W)

        assert self._sy + self._height <= obs_shape[height_axis], (
            "Crop is out of boundary along the vertical direction")
        assert self._sx + self._width <= obs_shape[height_axis + 1], (
            "Crop is out of boundary along the horizontal direction")
        return gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def transform_observation(self, observation):
        if self._channel_order == "channels_last":
            obs = observation[self._sy:self._sy +
                              self._height, self._sx:self._sx + self._width]
        else:
            obs = observation[:, self._sy:self._sy +
                              self._height, self._sx:self._sx + self._width]

        return obs


@alf.configurable
class FrameGrayScale(BaseObservationWrapper):
    """Gray scale image observation"""

    def __init__(self, env, fields=None):
        """Create a FrameGrayScale instance

        Args:
             env (gym.Env): the gym environment
             fields (list[str]):  fields to be gray scaled, A field str is a multi-level
                path denoted by "A.B.C". If None, then non-nested observation is gray scaled
        """
        super().__init__(env, fields=fields)

    def transform_space(self, observation_space):
        obs_shape = observation_space.shape
        assert len(obs_shape) == 3 and obs_shape[-1] == 3, \
            "observation shape should be (H, W, C) where C=3"
        return gym.spaces.Box(
            low=0, high=255, shape=list(obs_shape[:-1]) + [1], dtype=np.uint8)

    def transform_observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(obs, -1)


@alf.configurable
class DMAtariPreprocessing(gym.Wrapper):
    """
    Derived from tf_agents AtariPreprocessing. Three differences:
    1. Random number of NOOPs after reset
    2. FIRE after a reset or a lost life. This is for the purpose of evaluation
       with greedy prediction without getting stuck in the early training
       stage.
    3. A lost life doesn't result in a terminal state

    NOTE: Some implementations forces the time step that loses a life to have a
    zero value (i.e., mark a 'terminal' state) to help bootstrap value functions,
    but *only resetting the env when all lives are used (`done==True`)*. In this
    case, the episodic score is still summed over all lives.

    For our implementation, we only mark a terminal state when all lives are
    used (`done==True`). It's more difficult to learn in our case (time horizon
    is longer).

    To see a complete list of atari wrappers used by DeepMind, see
    https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/atari_wrappers.py
    Also see OpenAI Gym's implementation (not completely the same):
    https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py

    (This wrapper does not handle framestacking. It can be paired with
    FrameStack. See atari.gin for an example.)
    """

    def __init__(self,
                 env,
                 frame_skip=4,
                 noop_max=30,
                 screen_size=84,
                 gray_scale=True):
        """Constructor for an Atari 2600 preprocessor.

        Args:
            env (gym.Env): the environment whose observations are preprocessed.
            frame_skip (int): the frequency at which the agent experiences the game.
            noop_max (int): the maximum number of no-op actions after resetting the env
            screen_size (int): size of a resized Atari 2600 frame.
            gray_scale (bool):
        """
        super().__init__(env)
        if frame_skip <= 0:
            raise ValueError('Frame skip should be strictly positive, got {}'.
                             format(frame_skip))
        if screen_size <= 0:
            raise ValueError(
                'Target screen size should be strictly positive, got {}'.
                format(screen_size))

        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.noop_max = noop_max
        self.gray_scale = gray_scale
        num_channels = 1 if gray_scale else 3
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        obs_dims = self.env.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        if gray_scale:
            self.screen_buffer = [
                np.empty((obs_dims.shape[0], obs_dims.shape[1]),
                         dtype=np.uint8),
                np.empty((obs_dims.shape[0], obs_dims.shape[1]),
                         dtype=np.uint8)
            ]
        else:
            self.screen_buffer = [
                np.empty((obs_dims.shape[0], obs_dims.shape[1], 3),
                         dtype=np.uint8),
                np.empty((obs_dims.shape[0], obs_dims.shape[1], 3),
                         dtype=np.uint8)
            ]

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_size, self.screen_size, num_channels),
            dtype=np.uint8)

        self._lives = 0

    def _reset_with_random_noops(self):
        self.env.reset()
        self._lives = self.env.ale.lives()
        if self.noop_max > 0:
            n_noops = self.env.unwrapped.np_random.randint(
                1, self.noop_max + 1)
            for _ in range(n_noops):
                _, _, game_over, _ = self.env.step(0)
                if game_over:
                    self.env.reset()

    def fire(self):
        # The following code is from https://github.com/openai/gym/...
        # ...blob/master/gym/wrappers/atari_preprocessing.py
        action_meanings = self.env.unwrapped.get_action_meanings()
        if action_meanings[1] == 'FIRE' and len(action_meanings) >= 3:
            self.env.step(1)
            self.env.step(2)

    def _start_new_life(self):
        self.fire()
        # in either case, we need to clear the screen buffer
        self._fetch_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize()

    def reset(self):
        """
        Resets the environment.
        Returns:
            observation (np.array): the initial observation emitted by the
                environment.
        """
        self._reset_with_random_noops()
        return self._start_new_life()

    def step(self, action):
        """Applies the given action in the environment.

        Remarks:

        * If a terminal state (episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
        * Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.

        Args:
            action (int): The action to be executed.

        Returns:
            observation (np.array): the observation following the action.
            reward (float): the reward following the action.
            game_over (bool): whether the environment has reached a terminal state.
                This is true when an episode is over.
            info: Gym API's info data structure.
        """
        accumulated_reward = 0.
        life_lost = False

        info = {}
        num_env_frames = 0
        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, reward, game_over, info = self.env.step(action)
            life_lost = self.env.ale.lives() < self._lives
            self._lives = self.env.ale.lives()
            accumulated_reward += reward
            if 'num_env_frames' in info:
                # in case FrameSkip wrapper is being nested:
                n_steps = info['num_env_frames']
            else:
                n_steps = 1
            num_env_frames += n_steps
            if game_over or life_lost:
                break
            # We max-pool over the last two frames, in grayscale.
            elif time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                # when frame_skip==1, self.screen_buffer[1] will be filled!
                self._fetch_observation(self.screen_buffer[t])
        info['num_env_frames'] = num_env_frames

        if self.frame_skip == 1:
            self.screen_buffer[0] = self.screen_buffer[1]

        # Pool the last two observations.
        if life_lost:
            observation = self._start_new_life()
        else:
            observation = self._pool_and_resize()

        return observation, accumulated_reward, game_over, info

    def _fetch_observation(self, output):
        """Returns the current observation in grayscale or RGB.

        The returned observation is stored in 'output'.

        Args:
            output (np.array): screen buffer to hold the returned observation.

        Returns:
            observation (np.array): the current observation in grayscale or RGB
        """
        if self.gray_scale:
            self.env.ale.getScreenGrayscale(output)
        else:
            self.env.ale.getScreenRGB2(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

        For efficiency, the transformation is done in-place in self.screen_buffer.

        Returns:
            transformed_screen (np.array): pooled, resized screen.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(
                self.screen_buffer[0],
                self.screen_buffer[1],
                out=self.screen_buffer[0])

        transformed_image = cv2.resize(
            self.screen_buffer[0], (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        if self.gray_scale:
            return np.expand_dims(int_image, axis=2)
        else:
            return int_image


def _gym_space_to_nested_space(space):
    """Change gym Space to a nest which can be handled by alf.nest functions."""

    if isinstance(space, gym.spaces.Dict):
        return dict((k, _gym_space_to_nested_space(s))
                    for k, s in space.spaces.items())
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(_gym_space_to_nested_space(s) for s in space.spaces)
    else:
        return space


def _nested_space_to_gym_space(space):
    """Change nested space to gym Space"""

    if isinstance(space, (dict, OrderedDict)):
        spaces = dict(
            (k, _nested_space_to_gym_space(s)) for k, s in space.items())
        return gym.spaces.Dict(spaces)
    elif isinstance(space, tuple):
        spaces = tuple(_nested_space_to_gym_space(s) for s in space)
        return gym.spaces.Tuple(spaces)
    else:
        return space


@alf.configurable
class ContinuousActionClip(gym.ActionWrapper):
    """Clip continuous actions according to the action space.

    Note that any action outside of the bounds specified by action_space will be
    clipped to the bounds before passing to the underlying environment.
    """

    def __init__(self, env, min_v=-1.e9, max_v=1.e9):
        """Create an ContinuousActionClip gym wrapper.

        Args:
            env (gym.Env): A Gym env instance to wrap
        """
        super(ContinuousActionClip, self).__init__(env)

        def _space_bounds(space):
            if isinstance(space, gym.spaces.Box):
                return np.maximum(space.low, min_v), np.minimum(
                    space.high, max_v)
            else:
                return min_v, max_v

        self._nested_action_space = _gym_space_to_nested_space(
            self.action_space)
        self.bounds = alf.nest.map_structure(_space_bounds,
                                             self._nested_action_space)

    def action(self, action):
        def _clip_action(space, action, bounds):
            # Check if the action is corrupted or not.
            if np.any(np.isnan(action)):
                raise ValueError(
                    "NAN action detected! action: {}".format(action))
            if isinstance(space, gym.spaces.Box):
                action = np.clip(action, bounds[0], bounds[1])
            return action

        action = alf.nest.map_structure_up_to(action, _clip_action,
                                              self._nested_action_space,
                                              action, self.bounds)
        return action


@alf.configurable
class ContinuousActionMapping(gym.ActionWrapper):
    """Map continuous actions to a desired action space, while keeping discrete
    actions unchanged."""

    def __init__(self, env, low, high):
        """
        Args:
            env (gym.Env): Gym env to be wrapped
            low (float): the action lower bound to map to.
            high (float): the action higher bound to map to.
        """
        super(ContinuousActionMapping, self).__init__(env)

        def _space_bounds(space):
            if isinstance(space, gym.spaces.Box):
                assert np.all(np.isfinite(space.low))
                assert np.all(np.isfinite(space.high))
                return (space.low, space.high)

        nested_action_space = _gym_space_to_nested_space(self.action_space)
        self._bounds = alf.nest.map_structure(_space_bounds,
                                              nested_action_space)
        self._nested_action_space = alf.nest.map_structure(
            lambda space: (gym.spaces.Box(
                low=low, high=high, shape=space.shape, dtype=space.dtype)
                           if isinstance(space, gym.spaces.Box) else space),
            nested_action_space)
        self.action_space = _nested_space_to_gym_space(
            self._nested_action_space)

    def action(self, action):
        def _scale_back(a, b, space):
            if isinstance(space, gym.spaces.Box):
                # a and b should be mutually broadcastable
                b0, b1 = b
                a0, a1 = space.low, space.high
                return (a - a0) / (a1 - a0) * (b1 - b0) + b0
            return a

        # map action back to its original space
        action = alf.nest.map_structure_up_to(action, _scale_back, action,
                                              self._bounds,
                                              self._nested_action_space)
        return action


@alf.configurable
class NormalizedAction(ContinuousActionMapping):
    """Normalize actions to ``[-1, 1]``. This normalized action space is
    friendly to algorithms that computes action entropy, e.g., SAC."""

    def __init__(self, env):
        super().__init__(env, low=-1., high=1.)


@alf.configurable
class NonEpisodicEnv(gym.Wrapper):
    """Make a gym environment non-episodic by always setting ``done=False``."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return ob, reward, False, info
