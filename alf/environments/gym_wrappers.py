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

from collections import deque
import copy
import cv2
import gin
import gym
import numpy as np
import random

from alf.utils import common


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
            return func(space, field)

        assert isinstance(space, gym.spaces.Dict)
        level = levels[0]

        new_val = copy.deepcopy(space)
        new_val.spaces[level] = _traverse_transform(
            space=space.spaces[level], levels=levels[1:])
        return new_val

    return _traverse_transform(
        space=observation_space, levels=field.split('.') if field else [])


@gin.configurable
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

        self._fields = fields or [None]
        observation_space = env.observation_space
        for field in self._fields:
            observation_space = transform_space(
                observation_space=observation_space,
                field=field,
                func=self.transform_space)
        self.observation_space = observation_space

    def observation(self, observation):
        for field in self._fields:
            observation = common.transform_observation(
                observation=observation,
                field=field,
                func=self.transform_observation)
        return observation

    def transform_space(self, observation_space, field=None):
        """Transform space

        Subclass should implement this to perform transformation

        Args:
             observation_space (gym.Space): space to be transformed
             field (str): field to be transformed, it's a multi-level path denoted by "A.B.C"
        Returns:
            transformed space
        """
        raise NotImplementedError("transform_space is not implemented")

    def transform_observation(self, observation, field=None):
        """Transform observation

        Subclass should implement this to perform transformation

        Args:
             observation (ndarray): observation to be transformed
             field (str): field to be transformed, it's a multi-level path denoted by "A.B.C"
        Returns:
            transformed space
        """
        raise NotImplementedError("transform_observation is not implemented")


@gin.configurable
class ImageChannelFirst(BaseObservationWrapper):
    """Make images in observations channel_first. """

    def __init__(self, env, fields=None):
        super().__init__(env, fields=fields)

    def transform_space(self, observation_space, field=None):
        if isinstance(observation_space, gym.spaces.Box):
            if self._need_channel_transpose(observation_space.shape):
                low = observation_space.low
                high = observation_space.high
                if np.isscalar(low) and np.isscalar(high):
                    shape = observation_space.shape[::-1]
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

    def transform_observation(self, observation, field=None):
        transpose = self._need_channel_transpose(observation.shape)
        return self._make_channel_first(observation, transpose)

    def _need_channel_transpose(self, shape):
        if len(shape) == 3:
            return True
        return False

    def _make_channel_first(self, np_array, transpose=False):
        if transpose:
            rank = np_array.ndim
            np_array = np.transpose(np_array,
                                    (rank - 1, ) + tuple(range(rank - 1)))
        return np_array


@gin.configurable
class FrameStack(BaseObservationWrapper):
    """Stack previous `stack_size` frames, applied to Gym env."""

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

    def transform_space(self, observation_space, field=None):
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

    def transform_observation(self, observation, field=None):
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


@gin.configurable
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
        info = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            accumulated_reward += reward
            if done:
                break
        return obs, accumulated_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self.env, name)


@gin.configurable
class FrameResize(BaseObservationWrapper):
    def __init__(self, env, width=84, height=84, fields=None):
        """Create a FrameResize instance

        Args:
             env (gym.Env): the gym environment
             width (int): resize width
             height (int): resize height
             fields (list[str]):  fields to be resize, A field str is a multi-level
                path denoted by "A.B.C". If None, then non-nested observation is resized
        """
        self._width = width
        self._height = height
        super().__init__(env, fields=fields)

    def transform_space(self, observation_space, field=None):
        obs_shape = observation_space.shape
        assert len(obs_shape) == 3, "observation shape should be (H,W,C)"
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=[self._width, self._height] + list(obs_shape[2:]),
            dtype=np.uint8)

    def transform_observation(self, observation, field=None):
        obs = cv2.resize(
            observation, (self._width, self._height),
            interpolation=cv2.INTER_AREA)
        if len(obs.shape) != 3:
            obs = obs[:, :, np.newaxis]
        return obs


@gin.configurable
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

    def transform_space(self, observation_space, field=None):
        obs_shape = observation_space.shape
        assert len(obs_shape) == 3 and obs_shape[-1] == 3, \
            "observation shape should be (H, W, C) where C=3"
        return gym.spaces.Box(
            low=0, high=255, shape=list(obs_shape[:-1]) + [1], dtype=np.uint8)

    def transform_observation(self, obs, field=None):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(obs, -1)


@gin.configurable
class DMAtariPreprocessing(gym.Wrapper):
    """
    Derived from tf_agents AtariPreprocessing. Three differences:
    1. Random number of NOOPs after reset
    2. FIRE after a reset or a lost life. This is for the purpose of evaluation
       with greedy prediction without getting stuck in the early training
       stage.
    3. A lost life doesn't result in a terminal state

    NOTE: Some implementations forces the time step that loses a life to have a
    zero value (i.e., mark a 'terminal' state) to help boostrap value functions,
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

    def __init__(self, env, frame_skip=4, noop_max=30, screen_size=84):
        """Constructor for an Atari 2600 preprocessor.

        Args:
            env (gym.Env): the environment whose observations are preprocessed.
            frame_skip (int): the frequency at which the agent experiences the game.
            noop_max (int): the maximum number of no-op actions after resetting the env
            screen_size (int): size of a resized Atari 2600 frame.
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
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        obs_dims = self.env.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_size, self.screen_size, 1),
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
        self._fetch_grayscale_observation(self.screen_buffer[0])
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

        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, reward, game_over, info = self.env.step(action)
            life_lost = self.env.ale.lives() < self._lives
            self._lives = self.env.ale.lives()
            accumulated_reward += reward
            if game_over or life_lost:
                break
            # We max-pool over the last two frames, in grayscale.
            elif time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                # when frame_skip==1, self.screen_buffer[1] will be filled!
                self._fetch_grayscale_observation(self.screen_buffer[t])

        if self.frame_skip == 1:
            self.screen_buffer[0] = self.screen_buffer[1]

        # Pool the last two observations.
        if life_lost:
            observation = self._start_new_life()
        else:
            observation = self._pool_and_resize()

        return observation, accumulated_reward, game_over, info

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.

        The returned observation is stored in 'output'.

        Args:
            output (np.array): screen buffer to hold the returned observation.

        Returns:
            observation (np.array): the current observation in grayscale.
        """
        self.env.ale.getScreenGrayscale(output)
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
        return np.expand_dims(int_image, axis=2)
