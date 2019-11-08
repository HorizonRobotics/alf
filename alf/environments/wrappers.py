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

import collections
from collections import OrderedDict, deque
from copy import deepcopy

import gin
import gym
import numpy as np
import cv2

from absl import logging

from tf_agents.environments import wrappers
from tf_agents.trajectories.time_step import StepType


@gin.configurable
class FrameStack(gym.Wrapper):
    """Stack previous `stack_size` frames, applied to Gym env."""

    def __init__(
            self,
            env,
            stack_size=4,
            channel_order='channels_last',
            fields_to_stack=None,
    ):
        """Create a FrameStack object.

        Args:
            env (gym.Space): gym environment.
            stack_size (int):
            channel_order (str): one of `channels_last` or `channels_first`.
                The ordering of the dimensions in the images.
                `channels_last` corresponds to images with shape
                `(height, width, channels)` (Atari's default) while
                `channels_first` corresponds to images with shape
                `(channels, height, width)`.
            fields_to_stack (list of str): optional paths to the fields of the
                Dict env.observation_space.  If specified, only use the
                spaces corresponding to the keys in FrameStack.  If is None,
                stack all recognized spaces.
                When input space is not a Dict, we just try to stack if we can.
                Fields_to_stack doesn't apply anymore.
        """
        super().__init__(env)
        self._frames = deque(maxlen=stack_size)
        self._channel_order = channel_order
        self._stack_size = stack_size
        self._fields_to_stack = fields_to_stack
        raw_space = self.env.observation_space

        def _stack_space(sp):
            if isinstance(sp, gym.spaces.Box):
                # Shape of stacked_space is determined by low.shape
                low = np.concatenate([sp.low] * stack_size)
                high = np.concatenate([sp.high] * stack_size)
                assert channel_order in ['channels_last', 'channels_first']
                if channel_order == 'channels_last':
                    low = np.transpose(
                        np.concatenate([np.transpose(sp.low)] * stack_size))
                    high = np.transpose(
                        np.concatenate([np.transpose(sp.high)] * stack_size))
                stacked_space = gym.spaces.Box(
                    low=np.array(low), high=np.array(high), dtype=sp.dtype)

            elif isinstance(sp, gym.spaces.MultiDiscrete):
                nvec = [sp.nvec] * stack_size
                stacked_space = gym.spaces.MultiDiscrete(nvec)

            else:
                stacked_space = sp
            return stacked_space

        def _traverse(d, fields_to_stack=None, prefix=""):
            assert isinstance(d, gym.spaces.Dict), 'input is not dict'
            res = deepcopy(d)
            for name, sp in d.spaces.items():
                if prefix:
                    prefix += "."
                path = prefix + name
                transformed = sp
                if isinstance(sp, gym.spaces.Dict):
                    transformed, fields_to_stack = _traverse(
                        sp, fields_to_stack, path)
                else:
                    if fields_to_stack is None or path in fields_to_stack:
                        transformed = _stack_space(sp)
                        if fields_to_stack:
                            fields_to_stack = [
                                item for item in fields_to_stack
                                if item != path
                            ]
                res.spaces[name] = transformed
            return res, fields_to_stack

        if isinstance(raw_space, gym.spaces.Dict):
            self.observation_space, remain_fields = _traverse(
                raw_space, fields_to_stack)
            assert not remain_fields, "These paths are not in input: " + str(
                remain_fields) + ", but in " + str(fields_to_stack)
        else:
            assert isinstance(raw_space, gym.spaces.Box)
            self.observation_space = _stack_space(raw_space)

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self.env, name)

    def _get_space_for_path(self, path):
        segments = path.split(".")
        space = self.observation_space
        if segments:
            for name in segments:
                space = space.spaces[name]
        return space

    def _generate_observation(self):
        def _traverse_append(result, data, path_prefix=""):
            assert isinstance(result, OrderedDict)
            if path_prefix:
                path_prefix += "."
            for key, field in data.items():
                path = path_prefix + key
                if isinstance(field, dict):
                    if key not in result:
                        result[key] = OrderedDict()
                    result[key] = _traverse_append(result[key], field, path)
                else:
                    if key not in result:
                        result[key] = deque(maxlen=self._stack_size)
                    result[key].append(field)
            return result

        def _traverse_stack(result, path_prefix=""):
            if not path_prefix or isinstance(result, OrderedDict):
                for key, field in result.items():
                    if path_prefix:
                        path_prefix += "."
                    path = path_prefix + key
                    result[key] = _traverse_stack(field, path)
            else:
                if not self._fields_to_stack or (
                        path_prefix in self._fields_to_stack):
                    space = self._get_space_for_path(path_prefix)
                    if isinstance(space, gym.spaces.Box):
                        if (self._channel_order == 'channels_last'):
                            result = np.concatenate(result, axis=-1)
                        else:
                            result = np.concatenate(result, axis=0)
                    elif isinstance(space, gym.spaces.MultiDiscrete):
                        result = np.concatenate(result, axis=-1)
                    else:
                        assert False, ("space with path {} not recognized: " +
                                       "{}").format(path_prefix, str(space))
                else:  # not stacking field
                    result = result[-1]  # take last/current frame
            return result

        if isinstance(self.observation_space, gym.spaces.Dict):
            result = OrderedDict()
            for frame in self._frames:
                result = _traverse_append(result, frame)
            result = _traverse_stack(result)
            return result
        else:  # Assuming input is image space
            # Always stacks regardless of _fields_to_stack
            if self._channel_order == 'channels_last':
                return np.concatenate(self._frames, axis=-1)
            else:
                return np.concatenate(self._frames, axis=0)

    def reset(self):
        observation = self.env.reset()
        for _ in range(self._stack_size):
            self._frames.append(observation)
        return self._generate_observation()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._frames.append(observation)
        return self._generate_observation(), reward, done, info


@gin.configurable
class FrameSkip(gym.Wrapper):
    """
    Repeat same action n times and return the last observation
     and accumulated reward
    """

    def __init__(self, env, skip):
        """Create a FrameSkip object

        Args:
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
            if done: break
        return obs, accumulated_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self.env, name)


@gin.configurable
class FrameResize(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 3, "observation shape should be (H,W,C)"
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=[width, height] + list(obs_shape[2:]),
            dtype=np.uint8)

    def observation(self, observation):
        obs = cv2.resize(
            observation, (self._width, self._height),
            interpolation=cv2.INTER_AREA)
        if len(obs.shape) != 3:
            obs = obs[:, :, np.newaxis]
        return obs


@gin.configurable
class FrameGrayScale(gym.ObservationWrapper):
    """Gray scale image observation"""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 3 and obs_shape[-1] == 3, \
            "observation shape should be (H, W, C) where C=3"
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=list(obs_shape[:-1]) + [1], dtype=np.uint8)

    def observation(self, obs):
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
        n_noops = self.env.unwrapped.np_random.randint(1, self.noop_max + 1)
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


@gin.configurable
class NonEpisodicAgent(wrappers.PyEnvironmentBaseWrapper):
    """
    Make the agent non-episodic by replacing all termination time steps with
    a non-zero discount (essentially the same type as returned by the TimeLimit
    wrapper).

    This wrapper could be useful for pure intrinsic-motivated agent, as
    suggested in the following paper:

        EXPLORATION BY RANDOM NETWORK DISTILLATION, Burda et al. 2019,

    "... We argue that this is a natural way to do exploration in simulated
    environments, since the agent’s intrinsic return should be related to all
    the novel states that it could find in the future, regardless of whether
    they all occur in one episode or are spread over several.

    ... If Alice is modelled as an episodic reinforcement learning agent, then
    her future return will be exactly zero if she gets a game over, which might
    make her overly risk averse. The real cost of a game over to Alice is the
    opportunity cost incurred by having to play through the game from the
    beginning."

    NOTE: For PURE intrinsic-motivated agents only. If you use both extrinsic
    and intrinsic rewards, then DO NOT use this wrapper! Because without
    episodic setting, the agent could exploit extrinsic rewards by intentionally
    die to get easy early rewards in the game.

    Example usage:
        suite_mario.load.env_wrappers=(@NonEpisodicAgent, )
        suite_gym.load.env_wrappers=(@NonEpisodicAgent, )
    """

    def __init__(self, env, discount=1.0):
        super().__init__(env)
        self._discount = discount

    def _step(self, action):
        time_step = self._env.step(action)
        if time_step.step_type == StepType.LAST:
            # We set a non-zero discount so that the target value would not be
            # zero (non-episodic).
            time_step = time_step._replace(
                discount=np.asarray(self._discount, np.float32))
        return time_step
