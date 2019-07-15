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

import gin
import gym
import numpy as np
import cv2


@gin.configurable
class FrameStack(gym.Wrapper):
    """Stack previous `stack_size` frames, applied to Gym env."""

    def __init__(self, env, stack_size=4, channel_order='channels_last'):
        """Create a FrameStack object.

        Args:
            stack_size (int):
            channel_order (str): one of `channels_last` or `channels_first`.
                The ordering of the dimensions in the images.
                `channels_last` corresponds to images with shape
                `(height, width, channels)` (Atari's default) while
                `channels_first` corresponds to images with shape
                `(channels, height, width)`.
        """
        super().__init__(env)
        self._frames = collections.deque(maxlen=stack_size)
        self._channel_order = channel_order
        self._stack_size = stack_size

        space = self.env.observation_space
        assert channel_order in ['channels_last', 'channels_first']

        if channel_order == 'channels_last':
            shape = list(space.shape[0:-1]) + [stack_size * space.shape[-1]]
        else:
            shape = [
                stack_size * space.shape[0],
            ] + list(space.shape[1:])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8)

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self.env, name)

    def _generate_observation(self):
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
class DMAtariPreprocessing(gym.Wrapper):
    """
    Derived from tf_agents AtariPreprocessing. Three differences:
    1. Random number of NOOPs after reset
    2. FIRE after a reset or a lost life. This is for the purpose of evaluation
       with greedy prediction without getting stucked in the early training
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
    FrameStack. See ac_atari.gin for an example.)
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
