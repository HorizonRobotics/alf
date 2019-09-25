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

import gym
from gym import spaces
import numpy as np

import cv2


class NoisyArray(gym.Env):
    """
    A synthetic noisy array to test the agent's robustness to random noises. The
    binary array has a length of (K+M), where the subarray of length K is a
    onehot vector with 1 representing the agent's current location, and the
    remaining M bits constitute a noise vector in {0,1}^M. For example (K=5,
    M=3):

        0 0 1 0 0 | 0 1 1

    and the agent is at i==2 now.

    The agent always starts from i==0. The goal is to reach i==K-1 (it cannot
    step on the noise vector). It has three actions: LEFT, RIGHT, and FIRE. The
    FIRE action changes the noise vector into some random M bits, without
    changing the agent's position. Both LEFT and RIGHT won't change the noise
    vector.

    In the example above, if the next action is FIRE, then the resulting array
    might be

        0 0 1 0 0 | 1 1 0

    If the next action is RIGHT, then the resulting array should be:

        0 0 0 1 0 | 0 1 1

    The game ends whether the array looks like

        0 0 0 0 1 | X X X
    """
    LEFT = 0
    FIRE = 1
    RIGHT = 2

    def __init__(self, K=11, M=100, auto_noise=False):
        """
        Args:
            K (int): K-1 will be the minimum steps that take the agent from left
                to right and get a reward of 1
            M (int): the length of the noisy vector. The total observation length
                would be K+M
            auto_noise (bool): if True, the noise vector will change automatically
                at every step, and FIRE becomes "no-operation".
        """
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(K + M, ), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._K = K
        self._M = M
        self._auto_noise = auto_noise
        self.reset()

    def reset(self):
        # reset to leftmost
        self._position = 0
        self._action = None
        self._noise_vector = np.random.randint(2, size=self._M)
        self._game_over = False
        self._obs = self._gen_observation(act=self.FIRE)[0]
        return self._obs

    def step(self, action):
        # auto_reset will be called by wrappers
        self._action = action
        self._obs, r = self._gen_observation(act=self._action)
        return self._obs, r, self._game_over, {}

    def render(self, mode="human", close=False):
        # first convert obs to an RGB array
        obs = np.copy(1 - self._obs)
        obs[self._K:] *= 0.5  # turn the noise portion to gray
        obs *= 255
        obs = obs.astype("uint8")

        grid_size = 16
        length = obs.shape[0]
        rgb_array = np.expand_dims(obs, axis=0)
        rgb_array = cv2.resize(
            rgb_array,
            dsize=(length * grid_size, grid_size),
            interpolation=cv2.INTER_NEAREST)
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2RGB)

        if mode == "rgb_array":
            return rgb_array
        else:
            if self._action is not None:
                print("action: ", self._action)
            cv2.imshow("NoisyArray", rgb_array)
            cv2.waitKey(100)

    def _gen_observation(self, act):
        movement = act - 1
        # If the current position is beyond the right boundary, put the agent
        # back to the left
        self._position = max(self._position + movement, 0)
        self._position %= self._K

        self._game_over = (self._position == self._K - 1)

        reward = 1 if self._game_over else 0

        if act == self.FIRE or self._auto_noise:
            self._noise_vector = np.random.randint(2, size=self._M)

        position_array = np.zeros(self._K, dtype=np.float32)
        position_array[self._position] = 1

        observation = np.concatenate((position_array,
                                      self._noise_vector.astype(np.float32)))
        return observation, reward
