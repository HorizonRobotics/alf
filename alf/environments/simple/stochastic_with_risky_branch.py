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

S0 = 0
S1 = 1
S2 = 2
S3 = 3
T = 4

A0 = 0
A1 = 1


def _k(s, a):
    return s * 10 + a


class StochasticWithRiskyBranch(gym.Env):
    """
    A simple stochastic MDP
    s0 -> a0 - 50%  -> s1 -> a0 -> T (reward=2)
     |     |-- 50% -> s3 -> a1 -> T (reward=1)
     |--> a1 - 100% -> s2 -> a0 -> T (reward=1.8)
    All other actions terminates with reward 0.  T is the terminal state.

    Optimal action at s0 is a1 with q(s0, a1) = 1.5.  Optimal q(s0, a0) = 1.8.

    However, if it so happens that Q learning is conditioned on the action sequence, then
    q(s0, a0, a0) will contain mostly experience of <s0, a0, s1, a0, s0> and very few <s0, a0, s3, a0, s0>,
    leading to an average of about 2.
    q(s0, a0, a1) will be around 1.
    q(s0, a1, a0) will be around 1.8.
    The agent could end up choosing a0 at s0.
    """

    def __init__(self, seed=None):
        """
        Args:
            seed (int): random seed for the environment.
        """
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(1, ), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._deterministic_transitions = {
            _k(S0, A1): (S2, 0.),
            _k(S1, A0): (T, 2.),
            _k(S3, A1): (T, 1),
            _k(S2, A0): (T, 1.8)
        }
        if seed is not None:
            np.random.seed(seed)
        self.reset()

    def reset(self):
        # reset to leftmost
        self._s = S0
        self._action = None
        self._game_over = False
        self._obs = float(self._s)
        return np.array([self._obs])

    def step(self, action):
        self._action = action
        key = _k(self._obs, action)
        if key in self._deterministic_transitions:
            self._obs, r = self._deterministic_transitions[key]
        elif key == _k(S0, A0):
            # Stochastic transition:
            self._obs, r = (S1 if np.random.rand() > 0.5 else S3, 0)
        else:
            self._obs, r = (T, 0)

        self._game_over = self._obs == T
        return np.array([self._obs]), r, self._game_over, {}

    def render(self, mode="human", close=False):
        pass
