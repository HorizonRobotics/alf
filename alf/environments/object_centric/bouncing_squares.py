# Copyright (c) 2022 Horizon Robotics. All Rights Reserved.
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


class BouncingSquares(gym.Env):
    """An environment which contains two bouncing squares. Either square bounces
    when it hits the image border or the other square.
    The environment has a dummy action NOOP which doesn't affect either square.
    """

    def __init__(self,
                 N: int,
                 pixels_per_node: int = 1,
                 noise_level: float = 0.,
                 render_size: int = 640,
                 color: bool = False):
        """
        Args:
            N: the length of the map
            pixels_per_node: when generating an image input, how many pixels are
                drawn at each location.
            noise_level: If >0, the generated images will be added a Gaussian noise
                whose std is ``noise_level``.
            color: whether the squares are colorful or grayscale.
        """
        super().__init__()

        size = N * pixels_per_node
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 3 if color else 1),
            dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self._N = N
        self._colors = ([(0, 255, 0), (0, 0, 255), (120, 120, 120),
                         (255, 255, 0), (0, 255, 255), (255, 0, 255),
                         (120, 0, 120), (0, 120, 120)] if color else None)
        self._pixels_per_node = pixels_per_node
        self._noise_level = noise_level
        self._render_size = render_size
        self.reset()
        self.metadata.update({'render.modes': ["rgb_array"]})

    def _get_square_boundary(self, loc, size):
        """inclusive"""
        return (loc[0], loc[0] + size[0] - 1), (loc[1], loc[1] + size[1] - 1)

    def reset(self):
        def _initialize_loc(size):
            return np.array((np.random.randint(self._N - size[0]),
                             np.random.randint(self._N - size[1])))

        def _initialize_speed():
            return np.array((np.random.randint(-3, 4), np.random.randint(
                -3, 4)))

        def _initialize_size():
            sizes = [4, 5]
            return np.array((self._N // np.random.choice(sizes),
                             self._N // np.random.choice(sizes)))

        self._size1 = _initialize_size()
        self._size2 = _initialize_size()

        self._square1 = _initialize_loc(self._size1)

        while True:
            self._square2 = _initialize_loc(self._size2)
            if not self._collision():
                break

        if self._colors:
            self._color1 = self._colors[np.random.randint(len(self._colors))]
            self._color2 = self._colors[np.random.randint(len(self._colors))]
        else:
            self._color1 = self._color2 = None

        self._speed1 = _initialize_speed()
        self._speed2 = _initialize_speed()

        return self._obs()

    def _obs(self):
        img = np.zeros((self._N, self._N, 3 if self._colors else 1),
                       dtype=np.uint8)

        def _paint_square(sq, img, size, color):
            (i0, i1), (j0, j1) = self._get_square_boundary(sq, size)
            patch = img[i0:i1 + 1, j0:j1 + 1]
            patch[:, :] = color
            if self._noise_level > 0:
                noise = (np.clip(np.random.randn(*patch.shape), 0, None) *
                         self._noise_level * 255)
                patch += noise.astype(np.uint8)

        _paint_square(self._square1, img, self._size1, self._color1 or 255)
        _paint_square(self._square2, img, self._size2, self._color2 or 255)

        img = cv2.resize(
            img,
            dsize=(self._N * self._pixels_per_node, ) * 2,
            interpolation=cv2.INTER_NEAREST)
        return img

    def _collision(self):
        (i0, i1), (j0, j1) = self._get_square_boundary(self._square1,
                                                       self._size1)
        (y0, y1), (x0, x1) = self._get_square_boundary(self._square2,
                                                       self._size2)
        if (i0 > y1 or y0 > i1) or (j0 > x1 or x0 > j1):
            return False
        return True

    def step(self, action):
        def _boundary_collision(sq, sp, s, osq):
            if sq[0] < 0 or sq[0] + s[0] - 1 >= self._N:
                return osq, sp * np.array([-1, 1])
            if sq[1] < 0 or sq[1] + s[1] - 1 >= self._N:
                return osq, sp * np.array([1, -1])
            return sq, sp

        # assuming sq1 and sq2 are 'safe' locations
        sq1, sq2 = self._square1, self._square2

        self._square1, self._speed1 = _boundary_collision(
            self._square1 + self._speed1, self._speed1, self._size1, sq1)
        self._square2, self._speed2 = _boundary_collision(
            self._square2 + self._speed2, self._speed2, self._size2, sq2)

        if self._collision():
            self._speed1, self._speed2 = self._speed2, self._speed1
            self._square1 = sq1
            self._square2 = sq2

        return self._obs(), 0., False, {}

    def render(self, mode="human"):
        obs = self._obs()
        obs = cv2.resize(
            obs,
            dsize=(self._render_size, self._render_size),
            interpolation=cv2.INTER_NEAREST)
        if mode == "rgb_array":
            return obs
        else:
            cv2.imshow("BouncingSquares", obs)
            cv2.waitKey(500)
