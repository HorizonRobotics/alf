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

from collections import deque
import itertools
from copy import copy
import numpy as np
from PIL import Image
import gym
from gym import spaces

# See https://github.com/openai/large-scale-curiosity/blob/0c3d179fd61ee46233199d0891c40fbe7964d3aa/wrappers.py#L155-L238


class MarioXReward(gym.Wrapper):
    """Wrap mario environment and use X-axis coordinate increment as reward.

    .. code-block::

        if initial or upgrade_to_new_level
            reward, max_x = 0, 0
        else:
            current_x = xscrollHi * 256 + xscrollLo
            reward = current_x - max_x if current_x > max_x else 0
            max_x = current_x if current_x > max_x else max_x
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.current_level = (0, 0)
        self.current_max_x = 0.

    def reset(self):
        ob = self.env.reset()
        self.current_level = (0, 0)
        self.current_max_x = 0.
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        levellow, levelhigh, xscrollHi, xscrollLo = \
            info["levelLo"], info["levelHi"], \
            info["xscrollHi"], info["xscrollLo"]
        new_level = (levellow, levelhigh)
        if new_level != self.current_level:
            self.current_level = new_level
            self.current_max_x = 0.
            reward = 0.
        else:
            currentx = xscrollHi * 256 + xscrollLo
            if currentx > self.current_max_x:
                reward = currentx - self.current_max_x
                self.current_max_x = currentx
            else:
                reward = 0.

        return ob, reward, done, info


class LimitedDiscreteActions(gym.ActionWrapper):
    """
    Wrap mario environment and make it use discrete actions.
    Map available button combinations to discrete actions
    eg:
       0 -> None
       1 -> UP
       2 -> DOWN
       ...
       k -> A
       ...
       m -> A + LEFT
       ...
       n -> B + UP
       ...
    """

    BUTTONS = {"A", "B"}
    SHOULDERS = {"L", "R"}

    def __init__(self, env, all_buttons):
        gym.ActionWrapper.__init__(self, env)
        # 'B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A'
        self._num_buttons = len(all_buttons)
        button_keys = {
            i
            for i, b in enumerate(all_buttons) if b in self.BUTTONS
        }
        buttons = [(), *zip(button_keys),
                   *itertools.combinations(button_keys, 2)]
        # 'UP', 'DOWN', 'LEFT', 'RIGHT'
        arrows = [(), (4, ), (5, ), (6, ), (7, )]
        acts = []
        acts += arrows
        acts += buttons[1:]
        acts += [a + b for a in arrows[-2:] for b in buttons[1:]]
        self._actions = acts
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons)
        for i in self._actions[a]:
            mask[i] = 1
        return mask


class ProcessFrame84(gym.ObservationWrapper):
    """
    Resize frame from original resolution to 84x84 or
    resize to 84x110 and then crop to 84x84
    """

    def __init__(self, env, crop=True):
        self.crop = crop
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs, crop=self.crop)

    @staticmethod
    def process(frame, crop=True):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        elif frame.size == 224 * 240 * 3:  # mario resolution
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution." + str(frame.size)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        size = (84, 110 if crop else 84)
        resized_screen = np.array(
            Image.fromarray(img).resize(size, resample=Image.BILINEAR),
            dtype=np.uint8)
        x_t = resized_screen[18:102, :] if crop else resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class FrameFormat(gym.Wrapper):
    """
    Format frame to specified data_format

    Args:
       data_format: Data format for frame
          `channels_first` for CHW and `channels_last` for HWC
    """

    def __init__(self, env, data_format='channels_last'):
        gym.Wrapper.__init__(self, env)
        data_format = data_format.lower()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('The `data_format` argument must be one of '
                             '"channels_first", "channels_last". Received: ' +
                             str(data_format))
        self._transpose = False
        obs_shape = env.observation_space.shape
        if data_format == 'channels_first':
            self._transpose = True
            obs_shape = (obs_shape[-1], ) + (obs_shape[:-1])
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        return self._get_ob(ob)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = self._get_ob(ob)
        return ob, reward, done, info

    def _get_ob(self, ob):
        import numpy as np
        if self._transpose:
            return np.transpose(ob, (2, 0, 1))
        return ob
