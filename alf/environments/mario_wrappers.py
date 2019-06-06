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


# See https://github.com/openai/large-scale-curiosity/blob/ \
#  0c3d179fd61ee46233199d0891c40fbe7964d3aa/wrappers.py#L155-L238

class MarioXReward(gym.Wrapper):
    """
    Use X-axis coordinate increment as reward
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.

    def reset(self):
        ob = self.env.reset()
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        levellow, levelhigh, xscrollHi, xscrollLo = \
            info["levelLo"], info["levelHi"], \
            info["xscrollHi"], info["xscrollLo"]
        currentx = xscrollHi * 256 + xscrollLo
        new_level = [levellow, levelhigh]
        if new_level != self.current_level:
            self.current_level = new_level
            self.current_max_x = 0.
            reward = 0.
            self.visited_levels.add(tuple(self.current_level))
        else:
            if currentx > self.current_max_x:
                delta = currentx - self.current_max_x
                self.current_max_x = currentx
                reward = delta
            else:
                reward = 0.
        if done:
            info["levels"] = copy(self.visited_levels)
            info["retro_episode"] = dict(levels=copy(self.visited_levels))

        return ob, reward, done, info


class LimitedDiscreteActions(gym.ActionWrapper):
    """
    Map button combinations to discrete actions
    """
    BUTTONS = {"A", "B"}
    SHOULDERS = {"L", "R"}

    def __init__(self, env, all_buttons):
        gym.ActionWrapper.__init__(self, env)
        # 'B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A'
        self._num_buttons = len(all_buttons)
        button_keys = {i for i, b in enumerate(all_buttons) if b in self.BUTTONS}
        buttons = [(), *zip(button_keys),
                   *itertools.combinations(button_keys, 2)]
        # 'UP', 'DOWN', 'LEFT', 'RIGHT'
        arrows = [(), (4,), (5,), (6,), (7,)]
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


class FrameSkip(gym.Wrapper):
    """
    Repeat same action n times and return the last observation
     and accumulated reward
    """

    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n

    def step(self, action):
        obs = None
        accumulated_reward = 0
        done = False
        info = None
        for _ in range(self.n):
            obs, reward, done, info = self.env.step(action)
            accumulated_reward += reward
            if done: break
        return obs, accumulated_reward, done, info


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
        resized_screen = np.array(Image.fromarray(img).resize(
            size, resample=Image.BILINEAR), dtype=np.uint8)
        x_t = resized_screen[18:102, :] if crop else resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


#   See https://github.com/openai/baselines/blob/
#   9b68103b737ac46bc201dfb3121cfa5df2127e53/
#   baselines/common/atari_wrappers.py#L188-L257


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between
        the observations are only stored once.
        It exists purely to optimize memory usage which can be
        huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array
        before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)
        # Returns lazy array, which is much more memory efficient.
        # return LazyFrames(list(self.frames))


class FrameFormat(gym.Wrapper):
    """
    Format frame to specified data_format
    `channels_first` for CHW and `channels_last` for HWC
    """

    def __init__(self, env, data_format='channels_last'):
        gym.Wrapper.__init__(self, env)
        data_format = data_format.lower()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError(
                'The `data_format` argument must be one of '
                '"channels_first", "channels_last". Received: ' +
                str(data_format))
        self._transpose = False
        obs_shape = env.observation_space.shape
        if data_format == 'channels_first':
            self._transpose = True
            obs_shape = (obs_shape[-1],) + (obs_shape[:-1])
        self.observation_space = spaces.Box(
            low=0, high=255,
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
