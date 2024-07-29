# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import numpy as np
import re

import alf
import alf.environments.gym_wrappers
from alf.environments import alf_wrappers
from .suite_gym import wrap_env

try:
    import babyai
except ImportError:
    babyai = None


def is_available():
    return babyai is not None


@alf.configurable
def load(environment_name,
         env_id=None,
         max_instruction_length=80,
         mode='sent',
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         alf_env_wrappers=()):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name (str): Name for the environment to load.
        env_id (int): (optional) ID of the environment.
        max_instruction_length (int): the maximum number of words of an instruction.
        mode (str): one of ('sent', 'word', 'char'). If 'sent', the whole instruction
            (word ID array) is given in the observation at every step. If 'word',
            the word IDs are given in the observation sequentially. Each step
            only one word ID is given. A zero is given for every steps after all
            the word IDs are given. If 'char', similar to 'word', but only one
            character is given at each step.
        discount (float): Discount to use for the environment.
        max_episode_steps (int): If None the max_episode_steps will be set to the
            default step limit defined in the environment's spec. No limit is applied
            if set to 0 or if there is no max_episode_steps set in the environment's
            spec.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers
            classes to use directly on the gym environment.
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment.

    Returns:
        An AlfEnvironment instance.
    """
    # babyai doesn't register max_episode_steps in Gym
    gym_env = gym.make(environment_name)
    # but it does have an internal property ``max_steps`` which returns ``done=True```
    # when reached, see
    # https://github.com/maximecb/gym-minigrid/blob/6f5fe8588d05eb13a08f971fd3c7a82c404dc1bb/gym_minigrid/minigrid.py#L1158
    if max_episode_steps is None:
        if hasattr(gym_env, 'max_steps'):
            # minus 1 because we need to let ALF wrap ``TimeLimit`` before getting ``done=True```
            max_episode_steps = gym_env.max_steps - 1
        else:
            max_episode_steps = 0

    gym_env = BabyAIWrapper(gym_env, max_instruction_length, mode)

    return wrap_env(
        gym_env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        image_channel_first=False)


class BabyAIWrapper(gym.Wrapper):
    """A wrapper for BabyAI environment.

    BabyAI environment is introduced in
    `Chevalier-Boisver et al. Baby{AI}: First Steps Towards Grounded Language
    Learning With a Human In the Loop <https://openreview.net/pdf?id=rJeXCo0cYX>`_.

    It can be downloaded from https://github.com/mila-iqia/babyai
    """

    # From Figure 2 in the paper.
    # Note that "," is not treated as a word.
    VOCAB = [
        'then',
        'after',
        'you',
        'and',
        'go',
        'to',
        'pick',
        'up',
        'open',
        'put',
        'next',
        'door',
        'ball',
        'box',
        'key',
        'on',
        'your',
        'left',
        'right',
        'in',
        'front',
        'of',
        'you',
        'behind',
        'red',
        'green',
        'blue',
        'purple',
        'yellow',
        'grey',
        'the',
        'a',
    ]

    VOCAB_SIZE = len(VOCAB) + 1

    def __init__(self, env, max_instruction_length=80, mode='sent'):
        """
        Args:
            gym_env (gym.Env): An instance of OpenAI gym environment.
            max_instruction_length (int): the maximum number of words of an instruction.
            mode (str): one of ('sent', 'word', 'char'). If 'sent', the whole instruction
                (word ID array) is given in the observation at every step. If 'word',
                the word IDs are given in the observation sequentially. Each step
                only one word ID is given. A zero is given for every steps after all
                the word IDs are given. If 'char', similar to 'word', but only one
                character is given at each step. For 'char' mode, we assume that
                the unicode of each character is within [0, 127].
        """
        super().__init__(env)

        self._max_instruction_length = max_instruction_length
        assert mode in ('sent', 'word', 'char')
        self._mode = mode

        # the extra 1 is for padding
        vocab_size = len(self.VOCAB) + 1

        obs_space = {
            # 7x7x3 ego-centric observation, each location is represented by
            # 3 values: object type, color, state (open, closed, locked)
            'image':
                env.observation_space['image'],
            # the orientation of the agent
            'direction':
                gym.spaces.Discrete(4),
            # instruction
            'mission':
                gym.spaces.MultiDiscrete([vocab_size] * max_instruction_length)
        }
        if mode == 'word':
            obs_space['mission'] = gym.spaces.Discrete(vocab_size)
        elif mode == 'char':
            obs_space['mission'] = gym.spaces.Discrete(128)

        self.observation_space = gym.spaces.Dict(obs_space)

        self._vocab = {'': 0}
        for i, w in enumerate(self.VOCAB):
            self._vocab[w] = i + 1
        self._last_mission = ''
        self._tokens = []
        self._word_pattern = re.compile("([a-z]+)")

    def _tokenize(self, instruction):
        """Convert instruction string to a numpy array."""
        if self._mode == 'char':
            tokens = np.array([ord(c) for c in instruction])
            if np.amax(tokens) > 127:
                raise ValueError(
                    "Character out of range. The unicode of "
                    "character should be in [0, 127]: %s" % instruction)
            return tokens

        tokens = self._word_pattern.findall(instruction.lower())
        instr = np.array([self._vocab.get(token, 0) for token in tokens])
        if np.amin(instr) == 0:
            for token in tokens:
                if token not in self._vocab:
                    raise ValueError(
                        "The instruction '%s' contains word "
                        " out of vocabulary: %s" % (instruction, token))
        return instr

    def _vectorize(self, instruction):
        instr = self._tokenize(instruction)
        if len(instr) < self._max_instruction_length:
            instr = np.concatenate([
                instr,
                np.zeros([self._max_instruction_length - len(instr)],
                         dtype=np.int64)
            ])
        elif len(instr) > self._max_instruction_length:
            raise ValueError("The instruction is too long: %d" % len(instr))
        return instr

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._transform_observation(obs)
        info['success'] = 1.0 if reward > 0 else 0.
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_mission = ''
        self._tokens = []
        return self._transform_observation(obs)

    def _transform_observation(self, observation):
        # Note: The original BabyAI environment give the same instruction at every
        # steps of an episode.
        observation['direction'] = np.int64(observation['direction'])
        mission = observation['mission']
        if self._mode == 'sent':
            observation['mission'] = self._vectorize(mission)
        else:
            if mission != self._last_mission:
                if mission != '':
                    self._tokens.extend(self._tokenize(mission))
                self._last_mission = mission
            if len(self._tokens) > 0:
                observation['mission'] = self._tokens.pop(0)
            else:
                observation['mission'] = np.int64(0)

        return observation
