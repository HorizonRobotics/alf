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
"""Runs a single environments in a separate thread. """

from multiprocessing import dummy as mp_threads
import numbers
import numpy as np
import torch

import alf
from alf.environments import alf_environment
import alf.nest as nest


def _array_to_tensor(data):
    def _array_to_tensor(obj):
        return torch.as_tensor(obj).unsqueeze(
            dim=0) if isinstance(obj, (np.ndarray, numbers.Number)) else obj

    return nest.map_structure(_array_to_tensor, data)


def _tensor_to_array(data):
    return nest.map_structure(lambda x: x.squeeze(dim=0).cpu().numpy(), data)


class ThreadEnvironment(alf_environment.AlfEnvironment):
    """Create, Step a single env in a separate thread
    """

    def __init__(self, env_constructor):
        """Create a ThreadEnvironment

        Args:
            env_constructor (Callable): env_constructor for the OpenAI Gym environment
        """
        super().__init__()
        self._pool = mp_threads.Pool(1)
        self._env = self._pool.apply(env_constructor)
        assert not self._env.batched
        self._closed = False

    @property
    def is_tensor_based(self):
        return True

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return 1

    def env_info_spec(self):
        return self._apply('env_info_spec')

    def observation_spec(self):
        return self._apply('observation_spec')

    def action_spec(self):
        return self._apply('action_spec')

    def reward_spec(self):
        return self._apply('reward_spec')

    def _step(self, action):
        action = _tensor_to_array(action)
        return _array_to_tensor(self._apply('step', (action, )))

    def _reset(self):
        return _array_to_tensor(self._apply('reset'))

    def close(self):
        if self._closed:
            return
        self._apply('close')
        self._pool.close()
        self._pool.join()
        self._closed = True

    def render(self, mode='rgb_array'):
        return self._apply('render', (mode, ))

    def seed(self, seed):
        self._apply('seed', (seed, ))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _apply(self, name, args=()):
        func = getattr(self._env, name)
        return self._pool.apply(func, args)
