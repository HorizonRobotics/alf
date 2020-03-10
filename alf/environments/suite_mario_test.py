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

import functools
import gin
import torch

import alf
from alf.environments import suite_mario, parallel_torch_environment
import alf.nest as nest


class SuiteMarioTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_mario.is_available():
            self.skipTest('suite_mario is not available.')
        else:
            gin.clear_config()

    def tearDown(self):
        super().tearDown()
        self._env.close()

    def test_mario_env(self):
        game = 'SuperMarioBros-Nes'

        def ctor(game, env_id=None):
            return suite_mario.load(
                game=game, state='Level1-1', wrap_with_process=False)

        constructor = functools.partial(ctor, game)

        self._env = parallel_torch_environment.ParallelTorchEnvironment(
            [constructor] * 4)
        self.assertEqual(torch.uint8, self._env.observation_spec().dtype)
        self.assertEqual((4, 84, 84), self._env.observation_spec().shape)

        actions = self._env.action_spec().sample(outer_dims=(4, ))
        for _ in range(10):
            time_step = self._env.step(actions)


if __name__ == '__main__':
    alf.test.main()
