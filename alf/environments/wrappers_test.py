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

import gin
import tensorflow as tf

from alf.environments import suite_socialbot
from alf.environments.wrappers import FrameStack


class FrameStackTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_socialbot.is_available():
            self.skipTest('suite_socialbot is not available.')
        else:
            gin.clear_config()
            from social_bot.envs.play_ground import PlayGround

    def tearDown(self):
        super().tearDown()
        self._env.close()

    def test_framestack_all_fields(self):
        gin.bind_parameter('suite_socialbot.load.gym_env_wrappers',
                           (FrameStack, ))
        gin.bind_parameter('suite_socialbot.load.wrap_with_process', True)
        gin.bind_parameter('PlayGround.use_image_observation', True)
        gin.bind_parameter('PlayGround.image_with_internal_states', True)
        gin.bind_parameter('PlayGround.with_language', False)
        gin.bind_parameter('PlayGround.resized_image_size', (2, 2))
        self._env = suite_socialbot.load('SocialBot-PlayGround-v0')
        obs = self._env.reset().observation
        assert (
            obs['image'].shape,
            obs['states'].shape,
        ) == (
            (2, 2, 3 * 4),  # 3 channels * 4
            (4 * 4, ),  # 4 dimensions * 4 == 16
        )

    def test_framestack_some_fields(self):
        gin.bind_parameter('suite_socialbot.load.gym_env_wrappers',
                           (FrameStack, ))
        gin.bind_parameter('suite_socialbot.load.wrap_with_process', True)
        gin.bind_parameter('FrameStack.fields_to_stack', ['image'])
        gin.bind_parameter('PlayGround.use_image_observation', True)
        gin.bind_parameter('PlayGround.image_with_internal_states', True)
        gin.bind_parameter('PlayGround.with_language', False)
        gin.bind_parameter('PlayGround.resized_image_size', (2, 2))
        self._env = suite_socialbot.load('SocialBot-PlayGround-v0')
        obs = self._env.reset().observation
        assert (
            obs['image'].shape,
            obs['states'].shape,
        ) == (
            (2, 2, 3 * 4),  # 3 channels * 4
            (4, ),  # not stacking
        )

    # TODO: once nested observation fields are implemented, add tests
    # for FrameStacking nested fields here.


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    tf.test.main()
