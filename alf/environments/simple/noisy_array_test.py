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

import unittest
from absl.testing import parameterized
from alf.environments.simple.noisy_array import NoisyArray


class NoisyArrayTest(parameterized.TestCase, unittest.TestCase):
    @parameterized.parameters((5, 3), (201, 100))
    def test_noisy_array_environment(self, K, M):
        array = NoisyArray(K, M)
        array.reset()
        for _ in range(K - 1):
            done = array.step(NoisyArray.RIGHT)[2]
        self.assertTrue(done)

        array.reset()
        array.step(NoisyArray.LEFT)  # cannot go beyond the left boundary
        self.assertEqual(array._position, 0)

        array.step(NoisyArray.RIGHT)
        array.reset()

        game_ends = 0
        total_rewards = 0
        done = False
        for _ in range(2 * K - 1):
            if done:
                array.reset()
                done = False
            else:
                _, r, done, _ = array.step(NoisyArray.RIGHT)
                total_rewards += r
                game_ends += int(done)

        self.assertEqual(game_ends, 2)
        self.assertEqual(total_rewards, 2)
        self.assertEqual(r, 1.0)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
