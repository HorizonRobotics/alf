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

import alf
from alf.environments import suite_babyai
from alf.environments.alf_wrappers import TimeLimit


class SuiteBabyAITest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_babyai.is_available():
            self.skipTest('suite_babyai is not available.')

    def test_one_token_per_step(self):
        env = suite_babyai.load("BabyAI-GoToRedBall-v0", mode='word')
        vocab = suite_babyai.BabyAIWrapper.VOCAB
        self.assertEqual(env.observation_spec()['mission'].shape, ())
        self.assertEqual(env.observation_spec()['mission'].minimum, 0)
        self.assertEqual(env.observation_spec()['mission'].maximum, len(vocab))
        obs = env.reset().observation
        self.assertEqual(obs['mission'], vocab.index('go') + 1)
        obs = env.step(0).observation
        self.assertEqual(obs['mission'], vocab.index('to') + 1)
        obs = env.step(0).observation
        self.assertTrue(obs['mission'] == vocab.index('a') + 1
                        or obs['mission'] == vocab.index('the') + 1)
        obs = env.step(0).observation
        self.assertEqual(obs['mission'], vocab.index('red') + 1)
        obs = env.step(0).observation
        self.assertEqual(obs['mission'], vocab.index('ball') + 1)
        obs = env.step(0).observation
        self.assertEqual(obs['mission'], 0)
        obs = env.step(0).observation
        self.assertEqual(obs['mission'], 0)

    def test_one_char_per_step(self):
        env = suite_babyai.load("BabyAI-GoToRedBall-v0", mode='char')
        self.assertEqual(env.observation_spec()['mission'].shape, ())
        self.assertEqual(env.observation_spec()['mission'].minimum, 0)
        self.assertEqual(env.observation_spec()['mission'].maximum, 127)
        obs = env.reset().observation
        self.assertEqual(obs['mission'], ord('g'))
        obs = env.step(0).observation
        self.assertEqual(obs['mission'], ord('o'))
        obs = env.step(0).observation
        self.assertEqual(obs['mission'], ord(' '))

    def test_one_instruction_per_step(self):
        env = suite_babyai.load(
            "BabyAI-GoToRedBall-v0", max_instruction_length=10, mode='sent')
        vocab = suite_babyai.BabyAIWrapper.VOCAB
        self.assertEqual(env.observation_spec()['mission'].shape, (10, ))
        self.assertEqual(env.observation_spec()['mission'].minimum, 0)
        self.assertEqual(env.observation_spec()['mission'].maximum, len(vocab))
        instr1 = np.array(
            [vocab.index(w) + 1
             for w in ['go', 'to', 'the', 'red', 'ball']] + [0, 0, 0, 0, 0])
        instr2 = np.array(
            [vocab.index(w) + 1
             for w in ['go', 'to', 'a', 'red', 'ball']] + [0, 0, 0, 0, 0])
        obs = env.reset().observation
        self.assertTrue(
            np.alltrue(obs['mission'] == instr1)
            or np.alltrue(obs['mission'] == instr2))
        obs = env.step(0).observation
        self.assertTrue(
            np.alltrue(obs['mission'] == instr1)
            or np.alltrue(obs['mission'] == instr2))

    def test_timelimit_discount(self):
        env_name = "BabyAI-GoToObj-v0"
        gym_env = gym.make(env_name)
        gym_spec = gym.spec(env_name)
        self.assertTrue(gym_spec.max_episode_steps is None)

        # first test the original env will incorrectly return done=True when timeout
        self.assertTrue(hasattr(gym_env, 'max_steps'))
        gym_env.reset()
        for i in range(gym_env.max_steps):
            observation, reward, done, info = gym_env.step(0)
        self.assertTrue(done)  # timelimit

        # then test the new suite_babyai will correctly handle this
        env = suite_babyai.load(env_name)
        self.assertTrue(isinstance(env, TimeLimit))
        self.assertEqual(env.duration, gym_env.max_steps - 1)


if __name__ == '__main__':
    alf.test.main()
