# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
"""Test cases adpated from tf_agents' wrappers_test.py."""
# TODO: just test TimeLimit wrapper for now, add other tests later.

from absl.testing import parameterized
from absl.testing.absltest import mock
import gym
import math
import torch
import numpy as np

import alf
import alf.data_structures as ds
from alf.environments import torch_environment, torch_gym_wrapper, torch_wrappers
from alf.environments.random_torch_environment import RandomTorchEnvironment
import alf.tensor_specs as ts


class TorchEnvironmentBaseWrapperTest(parameterized.TestCase):
    @parameterized.named_parameters(
        {
            'testcase_name': 'scalar',
            'batch_size': None
        },
        {
            'testcase_name': 'batched',
            'batch_size': 2
        },
    )
    def test_batch_properties(self, batch_size):
        obs_spec = ts.BoundedTensorSpec((2, 3), torch.int32, -10, 10)
        action_spec = ts.BoundedTensorSpec((1, ), torch.int64, -10, 10)
        env = RandomTorchEnvironment(
            obs_spec,
            action_spec,
            reward_fn=lambda *_: torch.tensor([1.0], dtype=torch.float32),
            batch_size=batch_size)
        wrap_env = torch_wrappers.TorchEnvironmentBaseWrapper(env)
        self.assertEqual(wrap_env.batched, env.batched)
        self.assertEqual(wrap_env.batch_size, env.batch_size)

    def test_default_batch_properties(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = torch_gym_wrapper.TorchGymWrapper(cartpole_env)
        self.assertFalse(env.batched)
        self.assertEqual(env.batch_size, None)
        wrap_env = torch_wrappers.TorchEnvironmentBaseWrapper(env)
        self.assertEqual(wrap_env.batched, env.batched)
        self.assertEqual(wrap_env.batch_size, env.batch_size)

    def test_wrapped_method_propagation(self):
        mock_env = mock.MagicMock()
        env = torch_wrappers.TorchEnvironmentBaseWrapper(mock_env)
        env.reset()
        self.assertEqual(1, mock_env.reset.call_count)
        action = torch.tensor(0, dtype=torch.int64)
        env.step(action)
        self.assertEqual(1, mock_env.step.call_count)
        mock_env.step.assert_called_with(0)
        env.seed(0)
        self.assertEqual(1, mock_env.seed.call_count)
        mock_env.seed.assert_called_with(0)
        env.render()
        self.assertEqual(1, mock_env.render.call_count)
        env.close()
        self.assertEqual(1, mock_env.close.call_count)


class TimeLimitWrapperTest(alf.test.TestCase):
    def test_limit_duration_wrapped_env_forwards_calls(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = torch_gym_wrapper.TorchGymWrapper(cartpole_env)
        env = torch_wrappers.TimeLimit(env, 10)

        action_spec = env.action_spec()
        self.assertEqual((), action_spec.shape)
        self.assertEqual(0, action_spec.minimum)
        self.assertEqual(1, action_spec.maximum)

        observation_spec = env.observation_spec()
        self.assertEqual((4, ), observation_spec.shape)
        high = np.array([
            4.8,
            np.finfo(np.float32).max, 2 / 15.0 * math.pi,
            np.finfo(np.float32).max
        ])
        np.testing.assert_array_almost_equal(-high, observation_spec.minimum)
        np.testing.assert_array_almost_equal(high, observation_spec.maximum)

    def test_limit_duration_stops_after_duration(self):
        cartpole_env = gym.make('CartPole-v1')
        env = torch_gym_wrapper.TorchGymWrapper(cartpole_env)
        env = torch_wrappers.TimeLimit(env, 2)

        env.reset()
        action = torch.tensor(0, dtype=torch.int64)
        env.step(action)
        time_step = env.step(action)

        self.assertTrue(time_step.is_last())
        self.assertNotEqual(None, time_step.discount)
        self.assertNotEqual(0.0, time_step.discount)

    def test_extra_env_methods_work(self):
        cartpole_env = gym.make('CartPole-v1')
        env = torch_gym_wrapper.TorchGymWrapper(cartpole_env)
        env = torch_wrappers.TimeLimit(env, 2)

        self.assertEqual(None, env.get_info())
        env.reset()
        action = torch.tensor(0, dtype=torch.int64)
        env.step(action)
        self.assertEqual({}, env.get_info())

    def test_automatic_reset(self):
        cartpole_env = gym.make('CartPole-v1')
        env = torch_gym_wrapper.TorchGymWrapper(cartpole_env)
        env = torch_wrappers.TimeLimit(env, 2)

        # Episode 1
        action = torch.tensor(0, dtype=torch.int64)
        first_time_step = env.step(action)
        self.assertTrue(first_time_step.is_first())
        mid_time_step = env.step(action)
        self.assertTrue(mid_time_step.is_mid())
        last_time_step = env.step(action)
        self.assertTrue(last_time_step.is_last())

        # Episode 2
        first_time_step = env.step(action)
        self.assertTrue(first_time_step.is_first())
        mid_time_step = env.step(action)
        self.assertTrue(mid_time_step.is_mid())
        last_time_step = env.step(action)
        self.assertTrue(last_time_step.is_last())

    def test_duration_applied_after_episode_terminates_early(self):
        cartpole_env = gym.make('CartPole-v1')
        env = torch_gym_wrapper.TorchGymWrapper(cartpole_env)
        env = torch_wrappers.TimeLimit(env, 10000)

        # Episode 1 stepped until termination occurs.
        action = torch.tensor(1, dtype=torch.int64)
        time_step = env.step(action)
        while not time_step.is_last():
            time_step = env.step(action)

        self.assertTrue(time_step.is_last())
        env._duration = 2

        # Episode 2 short duration hits step limit.
        action = torch.tensor(0, dtype=torch.int64)
        first_time_step = env.step(action)
        self.assertTrue(first_time_step.is_first())
        mid_time_step = env.step(action)
        self.assertTrue(mid_time_step.is_mid())
        last_time_step = env.step(action)
        self.assertTrue(last_time_step.is_last())


if __name__ == '__main__':
    alf.test.main()
