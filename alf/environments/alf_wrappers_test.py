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
"""Test cases adapted from tf_agents' wrappers_test.py."""
# TODO: just test TimeLimit wrapper for now, add other tests later.

from absl.testing import parameterized
from absl.testing.absltest import mock
from collections import OrderedDict
import gym
import math
import torch
import numpy as np
from functools import partial

import alf
import alf.data_structures as ds
from alf.environments import alf_environment, alf_gym_wrapper, alf_wrappers, suite_gym
from alf.environments.random_alf_environment import RandomAlfEnvironment
from alf.environments.utils import create_environment
import alf.tensor_specs as ts


class AlfEnvironmentBaseWrapperTest(parameterized.TestCase):
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
        env = RandomAlfEnvironment(
            obs_spec,
            action_spec,
            reward_fn=lambda *_: torch.tensor([1.0], dtype=torch.float32),
            batch_size=batch_size)
        wrap_env = alf_wrappers.AlfEnvironmentBaseWrapper(env)
        self.assertEqual(wrap_env.batched, env.batched)
        self.assertEqual(wrap_env.batch_size, env.batch_size)

    def test_default_batch_properties(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        self.assertFalse(env.batched)
        self.assertEqual(env.batch_size, 1)
        wrap_env = alf_wrappers.AlfEnvironmentBaseWrapper(env)
        self.assertEqual(wrap_env.batched, env.batched)
        self.assertEqual(wrap_env.batch_size, env.batch_size)

    def test_wrapped_method_propagation(self):
        mock_env = mock.MagicMock()
        env = alf_wrappers.AlfEnvironmentBaseWrapper(mock_env)
        env.reset()
        self.assertEqual(1, mock_env.reset.call_count)
        action = np.array(0, dtype=np.int64)
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


class TimeLimitWrapperTest(parameterized.TestCase, alf.test.TestCase):
    def test_limit_duration_wrapped_env_forwards_calls(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        env = alf_wrappers.TimeLimit(env, 10)

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
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        env = alf_wrappers.TimeLimit(env, 2)

        env.reset()
        action = np.array(0, dtype=np.int64)
        env.step(action)
        time_step = env.step(action)

        self.assertTrue(time_step.is_last())
        self.assertNotEqual(None, time_step.discount)
        self.assertNotEqual(0.0, time_step.discount)

    @parameterized.parameters(True, False)
    def test_all_step_types_same_as_env(self, use_tensor_time_step):
        obs_spec = ts.BoundedTensorSpec((2, 3), torch.int32, -10, 10)
        action_spec = ts.BoundedTensorSpec((1, ), torch.int64, -10, 10)
        env = RandomAlfEnvironment(
            obs_spec,
            action_spec,
            reward_fn=lambda *_: np.array(1.0, dtype=np.float32),
            use_tensor_time_step=use_tensor_time_step)

        env = alf_wrappers.AlfEnvironmentBaseWrapper(env)
        duration = 2
        env = alf_wrappers.TimeLimit(env, duration)

        env.reset()
        action = action_spec.sample()
        step_types = []
        for i in range(duration):
            time_step = env.step(action)
            step_types.append(time_step.step_type)

        self.assertTrue(time_step.is_last())
        if use_tensor_time_step:
            self.assertTrue(all(map(torch.is_tensor, step_types)))
        else:
            self.assertTrue(all(map(ds._is_numpy_array, step_types)))

    def test_extra_env_methods_work(self):
        cartpole_env = gym.make('CartPole-v1')
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        env = alf_wrappers.TimeLimit(env, 2)

        self.assertEqual(None, env.get_info())
        env.reset()
        action = np.array(0, dtype=np.int64)
        env.step(action)
        self.assertEqual({}, env.get_info())

    def test_automatic_reset(self):
        cartpole_env = gym.make('CartPole-v1')
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        env = alf_wrappers.TimeLimit(env, 2)

        # Episode 1
        action = np.array(0, dtype=np.int64)
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
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        env = alf_wrappers.TimeLimit(env, 10000)

        # Episode 1 stepped until termination occurs.
        action = np.array(1, dtype=np.int64)
        time_step = env.step(action)
        while not time_step.is_last():
            time_step = env.step(action)

        self.assertTrue(time_step.is_last())
        env._duration = 2

        # Episode 2 short duration hits step limit.
        action = np.array(0, dtype=np.int64)
        first_time_step = env.step(action)
        self.assertTrue(first_time_step.is_first())
        mid_time_step = env.step(action)
        self.assertTrue(mid_time_step.is_mid())
        last_time_step = env.step(action)
        self.assertTrue(last_time_step.is_last())


class MultitaskWrapperTest(alf.test.TestCase):
    def test_multitask_wrapper(self):
        env = alf_wrappers.MultitaskWrapper.load(
            suite_gym.load, ['CartPole-v0', 'CartPole-v1'])
        self.assertEqual(env.num_tasks, 2)
        self.assertEqual(env.action_spec()['task_id'],
                         alf.BoundedTensorSpec((), maximum=1, dtype='int64'))
        self.assertEqual(env.action_spec()['action'],
                         env._envs[0].action_spec())
        time_step = env.reset()
        time_step = env.step(
            OrderedDict(task_id=1, action=time_step.prev_action['action']))
        self.assertEqual(time_step.prev_action['task_id'], 1)


class CurriculumWrapperTest(alf.test.TestCase):
    def test_curriculum_wrapper(self):
        task_names = ['CartPole-v0', 'CartPole-v1']
        env = create_environment(
            env_name=task_names,
            env_load_fn=suite_gym.load,
            num_parallel_environments=4,
            batched_wrappers=(alf_wrappers.CurriculumWrapper, ))

        self.assertTrue(type(env.action_spec()) == alf.BoundedTensorSpec)

        self.assertEqual(env.num_tasks, 2)
        self.assertEqual(len(env.env_info_spec()['curriculum_task_count']), 2)
        self.assertEqual(len(env.env_info_spec()['curriculum_task_score']), 2)
        self.assertEqual(len(env.env_info_spec()['curriculum_task_prob']), 2)
        for i in task_names:
            self.assertEqual(env.env_info_spec()['curriculum_task_count'][i],
                             alf.TensorSpec(()))
            self.assertEqual(env.env_info_spec()['curriculum_task_score'][i],
                             alf.TensorSpec(()))
            self.assertEqual(env.env_info_spec()['curriculum_task_prob'][i],
                             alf.TensorSpec(()))

        time_step = env.reset()
        self.assertEqual(len(env.env_info_spec()['curriculum_task_count']), 2)
        self.assertEqual(len(env.env_info_spec()['curriculum_task_score']), 2)
        self.assertEqual(len(env.env_info_spec()['curriculum_task_prob']), 2)
        for i in task_names:
            self.assertEqual(
                time_step.env_info['curriculum_task_count'][i].shape, (4, ))
            self.assertEqual(
                time_step.env_info['curriculum_task_score'][i].shape, (4, ))
            self.assertEqual(
                time_step.env_info['curriculum_task_prob'][i].shape, (4, ))

        for j in range(500):
            time_step = env.step(time_step.prev_action)
            self.assertEqual(time_step.env_id, torch.arange(4))
            self.assertEqual(
                len(env.env_info_spec()['curriculum_task_count']), 2)
            self.assertEqual(
                len(env.env_info_spec()['curriculum_task_score']), 2)
            self.assertEqual(
                len(env.env_info_spec()['curriculum_task_prob']), 2)
            for i in task_names:
                self.assertEqual(
                    time_step.env_info['curriculum_task_count'][i].shape,
                    (4, ))
                self.assertEqual(
                    time_step.env_info['curriculum_task_score'][i].shape,
                    (4, ))
                self.assertEqual(
                    time_step.env_info['curriculum_task_prob'][i].shape, (4, ))
            sum_probs = sum(
                time_step.env_info['curriculum_task_prob'].values())
            self.assertTrue(
                torch.all((sum_probs == 0.) | ((sum_probs - 1.).abs() < 1e-3)))


class DiscreteWrapperTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((
        5,
        "MountainCarContinuous-v0",
        True,
    ), (3, "LunarLanderContinuous-v2", False))
    def test_discrete_action_wrapper(self, actions_num, env_name, batched):
        unwrapped = gym.make(env_name)
        low, high = unwrapped.action_space.low, unwrapped.action_space.high

        class ActionInfoWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)

            def step(self, action):
                obs, reward, done, info = self.env.step(action)
                info['action'] = action
                return obs, reward, done, info

        if batched:
            batched_wrappers = [
                partial(
                    alf_wrappers.DiscreteActionWrapper,
                    actions_num=actions_num)
            ]
            load_fn = partial(
                suite_gym.load, gym_env_wrappers=[ActionInfoWrapper])
        else:
            batched_wrappers = []
            load_fn = partial(
                suite_gym.load,
                gym_env_wrappers=[ActionInfoWrapper],
                alf_env_wrappers=[
                    partial(
                        alf_wrappers.DiscreteActionWrapper,
                        actions_num=actions_num)
                ])

        env = create_environment(
            env_name=env_name,
            env_load_fn=load_fn,
            num_parallel_environments=4,
            batched_wrappers=batched_wrappers)

        self.assertTrue(env.action_spec().is_discrete)

        time_step = env.reset()
        self.assertTrue(torch.all(time_step.prev_action == 0))
        actual_actions = []
        for a in range(actions_num**unwrapped.action_space.shape[0]):
            a = torch.full_like(time_step.step_type, a, dtype=torch.int64)
            time_step = env.step(a)
            self.assertTensorEqual(time_step.prev_action, a)  # discrete
            self.assertEqual(torch.int64, time_step.prev_action.dtype)
            actual_actions.append(time_step.env_info['action'])  # continuous

        self.assertTrue(np.allclose(actual_actions[0].cpu().numpy(), low))
        self.assertTrue(np.allclose(actual_actions[-1].cpu().numpy(), high))
        # evenly distributed values
        self.assertTrue(
            np.allclose(actual_actions[1].cpu().numpy() - low,
                        high - actual_actions[-2].cpu().numpy()))


if __name__ == '__main__':
    alf.test.main()
