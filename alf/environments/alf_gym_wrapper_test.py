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
"""Tests for alf_gym_wrapper. Adapted from tf_agents gym_wrapper_test.py.
"""

from absl.testing.absltest import mock
import gym
import gym.spaces
import math
import numpy as np
import torch

import alf
from alf.environments import alf_gym_wrapper
from alf.tensor_specs import torch_dtype_to_str


class GymWrapperSpecTest(alf.test.TestCase):
    def test_tensor_spec_from_gym_space_discrete(self):
        discrete_space = gym.spaces.Discrete(3)
        spec = alf_gym_wrapper.tensor_spec_from_gym_space(discrete_space)

        self.assertEqual((), spec.shape)
        self.assertEqual(torch.int64, spec.dtype)
        self.assertEqual(0, spec.minimum)
        self.assertEqual(2, spec.maximum)

    def test_tensor_spec_from_gym_space_multi_discrete(self):
        multi_discrete_space = gym.spaces.MultiDiscrete([1, 2, 3, 4])
        spec = alf_gym_wrapper.tensor_spec_from_gym_space(multi_discrete_space)

        self.assertEqual((4, ), spec.shape)
        self.assertEqual(torch.int64, spec.dtype)
        np.testing.assert_array_equal(
            np.array([0], dtype=np.int64), spec.minimum)
        np.testing.assert_array_equal(
            np.array([0, 1, 2, 3], dtype=np.int64), spec.maximum)

    def test_tensor_spec_from_gym_space_multi_binary(self):
        multi_binary_space = gym.spaces.MultiBinary(4)
        spec = alf_gym_wrapper.tensor_spec_from_gym_space(multi_binary_space)

        self.assertEqual((4, ), spec.shape)
        self.assertEqual(torch.int8, spec.dtype)
        np.testing.assert_array_equal(
            np.array([0], dtype=np.int64), spec.minimum)
        np.testing.assert_array_equal(
            np.array([1], dtype=np.int64), spec.maximum)

    def test_tensor_spec_from_gym_space_box_scalars(self):
        for dtype in (np.float32, np.float64):
            box_space = gym.spaces.Box(-1.0, 1.0, (3, 4), dtype=dtype)

            # test if float_dtype is not specified, the spec's dtype
            # will match that of the space
            spec = alf_gym_wrapper.tensor_spec_from_gym_space(
                box_space, float_dtype=None)

            torch_dtype = getattr(torch, np.dtype(dtype).name)
            self.assertEqual((3, 4), spec.shape)
            self.assertEqual(torch_dtype, spec.dtype)
            np.testing.assert_array_equal(-np.ones((3, 4)), spec.minimum)
            np.testing.assert_array_equal(np.ones((3, 4)), spec.maximum)

            # test if float_dtype is specified, the spec's dtype will match
            # the specified one, regardless of the dtype of space
            for float_dtype in (np.float32, np.float64):
                spec = alf_gym_wrapper.tensor_spec_from_gym_space(
                    box_space, float_dtype=float_dtype)
                torch_float_dtype = getattr(torch, np.dtype(float_dtype).name)
                self.assertEqual(torch_float_dtype, spec.dtype)

    def test_tensor_spec_from_gym_space_box_scalars_simplify_bounds(self):
        box_space = gym.spaces.Box(-1.0, 1.0, (3, 4))
        spec = alf_gym_wrapper.tensor_spec_from_gym_space(
            box_space, simplify_box_bounds=True)

        self.assertEqual((3, 4), spec.shape)
        self.assertEqual(torch.float32, spec.dtype)
        np.testing.assert_array_equal(
            np.array([-1], dtype=np.int64), spec.minimum)
        np.testing.assert_array_equal(
            np.array([1], dtype=np.int64), spec.maximum)

    def test_tensor_spec_from_gym_space_when_simplify_box_bounds_false(self):
        # testing on gym.spaces.Dict which makes recursive calls to
        # _tensor_spec_from_gym_space
        box_space = gym.spaces.Box(-1.0, 1.0, (2, ))
        dict_space = gym.spaces.Dict({'box1': box_space, 'box2': box_space})
        spec = alf_gym_wrapper.tensor_spec_from_gym_space(
            dict_space, simplify_box_bounds=False)

        self.assertEqual((2, ), spec['box1'].shape)
        self.assertEqual((2, ), spec['box2'].shape)
        self.assertEqual(torch.float32, spec['box1'].dtype)
        self.assertEqual(torch.float32, spec['box2'].dtype)
        np.testing.assert_array_equal(
            np.array([-1, -1], dtype=np.int64), spec['box1'].minimum)
        np.testing.assert_array_equal(
            np.array([1, 1], dtype=np.int64), spec['box1'].maximum)
        np.testing.assert_array_equal(
            np.array([-1, -1], dtype=np.int64), spec['box2'].minimum)
        np.testing.assert_array_equal(
            np.array([1, 1], dtype=np.int64), spec['box2'].maximum)

    def test_tensor_spec_from_gym_space_box_array(self):
        for dtype in (np.float32, np.float64):
            box_space = gym.spaces.Box(
                np.array([-1.0, -2.0]), np.array([2.0, 4.0]), dtype=dtype)

            # test if float_dtype is not specified, the spec's dtype
            # will match that of the space
            spec = alf_gym_wrapper.tensor_spec_from_gym_space(
                box_space, float_dtype=None)

            torch_dtype = getattr(torch, np.dtype(dtype).name)
            self.assertEqual((2, ), spec.shape)
            self.assertEqual(torch_dtype, spec.dtype)
            np.testing.assert_array_equal(np.array([-1.0, -2.0]), spec.minimum)
            np.testing.assert_array_equal(np.array([2.0, 4.0]), spec.maximum)

            # test if float_dtype is specified, the spec's dtype will match
            # the specified one, regardless of the dtype of space
            for float_dtype in (np.float32, np.float64):
                spec = alf_gym_wrapper.tensor_spec_from_gym_space(
                    box_space, float_dtype=float_dtype)
                torch_float_dtype = getattr(torch, np.dtype(float_dtype).name)
                self.assertEqual(torch_float_dtype, spec.dtype)

    def test_tensor_spec_from_gym_space_tuple(self):
        tuple_space = gym.spaces.Tuple((gym.spaces.Discrete(2),
                                        gym.spaces.Discrete(3)))
        spec = alf_gym_wrapper.tensor_spec_from_gym_space(tuple_space)

        self.assertEqual(2, len(spec))
        self.assertEqual((), spec[0].shape)
        self.assertEqual(torch.int64, spec[0].dtype)
        self.assertEqual(0, spec[0].minimum)
        self.assertEqual(1, spec[0].maximum)

        self.assertEqual((), spec[1].shape)
        self.assertEqual(torch.int64, spec[1].dtype)
        self.assertEqual(0, spec[1].minimum)
        self.assertEqual(2, spec[1].maximum)

    def test_tensor_spec_from_gym_space_tuple_mixed(self):
        tuple_space = gym.spaces.Tuple((
            gym.spaces.Discrete(2),
            gym.spaces.Box(-1.0, 1.0, (3, 4)),
            gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(3))),
            gym.spaces.Dict({
                'spec_1':
                    gym.spaces.Discrete(2),
                'spec_2':
                    gym.spaces.Tuple((gym.spaces.Discrete(2),
                                      gym.spaces.Discrete(3))),
            }),
        ))
        spec = alf_gym_wrapper.tensor_spec_from_gym_space(tuple_space)

        self.assertEqual(4, len(spec))
        # Test Discrete
        self.assertEqual((), spec[0].shape)
        self.assertEqual(torch.int64, spec[0].dtype)
        self.assertEqual(0, spec[0].minimum)
        self.assertEqual(1, spec[0].maximum)

        # Test Box
        self.assertEqual((3, 4), spec[1].shape)
        self.assertEqual(torch.float32, spec[1].dtype)
        np.testing.assert_array_almost_equal(-np.ones((3, 4)), spec[1].minimum)
        np.testing.assert_array_almost_equal(np.ones((3, 4)), spec[1].maximum)

        # Test Tuple
        self.assertEqual(2, len(spec[2]))
        self.assertEqual((), spec[2][0].shape)
        self.assertEqual(torch.int64, spec[2][0].dtype)
        self.assertEqual(0, spec[2][0].minimum)
        self.assertEqual(1, spec[2][0].maximum)
        self.assertEqual((), spec[2][1].shape)
        self.assertEqual(torch.int64, spec[2][1].dtype)
        self.assertEqual(0, spec[2][1].minimum)
        self.assertEqual(2, spec[2][1].maximum)

        # Test Dict
        # Test Discrete in Dict
        discrete_in_dict = spec[3]['spec_1']
        self.assertEqual((), discrete_in_dict.shape)
        self.assertEqual(torch.int64, discrete_in_dict.dtype)
        self.assertEqual(0, discrete_in_dict.minimum)
        self.assertEqual(1, discrete_in_dict.maximum)

        # Test Tuple in Dict
        tuple_in_dict = spec[3]['spec_2']
        self.assertEqual(2, len(tuple_in_dict))
        self.assertEqual((), tuple_in_dict[0].shape)
        self.assertEqual(torch.int64, tuple_in_dict[0].dtype)
        self.assertEqual(0, tuple_in_dict[0].minimum)
        self.assertEqual(1, tuple_in_dict[0].maximum)
        self.assertEqual((), tuple_in_dict[1].shape)
        self.assertEqual(torch.int64, tuple_in_dict[1].dtype)
        self.assertEqual(0, tuple_in_dict[1].minimum)
        self.assertEqual(2, tuple_in_dict[1].maximum)

    def test_tensor_spec_from_gym_space_dict(self):
        dict_space = gym.spaces.Dict([
            ('spec_2', gym.spaces.Box(-1.0, 1.0, (3, 4))),
            ('spec_1', gym.spaces.Discrete(2)),
        ])

        spec = alf_gym_wrapper.tensor_spec_from_gym_space(dict_space)

        keys = list(spec.keys())
        self.assertEqual('spec_1', keys[1])
        self.assertEqual(2, len(spec))
        self.assertEqual((), spec['spec_1'].shape)
        self.assertEqual(torch.int64, spec['spec_1'].dtype)
        self.assertEqual(0, spec['spec_1'].minimum)
        self.assertEqual(1, spec['spec_1'].maximum)

        self.assertEqual('spec_2', keys[0])
        self.assertEqual((3, 4), spec['spec_2'].shape)
        self.assertEqual(torch.float32, spec['spec_2'].dtype)
        np.testing.assert_array_almost_equal(
            -np.ones((3, 4)),
            spec['spec_2'].minimum,
        )
        np.testing.assert_array_almost_equal(
            np.ones((3, 4)),
            spec['spec_2'].maximum,
        )


class GymWrapperOnCartpoleTest(alf.test.TestCase):
    def test_wrapped_cartpole_specs(self):
        # Note we use spec.make on gym envs to avoid getting a TimeLimit wrapper on
        # the environment.
        cartpole_env = gym.spec('CartPole-v1').make()
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)

        action_spec = env.action_spec()
        self.assertEqual((), action_spec.shape)
        self.assertEqual(0, action_spec.minimum)
        self.assertEqual(1, action_spec.maximum)

        observation_spec = env.observation_spec()
        self.assertEqual((4, ), observation_spec.shape)
        self.assertEqual(torch.float32, observation_spec.dtype)
        high = np.array([
            4.8,
            np.finfo(np.float32).max, 2 / 15.0 * math.pi,
            np.finfo(np.float32).max
        ])
        np.testing.assert_array_almost_equal(-high, observation_spec.minimum)
        np.testing.assert_array_almost_equal(high, observation_spec.maximum)

    def test_wrapped_cartpole_reset(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)

        first_time_step = env.reset()
        self.assertTrue(first_time_step.is_first())
        self.assertEqual(0.0, first_time_step.reward)
        self.assertEqual(1.0, first_time_step.discount)
        self.assertEqual((4, ), first_time_step.observation.shape)
        self.assertEqual("float32", str(first_time_step.observation.dtype))

    def test_wrapped_cartpole_transition(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        env.reset()
        action = np.array(0, dtype=np.int64)
        transition_time_step = env.step(action)

        self.assertTrue(transition_time_step.is_mid())
        self.assertNotEqual(None, transition_time_step.reward)
        self.assertEqual(1.0, transition_time_step.discount)
        self.assertEqual((4, ), transition_time_step.observation.shape)

    def test_wrapped_cartpole_final(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        time_step = env.reset()

        action = np.array(1, dtype=np.int64)
        while not time_step.is_last():
            time_step = env.step(action)

        self.assertTrue(time_step.is_last())
        self.assertNotEqual(None, time_step.reward)
        self.assertEqual(0.0, time_step.discount)
        self.assertEqual((4, ), time_step.observation.shape)

    def test_get_info(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        self.assertEqual(None, env.get_info())
        env.reset()
        self.assertEqual(None, env.get_info())
        action = np.array(0, dtype=np.int64)
        env.step(action)
        self.assertEqual({}, env.get_info())

    def test_automatic_reset_after_create(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)

        action = np.array(0, dtype=np.int64)
        first_time_step = env.step(action)
        self.assertTrue(first_time_step.is_first())

    def test_automatic_reset_after_done(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        time_step = env.reset()

        action = np.array(1, dtype=np.int64)
        while not time_step.is_last():
            time_step = env.step(action)

        self.assertTrue(time_step.is_last())
        action = np.array(0, dtype=np.int64)
        first_time_step = env.step(action)
        self.assertTrue(first_time_step.is_first())

    def test_automatic_reset_after_done_not_using_reset_directly(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        action = np.array(1, dtype=np.int64)
        time_step = env.step(action)

        while not time_step.is_last():
            time_step = env.step(action)

        self.assertTrue(time_step.is_last())
        action = np.array(0, dtype=np.int64)
        first_time_step = env.step(action)
        self.assertTrue(first_time_step.is_first())

    def test_method_propagation(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        for method_name in ('render', 'seed', 'close'):
            setattr(cartpole_env, method_name, mock.MagicMock())
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        env.render()
        self.assertEqual(1, cartpole_env.render.call_count)
        env.seed(0)
        self.assertEqual(1, cartpole_env.seed.call_count)
        cartpole_env.seed.assert_called_with(0)
        env.close()
        self.assertEqual(1, cartpole_env.close.call_count)

    def test_obs_dtype(self):
        cartpole_env = gym.spec('CartPole-v1').make()
        env = alf_gym_wrapper.AlfGymWrapper(cartpole_env)
        time_step = env.reset()
        self.assertEqual(
            torch_dtype_to_str(env.observation_spec().dtype),
            str(time_step.observation.dtype))


if __name__ == '__main__':
    GymWrapperOnCartpoleTest().test_obs_dtype()
    alf.test.main()
