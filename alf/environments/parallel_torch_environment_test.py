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
"""Tests for the parallel_torch_environment. 
Adapted from TF-Agents' parallel_py_environment_test.py
"""

import collections
import functools
import multiprocessing.dummy as dummy_multiprocessing
import numpy as np
import time
import torch

import alf
import alf.data_structures as ds
from alf.environments import parallel_torch_environment
from alf.environments.random_torch_environment import RandomTorchEnvironment
import alf.tensor_specs as ts


class SlowStartingEnvironment(RandomTorchEnvironment):
    def __init__(self, *args, **kwargs):
        time_sleep = kwargs.pop('time_sleep', 1.0)
        time.sleep(time_sleep)
        super(SlowStartingEnvironment, self).__init__(*args, **kwargs)


class ParallelTorchEnvironmentTest(alf.test.TestCase):
    def setUp(self):
        parallel_torch_environment.multiprocessing = dummy_multiprocessing

    def _set_default_specs(self):
        self.observation_spec = ts.TensorSpec((3, 3), torch.float32)
        self.action_spec = ts.BoundedTensorSpec([7],
                                                dtype=torch.float32,
                                                minimum=-1.0,
                                                maximum=1.0)
        self.time_step_spec = ds.time_step_spec(self.observation_spec,
                                                self.action_spec)

    def _make_parallel_torch_environment(self,
                                         constructor=None,
                                         num_envs=2,
                                         start_serially=True,
                                         blocking=True):
        self._set_default_specs()
        constructor = constructor or functools.partial(
            RandomTorchEnvironment, self.observation_spec, self.action_spec)
        return parallel_torch_environment.ParallelTorchEnvironment(
            env_constructors=[constructor] * num_envs,
            blocking=blocking,
            start_serially=start_serially)

    def test_close_no_hang_after_init(self):
        env = self._make_parallel_torch_environment()
        env.close()

    def test_get_specs(self):
        env = self._make_parallel_torch_environment()
        self.assertEqual(self.observation_spec, env.observation_spec())
        self.assertEqual(self.time_step_spec, env.time_step_spec())
        self.assertEqual(self.action_spec, env.action_spec())

        env.close()

    def test_step(self):
        num_envs = 2
        env = self._make_parallel_torch_environment(num_envs=num_envs)

        alf.set_default_device('cuda')

        action_spec = env.action_spec()
        observation_spec = env.observation_spec()
        action = torch.stack([action_spec.sample() for _ in range(num_envs)])
        env.reset()

        # Take one step and assert observation is batched the right way.
        time_step = env.step(action)
        self.assertEqual(num_envs, time_step.observation.shape[0])
        self.assertEqual(observation_spec.shape,
                         time_step.observation.shape[1:])
        self.assertEqual(num_envs, action.shape[0])
        self.assertEqual(torch.Size(action_spec.shape), action.shape[1:])

        # Take another step and assert that observations have the same shape.
        time_step2 = env.step(action)
        self.assertEqual(time_step.observation.shape,
                         time_step2.observation.shape)
        env.close()

    def test_non_blocking_start_processes_in_parallel(self):
        self._set_default_specs()
        constructor = functools.partial(
            SlowStartingEnvironment,
            self.observation_spec,
            self.action_spec,
            time_sleep=1.0)
        start_time = time.time()
        env = self._make_parallel_torch_environment(
            constructor=constructor,
            num_envs=10,
            start_serially=False,
            blocking=False)
        end_time = time.time()
        self.assertLessEqual(
            end_time - start_time,
            5.0,
            msg=('Expected all processes to start together, '
                 'got {} wait time').format(end_time - start_time))
        env.close()

    def test_blocking_start_processes_one_after_another(self):
        self._set_default_specs()
        constructor = functools.partial(
            SlowStartingEnvironment,
            self.observation_spec,
            self.action_spec,
            time_sleep=1.0)
        start_time = time.time()
        env = self._make_parallel_torch_environment(
            constructor=constructor,
            num_envs=10,
            start_serially=True,
            blocking=True)
        end_time = time.time()
        self.assertGreater(
            end_time - start_time,
            10,
            msg=('Expected all processes to start one '
                 'after another, got {} wait time').format(end_time -
                                                           start_time))
        env.close()

    def test_unstack_actions(self):
        num_envs = 2
        env = self._make_parallel_torch_environment(num_envs=num_envs)
        action_spec = env.action_spec()
        batched_action = torch.stack(
            [action_spec.sample() for _ in range(num_envs)])

        # Test that actions are correctly unstacked when just batched in np.array.
        unstacked_actions = env._unstack_actions(batched_action)
        for action in unstacked_actions:
            self.assertEqual(torch.Size(action_spec.shape), action.shape)
        env.close()

    def test_unstack_nested_actions(self):
        num_envs = 2
        env = self._make_parallel_torch_environment(num_envs=num_envs)
        action_spec = env.action_spec()
        batched_action = torch.stack(
            [action_spec.sample() for _ in range(num_envs)])

        # Test that actions are correctly unstacked when nested in namedtuple.
        class NestedAction(
                collections.namedtuple('NestedAction',
                                       ['action', 'other_var'])):
            pass

        nested_action = NestedAction(
            action=batched_action, other_var=np.array([13.0] * num_envs))
        unstacked_actions = env._unstack_actions(nested_action)
        for nested_action in unstacked_actions:
            self.assertEqual(
                torch.Size(action_spec.shape), nested_action.action.shape)
            self.assertEqual(13.0, nested_action.other_var)
        env.close()

    def test_seedable(self):
        seeds = [0, 1]
        env = self._make_parallel_torch_environment()
        env.seed(seeds)
        self.assertEqual(
            np.random.RandomState(0).get_state()[1][-1],
            env._envs[0]._rng.get_state()[1][-1])

        self.assertEqual(
            np.random.RandomState(1).get_state()[1][-1],
            env._envs[1]._rng.get_state()[1][-1])
        env.close()


if __name__ == '__main__':
    alf.test.main()
