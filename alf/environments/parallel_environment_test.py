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
"""Tests for the parallel_environment.
Adapted from TF-Agents' parallel_py_environment_test.py
"""
from absl import logging
from absl.testing import parameterized
import collections
import functools
import multiprocessing.dummy as dummy_multiprocessing
import numpy as np
import time
import torch

import alf
import alf.data_structures as ds
from alf.environments import parallel_environment
from alf.environments.alf_wrappers import BatchEnvironmentWrapper
from alf.environments.fast_parallel_environment import FastParallelEnvironment
from alf.environments.random_alf_environment import RandomAlfEnvironment
from alf.environments.alf_environment import AlfEnvironment
import alf.tensor_specs as ts
from alf.utils.schedulers import update_progress, get_progress


class SlowStartingEnvironment(RandomAlfEnvironment):
    def __init__(self, *args, **kwargs):
        self._time_sleep = kwargs.pop('time_sleep', 1.0)
        self._reset_sleep = kwargs.pop('reset_sleep', 0.0)
        time.sleep(self._time_sleep)
        super(SlowStartingEnvironment, self).__init__(*args, **kwargs)

    def reset(self):
        time.sleep(self._reset_sleep)
        return super().reset()


class ProgressEnvironment(AlfEnvironment):
    def __init__(self, env_id):
        super().__init__()
        self._env_id = np.int32(env_id)

    def action_spec(self):
        return ts.BoundedTensorSpec((),
                                    torch.float32,
                                    minimum=-1.0,
                                    maximum=1.0)

    def observation_spec(self):
        return ts.TensorSpec((), torch.float32)

    def env_info_spec(self):
        return {}

    def _step(self, action):
        return ds.TimeStep(
            step_type=ds.StepType.MID,
            prev_action=action,
            reward=np.float32(1.0),
            observation=np.float32(get_progress("global_counter")),
            env_id=self._env_id,
            discount=np.float32(1.0))

    def seed(self, seed):
        pass

    def _reset(self):
        return ds.TimeStep(
            step_type=ds.StepType.MID,
            prev_action=np.float32(0.0),
            reward=np.float32(1.0),
            observation=np.float32(get_progress("global_counter")),
            env_id=self._env_id,
            discount=np.float32(1.0))


def slow_env_load(observation_spec,
                  action_spec,
                  env_name,
                  env_id=0,
                  time_sleep=0,
                  reset_sleep=0):
    return SlowStartingEnvironment(
        observation_spec,
        action_spec,
        env_id=env_id,
        max_duration=1,
        time_sleep=time_sleep,
        reset_sleep=reset_sleep,
        use_tensor_time_step=False)


def _batch_env_ctor(env_ctor, batch_size_per_env, env_id):
    envs = [
        env_ctor(env_id=env_id * batch_size_per_env + i)
        for i in range(batch_size_per_env)
    ]
    return BatchEnvironmentWrapper(envs)


class ParallelAlfEnvironmentTest(parameterized.TestCase, alf.test.TestCase):
    def setUp(self):
        self._parallel_environment_ctor = parallel_environment.ParallelAlfEnvironment

    def _set_default_specs(self):
        self.observation_spec = ts.TensorSpec((3, 3), torch.float32)
        self.action_spec = ts.BoundedTensorSpec([7],
                                                dtype=torch.float32,
                                                minimum=-1.0,
                                                maximum=1.0)
        self.time_step_spec = ds.time_step_spec(self.observation_spec,
                                                self.action_spec,
                                                ts.TensorSpec(()))

    def _make_parallel_environment(self,
                                   constructor=None,
                                   num_envs=2,
                                   batch_size_per_env=1,
                                   flatten=True,
                                   start_serially=True,
                                   blocking=True):
        self._set_default_specs()
        constructor = constructor or functools.partial(
            RandomAlfEnvironment, self.observation_spec, self.action_spec)
        if batch_size_per_env > 1:
            constructor = functools.partial(_batch_env_ctor, constructor,
                                            batch_size_per_env)
        num_envs //= batch_size_per_env
        env = self._parallel_environment_ctor(
            env_constructors=[constructor] * num_envs,
            blocking=blocking,
            flatten=flatten,
            start_serially=start_serially)
        env.seed(list(range(num_envs)))
        return env

    def test_close_no_hang_after_init(self):
        env = self._make_parallel_environment()
        env.close()

    def test_get_specs(self):
        env = self._make_parallel_environment()
        self.assertEqual(self.observation_spec, env.observation_spec())
        self.assertEqual(self.time_step_spec, env.time_step_spec())
        self.assertEqual(self.action_spec, env.action_spec())

        env.close()

    def test_step(self):
        num_envs = 2
        env = self._make_parallel_environment(num_envs=num_envs)

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

    def test_non_blocking_reset_with_spare_envs(self):
        num_envs = 2
        sleep_time = 1.
        self._set_default_specs()

        start_t = time.time()
        env = alf.environments.utils.create_environment(
            env_name="IgnoredName",
            env_load_fn=functools.partial(
                slow_env_load,
                self.observation_spec,
                self.action_spec,
                time_sleep=sleep_time,
                reset_sleep=sleep_time),
            parallel_environment_ctor=self._parallel_environment_ctor,
            num_parallel_environments=num_envs,
            start_serially=False,
            num_spare_envs=num_envs)
        init_t = time.time()
        init_time = init_t - start_t
        self.assertLessEqual(
            init_time,
            sleep_time * 2 - 0.5,
            msg=('Expected all processes to start together, '
                 'got {} wait time').format(init_time))

        time_step0 = env.reset()
        assert torch.all(time_step0.step_type == ds.StepType.FIRST)
        action_spec = env.action_spec()
        action = torch.stack([action_spec.sample() for _ in range(num_envs)])
        time_step1 = env.step(action)
        self.assertEqual(time_step0.observation.shape,
                         time_step1.observation.shape)
        # need to wait a little so that the spare env can have enough time
        # to reset
        time.sleep(sleep_time)
        step1_t = time.time()
        # This step internally calls reset, because episodes are of length 1.
        time_step2 = env.step(action)
        reset_t = time.time()
        reset_time = reset_t - step1_t
        self.assertEqual(time_step1.observation.shape,
                         time_step2.observation.shape)
        assert torch.all(time_step2.env_id < num_envs)
        self.assertLessEqual(
            reset_time,
            sleep_time - 0.1,
            msg=(f'Reset with spare envs took {reset_time}, too long'))
        time_step3 = env.step(action)
        step3_t = time.time()
        step3_time = step3_t - reset_t
        self.assertLessEqual(
            step3_time, 0.5, msg=(f'Regular step took {step3_time}, too long'))
        time_step4 = env.step(action)
        step4_t = time.time()
        step4_time = step4_t - step3_t
        self.assertLessEqual(
            step4_time,
            sleep_time - 0.1,
            msg=(f'Step with spare envs took {step4_time}, too long'))
        time_step = env.step(action)  # reset is called here
        time.sleep(sleep_time)
        step5_t = time.time()
        # should be fast due to reset being called before sleep
        time_step = env.step(action)
        step5_time = time.time() - step5_t
        self.assertLessEqual(
            step5_time,
            sleep_time - 0.1,
            msg=(f'Reset already called, took {step5_time}, too long'))
        env.close()

        env = alf.environments.utils.create_environment(
            env_name="IgnoredName",
            env_load_fn=functools.partial(
                slow_env_load,
                self.observation_spec,
                self.action_spec,
                time_sleep=0,
                reset_sleep=sleep_time),
            parallel_environment_ctor=self._parallel_environment_ctor,
            num_parallel_environments=num_envs,
            start_serially=False,
            num_spare_envs=0)
        time_step0 = env.reset()
        time_step1 = env.step(action)
        time.sleep(sleep_time)
        start_t = time.time()
        time_step2 = env.step(action)
        reset_time = time.time() - start_t
        self.assertLessEqual(
            reset_time,
            sleep_time - 0.1,
            msg=(f'Without spare env, Reset already called, '
                 'took {reset_time}, too long'))
        # make sure promises are properly cleaned up
        time_step3 = env.step(action)
        env.close()

    def test_non_blocking_start_processes_in_parallel(self):
        self._set_default_specs()
        constructor = functools.partial(
            SlowStartingEnvironment,
            self.observation_spec,
            self.action_spec,
            time_sleep=1.0)
        start_time = time.time()
        env = self._make_parallel_environment(
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
        env = self._make_parallel_environment(
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
        env = self._make_parallel_environment(num_envs=num_envs, flatten=False)
        action_spec = env.action_spec()
        batched_action = torch.stack(
            [action_spec.sample() for _ in range(num_envs)])

        # Test that actions are correctly unstacked when just batched in np.array.
        unstacked_actions = env._unstack_actions(batched_action)
        for action in unstacked_actions:
            self.assertEqual(action_spec.shape, action.shape)
        env.close()

    def test_unstack_nested_actions(self):
        num_envs = 2
        env = self._make_parallel_environment(num_envs=num_envs, flatten=False)
        action_spec = env.action_spec()
        batched_action = torch.stack(
            [action_spec.sample() for _ in range(num_envs)])

        # Test that actions are correctly unstacked when nested in namedtuple.
        class NestedAction(
                collections.namedtuple('NestedAction',
                                       ['action', 'other_var'])):
            pass

        nested_action = NestedAction(
            action=batched_action, other_var=torch.tensor([13.0] * num_envs))
        unstacked_actions = env._unstack_actions(nested_action)
        for nested_action in unstacked_actions:
            self.assertEqual(action_spec.shape, nested_action.action.shape)
            self.assertEqual(13.0, nested_action.other_var)
        env.close()

    def test_seedable(self):
        seeds = [0, 1]
        env = self._make_parallel_environment()
        env.seed(seeds)
        self.assertEqual(
            np.random.RandomState(0).get_state()[1][-1],
            env._envs[0]._rng.get_state()[1][-1])

        self.assertEqual(
            np.random.RandomState(1).get_state()[1][-1],
            env._envs[1]._rng.get_state()[1][-1])
        env.close()


class FastParallelEnvironmentTest(ParallelAlfEnvironmentTest):
    def setUp(self):
        self._parallel_environment_ctor = FastParallelEnvironment

    def test_unstack_actions(self):
        # FastParallelEnvironment does not have _unstack_actions
        pass

    def test_unstack_nested_actions(self):
        # FastParallelEnvironment does not have _unstack_actions
        pass

    def _check_same_nest(self, nest1, nest2):
        alf.nest.assert_same_structure(nest1, nest2)

        def _same_tensor(path, t1, t2):
            self.assertTrue((t1 == t2).all(),
                            f"different value at path '{path}': {t1} vs. {t2}")

        alf.nest.py_map_structure_with_path(_same_tensor, nest1, nest2)

    def test_fast_parallel_environment(self):
        self._parallel_environment_ctor = parallel_environment.ParallelAlfEnvironment
        env1 = self._make_parallel_environment()
        self._parallel_environment_ctor = FastParallelEnvironment
        env2 = self._make_parallel_environment()
        ts1 = env1.reset()
        ts2 = env2.reset()
        for _ in range(100):
            ts1 = env1.step(ts1.prev_action)
            ts2 = env2.step(ts2.prev_action)
            self._check_same_nest(ts1, ts2)
        env1.close()
        env2.close()

    def test_fast_parallel_environment2(self):
        # test batch_size_per_env > 1
        self._parallel_environment_ctor = parallel_environment.ParallelAlfEnvironment
        env1 = self._make_parallel_environment(num_envs=6)
        self._parallel_environment_ctor = FastParallelEnvironment
        env2 = self._make_parallel_environment(
            num_envs=6, batch_size_per_env=2)
        ts1 = env1.reset()
        ts2 = env2.reset()
        self.assertEqual(env1.batch_size, env2.batch_size)
        for _ in range(100):
            ts1 = env1.step(ts1.prev_action)
            ts2 = env2.step(ts2.prev_action)
            self._check_same_nest(ts1, ts2)
        env1.close()
        env2.close()

    @parameterized.parameters((FastParallelEnvironment, ),
                              (parallel_environment.ParallelAlfEnvironment, ))
    def test_sync_progress(self, parallel_environment_ctor):
        self._parallel_environment_ctor = parallel_environment_ctor
        env = self._make_parallel_environment(
            constructor=ProgressEnvironment, num_envs=6, batch_size_per_env=2)
        ts = env.reset()
        for i in range(10):
            update_progress("global_counter", i)
            env.sync_progress()
            ts = env.step(ts.prev_action)
            self.assertTrue((ts.observation == i).all())
        for i in range(10, 20):
            update_progress("global_counter", i)
            ts = env.step(ts.prev_action)
            self.assertTrue((ts.observation == 9).all())
        env.close()


if __name__ == '__main__':
    alf.test.main()
