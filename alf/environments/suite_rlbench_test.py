# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from gym import spaces
import numpy as np

import alf
from alf.environments import suite_rlbench


class SuiteRLBenchTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_rlbench.is_available():
            self.skipTest("suite_rlbench is not available")

    def test_simulation_timestep(self):
        env = suite_rlbench.load("reach_target-state-v0")
        # default simulation timestep
        self.assertAlmostEqual(env.pyrep.get_simulation_timestep(), 0.05)
        env.close()

    def test_observation_configuration(self):
        env = suite_rlbench.load('reach_target-state-v0')
        state = env.reset().observation
        # state should be a flat vector
        self.assertFalse(isinstance(state, dict))
        env.close()

        env = suite_rlbench.load('reach_target-vision-v0')
        obs = env.reset().observation
        self.assertTrue(isinstance(obs, dict))
        # the default 'vision' mode will return five rgb images
        expected_obs_keys = [
            'left_shoulder_rgb', 'right_shoulder_rgb', 'overhead_rgb',
            'wrist_rgb', 'front_rgb', 'proprioceptive_state'
        ]
        self.assertEqual(set(obs.keys()), set(expected_obs_keys))
        self.assertEqual(obs['wrist_rgb'].dtype, np.uint8)
        env.close()

    def test_custom_observation_config(self):
        # We customize the observation so that only left shoulder camera returns
        # a depth image, along with the state vector
        off_camera_config = suite_rlbench.CameraConfig(
            rgb=False, depth=False, mask=False, point_cloud=False)
        obs_config = suite_rlbench.ObservationConfig(
            left_shoulder_camera=suite_rlbench.CameraConfig(
                rgb=False, depth=True, mask=False, point_cloud=False),
            right_shoulder_camera=off_camera_config,
            overhead_camera=off_camera_config,
            wrist_camera=off_camera_config,
            front_camera=off_camera_config,
            task_low_dim_state=True)
        env = suite_rlbench.load(
            'reach_target-v0', observation_config=obs_config)
        obs = env.reset().observation
        self.assertTrue(isinstance(obs, dict))
        expected_keys = {
            'left_shoulder_depth', 'proprioceptive_state', 'task_state'
        }
        self.assertEqual(set(obs.keys()), )
        self.assertEqual(obs['left_shoulder_depth'].dtype, np.float32)
        env.close()

    def test_low_dim_observation_config(self):
        obs_config = suite_rlbench.ObservationConfig()
        obs_config.set_all_high_dim(False)
        env = suite_rlbench.load(
            'push_buttons-v0', observation_config=obs_config)
        state1 = env.reset().observation['proprioceptive_state']
        env.close()

        # By default, ``gripper_open`` is turned on
        obs_config = suite_rlbench.ObservationConfig(gripper_open=False)
        # Turn off all high-dim observations
        obs_config.set_all_high_dim(False)
        env = suite_rlbench.load(
            'push_buttons-v0', observation_config=obs_config)
        state2 = env.reset().observation['proprioceptive_state']
        env.close()

        self.assertEqual(len(state1), len(state2) + 1)

    def test_multi_parallel_envs_seeds(self):
        env = alf.environments.utils.create_environment(
            'reach_target-state-v0',
            env_load_fn=suite_rlbench.load,
            num_parallel_environments=2)

        time_step = env.reset()
        self.assertTensorNotClose(time_step.observation[0],
                                  time_step.observation[1])

        env.seed([100, 100])
        time_step = env.reset()
        self.assertTensorClose(time_step.observation[0],
                               time_step.observation[1])

        env.close()


if __name__ == "__main__":
    alf.test.main()
