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

from typing import Union

import numpy as np

from gym.envs.robotics.fetch_env import FetchEnv
from gym.envs.robotics import robot_env, rotations, utils


class AdvFetchEnv(FetchEnv):
    def __init__(
            self,
            model_path: str,
            n_substeps: int,
            gripper_extra_height: float,
            block_gripper: bool,
            has_object: bool,
            target_in_the_air: bool,
            target_offset: Union[float, np.ndarray],
            obj_range: float,
            target_range: float,
            distance_threshold: float,
            initial_qpos: dict,
            reward_type: str,
    ):
        """Class copied from OpenAI ``FetchEnv``. Almost the same with ``FetchEnv``.
        The only change is from 4 to 7 for ``n_actions`` where the extra 3 dims
        represent 'xyz' euler angles rotated compared to the previous step.

        The updated action dims: 0-2 for position control, 3-5 for euler angle control,
        and 6 for gripper openness control.

        .. note::

            After adding the rotation angles, the robot gripper might more easily
            have collisions with the table or itself. Maybe in the future we should
            have the environment return collision info.

            You can overwrite the function `_sample_goal()` to define new goals
            given these extra rotation actions, also `_get_obs()` to redefine the
            achieved goal.

        Args:
            model_path: path to the environments XML file
            n_substeps: number of substeps the simulation runs on every call to step
            gripper_extra_height: additional height above the table when positioning the gripper
            block_gripper: whether or not the gripper is blocked (i.e. not movable) or not
            has_object: whether or not the environment has an object
            target_in_the_air: whether or not the target should be in the air above the table or on the table surface
            target_offset: offset of the target
            obj_range: range of a uniform distribution for sampling initial object positions
            target_range: range of a uniform distribution for sampling a target
            distance_threshold: the threshold after which a goal is considered achieved
            initial_qpos: a dictionary of joint names and values that define the initial configuration
            reward_type: the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self._quat = None

        robot_env.RobotEnv.__init__(
            self,
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=7,
            initial_qpos=initial_qpos)

    def _reset_sim(self):
        # reset to the upright orientation
        self._quat = np.array([1., 0., 1., 0.])
        return super()._reset_sim()

    def _set_action(self, action):
        assert action.shape == (7, )
        action = action.copy(
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, euler_angles, gripper_ctrl = action[:3], action[3:6], action[
            -1]

        pos_ctrl *= 0.05  # limit maximum change in position
        # first downscale the euler angle to limit the max rotation of each step
        self._quat = rotations.quat_mul(
            rotations.euler2quat(euler_angles * np.pi * 0.05), self._quat)

        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2, )
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, self._quat, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        """We should also return the gripper's orientation.
        """
        obs = super()._get_obs()
        gripper_rot = rotations.mat2euler(
            self.sim.data.get_site_xmat('robot0:grip'))
        obs['observation'] = np.concatenate(
            [obs['observation'], gripper_rot.ravel()])
        return obs
