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

import numpy as np

from rlbench.tasks.reach_target import ReachTarget


class ReachTargetDense(ReachTarget):
    """Setup is the same with the original ``ReachTarget`` task, but with a dense
    reward as the decrement of the distance between the gripper tip and the target
    object.
    """

    def __init__(self, *args, **kwargs):
        super(ReachTargetDense, self).__init__(*args, **kwargs)
        # This hack is needed to reuse the .ttm file of ``ReachTarget``
        # (sparse reward) for the scene
        self.name = "reach_target"

    def init_episode(self, index):
        super().init_episode(index)
        self._prev_distance = None

    def reward(self):
        tip = self.robot.arm.get_tip()
        target_pos = self.target.get_position(relative_to=tip)
        distance = np.linalg.norm(target_pos)
        if self._prev_distance is None:
            reward = 0
        else:
            reward = self._prev_distance - distance
        self._prev_distance = distance
        # additional big reward when success
        #reward += float(self.success()[0]) * 0.1
        return reward
