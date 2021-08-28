# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from igibson.envs.igibson_env import iGibsonEnv
from igibson.tasks.room_rearrangement_task import RoomRearrangementTask
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from igibson.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from igibson.tasks.reaching_random_task import ReachingRandomTask
from alf.environments.igibson.igibson_tasks import VisualObjectNavTask


class iGibsonCustomEnv(iGibsonEnv):
    """iGibson Environment (OpenAI Gym interface)"""

    def __init__(
            self,
            config_file,
            scene_id=None,
            mode='headless',
            task=None,
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            device_idx=0,
            render_to_tensor=False,
            automatic_reset=False,
    ):
        """
        Args:
            config_file (str): config_file path
            scene_id (str): override scene_id in config file
            mode (str): headless, gui, iggui
            task: (str) task type
            action_timestep (float): environment executes action per action_timestep second
            physics_timestep (float): physics timestep for pybullet
            device_idx (int): which GPU to run the simulation and rendering on
            render_to_tensor (bool): whether to render directly to pytorch tensors
            automatic_reset (bool): whether to automatic reset after an episode finishes
        """
        self.task = task
        super(iGibsonCustomEnv, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor)

    def load_task_setup(self):
        """Load task setup"""

        self.initial_pos_z_offset = self.config.get('initial_pos_z_offset',
                                                    0.1)
        drop_distance = 0.5 * 9.8 * (self.action_timestep**2)
        assert drop_distance < self.initial_pos_z_offset, \
            'initial_pos_z_offset is too small for collision checking'

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(
            self.config.get('collision_ignore_body_b_ids', []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(
            self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get(
            'texture_randomization_freq', None)
        self.object_randomization_freq = self.config.get(
            'object_randomization_freq', None)

        # task
        if self.task is None:
            if self.config['task'] == 'point_nav_fixed':
                self.task = PointNavFixedTask(self)
            elif self.config['task'] == 'point_nav_random':
                self.task = PointNavRandomTask(self)
            elif self.config['task'] == 'interactive_nav_random':
                self.task = InteractiveNavRandomTask(self)
            elif self.config['task'] == 'dynamic_nav_random':
                self.task = DynamicNavRandomTask(self)
            elif self.config['task'] == 'reaching_random':
                self.task = ReachingRandomTask(self)
            elif self.config['task'] == 'room_rearrangement':
                self.task = RoomRearrangementTask(self)
            elif self.config['task'] == 'visual_object_nav':
                self.task = VisualObjectNavTask(self)
            else:
                self.task = None
