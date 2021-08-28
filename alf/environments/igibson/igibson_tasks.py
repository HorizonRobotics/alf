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

import igibson
from igibson.tasks.task_base import BaseTask
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.timeout import Timeout
from igibson.termination_conditions.out_of_bound import OutOfBound
from igibson.termination_conditions.point_goal import PointGoal
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from igibson.objects.visual_marker import VisualMarker

import numpy as np
import os
import time
import logging
import pybullet as p

from alf.environments.igibson.igibson_object import iGibsonObject


class VisualPointNavFixedTask(BaseTask):
    """Fixed Point Navigation Task.

    The goal is to navigate to a fixed highlighted goal position in the stadium scene.
    """

    def __init__(self, env, target_pos=[5, 5, 0], target_pos_vis_obj=None):
        """
        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
            target_pos (list[float]): x, y, z coordinates of the target position
            target_pos_vis_obj (igibson.objects.object_base.Object): the visual object
                at the target position
        """
        super(VisualPointNavFixedTask, self).__init__(env)
        self.reward_type = self.config.get('reward_type', 'l2')
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            PointGoal(self.config),
        ]
        self.reward_functions = [
            PotentialReward(self.config),
            CollisionReward(self.config),
            PointGoalReward(self.config),
        ]

        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))
        self.target_pos = np.array(self.config.get('target_pos', target_pos))
        self.goal_format = self.config.get('goal_format', 'polar')
        self.dist_tol = self.termination_conditions[-1].dist_tol

        self.visual_object_at_initial_target_pos = self.config.get(
            'visual_object_at_initial_target_pos', True)
        self.target_visual_object_visible_to_agent = self.config.get(
            'target_visual_object_visible_to_agent', False)
        self.target_pos_vis_obj = target_pos_vis_obj
        if self.target_visual_object_visible_to_agent:
            env.simulator.import_object(self.target_pos_vis_obj)
        else:
            self.target_pos_vis_obj.load()

        # adjust z position of the object so that it's above ground
        self.z_offset = 0
        zmin = p.getAABB(self.target_pos_vis_obj.body_id)[0][2]
        if zmin < 0:
            self.z_offset = -zmin
            x, y, z = self.target_pos
            self.target_pos = np.array([x, y, z + self.z_offset])
            self.target_pos_vis_obj.set_position(self.target_pos)

        self.floor_num = 0
        self.load_visualization(env)

    def load_visualization(self, env):
        """Load visualization, including the object at the initial position and the shortest path.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        if env.mode != 'gui':
            return

        # turn off multiview GUIs
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        cyl_length = 0.2
        self.initial_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[1, 0, 0, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0])
        self.initial_pos_vis_obj.load()

        if env.scene.build_graph:
            self.num_waypoints_vis = 250
            self.waypoints_vis = [
                VisualMarker(
                    visual_shape=p.GEOM_CYLINDER,
                    rgba_color=[0, 1, 0, 0.3],
                    radius=0.1,
                    length=cyl_length,
                    initial_offset=[0, 0, cyl_length / 2.0])
                for _ in range(self.num_waypoints_vis)
            ]
            for waypoint in self.waypoints_vis:
                waypoint.load()

    def get_geodesic_potential(self, env):
        """Get potential based on geodesic distance.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance

        Returns:
            float: geodesic distance to the target position.
        """
        _, geodesic_dist = self.get_shortest_path(env)
        return geodesic_dist

    def get_l2_potential(self, env):
        """Get potential based on L2 distance.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance

        Returns:
            float: L2 distance to the target position
        """
        return l2_distance(env.robots[0].get_position()[:2],
                           self.target_pos[:2])

    def get_potential(self, env):
        """Compute task-specific potential: distance to the goal.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance

        Returns:
            float: task potential
        """
        if self.reward_type == 'l2':
            return self.get_l2_potential(env)
        elif self.reward_type == 'geodesic':
            return self.get_geodesic_potential(env)

    def reset_scene(self, env):
        """Task-specific scene reset: reset scene objects or floor plane.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)

    def reset_agent(self, env):
        """Task-specific agent reset: land the robot to initial pose, compute initial potential.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        env.land(env.robots[0], self.initial_pos, self.initial_orn)
        self.path_length = 0.0
        self.robot_pos = self.initial_pos[:2]
        self.target_pos_vis_obj.set_position(self.target_pos)
        self.geodesic_dist = self.get_geodesic_potential(env)
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """Aggreate termination conditions and fill info.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
            collision_links (list[int]): collision links after executing action
            action (tuple(float)): the executed action
            info (dict): additional info
        """
        done, info = super(VisualPointNavFixedTask, self).get_termination(
            env, collision_links, action, info)

        info['path_length'] = self.path_length
        if done:
            info['spl'] = float(info['success']) * \
                min(1.0, self.geodesic_dist / self.path_length)
        else:
            info['spl'] = 0.0

        return done, info

    def global_to_local(self, env, pos):
        """Convert a 3D point in global frame to agent's local frame.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
            pos (numpy.ndarray): a 3D point in global frame
        Returns:
            numpy.ndarray: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(pos - env.robots[0].get_position(),
                                *env.robots[0].get_rpy())

    def get_task_obs(self, env):
        """Get task-specific observation, including goal position, current velocities, etc.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        Returns:
            numpy.ndarray: task-specific observation
        """
        task_obs = self.global_to_local(env, self.target_pos)[:2]
        if self.goal_format == 'polar':
            task_obs = np.array(cartesian_to_polar(task_obs[0], task_obs[1]))

        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(),
                                           *env.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(
            env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
        task_obs = np.append(task_obs, [linear_velocity, angular_velocity])

        return task_obs

    def get_shortest_path(self, env, from_initial_pos=False,
                          entire_path=False):
        """Get the shortest path and geodesic distance.

        Get the shortest path and geodesic distance from the robot or
        the initial position to the target position.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
            from_initial_pos (bool): whether source is initial position rather than current position
            entire_path (bool): whether to return the entire shortest path
        Returns:
            shortest path and geodesic distance to the target position:
            - shortest path (numpy.ndarray): shortest path as a sequence of waypoints
            - geodesic distance (float): geodesic distance
        """
        if from_initial_pos:
            source = self.initial_pos[:2]
        else:
            source = env.robots[0].get_position()[:2]
        target = self.target_pos[:2]
        return env.scene.get_shortest_path(
            self.floor_num, source, target, entire_path=entire_path)

    def step_visualization(self, env):
        """Step visualization.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        if env.mode != 'gui':
            return

        self.initial_pos_vis_obj.set_position(self.initial_pos)
        self.target_pos_vis_obj.set_position(self.target_pos)

        if env.scene.build_graph:
            shortest_path, _ = self.get_shortest_path(env, entire_path=True)
            floor_height = env.scene.get_floor_height(self.floor_num)
            num_nodes = min(self.num_waypoints_vis, shortest_path.shape[0])
            for i in range(num_nodes):
                self.waypoints_vis[i].set_position(
                    pos=np.array([
                        shortest_path[i][0], shortest_path[i][1], floor_height
                    ]))
            for i in range(num_nodes, self.num_waypoints_vis):
                self.waypoints_vis[i].set_position(
                    pos=np.array([0.0, 0.0, 100.0]))

    def step(self, env):
        """Perform task-specific step: step visualization and aggregate path length.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        self.step_visualization(env)
        new_robot_pos = env.robots[0].get_position()[:2]
        self.path_length += l2_distance(self.robot_pos, new_robot_pos)
        self.robot_pos = new_robot_pos


class VisualPointNavRandomTask(VisualPointNavFixedTask):
    """Random Point Navigation Task.

    The goal is to navigate to a random goal position from a random initial position.
    """

    def __init__(self, env, target_pos_vis_obj=None):
        """
        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
            target_pos_vis_obj (igibson.objects.object_base.Object): an instance of igibson.objects
        """
        super(VisualPointNavRandomTask, self).__init__(
            env=env, target_pos_vis_obj=target_pos_vis_obj)
        self.target_dist_min = self.config.get('target_dist_min', 1.0)
        self.target_dist_max = self.config.get('target_dist_max', 10.0)

    def sample_initial_pose_and_target_pos(self, env):
        """Sample robot initial pose and target position.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        Returns:
            initial and target pose information:
            - initial_pos (numpy.ndarray): initial position
            - initial_orn (numpy.ndarray): initial orientation
            - target_pos (numpy.ndarray): target position
        """
        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        max_trials = 100
        dist = 0.0
        for _ in range(max_trials):
            _, target_pos = env.scene.get_random_point(floor=self.floor_num)
            if env.scene.build_graph:
                _, dist = env.scene.get_shortest_path(
                    self.floor_num,
                    initial_pos[:2],
                    target_pos[:2],
                    entire_path=False)
            else:
                dist = l2_distance(initial_pos, target_pos)
            if self.target_dist_min < dist < self.target_dist_max:
                break
        if not (self.target_dist_min < dist < self.target_dist_max):
            print("WARNING: Failed to sample initial and target positions")
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn, target_pos

    def reset_scene(self, env):
        """Task-specific scene reset: get a random floor number first.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        self.floor_num = env.scene.get_random_floor()
        super(VisualPointNavRandomTask, self).reset_scene(env)

    def reset_agent(self, env):
        """Reset robot initial pose.

        Sample initial pose and target position, check validity, and land it.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        reset_success = False
        max_trials = 100

        # cache pybullet state
        state_id = p.saveState()
        for i in range(max_trials):
            initial_pos, initial_orn, target_pos = \
                self.sample_initial_pose_and_target_pos(env)
            reset_success = env.test_valid_position(
                env.robots[0], initial_pos, initial_orn) and \
                env.test_valid_position(
                    env.robots[0], target_pos)
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        p.removeState(state_id)

        x, y, z = target_pos
        self.target_pos = np.array([x, y, z + self.z_offset])
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn

        super(VisualPointNavRandomTask, self).reset_agent(env)


class VisualObjectNavTask(BaseTask):
    """Object Navigation Task.

    The goal is to navigate to one of the many loaded objects given object name
    from a random initial position.
    """

    def __init__(self, env):
        """
        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        super(VisualObjectNavTask, self).__init__(env)
        self.env = env

        # minimum distance between object and initial robot position
        self.object_dist_min = self.config.get('object_dist_min', 1.0)
        # maximum distance between object and initial robot position
        self.object_dist_max = self.config.get('object_dist_max', 10.0)

        self.reward_type = self.config.get('reward_type', 'l2')
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            PointGoal(self.config),
        ]
        self.reward_functions = [
            PotentialReward(self.config),
            CollisionReward(self.config),
            PointGoalReward(self.config),
        ]

        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))
        self.goal_format = self.config.get('goal_format', 'polar')
        # additional tolerance for goal checking
        self.goal_buffer_dist = self.config.get('goal_buffer_dist', 0.5)
        # minimum distance between objects when sampling object position
        self.object_keepout_buffer_dist = self.config.get(
            'object_keepout_buffer_dist', 0.5)
        self.visual_object_at_initial_pos = self.config.get(
            'visual_object_at_initial_pos', True)
        self.floor_num = 0
        self.num_objects = self.config.get('num_objects', 5)
        self.object_randomization_freq = self.config.get(
            'object_randomization_freq', None)
        self.initialize_scene_objects()
        self.load_visualization(env)
        self.reset_time_vars()

    def reset_time_vars(self):
        """Reset/Initialize time related variables."""
        self.start_time = time.time()
        self.reset_time = time.time()
        self.episode_time = time.time()

    def initialize_scene_objects(self):
        """Initialize objects in the scene."""

        # get object names
        all_object_names = [
            os.path.basename(f.path) for f in os.scandir(
                os.path.join(igibson.ig_dataset_path, 'objects'))
            if f.is_dir()
        ]
        assert self.num_objects <= len(all_object_names)
        if self.object_randomization_freq is not None:
            self.object_names = np.random.choice(all_object_names,
                                                 self.num_objects)
        else:
            if self.num_objects == 5:
                self.object_names = [
                    'standing_tv', 'piano', 'office_chair', 'toilet',
                    'speaker_system'
                ]
            else:
                self.object_names = all_object_names[-self.num_objects:]

        # load objects into scene and save their info
        max_radius = 0.0
        self.object_dict = {}
        self.object_pos_dict = {}
        self.object_orn_dict = {}
        self.object_id_dict = {}
        for object_name in self.object_names:
            self.object_dict[object_name] = iGibsonObject(name=object_name)
            pybullet_id = self.env.simulator.import_object(
                self.object_dict[object_name])
            self.object_pos_dict[object_name] = self.object_dict[
                object_name].get_position()
            self.object_orn_dict[object_name] = self.object_dict[
                object_name].get_orientation()
            self.object_id_dict[object_name] = pybullet_id
            # get object max radius from bounding box
            (xmax, ymax, _), (xmin, ymin, _) = p.getAABB(pybullet_id)
            object_max_radius = max(
                abs(xmax - xmin) / 2.,
                abs(ymax - ymin) / 2.)
            max_radius = max(object_max_radius, max_radius)
        self.max_radius = max_radius

        # update distance tolerance
        self.dist_tol = self.max_radius + self.goal_buffer_dist
        self.termination_conditions[-1].dist_tol = self.dist_tol
        self.reward_functions[-1].dist_tol = self.dist_tol
        self.object_dist_keepout = self.max_radius * 2 + self.object_keepout_buffer_dist

        self.sample_goal_object()

    def sample_goal_object(self):
        """Sample goal object and save goal object info."""
        goal_object_idx = np.random.randint(self.num_objects)
        self.target_name = self.object_names[goal_object_idx]
        self.target_object = self.object_dict[self.target_name]
        self.target_pos = self.object_pos_dict[self.target_name]
        self.target_orn = self.object_orn_dict[self.target_name]
        # one-hot encoding
        self.target_obs = np.eye(self.num_objects)[goal_object_idx]

    def sample_initial_pose_and_object_pos(self, env):
        """Sample robot initial pose and object positions.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance

        Returns:
            robot initial pose and object positions
            - initial_pos (numpy.ndarray): robot initial position
            - initial_orn (numpy.ndarray): robot initial orientation
            - object_pos_dict (dict): dictionary of object positions
            - object_orn_dict (dict): dictionary of object orientations
        """
        object_pos_dict = {}
        object_orn_dict = {}

        def placement_is_valid(pos, initial_pos):
            """Test if an object position is valid."""
            dist = l2_distance(pos, initial_pos)
            if dist < self.object_dist_min or dist > self.object_dist_max:
                return False
            for object_name, object_pos in object_pos_dict.items():
                dist = l2_distance(pos, object_pos)
                if dist < self.object_dist_keepout:
                    return False
            return True

        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        max_trials = 500
        for object_name in self.object_names:
            object_orn_dict[object_name] = np.array(
                [0, 0, np.random.uniform(0, np.pi * 2)])
            for _ in range(max_trials):
                _, object_pos = env.scene.get_random_point(
                    floor=self.floor_num)
                valid_pos = placement_is_valid(object_pos, initial_pos)
                if valid_pos:
                    object_pos_dict[object_name] = object_pos
                    break
            if not valid_pos:
                print(
                    f"WARNING: Failed to sample valid position for {object_name}"
                )

        return initial_pos, initial_orn, object_pos_dict, object_orn_dict

    def load_visualization(self, env):
        """Load visualization, such as initial and target position, shortest path, etc.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        if env.mode != 'gui':
            return

        # Turn off multi-view GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # load visual object at initial position
        cyl_length = 0.2
        self.initial_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[1, 0, 0, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0])
        self.initial_pos_vis_obj.load()

        # load text at target position
        x, y, z = self.target_pos
        p.addUserDebugText(
            text=f'Target: {self.target_name}',
            textPosition=[x, y, z + 2],
            textSize=2)

        if env.scene.build_graph:
            self.num_waypoints_vis = 250
            self.waypoints_vis = [
                VisualMarker(
                    visual_shape=p.GEOM_CYLINDER,
                    rgba_color=[0, 1, 0, 0.3],
                    radius=0.1,
                    length=cyl_length,
                    initial_offset=[0, 0, cyl_length / 2.0])
                for _ in range(self.num_waypoints_vis)
            ]
            for waypoint in self.waypoints_vis:
                waypoint.load()

    def get_geodesic_potential(self, env):
        """Get potential based on geodesic distance.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance

        Returns:
            float: geodesic distance to the target position
        """
        _, geodesic_dist = self.get_shortest_path(env)
        return geodesic_dist

    def get_l2_potential(self, env):
        """Get potential based on L2 distance.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        Returns:
            float: L2 distance to the target position
        """
        return l2_distance(env.robots[0].get_position()[:2],
                           self.target_pos[:2])

    def get_potential(self, env):
        """Compute task-specific potential: distance to the goal.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance

        Returns:
            float: task potential
        """
        if self.reward_type == 'l2':
            return self.get_l2_potential(env)
        else:
            raise ValueError(f'Invalid reward type: {self.reward_type}')

    def reset_scene(self, env):
        """Task-specific scene reset: get a random floor number first.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        self.floor_num = env.scene.get_random_floor()
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)

    def reset_agent(self, env):
        """Reset robot initial pose.

        Sample initial pose and target position, check validity, and land it.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        self.episode_time = time.time()
        logging.info(
            f'Episode time: {self.episode_time - self.start_time:.5f} | '
            f'Reset time: {self.reset_time - self.start_time:.5f}')
        self.reset_time_vars()

        reset_success = False
        max_trials = 100

        # cache pybullet state
        state_id = p.saveState()
        for i in range(max_trials):
            initial_pos, initial_orn, object_pos_dict, object_orn_dict = self.sample_initial_pose_and_object_pos(
                env)
            reset_success = env.test_valid_position(env.robots[0], initial_pos,
                                                    initial_orn)
            for object_name, object in self.object_dict.items():
                reset_success &= env.test_valid_position(
                    object, object_pos_dict[object_name],
                    object_orn_dict[object_name])
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        p.removeState(state_id)

        self.object_pos_dict = object_pos_dict
        self.object_orn_dict = object_orn_dict
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn

        # land robot and objects
        env.land(env.robots[0], self.initial_pos, self.initial_orn)
        self.path_length = 0.0
        self.robot_pos = self.initial_pos[:2]
        for object_name, object in self.object_dict.items():
            env.land(object, self.object_pos_dict[object_name],
                     self.object_orn_dict[object_name])

        self.geodesic_dist = self.get_geodesic_potential(env)
        self.sample_goal_object()
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

        if env.mode == 'gui':
            p.removeAllUserDebugItems()
            x, y, z = self.target_pos
            p.addUserDebugText(
                text=f'Target: {self.target_name}',
                textPosition=[x, y, z + 2],
                textSize=2)

        self.reset_time = time.time()

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """Aggreate termination conditions and fill info.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
            collision_links (list[int]): collision links after executing action
            action (tuple(float)): the executed action
            info (dict): additional info
        Returns:
            episode termination and info
            - done (bool): whether the episode has terminated
            - info (dict): additional info
        """
        done, info = super(VisualObjectNavTask, self).get_termination(
            env, collision_links, action, info)

        info['path_length'] = self.path_length
        if done:
            info['spl'] = float(info['success']) * \
                min(1.0, self.geodesic_dist / self.path_length)
        else:
            info['spl'] = 0.0

        return done, info

    def get_task_obs(self, env):
        """Get task-specific observation, including goal position, current velocities, etc.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        Returns:
            numpy.ndarray: task-specific observation
        """
        return self.target_obs

    def get_shortest_path(self, env, from_initial_pos=False,
                          entire_path=False):
        """Get the shortest path and geodesic distance.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
            from_initial_pos (bool): whether source is initial position rather than current position
            entire_path: whether to return the entire shortest path
        Returns:
            shortest path information:
            - path_world (numpy.ndarray): shortest path
            - geodesic_distance (float): geodesic distance to the target position
        """
        if from_initial_pos:
            source = self.initial_pos[:2]
        else:
            source = env.robots[0].get_position()[:2]
        target = self.target_pos[:2]
        return env.scene.get_shortest_path(
            self.floor_num, source, target, entire_path=entire_path)

    def step_visualization(self, env):
        """Step visualization.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        if env.mode != 'gui':
            return

        self.initial_pos_vis_obj.set_position(self.initial_pos)

        if env.scene.build_graph:
            shortest_path, _ = self.get_shortest_path(env, entire_path=True)
            floor_height = env.scene.get_floor_height(self.floor_num)
            num_nodes = min(self.num_waypoints_vis, shortest_path.shape[0])
            for i in range(num_nodes):
                self.waypoints_vis[i].set_position(
                    pos=np.array([
                        shortest_path[i][0], shortest_path[i][1], floor_height
                    ]))
            for i in range(num_nodes, self.num_waypoints_vis):
                self.waypoints_vis[i].set_position(
                    pos=np.array([0.0, 0.0, 100.0]))

    def step(self, env):
        """Perform task-specific step: step visualization and aggregate path length.

        Args:
            env (igibson.igibson_env.iGibsonCustomEnv): environment instance
        """
        self.step_visualization(env)
        new_robot_pos = env.robots[0].get_position()[:2]
        self.path_length += l2_distance(self.robot_pos, new_robot_pos)
        self.robot_pos = new_robot_pos
