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
"""CarlaEnvironment suite.

To use this, there are two ways:

1. Run the code within docker image horizonrobotics/alf:0.0.3-carla
   Both `Docker <https://docs.docker.com/engine/install/ubuntu/>`_ and
   `Nvidia-Docker2 <https://github.com/NVIDIA/nvidia-docker>`_ need to be installed.

2. Install carla:

.. code-block:: bash

    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.9.tar.gz
    mkdir carla
    tar zxf CARLA_0.9.9.tar.gz -C carla
    cd carla/Import
    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.9.tar.gz
    cd ..
    ./ImportAssert.sh
    easy_install PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg
    pip install networkx==2.2

Make sure you are using python3.7

"""

from collections import OrderedDict
from absl import logging
import math
import numpy as np
import os
import random
import scipy.interpolate
import subprocess
import sys
import time
import torch
from unittest.mock import Mock
import weakref

try:
    import carla
except ImportError:
    # create 'carla' as a mock to not break python argument type hints
    carla = Mock()

import alf
import alf.data_structures as ds
from alf.utils import common
from .alf_environment import AlfEnvironment
from .carla_sensors import (
    BEVSensor, CameraSensor, CollisionSensor, GnssSensor, IMUSensor,
    LaneInvasionSensor, NavigationSensor, RadarSensor, RedlightSensor,
    DynamicObjectSensor, ObstacleDetectionSensor, World, get_scaled_image_size,
    MINIMUM_RENDER_WIDTH, MINIMUM_RENDER_HEIGHT)

from alf.environments.carla_env.carla_utils import (
    _calculate_relative_position, _calculate_relative_velocity, _get_self_pose,
    geo_distance, _to_numpy_loc)


def is_available():
    return not isinstance(carla, Mock)


class WeatherParameters(object):
    """A class for a set of weather related parameters. Currently it contains
    all the weather fields from ``carla.WeatherParameters`` except for
    ``sun_azimuth_angle`` and ``sun_altitude_angle``, which are controlled
    separately by ``day_length`` in a more realistic way.
    """

    def __init__(self,
                 cloudiness=0,
                 precipitation=0,
                 precipitation_deposits=0,
                 wind_intensity=0,
                 fog_density=0,
                 fog_distance=0):
        self.cloudiness = cloudiness  # [0, 100]
        self.precipitation = precipitation  # [0, 100]
        self.precipitation_deposits = precipitation_deposits  # [0, 100]
        self.wind_intensity = wind_intensity  # [0, 100]
        self.fog_density = fog_density  # [0, 100]
        self.fog_distance = fog_distance  # [0, 100]
        self._fields = [
            m for m in self.__dict__.keys() if not m.startswith('_')
        ]

    def get_weather_fields(self):
        """ Get the list of configurable weather fields

        Returns:
            A list of strings, each as the name of a configurable field
        """
        return self._fields

    def __add__(self, other):
        """ Add other to current parameters and return a new instance.

        Args:
            other (WeatherParameters)
        """
        new_param = type(self)()
        for m in self.get_weather_fields():
            setattr(new_param, m, getattr(self, m) + getattr(other, m))
        return new_param

    def __sub__(self, other):
        """ Subtract other from current parameters and return a new instance.

        Args:
            other (WeatherParameters)
        """
        new_param = type(self)()
        for m in self.get_weather_fields():
            setattr(new_param, m, getattr(self, m) - getattr(other, m))
        return new_param

    def __truediv__(self, value):
        """ Divide the current parameters by value and return a new instance.

        Args:
            value(float|int): a number to divide the parameters by
        """
        assert type(value) == int or type(value) == float
        value = float(value)
        new_param = type(self)()
        for m in self.get_weather_fields():
            setattr(new_param, m, getattr(self, m) / value)
        return new_param

    def __len__(self):
        """ Get the number of configurable weather fields
        """
        return len(self.get_weather_fields())

    def __str__(self):
        return ''.join([
            '{a}: {b} '.format(a=m, b=getattr(self, m))
            for m in self.get_weather_fields()
        ])


def extract_weather_parameters(weather_param: carla.WeatherParameters):
    """Extract the parameters according to the fields in ``WeatherParameters``
    and use them to construct an instance of ``WeatherParameters``.
    """
    wp = WeatherParameters()
    for m in wp.get_weather_fields():
        setattr(wp, m, getattr(weather_param, m))
    return wp


def adjust_weather_parameters(weather_param: carla.WeatherParameters,
                              delta: WeatherParameters):
    """Adjust the parameters of ``weather_param`` according to the fields
    in ``WeatherParameters``. The value is adjusted by adding the field
    value of ``delta`` to ``weather_param``.

    Args:
        weather_param (carla.WeatherParameters): a ``carla.WeatherParameters``
            instance containing the parameters to be adjusted
        delta (WeatherParameters): an instance of ``WeatherParameters`` with
            the value of each field representing the amount to be adjusted

    Returns:
        The input weather_param instance with adjusted field values.
    """
    for m in delta.get_weather_fields():
        setattr(weather_param, m,
                getattr(weather_param, m) + getattr(delta, m))
    return weather_param


@alf.configurable(blacklist=['actor', 'alf_world'])
class Player(object):
    """Player is a vehicle with some sensors.

    An episode terminates if it reaches one of the following situations:
    1. the vehicle arrives at the goal.
    2. the time exceeds ``route_length / min_speed + additional_time``.
    3. it get stuck because of a collision.

    At each step, the reward is given based on the following components:
    1. Arriving goal:  ``success_reward``
    2. Moving in the navigation direction: the number of meters moved
       This moving reward can be either dense of sparse depending on the argument
       ``sparse_reward``.
    3. Negative reward caused by collision: ``-min(max_collision_reward, max(epside_reward, 0))``

    Currently, the player has these sensors: ``CollisionSensor``, ``GnssSensor``,
    ``IMUSensor``, ``CameraSensor``, ``BEV_sensor``, ``LaneInvasionSensor``,
    ``RadarSensor``, ``NavigationSensor``. See the documentation for these class
    for the definition the data generated by these sensors.
    """

    # over all reward
    REWARD_OVERALL = 0

    # distance in meter for moving along route
    # If using sparse reward (`sparse_reward` is True), this reward is only given
    # about every `sparse_reward_interval` meters
    # If not using sparse reward, this reward is given every steps.
    REWARD_DISTANCE = 1

    # 0/1 valued indicating whether there is collision
    REWARD_COLLISION = 2

    # 0/1 valued indicating reaching goal
    REWARD_SUCCESS = 3

    # 0/1 valued indicating red light violation
    REWARD_RED_LIGHT = 4

    # 0/1 valued indicating overspeed
    REWARD_OVERSPEED = 5

    # dimension of the reward vector
    REWARD_DIMENSION = 6

    # See https://leaderboard.carla.org/#driving-score for reference
    PENALTY_RATE_COLLISION = 0.50
    PENALTY_RATE_RED_LIGHT = 0.30

    def __init__(self,
                 actor,
                 alf_world,
                 controller_ctor=None,
                 success_reward=100.,
                 success_distance_thresh=5.0,
                 max_collision_penalty=20.,
                 max_stuck_at_collision_seconds=5.0,
                 stuck_at_collision_distance=1.0,
                 max_red_light_penalty=10.,
                 overspeed_penalty_weight=0.,
                 sparse_reward=False,
                 sparse_reward_interval=10.,
                 allow_negative_distance_reward=True,
                 min_speed=5.,
                 additional_time=0.,
                 with_gnss_sensor=True,
                 with_imu_sensor=True,
                 with_camera_sensor=True,
                 with_radar_sensor=True,
                 with_bev_sensor=False,
                 with_dynamic_object_sensor=False,
                 data_collection_mode=False,
                 with_red_light_sensor=False,
                 with_obstacle_sensor=False,
                 terminate_upon_infraction="",
                 render_waypoints=True):
        """
        Args:
            actor (carla.Actor): the carla actor object
            alf_world (World): the world containing the player
            controller_ctor (Callable|None): if provided, will be as ``controller_ctor(vehicle, step_time)``
                to create a vehicle controller. It will be used to process the
                action and generate the control.
            success_reward (float): the reward for arriving the goal location.
            success_distance_thresh (float): success is achieved if the current
                location is with such distance of the goal
            max_collision_penalty (float): the maximum penalty (i.e. negative reward)
                for collision. We don't want the collision penalty to be too large
                if the player cannot even get enough positive moving reward. So the
                penalty is capped at ``Player.PENALTY_RATE_COLLISION * max(0., episode_reward))``.
                Note that this reward is only given once at the first step of
                contiguous collisions.
            max_stuck_at_collision_seconds (float): the episode will end and is
                considerred as failure if the car is stuck at the collision for
                so many seconds,
            stuck_at_collision_distance (float): the car is considerred as being
                stuck at the collision if it is within such distance of the first
                collision location.
            max_red_light_penalty (float): the maximum penalty (i.e. negative reward)
                for red light violation. We don't want the red light penalty to
                be too large if the player cannot even get enough positive moving
                reward. So the penalty is capped at ``Player.PENALTY_RATE_RED_LIGHT * max(0., episode_reward))``.
                Note that this reward is only given once at the first step of
                contiguous red light violation.
            overspeed_penalty_weight (float): if > 0, a penalty proportional to
                the overspeed magnitude will be applied, multiplied by the step
                time (seconds each step of simulation represents) to make the
                penalty invariant to it, and then multiplied by the weight
                of ``overspeed_penalty_weight``.
                A negative value is the same as 0.
            sparse_reward (bool): If False, the distance reward is given at every
                step based on how much it moves along the navigation route. If
                True, the distance reward is only given after moving ``sparse_reward_distance``.
            sparse_reward_interval (float): the sparse reward is given after
                approximately every such distance along the route has been driven.
            allow_negative_distance_reward (True): whether to allow negative distance
                reward. If True, the agent will receive positive reward for moving
                ahead along the route, and negative reward for moving back along
                the route. If False, the agent still receives positive reward for
                moving ahead along the route, but will not receive negative reward
                for moving back along the route. Instead, the negative distance
                will be accumulated to the future distance reward. This may ease
                the learning if the right behavior is to temporarily go back along
                the route in order, for example, to avoid obstacle.
            min_speed (float): unit is m/s. Failure if
                route_length / min_speed + additional_time seconds passed
            additional_time (float): additional time (unit is second) provided
                to the agent in each episode. This is useful especially for the
                episodes with short route_lengths (e.g. < 50m), as it takes
                some time for the car to be able to move (because of initial
                spawning phase with z > 0 and acceleration phase).
            with_gnss_sensor (bool): whether to use ``GnssSensor``.
            with_imu_sensor (bool): whether to use ``IMUSensor``.
            with_camera_sensor (bool): whether to use ``CameraSensor``.
            with_radar_sensor (bool): whether to use ``RadarSensor``.
            with_bev_sensor (bool): whether to use ``BEVSensor``.
            data_collection_mode (bool): if True, will use Rule-based agents
                to control the Players. This can be used for purposes such as
                collecting data.
            with_red_light_sensor (bool): whether to use ``RedlightSensor``.
            with_obstacle_sensor (bool): whether to use ``ObstacleDetectionSensor``.
            terminate_upon_infraction (str): whether to terminate the episode
                based on the specified mode ("collision", "redlight", "all", ""),
                when the agent has the corresponding infractions.
                If "", no infraction-based termination is activated.
            render_waypoints (bool): whether to render (interpolated) waypoints
                in the generated video during rendering. Note that it is only
                used for visualization and has no impacts on the perception data.
        """

        self._actor = actor
        self._alf_world = alf_world
        self._observation_sensors = {}
        self._render_waypoints = render_waypoints

        assert terminate_upon_infraction in ('collision', 'redlight', 'all',
                                             '')
        self._terminate_upon_infraction = terminate_upon_infraction

        self._collision_sensor = CollisionSensor(actor)
        self._observation_sensors['collision'] = self._collision_sensor

        if with_gnss_sensor:
            self._gnss_sensor = GnssSensor(actor)
            self._observation_sensors['gnss'] = self._gnss_sensor
        else:
            self._gnss_sensor = None

        if with_imu_sensor:
            self._imu_sensor = IMUSensor(actor)
            self._observation_sensors['imu'] = self._imu_sensor
        else:
            self._imu_sensor = None

        if with_camera_sensor:
            self._camera_sensor = CameraSensor(actor)
            self._observation_sensors['camera'] = self._camera_sensor
        else:
            self._camera_sensor = None

        self._lane_invasion_sensor = LaneInvasionSensor(actor)

        if with_radar_sensor:
            self._radar_sensor = RadarSensor(actor)
            self._observation_sensors['radar'] = self._radar_sensor
        else:
            self._radar_sensor = None

        self._navigation = NavigationSensor(actor, alf_world)
        self._observation_sensors['navigation'] = self._navigation

        if with_bev_sensor:
            self._bev_sensor = BEVSensor(actor, self._alf_world,
                                         self._navigation)
            self._observation_sensors['bev'] = self._bev_sensor
        else:
            self._bev_sensor = None

        if with_dynamic_object_sensor:
            self._dynamic_object_sensor = DynamicObjectSensor(
                actor, self._alf_world)
            self._observation_sensors[
                'dynamic_object'] = self._dynamic_object_sensor
        else:
            self._dynamic_object_sensor = None

        self._data_collection_mode = data_collection_mode
        if self._data_collection_mode:
            from .carla_env.carla_agents import SimpleNavigationAgent
            self._data_agent = SimpleNavigationAgent(actor, self._navigation,
                                                     alf_world)
        if with_red_light_sensor:
            self._red_light_sensor = RedlightSensor(actor, weakref.ref(self))
            self._observation_sensors['redlight'] = self._red_light_sensor

        self._with_obstacle_sensor = with_obstacle_sensor
        if with_obstacle_sensor:
            self._obstacle_sensor = ObstacleDetectionSensor(actor)
            self._observation_sensors['obstacle'] = self._obstacle_sensor

        self._success_reward = success_reward
        self._success_distance_thresh = success_distance_thresh
        self._min_speed = min_speed
        self._additional_time = additional_time
        self._delta_seconds = actor.get_world().get_settings(
        ).fixed_delta_seconds
        self._max_collision_penalty = max_collision_penalty
        self._max_stuck_at_collision_frames = max_stuck_at_collision_seconds / self._delta_seconds
        self._stuck_at_collision_distance = stuck_at_collision_distance

        self._max_red_light_penalty = max_red_light_penalty
        self._overspeed_penalty_weight = max(overspeed_penalty_weight, 0)

        self._sparse_reward = sparse_reward
        self._sparse_reward_index_interval = int(
            max(1, sparse_reward_interval // self._alf_world.route_resolution))
        self._allow_negative_distance_reward = allow_negative_distance_reward

        self._observation_spec = dict()
        self._observation_desc = dict()
        for sensor_name, sensor in self._observation_sensors.items():
            self._observation_spec[sensor_name] = sensor.observation_spec()
            self._observation_desc[sensor_name] = sensor.observation_desc()
        self._observation_spec['goal'] = alf.TensorSpec([3])
        self._observation_spec['velocity'] = alf.TensorSpec([3])

        # UE4 coordinate system is left handed:
        # https://forums.unrealengine.com/development-discussion/c-gameplay-programming/103787-ue4-coordinate-system-not-right-handed
        self._observation_desc['goal'] = (
            "Target location relative to the vehicle coordinate system in "
            "meters. X axis: front, Y axis: right, Z axis: up. Only the "
            "rotation around Z axis is taken into account when calculating the "
            "vehicle's coordinate system.")
        self._observation_desc['navigation'] = (
            'Relative positions of the future waypoints in the route')
        self._observation_desc[
            'velocity'] = "3D Velocity relative to self coordinate in m/s"
        self._info_spec = OrderedDict(
            success=alf.TensorSpec(()),
            collision=alf.TensorSpec(()),
            collision_front=alf.TensorSpec(()),
            red_light_violated=alf.TensorSpec(()),
            red_light_encountered=alf.TensorSpec(()),
            overspeed=alf.TensorSpec(()))

        self._control = carla.VehicleControl()
        self._controller = None
        if controller_ctor is not None:
            self._controller = controller_ctor(actor, self._delta_seconds)

        self.reset()

        # for rendering
        self._surface = None
        self._font = None
        self._clock = None

    def reset(self):
        """Reset the player location and goal.

        Use ``carla.Client.apply_batch_sync()`` to actually reset.

        Returns:
            list[carla.command]:
        """

        if self._controller:
            self._controller.reset()

        wp = random.choice(self._alf_world.get_waypoints())
        goal_loc = wp.transform.location
        self._goal_location = np.array([goal_loc.x, goal_loc.y, goal_loc.z],
                                       dtype=np.float32)

        forbidden_locations = []
        for v in self._alf_world.get_actors():
            if v.id == self._actor.id:
                continue
            forbidden_locations.append(
                self._alf_world.get_actor_location(v.id))

        # find a waypoint far enough from other vehicles
        ok = False
        i = 0
        while not ok and i < 100:
            wp = random.choice(self._alf_world.get_waypoints())
            loc = wp.transform.location
            ok = True
            for other_loc in forbidden_locations:
                if loc.distance(other_loc) < 10.:
                    ok = False
                    break
            i += 1
        assert ok, "Fail to find new position"
        # loc.z + 0.27531 to avoid Z-collision, see Carla documentation for
        # carla.Map.get_spawn_points(). The value used by carla is slightly
        # smaller: 0.27530714869499207
        loc = carla.Location(loc.x, loc.y, loc.z + 0.3)

        commands = [
            carla.command.ApplyTransform(
                self._actor, carla.Transform(loc, wp.transform.rotation)),
            carla.command.ApplyVelocity(self._actor, carla.Vector3D()),
            carla.command.ApplyAngularVelocity(self._actor, carla.Vector3D())
        ]

        self._max_frame = None
        self._done = False
        self._prev_location = loc
        self._prev_action = np.zeros(
            self.action_spec().shape, dtype=np.float32)
        self._alf_world.update_actor_location(self._actor.id, loc)

        self._route_length = self._navigation.set_destination(goal_loc)

        if self._data_collection_mode:
            self._data_agent.set_destination()

        self._prev_collision = False  # whether there is collision in the previous frame
        self._collision = False  # whether there is collision in the current frame
        self._collision_loc = None  # the location of the car when it starts to have collision

        self._prev_violated_red_light_id = None
        self._prev_encountered_red_light_id = None
        self._prev_encountered_red_light_dist = 1e10

        # The intermediate goal for sparse reward
        self._intermediate_goal_index = min(self._sparse_reward_index_interval,
                                            self._navigation.num_waypoints - 1)

        # The location of the car when the intermediate goal is set
        self._intermediate_start = _to_numpy_loc(loc)

        self._episode_reward = 0.
        self._unrecorded_distance_reward = 0.
        self._is_first_step = True
        self._speed_limit = None
        # when resetting, use the globally closest speed limit sign for update
        self.update_speed_limit(dis_threshold=-1)

        return commands

    def destroy(self):
        """Get the commands for destroying the player.

        Use carla.Client.apply_batch_sync() to actually destroy the sensor.

        Returns:
            list[carla.command]:
        """
        commands = []
        for sensor in self._observation_sensors.values():
            commands.extend(sensor.destroy())
        commands.extend(self._lane_invasion_sensor.destroy())
        commands.append(carla.command.DestroyActor(self._actor))
        if self._surface is not None:
            import pygame
            pygame.quit()

        return commands

    def observation_spec(self):
        """Get the observation spec.

        Returns:
            nested TensorSpec:
        """
        return self._observation_spec

    def observation_desc(self):
        """Get the description about the observation.

        Returns:
            nested str: each str corresponds to one TensorSpec from
            ``observatin_spec()``.
        """
        return self._observation_desc

    def action_spec(self):
        """Get the action spec.

        If ``controller`` is provided at ``__init__()``, the action_spec is given
        by ``controller``.

        Otherwise, the action is a 4-D vector of [throttle, steer, brake, reverse], where
        throttle is in [-1.0, 1.0] (negative value is same as zero), steer is in
        [-1.0, 1.0], brake is in [-1.0, 1.0] (negative value is same as zero),
        and reverse is interpreted as a boolean value with values greater than
        0.5 corresponding to True.

        Returns:
            nested BoundedTensorSpec:
        """
        if self._controller is not None:
            return self._controller.action_spec()
        else:
            return alf.BoundedTensorSpec([4],
                                         minimum=[-1., -1., -1., 0.],
                                         maximum=[1., 1., 1., 1.])

    def info_spec(self):
        """Get the info spec."""
        return self._info_spec

    def action_desc(self):
        """Get the description about the action.

        Returns:
            nested str: each str corresponds to one TensorSpec from
            ``action_spec()``.
        """
        if self._controller is not None:
            return self._controller.action_desc()
        else:
            return (
                "4-D vector of [throttle, steer, brake, reverse], where "
                "throttle is in [-1.0, 1.0] (negative value is same as zero), "
                "steer is in [-1.0, 1.0], brake is in [-1.0, 1.0] (negative value "
                "is same as zero), and reverse is interpreted as a boolean value "
                "with values greater than 0.5 corresponding to True.")

    def reward_spec(self):
        """Get the reward spec."""
        return alf.TensorSpec([Player.REWARD_DIMENSION])

    def _get_goal(self):
        return _calculate_relative_position(self._actor.get_transform(),
                                            self._goal_location)

    def update_speed_limit(self, dis_threshold=10):
        """Update the speed limit of the actor according to the active speed
        limit sign. The speed limit is updated when passing by a speed limit sign.

        Args:
            dis_threshold (float): the distance in meter within which to consider
                the speed limit sign as active. The one closest to the actor in
                the active set will be used as the current speed limit.
                If a negative value is provided, all speed limit signs are
                taken into considerations for determining the closest one.
        Returns:
            float: speed limit in m/s
        """
        updated_speed_limit = self._alf_world.get_active_speed_limit(
            self._actor, dis_threshold)
        if updated_speed_limit is not None:
            self._speed_limit = updated_speed_limit

    def _get_agent_speed(self):
        v = self._actor.get_velocity()
        speed = np.linalg.norm(np.array([v.x, v.y, v.z], dtype=np.float))
        return speed

    def get_overspeed_amount(self):
        """Get the difference between the actor's speed and the speed limit,
        lower bounded by 0.
        Returns:
            float:
                - 0. if actor's ``_speed_limit`` is None or speed is lower than
                    speed limit
                - the amount of the actor's speed over the speed limit otherwise
        """

        speed = self._get_agent_speed()

        if self._speed_limit is None or speed < self._speed_limit:
            return 0.
        else:
            return speed - self._speed_limit

    def get_current_time_step(self, current_frame):
        """Get the current time step for the player.

        Args:
            current_frame (int): current simulation frame no.
        Returns:
            TimeStep: all elements are ``np.ndarray`` or ``np.number``.
        """
        obs = dict()
        for sensor_name, sensor in self._observation_sensors.items():
            obs[sensor_name] = sensor.get_current_observation(current_frame)
        obs['goal'] = self._get_goal()
        self._alf_world.update_actor_location(self._actor.id,
                                              self._actor.get_location())
        v = self._actor.get_velocity()
        obs['velocity'] = _calculate_relative_velocity(
            self._actor.get_transform(), _to_numpy_loc(v))
        self._current_distance = np.linalg.norm(obs['goal'])

        prev_loc = _to_numpy_loc(self._prev_location)
        curr_loc = _to_numpy_loc(self._actor.get_location())

        reward_vector = np.zeros(Player.REWARD_DIMENSION, np.float32)
        reward = 0.
        discount = 1.0
        # this dictionary structure is used for describing the occurrences
        # of different types events appeared in the current time step.
        info = OrderedDict(
            success=np.float32(0.0),  # success event (0/1)
            collision=np.float32(0.0),  # all collision events (0/1)
            collision_front=np.float32(0.0),  # front collision event (0/1)
            red_light_violated=np.float32(0.0),  # violated red light (0/1)
            red_light_encountered=np.float32(
                0.0),  # encountered red light (0/1)
            overspeed=np.float32(0.0)  # overspeed event (0/1)
        )

        #===========================Infractions=================================

        # -------- Infraction 1: collision --------
        # When the previous episode ends because of stucking at a collision with
        # another vehicle, it may get an additional collision event in the new frame
        # because the relocation of the car may happen after the simulation of the
        # moving. So we ignore the collision at the first step.
        self._collision = not np.all(
            obs['collision'] == 0) and not self._is_first_step

        if self._collision and not self._prev_collision:
            # We only report the first collision event among contiguous collision
            # events.
            info['collision'] = np.float32(1.0)

            collision_location_available = obs['collision'].ndim == 3
            if collision_location_available:
                info['collision_front'] = np.float32(
                    np.any(obs['collision'][:, 1, 0] > 0.5))

            logging.info("actor=%d frame=%d COLLISION" % (self._actor.id,
                                                          current_frame))
            self._collision_loc = curr_loc
            self._collision_frame = current_frame
            # We don't want the collision penalty to be too large if the player
            # cannot even get enough positive moving reward. So we cap the penalty
            # at ``max(0., self._episode_reward)``
            reward -= min(
                self._max_collision_penalty,
                Player.PENALTY_RATE_COLLISION * max(0., self._episode_reward))
            reward_vector[Player.REWARD_COLLISION] = 1.

        # -------- Infraction 2: running red light --------
        red_light_id, encountered_red_light_id, encountered_red_light_dist = \
                        self._alf_world.is_running_red_light(self._actor)

        if encountered_red_light_id is not None and encountered_red_light_id != self._prev_encountered_red_light_id:
            logging.info("actor=%d frame=%d Encountering RED_LIGHT" %
                         (self._actor.id, current_frame))
            info['red_light_encountered'] = np.float32(1.0)

        self._prev_encountered_red_light_id = encountered_red_light_id
        self._prev_encountered_red_light_dist = encountered_red_light_dist

        if red_light_id is not None and red_light_id != self._prev_violated_red_light_id:
            speed = self._get_agent_speed()
            logging.info("actor=%d frame=%d Running RED_LIGHT speed %2.1f" %
                         (self._actor.id, current_frame, speed))
            reward_vector[Player.REWARD_RED_LIGHT] = 1.
            info['red_light_violated'] = np.float32(1.0)
            if self._terminate_upon_infraction != "redlight":
                reward -= min(
                    self._max_red_light_penalty,
                    Player.PENALTY_RATE_RED_LIGHT * max(
                        0., self._episode_reward))
            else:
                # to encourage stop at red-light, can set max_red_light_penalty
                # to a large value (e.g. 1000) and set terminate_upon_infraction
                # to "redlight"
                reward -= self._max_red_light_penalty
                # reward proportional to 1 - speed / capped_speed for encourating
                # stopping at redlight
                red_light_reward = (
                    1 - min(speed / 5, 1)) * 0.5 * self._max_red_light_penalty
                reward += red_light_reward

        self._prev_violated_red_light_id = red_light_id

        if self._max_frame is None:
            step_type = ds.StepType.FIRST
            max_frames = math.ceil(
                (self._route_length / self._min_speed + self._additional_time)
                / self._delta_seconds)
            self._max_frame = current_frame + max_frames
        elif (self._current_distance < self._success_distance_thresh
              and self._actor.get_velocity() == carla.Location(0., 0., 0.)):
            # TODO: include waypoint orientation as success critiria
            step_type = ds.StepType.LAST
            reward += self._success_reward
            reward_vector[Player.REWARD_SUCCESS] = 1.
            discount = 0.0
            info['success'] = np.float32(1.0)
            logging.info(
                "actor=%d frame=%d SUCCESS" % (self._actor.id, current_frame))
        elif current_frame >= self._max_frame:
            logging.info("actor=%d frame=%d FAILURE: out of time" %
                         (self._actor.id, current_frame))
            step_type = ds.StepType.LAST
        elif (self._terminate_upon_infraction == "redlight" or
                self._terminate_upon_infraction == "all") and \
                reward_vector[Player.REWARD_RED_LIGHT] > 0:
            # directly terminate upon redlight infractions; the corresponding
            # infraction penalty has already been assigned earlier
            step_type = ds.StepType.LAST
            discount = 0.0
            logging.info("actor=%d frame=%d FAILURE: red light infraction" %
                         (self._actor.id, current_frame))

        elif (self._terminate_upon_infraction == "collision" or
                self._terminate_upon_infraction == "all") and \
                reward_vector[Player.REWARD_COLLISION] > 0:
            # directly terminate upon collision infractions; the corresponding
            # infraction penalty has already been assigned earlier
            step_type = ds.StepType.LAST
            discount = 0.0
            logging.info("actor=%d frame=%d FAILURE: collision infraction" %
                         (self._actor.id, current_frame))

        elif (self._collision_loc is not None
              and current_frame - self._collision_frame >
              self._max_stuck_at_collision_frames
              and np.linalg.norm(curr_loc - self._collision_loc) <
              self._stuck_at_collision_distance):
            logging.info("actor=%d frame=%d FAILURE: stuck at collision" %
                         (self._actor.id, current_frame))
            step_type = ds.StepType.LAST
        else:
            step_type = ds.StepType.MID

        distance_reward = 0
        if self._sparse_reward:
            current_index = self._navigation.get_next_waypoint_index()
            if step_type == ds.StepType.LAST and info['success'] == 1.0:
                # Since the episode is finished, we need to incorporate the final
                # progress towards the goal as reward to encourage stopping near the goal.
                distance_reward = (
                    np.linalg.norm(self._intermediate_start -
                                   self._goal_location) -
                    np.linalg.norm(curr_loc - self._goal_location))
            elif self._intermediate_goal_index < current_index:
                # This means that the car has passed the intermediate goal.
                # And we give it a reward which is equal to the distance it
                # travels.
                intermediate_goal = self._navigation.get_waypoint(
                    self._intermediate_goal_index)
                distance_reward = np.linalg.norm(intermediate_goal -
                                                 self._intermediate_start)
                self._intermediate_start = intermediate_goal
                self._intermediate_goal_index = min(
                    self._intermediate_goal_index +
                    self._sparse_reward_index_interval,
                    self._navigation.num_waypoints - 1)
        else:
            goal0 = obs['navigation'][2]  # This is about 10m ahead
            distance_reward = (np.linalg.norm(prev_loc - goal0) -
                               np.linalg.norm(curr_loc - goal0))

        reward_vector[Player.REWARD_DISTANCE] = distance_reward
        if not self._allow_negative_distance_reward:
            distance_reward += self._unrecorded_distance_reward
            if distance_reward < 0:
                self._unrecorded_distance_reward = distance_reward
                distance_reward = 0
            else:
                self._unrecorded_distance_reward = 0
        reward += distance_reward

        overspeed = self.get_overspeed_amount()
        if overspeed > 0 and self._overspeed_penalty_weight > 0:
            logging.info("actor=%d frame=%d OVERSPEED" % (self._actor.id,
                                                          current_frame))
            reward_vector[Player.REWARD_OVERSPEED] = 1.
            info['overspeed'] = np.float32(1.0)
            reward -= self._overspeed_penalty_weight * overspeed * self._delta_seconds

        obs['navigation'] = _calculate_relative_position(
            self._actor.get_transform(), obs['navigation'])

        self._done = step_type == ds.StepType.LAST
        self._episode_reward += reward

        reward_vector[Player.REWARD_OVERALL] = reward

        self._current_time_step = ds.TimeStep(
            step_type=step_type,
            reward=reward_vector,
            discount=np.float32(discount),
            observation=obs,
            prev_action=self._prev_action,
            env_info=info)
        return self._current_time_step

    def act(self, action):
        """Generate the carla command for taking the given action.

        Use ``carla.Client.apply_batch_sync()`` to actually destroy the sensor.

        Args:
            action (nested np.ndarray):
        Returns:
            list[carla.command]:
        """
        self._prev_collision = self._collision
        self._prev_location = self._actor.get_location()
        self._is_first_step = False
        if self._done:
            return self.reset()

        if self._data_collection_mode:
            # TODO: add support to the usage of controller
            assert self._controller is None, ("controller is not supported "
                                              "in data collection currently")
            control = self._data_agent.run_step()
            control.manual_gear_shift = False
            action[0] = control.throttle
            action[1] = control.steer
            action[2] = control.brake
            if control.reverse:
                action[3] = 1
            else:
                action[3] = 0
            self._control = control
        else:
            if self._controller is not None:
                self._control = self._controller.act(action)
            else:
                self._control.throttle = max(float(action[0]), 0.0)
                self._control.steer = float(action[1])
                self._control.brake = max(float(action[2]), 0.0)
                self._control.reverse = bool(action[3] > 0.5)
        self._prev_action = action
        self.update_speed_limit()

        return [carla.command.ApplyVehicleControl(self._actor, self._control)]

    def render(self, mode):
        """Render the simulation.

        Args:
            mode (str): one of ['rgb_array', 'human']
        Returns:
            one of the following:
                - None: if mode is 'human'
                - np.ndarray: the image of shape [height, width, channels] if
                    mode is 'rgb_array'
        """
        import pygame
        if self._surface is None:
            pygame.init()
            pygame.font.init()
            self._clock = pygame.time.Clock()
            if self._camera_sensor:
                height, width = self._camera_sensor.observation_spec(
                ).shape[1:3]
                height, width = get_scaled_image_size(height, width)
            else:
                height = MINIMUM_RENDER_HEIGHT
                width = MINIMUM_RENDER_WIDTH
            if mode == 'human':
                self._surface = pygame.display.set_mode(
                    (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            else:
                self._surface = pygame.Surface((width, height))

        if mode == 'human':
            self._clock.tick_busy_loop(1000)

        if self._camera_sensor:
            self._camera_sensor.render(self._surface)
        obs = self._current_time_step.observation
        env_info = self._current_time_step.env_info
        np_precision = np.get_printoptions()['precision']
        np.set_printoptions(precision=1)
        info_text = [
            'FPS: %6.2f' % self._clock.get_fps(),
            'GPS:  (%7.4f, %8.4f, %5.2f)' % tuple(obs['gnss'].tolist()) \
                                if 'gnss' in obs.keys() else '',
            'Goal: (%7.1f, %8.1f, %5.1f)' % tuple(obs['goal'].tolist()) \
                                if 'goal' in obs.keys() else '',
            'Ahead: (%7.1f, %8.1f, %5.1f)' % tuple(
                obs['navigation'][2].tolist()) \
                                if 'navigation' in obs.keys() else '',
            'Distance: %7.2f' % np.linalg.norm(obs['goal']) \
                                if 'goal' in obs.keys() else '',
            'Velocity: (%4.1f, %4.1f, %4.1f) m/s' % tuple(
                    obs['velocity'].tolist()) \
                                if 'velocity' in obs.keys() else '',
            'Acceleration: (%4.1f, %4.1f, %4.1f)' % tuple(
                    obs['imu'][0:3].tolist()) \
                                if 'imu' in obs.keys() else '',
            'Compass: %5.1f' % math.degrees(float(obs['imu'][6])) \
                                if 'imu' in obs.keys() else '',
            'Throttle: %4.2f' % self._control.throttle,
            'Brake:    %4.2f' % self._control.brake,
            'Steer:    %4.2f' % self._control.steer,
            'Reverse:  %4s' % self._control.reverse,
            'Reward: (%s)' % self._current_time_step.reward,
            'Route Length: %4.2f m' % self._route_length,
            'Speed Limit: %4.2f m/s' % self._speed_limit,
            'Red light zone: %1d' % (self._prev_encountered_red_light_id != None),
            'Red light violation: %1d' % env_info['red_light_violated'],
            'Red light dist: %4.2f' % self._prev_encountered_red_light_dist,
        ]
        info_text = [info for info in info_text if info != '']
        np.set_printoptions(precision=np_precision)
        self._draw_text(info_text)

        if mode == 'human':
            pygame.display.flip()
        elif mode == 'rgb_array':
            if self._camera_sensor is not None:
                # (x, y, c) => (y, x, c)
                rgb_img = pygame.surfarray.array3d(self._surface).swapaxes(
                    0, 1)

                if 'navigation' in obs.keys() and self._render_waypoints:
                    # index of waypoint to be rendered
                    waypoint_index = np.arange(2, 5)
                    nav_traj = obs['navigation'][waypoint_index]
                    self._draw_ego_traj_on_image(
                        nav_traj,
                        rgb_img,
                        camera_sensor=self._camera_sensor,
                        color=(0, 255, 0),
                        size=5,
                        zero_world_z=True,
                        interp_num=500)
            else:
                rgb_img = None

            if self._bev_sensor is not None:
                bev_img = self._bev_sensor.render()
                if rgb_img is not None:
                    concat_img = np.zeros(
                        (max(rgb_img.shape[0], bev_img.shape[0]),
                         rgb_img.shape[1] + bev_img.shape[1], 3), np.uint8)
                    concat_img[:rgb_img.shape[0], :rgb_img.shape[1]] = rgb_img
                    concat_img[:bev_img.shape[0], -bev_img.shape[1]:] = bev_img
                    rgb_img = concat_img
                else:
                    rgb_img = bev_img

            return rgb_img
        else:
            raise ValueError("Unsupported render mode: %s" % mode)

    def _draw_ego_traj_on_image(self,
                                ego_points,
                                rgb_img,
                                camera_sensor,
                                color=(255, 0, 0),
                                size=3,
                                forward_shift_delta=0,
                                zero_world_z=False,
                                append_self=False,
                                interp_num=200):
        """Render points in ego coordinates on camera image.

        Args:
            ego_points (np.ndarray): [N, 3]
            rgb_img (np.ndarray): with the meaning of axis as (y, x, c)
            camera_sensor (CameraSensor): the camera sensor
            color (tuple[int]): color values for the [R, G, B] channels
                of the rendered points
            size (int): size of the rendered point in terms of pixels
            forward_shift_delta (int): the amount to be shifted for the points
                along the forward axis. This might be useful in some cases
                to shift the points to be within the camera's field of view
            zero_world_z (bool): whether set the z values of the transformed
                points in world coordinate to zero. This is useful to render
                points on the ground plane
            append_self (bool): whether append self location to the point set
            interp_num (int): the number of target elements to be obtained
                by interpolation
        Returns:
            np.ndarray: interpolated ego points or input ego points if no
                interpolation is applied
        """

        # [N, 3] -> [3, N]
        ego_points = np.transpose(ego_points)

        if interp_num >= ego_points.shape[1]:
            time_index = np.linspace(0, 1, ego_points.shape[1])

            interp_func = scipy.interpolate.interp1d(
                x=time_index, y=ego_points, axis=1)
            ego_interp = interp_func(
                np.linspace(time_index[0], time_index[-1], 100))
        else:
            if interp_num > 0:
                common.warning_once(
                    ("the specified number of elements after "
                     "interpolation is smaller than the number of elements "
                     "in the original trajectory; skipping interpolation"))
            ego_interp = ego_points

        # [3, N] -> [N, 3]
        ego_interp = np.transpose(ego_interp)

        nav_world = self._ego_to_world_position(
            ego_interp,
            append_self=append_self,
            forward_shift_delta=forward_shift_delta)
        if zero_world_z:
            nav_world[:, 2] = 0

        camera_sensor._draw_world_points_on_image(nav_world, rgb_img, color,
                                                  size)

        return ego_interp

    def _ego_to_world_position(self,
                               ego_points,
                               append_self=False,
                               forward_shift_delta=0):
        """
        Args:
            ego_points (np.ndarray): [N, 3]
            forward_shift_delta (int): the amount to be shifted for the points
                along the forward axis. This might be useful in some cases
                to shift the points to be within the camera's field of view
        Returns:
            np.ndarray: shape [N, 3]
        """

        trans = self._actor.get_transform()
        self_loc = trans.location
        yaw = math.radians(trans.rotation.yaw)

        self_loc = np.array([self_loc.x, self_loc.y, self_loc.z])
        self_loc = np.expand_dims(self_loc, axis=0)
        cos, sin = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos, -sin, 0.], [sin, cos, 0.], [0., 0., 1.]])

        # shift along forward axis
        ego_points[:, 0] = ego_points[:, 0] + forward_shift_delta

        ego_points = (
            np.matmul(ego_points, np.linalg.inv(rot)) + self_loc).astype(
                np.float32)
        if append_self:
            ego_points = np.concatenate([self_loc, ego_points], axis=0)
        return ego_points

    def _draw_text(self, texts):
        import os
        import pygame
        if self._font is None:
            font_name = 'courier' if os.name == 'nt' else 'mono'
            fonts = [x for x in pygame.font.get_fonts() if font_name in x]
            default_font = 'ubuntumono'
            mono = default_font if default_font in fonts else fonts[0]
            mono = pygame.font.match_font(mono)
            self._font = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        info_surface = pygame.Surface((240, 240))
        info_surface.set_alpha(100)
        self._surface.blit(info_surface, (0, 0))
        v_offset = 4
        for item in texts:
            surface = self._font.render(item, True, (255, 255, 255))
            self._surface.blit(surface, (8, v_offset))
            v_offset += 18


def _exec(command):
    stream = os.popen(command)
    ret = stream.read()
    stream.close()
    return ret


@alf.configurable
class CarlaServer(object):
    """CarlaServer for doing the simulation."""

    def __init__(self,
                 rpc_port=2000,
                 streaming_port=2001,
                 docker_image="horizonrobotics/alf:0.0.6-carla0.9.9",
                 quality_level="Low",
                 carla_root="/home/carla",
                 use_opengl=True):
        """

        Args:
            rpc_port (int): port for RPC
            streaming_port (int): port for data streaming
            docker_image (str): If provided, will use the docker image to start
                the Carla server. Some valid images are "carlasim/carla:0.9.9"
                and "horionrobotics/alf:0.0.3-carla"
            quality_level (str): one of ['Low', 'Epic']. See the explanation at
                `<https://carla.readthedocs.io/en/latest/adv_rendering_options/#graphics-quality>`_
            carla_root (str): directorcy where CarlaUE4.sh is in. The default
                value is correct for using docker image. If not using docker
                image, make sure you provide the correct path. This is the directory
                where you unzipped the file you downloaded from
                `<https://github.com/carla-simulator/carla/releases/tag/0.9.9>`_.
            use_opengl (bool): the default graphics engine of Carla is Vulkan,
                which is supposed to be better than OpenGL. However, Vulkan is not
                always available. It may not be installed or the nvidia driver does
                not support vulkan.
        """
        assert quality_level in ['Low', 'Epic'], "Unknown quality level"
        use_docker = (not alf.utils.common.is_inside_docker_container()
                      and docker_image)
        opengl = "-opengl" if use_opengl else ""
        if use_docker:
            dev = os.environ.get('CUDA_VISIBLE_DEVICES')
            if not dev:
                dev = 'all'
            command = ("docker run -d "
                       "-p {rpc_port}:{rpc_port} "
                       "-p {streaming_port}:{streaming_port} "
                       "-u carla "
                       "--rm --gpus device=" + dev + " " + docker_image +
                       " {carla_root}/CarlaUE4.sh "
                       "--carla-rpc-port={rpc_port} "
                       "--carla-streaming-port={streaming_port} "
                       "--quality-level={quality_level} {opengl}")
        else:
            assert os.path.exists(carla_root + "/CarlaUE4.sh"), (
                "%s/CarlaUE4.sh "
                "does not exist. Please provide correct value for `carla_root`"
                % carla_root)
            # We do not use CarlaUE4.sh here in order to get the actual Carla
            # server process so that we can kill it.
            command = (
                "{carla_root}/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping "
                "CarlaUE4 "  # perhaps most system does not have vulkan support, so we use opengl
                "-carla-rpc-port={rpc_port} "
                "-carla-streaming-port={streaming_port} "
                "-quality-level={quality_level} {opengl}")

        command = command.format(
            rpc_port=rpc_port,
            streaming_port=streaming_port,
            quality_level=quality_level,
            carla_root=carla_root,
            opengl=opengl)

        logging.info("Starting Carla server: %s" % command)
        self._container_id = None
        self._process = None
        if use_docker:
            self._container_id = _exec(command)
            assert self._container_id, "Fail to start container"
            logging.info("Starting carla in container %s" % self._container_id)
        else:
            new_env = os.environ.copy()
            new_env['SDL_VIDEODRIVER'] = 'offscreen'
            self._process = subprocess.Popen(
                command.split(),
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=new_env)

    def stop(self):
        """Stop the carla server."""
        if self._container_id:
            command = "docker kill %s" % self._container_id
            logging.info("Stopping Carla server: %s" % command)
            _exec(command)
            self._container_id = None
        if self._process:
            self._process.kill()
            self._process.communicate()
            self._process = None

    def __del__(self):
        self.stop()


@alf.configurable
class CarlaEnvironment(AlfEnvironment):
    """Carla simulation environment.

    In order to use it, you need to either download a valid docker image or
    a Carla package.
    """

    # not all vehicles have functioning lights. (See https://carla.readthedocs.io/en/0.9.9/core_world/#weather)
    vehicles_with_functioning_lights = [
        'vehicle.audi.tt',
        'vehicle.chevrolet.impala',
        'vehicle.dodge_charger.police',
        'vehicle.audi.etron',
        'vehicle.lincoln.mkz2017',
        'vehicle.mustang.mustang',
        'vehicle.tesla.model3',
        'vehicle.volkswagen.t2',
    ]

    def __init__(self,
                 batch_size,
                 map_name,
                 vehicle_filter='vehicle.*',
                 walker_filter='walker.pedestrian.*',
                 num_other_vehicles=0,
                 num_walkers=0,
                 percentage_walkers_running=0.1,
                 percentage_walkers_crossing=0.1,
                 global_distance_to_leading_vehicle=2.0,
                 use_hybrid_physics_mode=True,
                 safe=True,
                 day_length=0.,
                 max_weather_length=0,
                 weather_transition_ratio=0.1,
                 step_time=0.05):
        """
        Args:
            batch_size (int): the number of learning vehicles.
            map_name (str): the name of the map (e.g. "Town01")
            vehicle_filter (str): the filter for getting the blueprints for
                training vehicles. The filter for other vehicles will always be
                obtained using 'vehicle.*'.
            walker_filter (str): the filter for getting walker blueprints.
            num_other_vehicles (int): the number of autopilot vehicles
            num_walkers (int): the number of walkers
            global_distance_to_leading_vehicle (str): the autopiloted vehicles
                will try to keep such distance from other vehicles.
            percentage_walkers_running (float): percent of running walkers
            percentage_walkers_crossing (float): percent of walkers walking
                across the road.
            use_hybrid_physics_mode (bool): If true, the autopiloted vehicle will
                not use physics for simulation if it is far from other vehicles.
            safe (bool): avoid spawning vehicles prone to accidents.
            day_length (float): number of seconds of a day. If 0, the time of the
                day will not change.
            max_weather_length (float): the number of seconds each weather will
                last at the most. The actual lasting time (actual_weather_length)
                of each randomized weather setting is randomly sampled from
                [0.25 * max_weather_length, max_weather_length].
                If max_weather_length is set to 0, the weather won't change.
                Otherwise, weather randomization is turned on and we will
                sample a new set of parameters after reaching
                actual_weather_length for each sampled weather. Note that we
                exclude ``sun_azimuth_angle`` and ``sun_altitude_angle``
                from weather randomization and they are controlled separately
                by ``day_length`` in a more realistic way.
            weather_transition_ratio (float): the ratio between the length of
                the weather transition part and the actual lasting time of the
                new weather including the transition phase. It has no effect
                if max_weather_length is 0.
            step_time (float): how many seconds does each step of simulation represents.
        """
        super().__init__()

        with common.get_unused_port(2000, n=2) as (rpc_port, streaming_port):
            self._server = CarlaServer(rpc_port, streaming_port)

        self._batch_size = batch_size
        self._num_other_vehicles = num_other_vehicles
        self._num_walkers = num_walkers
        self._percentage_walkers_running = percentage_walkers_running
        self._percentage_walkers_crossing = percentage_walkers_crossing
        self._day_length = day_length
        self._time_of_the_day = 0.5 * day_length
        self._step_time = step_time
        self._max_weather_length = max_weather_length
        self._actual_weather_length = 0
        self._weather_length_count = 0
        self._weather_transition_ratio = min(1.0, weather_transition_ratio)

        self._world = None
        try:
            for i in range(20):
                try:
                    logging.info(
                        "Waiting for server to start. Try %d" % (i + 1))
                    self._client = carla.Client("localhost", rpc_port)
                    self._world = self._client.load_world(map_name)
                    break
                except RuntimeError:
                    continue
        finally:
            if self._world is None:
                self._server.stop()
                assert self._world is not None, "Fail to start server."

        logging.info("Server started.")

        self._traffic_manager = None
        if self._num_other_vehicles + self._num_walkers > 0:
            with common.get_unused_port(8000, n=1) as tm_port:
                self._traffic_manager = self._client.get_trafficmanager(
                    tm_port)
                # Need to set traffic manager (TM) to synchronous mode, since
                # traffic manager is designed to work in synchronous mode.
                # Both the CARLA server and TM should be set to synchronous
                # in order to function properly. Using TM in asynchronous mode
                # can lead to unexpected and undesirable results according to
                # https://carla.readthedocs.io/en/latest/adv_traffic_manager/#synchronous-mode
                self._traffic_manager.set_synchronous_mode(True)
            self._traffic_manager.set_hybrid_physics_mode(
                use_hybrid_physics_mode)
            self._traffic_manager.set_global_distance_to_leading_vehicle(
                global_distance_to_leading_vehicle)

        self._client.set_timeout(20)
        self._alf_world = World(self._world)
        self._safe = safe
        self._vehicle_filter = vehicle_filter
        self._walker_filter = walker_filter

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = step_time

        self._world.apply_settings(settings)
        self._map_name = map_name

        self._spawn_vehicles()
        self._spawn_walkers()

        self._observation_spec = self._players[0].observation_spec()
        self._action_spec = self._players[0].action_spec()
        self._env_info_spec = self._players[0].info_spec()
        self._reward_spec = self._players[0].reward_spec()

        # metadata property is required by video recording
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 1 / step_time
        }

    def _spawn_vehicles(self):
        blueprints = self._world.get_blueprint_library().filter(
            self._vehicle_filter)
        assert len(
            blueprints) > 0, "Cannot find vehicle '%s'" % self._vehicle_filter

        def _filter_safe(blueprints):
            blueprints = [
                x for x in blueprints
                if int(x.get_attribute('number_of_wheels')) == 4
            ]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [
                x for x in blueprints if not x.id.endswith('carlacola')
            ]
            blueprints = [
                x for x in blueprints if not x.id.endswith('cybertruck')
            ]
            return [x for x in blueprints if not x.id.endswith('t2')]

        if self._safe:
            blueprints = _filter_safe(blueprints)

        assert len(
            blueprints
        ) > 0, "Cannot find safe vehicle '%s'" % self._vehicle_filter

        blueprints = [
            x for x in blueprints
            if x.id in self.vehicles_with_functioning_lights
        ]
        assert len(blueprints) > 0, (
            "Cannot find vehicle with functioning lights")

        other_blueprints = self._world.get_blueprint_library().filter(
            'vehicle.*')
        other_blueprints = [
            x for x in other_blueprints
            if x.id in self.vehicles_with_functioning_lights
        ]

        spawn_points = self._world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        num_vehicles = self._batch_size + self._num_other_vehicles
        if num_vehicles <= number_of_spawn_points:
            random.shuffle(spawn_points)
        else:
            raise ValueError(
                "requested %d vehicles, but could only find %d spawn points" %
                (self._batch_size, number_of_spawn_points))

        commands = []
        for i, transform in enumerate(spawn_points[:num_vehicles]):
            if i < self._batch_size:
                blueprint = random.choice(blueprints)
            else:
                blueprint = random.choice(other_blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(
                    blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(
                    blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if i < self._batch_size:
                blueprint.set_attribute('role_name', 'hero')
            else:
                blueprint.set_attribute('role_name', 'autopilot')
            command = carla.command.SpawnActor(blueprint, transform)
            if i >= self._batch_size:
                # managed by traffic manager
                command = command.then(
                    carla.command.SetAutopilot(
                        carla.command.FutureActor, True,
                        self._traffic_manager.get_port()))
            commands.append(command)

        self._players = []
        self._other_vehicles = []
        responses = self._client.apply_batch_sync(commands, True)
        for i, response in enumerate(responses):
            if response.error:
                logging.error(response.error)
                continue
            vehicle = self._world.get_actor(response.actor_id)
            if i < self._batch_size:
                self._players.append(Player(vehicle, self._alf_world))
            else:
                self._other_vehicles.append(vehicle)
            self._alf_world.add_actor(vehicle)
            self._alf_world.update_actor_location(vehicle.id,
                                                  spawn_points[i].location)

        assert len(self._players) + len(
            self._other_vehicles) == num_vehicles, (
                "Fail to create %s vehicles" % num_vehicles)

    def _update_weather(self):
        """Update the weather settings.

        This function sample a new set of weather parameters once the
        actual lasting time of the previous weather setting is up.
        The actual lasting time of each weather setting is randomly sampled
        from [0.25 * max_weather_length, max_weather_length].
        After the termination of the previous weather setting, there is a
        transition phase to linearly transit from the old to the new weather
        settings.
        """

        # sample new weather parameter
        if self._weather_length_count == 0:
            # the actual lasting time of the new weather
            self._actual_weather_length = max(
                0.25, np.random.rand()) * self._max_weather_length
            new_weather_parameter = WeatherParameters(
                *np.random.uniform(0, 100, len(WeatherParameters())))
            weather = self._world.get_weather()

            prev_weather_parameter = extract_weather_parameters(weather)

            trans_steps = max(
                1, self._actual_weather_length * self._weather_transition_ratio
                / self._step_time)
            self._dp = (
                new_weather_parameter - prev_weather_parameter) / trans_steps

        # for the initial transition period, we smoothly transit between two
        # weather settings
        if (self._weather_length_count <=
                self._actual_weather_length * self._weather_transition_ratio):
            weather = self._world.get_weather()
            updated_weather = adjust_weather_parameters(weather, self._dp)
            self._world.set_weather(updated_weather)

        self._weather_length_count += self._step_time
        if self._weather_length_count >= self._actual_weather_length:
            self._weather_length_count = 0
            self._actual_weather_length = 0

    def _spawn_walkers(self):
        walker_blueprints = self._world.get_blueprint_library().filter(
            self._walker_filter)

        # 1. take all the random locations to spawn
        spawn_points = []
        for _ in range(self._num_walkers):
            spawn_point = carla.Transform()
            loc = self._world.get_random_location_from_navigation()
            if loc != None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # 2. we spawn the walker object
        commands = []
        walker_speeds = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_blueprints)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > self._percentage_walkers_running):
                    # walking
                    walker_speeds.append(
                        walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speeds.append(
                        walker_bp.get_attribute('speed').recommended_values[2])
            else:
                logging.info("Walker has no speed")
                walker_speeds.append(0.0)
            commands.append(carla.command.SpawnActor(walker_bp, spawn_point))
        responses = self._client.apply_batch_sync(commands, True)
        walker_speeds2 = []
        self._walkers = []
        for response, walker_speed, spawn_point in zip(
                responses, walker_speeds, spawn_points):
            if response.error:
                logging.error(
                    "%s: %s" % (response.error, spawn_point.location))
                continue
            walker = self._world.get_actor(response.actor_id)
            self._walkers.append({"walker": walker})
            walker_speeds2.append(walker_speed)

        walker_speeds = walker_speeds2

        # 3. we spawn the walker controller
        commands = []
        walker_controller_bp = self._world.get_blueprint_library().find(
            'controller.ai.walker')
        for walker in self._walkers:
            commands.append(
                carla.command.SpawnActor(walker_controller_bp,
                                         carla.Transform(),
                                         walker["walker"].id))
        responses = self._client.apply_batch_sync(commands, True)
        for response, walker in zip(responses, self._walkers):
            if response.error:
                logging.error(response.error)
                continue
            walker["controller"] = self._world.get_actor(response.actor_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self._world.tick()

        # 5. initialize each controller and set target to walk to (list is [controller, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self._world.set_pedestrians_cross_factor(
            self._percentage_walkers_crossing)
        for walker, walker_speed in zip(self._walkers, walker_speeds):
            # start walker
            walker['controller'].start()
            # set walk to random point
            location = self._world.get_random_location_from_navigation()
            walker['controller'].go_to_location(location)
            # max speed
            walker['controller'].set_max_speed(float(walker_speed))
            self._alf_world.add_actor(walker['walker'])
            self._alf_world.update_actor_location(walker['walker'].id,
                                                  location)

    def _clear(self):
        if self._world is None:
            return
        if self._players:
            commands = []
            for player in self._players:
                commands.extend(player.destroy())
            for response in self._client.apply_batch_sync(commands, True):
                if response.error:
                    logging.error(response.error)
            self._players.clear()
        commands = []
        for vehicle in self._other_vehicles:
            commands.append(carla.command.DestroyActor(vehicle))
        for walker in self._walkers:
            walker['controller'].stop()
            commands.append(carla.command.DestroyActor(walker['controller']))
            commands.append(carla.command.DestroyActor(walker['walker']))

        if commands:
            for response in self._client.apply_batch_sync(commands, True):
                if response.error:
                    logging.error(response.error)
        self._other_vehicles.clear()
        self._walkers.clear()

    @property
    def is_tensor_based(self):
        return True

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._batch_size

    def env_info_spec(self):
        return self._env_info_spec

    def observation_spec(self):
        return self._observation_spec

    def observation_desc(self):
        return self._players[0].observation_desc()

    def action_spec(self):
        return self._action_spec

    def action_desc(self):
        return self._players[0].action_desc()

    def reward_spec(self):
        return self._reward_spec

    def close(self):
        self._clear()
        self._server.stop()

    def __del__(self):
        self.close()

    @property
    def players(self):
        """Get all the players in the environment.

        Returns:
             list[Player]:
        """
        return self._players

    def render(self, mode):
        return self._players[0].render(mode)

    def _step(self, action):
        action = alf.nest.map_structure(lambda x: x.cpu().numpy(), action)
        commands = []
        for player, act in zip(self._players, action):
            commands.extend(player.act(act))
        for response in self._client.apply_batch_sync(commands):
            if response.error:
                logging.error(response.error)
        if self._day_length > 0:
            self._update_time_of_the_day()
        if self._max_weather_length > 0:
            self._update_weather()
        self._current_frame = self._world.tick()
        self._alf_world.on_tick()
        for vehicle in self._other_vehicles:
            self._alf_world.update_actor_location(vehicle.id,
                                                  vehicle.get_location())
        for walker in self._walkers:
            actor = walker['walker']
            self._alf_world.update_actor_location(actor.id,
                                                  actor.get_location())

        return self._get_current_time_step()

    def _update_time_of_the_day(self):
        light_state = None
        if 0.25 * self._day_length - self._step_time < self._time_of_the_day <= 0.25 * self._day_length:
            light_state = carla.VehicleLightState.NONE
        elif 0.75 * self._day_length - self._step_time < self._time_of_the_day <= 0.75 * self._day_length:
            light_state = carla.VehicleLightState(
                carla.VehicleLightState.Position
                | carla.VehicleLightState.LowBeam)
        if light_state is not None:
            for player in self._players:
                player._actor.set_light_state(light_state)
            for vehicle in self._other_vehicles:
                vehicle.set_light_state(light_state)
        self._time_of_the_day += self._step_time
        if self._time_of_the_day >= self._day_length:
            self._time_of_the_day -= self._day_length

        weather = self._world.get_weather()
        azimuth = weather.sun_azimuth_angle + 360 / self._day_length * self._step_time
        if azimuth > 360:
            azimuth -= 360
        weather.sun_azimuth_angle = azimuth
        altitude = self._time_of_the_day / self._day_length * 2
        if altitude > 1:
            altitude = 2. - altitude
        weather.sun_altitude_angle = altitude * 180 - 90
        self._world.set_weather(weather)

    def _get_current_time_step(self):
        time_step = [
            player.get_current_time_step(self._current_frame)
            for player in self._players
        ]
        time_step = alf.nest.map_structure(lambda *a: np.stack(a), *time_step)
        time_step = alf.nest.map_structure(torch.as_tensor, time_step)

        common.check_numerics(time_step)

        return time_step._replace(env_id=torch.arange(self._batch_size))

    def _reset(self):
        commands = []
        for player in self._players:
            commands.extend(player.reset())
        for response in self._client.apply_batch_sync(commands):
            if response.error:
                logging.error(response.error)
        self._current_frame = self._world.tick()
        self._alf_world.on_tick()
        return self._get_current_time_step()


@alf.configurable(whitelist=['wrappers'])
def load(map_name, batch_size, wrappers=[]):
    """Load CarlaEnvironment

    Args:
        map_name (str): name of the map. Currently available maps are:
            'Town01, Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07',
            and 'Town10HD'
        batch_size (int): the number of vehicles in the simulation.
        wrappers (list[AlfEnvironmentBaseWrapper]): environment wrappers
    Returns:
        CarlaEnvironment
    """
    env = CarlaEnvironment(batch_size, map_name)
    for wrapper in wrappers:
        env = wrapper(env)
    return env


load.batched = True
