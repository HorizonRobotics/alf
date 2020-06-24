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

Make sure you are using python3.7

"""

import abc
from absl import logging
import gin
import math
import numpy as np
import os
import random
import subprocess
import sys
import time
import torch
import weakref

try:
    import carla
except ImportError:
    carla = None

if carla is not None:
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
        from agents.navigation.local_planner import RoadOption
    except ImportError:
        logging.fatal("Cannot import carla agents package. Please add "
                      "$CARLA_ROOT/PythonAPI/carla to your PYTHONPATH")
        carla = None

import alf
import alf.data_structures as ds
from alf.utils import common
from .suite_socialbot import _get_unused_port
from .alf_environment import AlfEnvironment


def is_available():
    return carla is not None


class SensorBase(abc.ABC):
    """Base class for sersors."""

    def __init__(self, parent_actor):
        """
        Args:
            parent_actor (carla.Actor): the parent actor of this sensor
        """
        self._sensor = None
        self._parent = parent_actor

    def destroy(self):
        """Return the commands for destroying this sensor.

        Use ``carla.Client.apply_batch_sync()`` to actually destroy the sensor.

        Returns:
            list[carla.command]: the commands used to destroy the sensor.
        """
        if self._sensor is not None:
            self._sensor.stop()
            return [carla.command.DestroyActor(self._sensor)]
        else:
            return []

    @abc.abstractmethod
    def get_current_observation(self, current_frame):
        """Get the current observation.

        Args:
            current_frame (int): current frame no. For some sensors, they may
                not receive any data in the most recent tick. ``current_frame``
                will be compared against the frame no. of the last received data
                to make sure that the data is correctly interpretted.
        Returns:
            nested np.ndarray: sensor data received in the last tick.
        """

    @abc.abstractmethod
    def observation_spec(self):
        """Get the observation spec of this sensor.

        Returns:
            nested TensorSpec:
        """

    @abc.abstractmethod
    def observation_desc(self):
        """Get the description about the observation of this sensor.

        Returns:
            nested str: each str corresponds to one TensorSpec from
            ``observatin_spec()``.
        """


@gin.configurable
class CollisionSensor(SensorBase):
    """CollisionSensor for getting collision signal.

    It gets the impulses from the collisions during the last tick.

    TODO: include event.other_actor in the sensor result.
    """

    def __init__(self, parent_actor, max_num_collisions=4):
        """
        Args:
            parent_actor (carla.Actor): the parent actor of this sensor
            max_num_collisions (int): maximal number of collisions to be included
        """
        super().__init__(parent_actor)
        self._max_num_collisions = max_num_collisions
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self._sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda event: CollisionSensor._on_collision(
            weak_self, event))
        self._frame = 0
        self._empty_impulse = np.zeros([max_num_collisions, 3],
                                       dtype=np.float32)
        self._impulse = self._empty_impulse
        self._collisions = []
        self._empty_impulses = np.zeros([max_num_collisions, 3],
                                        dtype=np.float32)

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        if self._frame != event.frame:
            self._collisions = []
        self._collisions.append([impulse.x, impulse.y, impulse.z])
        self._frame = event.frame

    def observation_spec(self):
        return alf.TensorSpec([self._max_num_collisions, 3])

    def observation_desc(self):
        return (
            "Impulses from collision during the last tick. Each impulse is "
            "a 3-D vector. At most %d collisions are used. The result is padded "
            "with zeros if there are less than %d collisions" %
            (self._max_num_collisions, self._max_num_collisions))

    def get_current_observation(self, current_frame):
        """Get the current observation.

        Args:
            current_frame (int): current frame no. CollisionSensor may not
                not receive any data in the most recent tick. ``current_frame``
                will be compared against the frame no. of the last received data
                to make sure that the data is correctly interpretted.
        Returns:
            np.ndarray: Impulses from collision during the last tick. Each
                impulse is a 3-D vector. At most ``max_num_collisions``
                collisions are used. The result is padded with zeros if there
                are less than ``max_num_collisions`` collisions

        """
        if current_frame == self._frame:
            impulses = np.array(self._collisions, dtype=np.float32)
            n = len(self._collisions)
            if n < self._max_num_collisions:
                impulses = np.concatenate([
                    np.zeros([self._max_num_collisions - n, 3],
                             dtype=np.float32), impulses
                ],
                                          axis=0)
            elif n > self._max_num_collisions:
                impulses = impulses[-self._max_num_collisions:]
            return impulses
        else:
            return self._empty_impulse


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(SensorBase):
    """LaneInvasionSensor for detecting lane invasion.

    Lane invasion cannot be directly observed by raw sensors used by real cars.
    So main purpose of this is to provide training signal (e.g. reward).

    TODO: not completed.
    """

    def __init__(self, parent_actor):
        """
        Args:
            parent_actor (carla.Actor): the parent actor of this sensor
        """
        super().__init__(parent_actor)
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self._sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda event: LaneInvasionSensor._on_invasion(
            weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

    def get_current_observation(self, current_frame):
        raise NotImplementedError()

    def observation_spec(self):
        raise NotImplementedError()

    def observation_desc(self):
        raise NotImplementedError()


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================
class GnssSensor(SensorBase):
    """GnssSensor for sensing GPS location."""

    def __init__(self, parent_actor):
        """
        Args:
            parent_actor (carla.Actor): the parent actor of this sensor
        """
        super().__init__(parent_actor)
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self._sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=1.0, z=2.8)),
            attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda event: GnssSensor._on_gnss_event(
            weak_self, event))
        self._gps_location = np.zeros([3], dtype=np.float32)
        self._frame = 0

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self._gps_location = np.array(
            [event.latitude, event.longitude, event.altitude],
            dtype=np.float32)
        self._frame = event.frame

    def observation_spec(self):
        return alf.TensorSpec([3])

    def observation_desc(self):
        return "A vector of [latitude (degrees), longitude (degrees), altitude (meters to be confirmed)]"

    def get_current_observation(self, current_frame):
        """
        Args:
            current_frame (int): not used
        Returns:
            np.ndarray: A vector of [latitude (degrees), longitude (degrees),
                altitude (meters to be confirmed)]
        """
        return self._gps_location


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================
class IMUSensor(SensorBase):
    """IMUSensor for sensing accelaration and rotation."""

    def __init__(self, parent_actor):
        """
        Args:
            parent_actor (carla.Actor): the parent actor of this sensor
        """
        super().__init__(parent_actor)
        self._compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self._sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda sensor_data: IMUSensor._IMU_callback(
            weak_self, sensor_data))
        self._imu_reading = np.zeros([7], dtype=np.float32)
        self._frame = 0

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        if not math.isnan(sensor_data.compass):
            self._compass = sensor_data.compass
        else:
            logging.warning(
                "Got nan for compass. Use the previous compass reading.")
        imu_reading = np.array([
            sensor_data.accelerometer.x, sensor_data.accelerometer.y,
            sensor_data.accelerometer.z, sensor_data.gyroscope.x,
            sensor_data.gyroscope.y, sensor_data.gyroscope.z, self._compass
        ],
                               dtype=np.float32)
        self._imu_reading = np.clip(imu_reading, -99.9, 99.9)
        self._frame = sensor_data.frame

    def observation_spec(self):
        return alf.TensorSpec([7])

    def observation_desc(self):
        return (
            "7-D vector of [accelaration, gyroscope, compass], where "
            "accelaration is a 3-D vector in m/s^2, gyroscope is angular "
            "velocity in rad/s^2, and compass is orientation with regard to the "
            "North ((0.0, 1.0, 0.0) in Unreal Engine) in radians.")

    def get_current_observation(self, current_frame):
        return self._imu_reading


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================
@gin.configurable
class RadarSensor(SensorBase):
    """RadarSensor for detecting obstacles."""

    def __init__(self,
                 parent_actor,
                 xyz=(2.8, 0., 1.0),
                 pyr=(5., 0., 0.),
                 max_num_detections=200):
        """
        Args:
            parent_actor (carla.Actor): the parent actor of this sensor.
            xyz (tuple[float]): the attachment positition (x, y, z) relative to
                the parent_actor.
            pyr (tuple[float]): the attachment rotation (pitch, yaw, roll) in
                degrees.
            max_num_detections (int): maximal number of detection points.
        """
        super().__init__(parent_actor)
        self._velocity_range = 7.5  # m/s
        self._max_num_detections = max_num_detections

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self._sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(*xyz), carla.Rotation(*pyr)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda radar_data: RadarSensor._Radar_callback(
            weak_self, radar_data))

        self._empty_points = np.zeros([max_num_detections, 4],
                                      dtype=np.float32)
        self._detected_points = self._empty_points
        self._frame = 0

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        points = np.frombuffer(radar_data.raw_data, dtype=np.float32)
        points = np.reshape(points, (len(radar_data), 4))
        n = len(radar_data)
        if n < self._max_num_detections:
            points = np.concatenate([
                np.zeros([self._max_num_detections - n, 4], dtype=np.float32),
                points
            ],
                                    axis=0)
        elif n > self._max_num_detections:
            points = points[-self._max_num_detections:, :]
        self._detected_points = points
        self._frame = radar_data.frame

    def observation_spec(self):
        return alf.TensorSpec([self._max_num_detections, 4])

    def observation_desc(self):
        return (
            "A set of detected points. Each detected point is a 4-D vector "
            "of [vel, altitude, azimuth, depth], where vel is the velocity of "
            "the detected object towards the sensor in m/s, altitude is the "
            "altitude angle of the detection in radians, azimuth is the azimuth "
            "angle of the detection in radians, and depth id the distance from "
            "the sensor to the detection in meters.")

    def get_current_observation(self, current_frame):
        """
        Args:
            current_frame (int): current frame no. RadarSensor may not receive
                any data in the most recent tick. ``current_frame`` will be
                compared against the frame no. of the last received data to make
                sure that the data is correctly interpretted.
        Returns:
            np.ndarray: A set of detected points. Each detected point is a 4-D
                vector of [vel, altitude, azimuth, depth], where vel is the
                velocity of the detected object towards the sensor in m/s,
                altitude is the altitude angle of the detection in radians,
                azimuth is the azimuth angle of the detection in radians, and
                depth id the distance from the sensor to the detection in meters.
        """
        if current_frame == self._frame:
            return self._detected_points
        else:
            return self._empty_points


def geo_distance(loc1, loc2):
    """
    Args:
        loc1 (np.array): [latitude, longitude, altitude]. The units for altitude
            is meter.
        loc2 (np.array):
    Returns:
        float: distance in meters
    """
    earth_radius = 6371 * 1000
    d2r = math.pi / 180

    d = loc1 - loc2
    dlat = d[0] * d2r
    dlon = d[1] * d2r
    lat1 = loc1[0] * d2r
    lat2 = loc2[0] * d2r
    a = np.sin(
        0.5 * dlat)**2 + np.sin(0.5 * dlon)**2 * np.cos(lat1) * np.cos(lat2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    c = earth_radius * c
    return np.sqrt(c * c + d[2] * d[2])


# ==============================================================================
# -- CameraSensor -------------------------------------------------------------
# ==============================================================================
@gin.configurable
class CameraSensor(SensorBase):
    """CameraSensor."""

    def __init__(
            self,
            parent_actor,
            sensor_type='sensor.camera.rgb',
            xyz=(1.6, 0., 1.7),
            pyr=(0., 0., 0.),
            attachment_type='rigid',
            fov=90.0,
            fstop=1.4,
            gamma=2.2,
            image_size_x=640,
            image_size_y=480,
            iso=1200.0,
    ):
        """
        Args:
            parent_actor (carla.Actor): the parent actor of this sensor
            sensor_type (str): 'sensor.camera.rgb', 'sensor.camera.depth',
                'sensor.camera.semantic_segmentation'
            attachment_type (str): There are two types of attachement. 'rigid':
                the object follow its parent position strictly. 'spring_arm':
                the object expands or retracts depending on camera situation.
            xyz (tuple[float]): the attachment positition (x, y, z) relative to
                the parent_actor.
            pyr (tuple[float]): the attachment rotation (pitch, yaw, roll) in
                degrees.
            fov (str): horizontal field of view in degrees.
            image_size_x (int): image width in pixels.
            image_size_t (int): image height in pixels.
            gamma (float): target gamma value of the camera.
            iso (float): the camera sensor sensitivity.
        """
        super().__init__(parent_actor)
        attachment_type_map = {
            'rigid': carla.AttachmentType.Rigid,
            'spring_arm': carla.AttachmentType.SpringArm,
        }
        assert attachment_type in attachment_type_map, (
            "Unknown attachment_type %s" % attachment_type)
        self._attachment_type = attachment_type_map[attachment_type]
        self._camera_transform = carla.Transform(
            carla.Location(*xyz), carla.Rotation(*pyr))
        self._sensor_type = sensor_type

        sensor_map = {
            'sensor.camera.rgb': (carla.ColorConverter.Raw, 3),
            'sensor.camera.depth': (carla.ColorConverter.LogarithmicDepth, 1),
            'sensor.camera.semantic_segmentation': (carla.ColorConverter.Raw,
                                                    1),
        }
        assert sensor_type in sensor_map, "Unknown sensor type %s" % sensor_type
        conversion, num_channels = sensor_map[sensor_type]

        self._conversion = conversion
        self._observation_spec = alf.TensorSpec(
            [num_channels, image_size_y, image_size_x], dtype='uint8')

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find(sensor_type)

        attributes = dict(
            fov=fov,
            fstop=fstop,
            gamma=gamma,
            image_size_x=image_size_x,
            image_size_y=image_size_y,
            iso=iso)
        for name, val in attributes.items():
            if bp.has_attribute(name):
                bp.set_attribute(name, str(val))

        self._sensor = self._parent.get_world().spawn_actor(
            bp,
            self._camera_transform,
            attach_to=self._parent,
            attachment_type=self._attachment_type)
        # We need to pass the lambda a weak reference to self to avoid
        # circular reference.
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda image: CameraSensor._parse_image(
            weak_self, image))
        self._frame = 0
        self._image = np.zeros([num_channels, image_size_y, image_size_x],
                               dtype=np.uint8)

    def render(self, display):
        """Render the camera image to a pygame display.

        Args:
            display (pygame.Surface): the display surface to draw the image
        """
        if self._image is not None:
            import pygame
            surface = pygame.surfarray.make_surface(
                np.transpose(self._image, (2, 1, 0)))
            display.blit(surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self._conversion)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = np.transpose(array, (2, 0, 1))
        self._image = array.copy()

    def observation_desc(self):
        height, width = self.observation_spec().shape[1:3]
        if self._sensor_type == "sensor.camera.rgb":
            return "3x%dx%d RGB image " % (height, width)
        elif self._sensor_type == "sensor.camera.depth":
            return (
                "1x%dx%d depth image. The depth is in logarithmic scale. "
                "See https://carla.readthedocs.io/en/latest/ref_sensors/#rgb-camera "
                "for detail" % (height, width))
        elif self._sensor_type == "sensor.camera.semantic_segmentation":
            return (
                "1x%dx%d semantic label image. Current possible labels are "
                "0-12. See https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera "
                "for detail" % (height, width))

    def observation_spec(self):
        return self._observation_spec

    def get_current_observation(self, current_frame):
        """
        Args:
            current_frame (int): not used.
        Returns:
            np.ndarray: The shape is [num_channels, image_size_y, image_size_x],
                where num_channels is 3 for rgb sensor, and 1 for other sensors.
        """
        return self._image


def _calculate_relative_position(self_transform, location):
    """
    Args:
        self_transform (carla.Transform): transform of self actor
        location (np.ndarray): shape is [3] or [N, 3]
    Returns:
        np.ndarray: shape is same as location
    """
    trans = self_transform
    self_loc = trans.location
    yaw = math.radians(trans.rotation.yaw)

    self_loc = np.array([self_loc.x, self_loc.y, self_loc.z])
    cos, sin = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos, -sin, 0.], [sin, cos, 0.], [0., 0., 1.]])
    return np.matmul(location - self_loc, rot).astype(np.float32)


class World(object):
    """Keeping data for the world."""

    def __init__(self, world: carla.World):
        self._world = world
        self._map = world.get_map()
        self._waypoints = self._map.generate_waypoints(2.0)
        self._vehicles = []
        self._vehicle_dict = {}
        self._vehicle_locations = {}

        dao = GlobalRoutePlannerDAO(world.get_map(), sampling_resolution=1.0)
        self._global_route_planner = GlobalRoutePlanner(dao)
        self._global_route_planner.setup()

    def trace_route(self, origin, destination):
        """Find the route from ``origin`` to ``destination``.

        Args:
            origin (carla.Location):
            destination (carla.Location):
        Returns:
            list[tuple(carla.Waypoint, RoadOption)]
        """
        return self._global_route_planner.trace_route(origin, destination)

    def add_vehicle(self, actor: carla.Actor):
        self._vehicles.append(actor)
        self._vehicle_dict[actor.id] = actor

    def get_vehicles(self):
        return self._vehicles

    def update_vehicle_location(self, vid, loc):
        """Update the next location of the vehicle.

        Args:
            vid (int): vehicle id
            loc (carla.Location): location of the vehicle
        """
        self._vehicle_locations[vid] = loc

    def get_vehicle_location(self, vid):
        """Get the latest location of the vehicle.

        The reason of using this instead of calling ``carla.Actor.get_location()``
        directly is that the location of vehicle may not have been updated before
        world.tick().

        Args:
            vid (int): vehicle id
        Returns:
            carla.Location:
        """
        loc = self._vehicle_locations.get(vid, None)
        if loc is None:
            loc = self._vehicle_dict[vid].get_location()
        return loc

    def get_waypoints(self):
        """Get the coordinates of way points

        Returns:
            list[carla.Waypoint]:
        """
        return self._waypoints

    def transform_to_geolocation(self, location: carla.Location):
        """Transform a map coordiate to geo coordinate.

        Returns:
            np.ndarray: ``[latitude, longitude, altidude]``
        """
        loc = self._map.transform_to_geolocation(location)
        return np.array([loc.latitude, loc.longitude, loc.altitude])


class NavigationSensor(SensorBase):
    """Generating future waypoints on the route.

    Note that the route is fixed (not change based on current vehicle location)
    """

    WINDOW = 5

    def __init__(self, parent_actor, alf_world: World):
        """
        Args:
            parent_actor (carla.Actor): the parent actor of this sensor
            alf_world (World):
        """
        super().__init__(parent_actor)
        self._alf_world = alf_world
        self._future_indices = np.array([1, 3, 10, 30, 100, 300, 1000, 3000])

    def set_destination(self, destination):
        """Set the navigation destination.

        Args:
            destination (carla.Location):
        """
        start = self._alf_world.get_vehicle_location(self._parent.id)
        self._route = self._alf_world.trace_route(start, destination)
        self._waypoints = np.array([[
            wp.transform.location.x, wp.transform.location.y,
            wp.transform.location.z
        ] for wp, _ in self._route])
        self._road_option = np.array(
            [road_option for _, road_option in self._route])
        self._nearest_index = 0

    def observation_spec(self):
        return alf.TensorSpec([len(self._future_indices), 3])

    def observation_desc(self):
        return ("Positions of the %s future locations in the route." % len(
            self._future_indices))

    def get_current_observation(self, current_frame):
        """Get the current observation.

        The observation is an 8x3 array consists of the posistions of 8 future
        locations on the routes.

        Args:
            current_frame (int): not used.
        Returns:
            np.ndarray: positions of future waypoints on the route.
        """
        loc = self._alf_world.get_vehicle_location(self._parent.id)
        loc = np.array([loc.x, loc.y, loc.y])
        nearby_waypoints = self._waypoints[self._nearest_index:self.
                                           _nearest_index + self.WINDOW]
        dist = np.linalg.norm(nearby_waypoints - loc, axis=1)
        self._nearest_index = self._nearest_index + np.argmin(dist)
        indices = np.minimum(self._nearest_index + self._future_indices,
                             self._waypoints.shape[0] - 1)
        return self._waypoints[indices]


def _to_numpy_loc(loc: carla.Location):
    return np.array([loc.x, loc.y, loc.z])


class Player(object):
    """Player is a vehicle with some sensors.

    An episode terminates if the vehicle arrives at the goal or the time exceeds
    ``initial_distance / min_speed``.

    At each step, the reward is given based on the following components:
    1. Arriving goal:  ``success_reward``
    2. Moving in the navigation direction: the number of meters moved
    """

    def __init__(self,
                 actor,
                 alf_world,
                 success_reward=100.,
                 success_distance_thresh=5.0,
                 min_speed=5.):
        """
        Args:
            actor (carla.Actor): the carla actor object
            alf_world (Wolrd): the world containing the player
            success_reward (float): the reward for arriving the goal location.
            success_distance_thresh (float): success is achieved if the current
                location is with such distance of the goal
            min_speed (float): unit is m/s. Failure if initial_distance / min_speed
                seconds passed
        """
        self._actor = actor
        self._alf_world = alf_world
        self._collision_sensor = CollisionSensor(actor)
        self._gnss_sensor = GnssSensor(actor)
        self._imu_sensor = IMUSensor(actor)
        self._camera_sensor = CameraSensor(actor)
        self._lane_invasion_sensor = LaneInvasionSensor(actor)
        self._radar_sensor = RadarSensor(actor)
        self._navigation = NavigationSensor(actor, alf_world)
        self._success_reward = success_reward
        self._success_distance_thresh = success_distance_thresh
        self._min_speed = min_speed
        self._delta_seconds = actor.get_world().get_settings(
        ).fixed_delta_seconds

        self._observation_sensors = {
            'collision': self._collision_sensor,
            'gnss': self._gnss_sensor,
            'imu': self._imu_sensor,
            'camera': self._camera_sensor,
            'radar': self._radar_sensor,
            'navigation': self._navigation,
        }

        self._observation_spec = dict()
        self._observation_desc = dict()
        for sensor_name, sensor in self._observation_sensors.items():
            self._observation_spec[sensor_name] = sensor.observation_spec()
            self._observation_desc[sensor_name] = sensor.observation_desc()
        self._observation_spec['goal'] = alf.TensorSpec([3])
        self._observation_spec['speed'] = alf.TensorSpec(())

        # UE4 coordinate system is right handed:
        # https://forums.unrealengine.com/development-discussion/c-gameplay-programming/103787-ue4-coordinate-system-not-right-handed
        self._observation_desc['goal'] = (
            "Target location relative to the vehicle coordinate system in "
            "meters. X axis: front, Y axis: right, Z axis: up. Only the "
            "rotation around Z axis is taken into account when calculating the "
            "vehicle's coordinate system.")
        self._observation_desc['navigation'] = (
            'Relative positions of the future waypoints in the route')
        self._observation_desc['speed'] = "Speed in m/s"
        self._control = carla.VehicleControl()
        self.reset()

        # for rendering
        self._display = None
        self._font = None
        self._clock = None

    def reset(self):
        """Reset the player location and goal.

        Use ``carla.Client.apply_batch_sync()`` to actually reset.

        Returns:
            list[carla.command]:
        """

        wp = random.choice(self._alf_world.get_waypoints())
        goal_loc = wp.transform.location
        self._goal_location = np.array([goal_loc.x, goal_loc.y, goal_loc.z],
                                       dtype=np.float32)

        forbidden_locations = []
        for v in self._alf_world.get_vehicles():
            if v.id == self._actor.id:
                continue
            forbidden_locations.append(
                self._alf_world.get_vehicle_location(v.id))

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

        self._fail_frame = None
        self._done = False
        p = np.array([loc.x, loc.y, loc.z])
        self._prev_location = loc
        self._prev_action = np.zeros(
            self.action_spec().shape, dtype=np.float32)
        self._alf_world.update_vehicle_location(self._actor.id, loc)

        self._navigation.set_destination(goal_loc)

        return commands

    def destroy(self):
        """Get the commands for destroying the player.

        Use carla.Client.apply_batch_sync() to actually destroy the sensor.

        Returns:
            list[carla.command]:
        """
        commands = []
        commands.extend(self._collision_sensor.destroy())
        commands.extend(self._gnss_sensor.destroy())
        commands.extend(self._imu_sensor.destroy())
        commands.extend(self._camera_sensor.destroy())
        commands.extend(self._lane_invasion_sensor.destroy())
        commands.extend(self._radar_sensor.destroy())
        commands.append(carla.command.DestroyActor(self._actor))
        if self._display is not None:
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

        The action is a 4-D vector of [throttle, steer, brake, reverse], where
        throttle is in [-1.0, 1.0] (negative value is same as zero), steer is in
        [-1.0, 1.0], brake is in [-1.0, 1.0] (negative value is same as zero),
        and reverse is interpreted as a boolean value with values greater than
        0.5 corrsponding to True.

        Returns:
            nested BoundedTensorSpec:
        """
        return alf.BoundedTensorSpec([4],
                                     minimum=[-1., -1., -1., 0.],
                                     maximum=[1., 1., 1., 1.])

    def action_desc(self):
        """Get the description about the action.

        Returns:
            nested str: each str corresponds to one TensorSpec from
            ``action_spec()``.
        """
        return (
            "4-D vector of [throttle, steer, brake, reverse], where "
            "throttle is in [-1.0, 1.0] (negative value is same as zero), "
            "steer is in [-1.0, 1.0], brake is in [-1.0, 1.0] (negative value "
            "is same as zero), and reverse is interpreted as a boolean value "
            "with values greater than 0.5 corrsponding to True.")

    def _get_goal(self):
        return _calculate_relative_position(self._actor.get_transform(),
                                            self._goal_location)

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
        self._alf_world.update_vehicle_location(self._actor.id,
                                                self._actor.get_location())
        v = self._actor.get_velocity()
        obs['speed'] = np.float32(
            np.float32(v.x * v.x + v.y * v.y + v.z * v.z))
        self._current_distance = np.linalg.norm(obs['goal'])
        reward = 0.
        discount = 1.0
        if self._fail_frame is None:
            step_type = ds.StepType.FIRST
            max_frames = math.ceil(
                self._current_distance / self._min_speed / self._delta_seconds)
            self._fail_frame = current_frame + max_frames
        elif (self._current_distance < self._success_distance_thresh
              and self._actor.get_velocity() == carla.Location(0., 0., 0.)):
            # TODO: include waypoint orientation as success critiria
            self._done = True
            step_type = ds.StepType.LAST
            reward += self._success_reward
            discount = 0.0
        elif current_frame >= self._fail_frame:
            self._done = True
            step_type = ds.StepType.LAST
        else:
            step_type = ds.StepType.MID

        prev_loc = _to_numpy_loc(self._prev_location)
        curr_loc = _to_numpy_loc(self._actor.get_location())
        goal0 = obs['navigation'][2]  # This is about 10m ahead in the route
        reward += (np.linalg.norm(prev_loc - goal0) -
                   np.linalg.norm(curr_loc - goal0))
        obs['navigation'] = _calculate_relative_position(
            self._actor.get_transform(), obs['navigation'])

        self._current_time_step = ds.TimeStep(
            step_type=step_type,
            reward=np.float32(reward),
            discount=np.float32(discount),
            observation=obs,
            prev_action=self._prev_action)
        return self._current_time_step

    def act(self, action):
        """Generate the carla command for taking the given action.

        Use ``carla.Client.apply_batch_sync()`` to actually destroy the sensor.

        Args:
            action (nested np.ndarray):
        Returns:
            list[carla.command]:
        """
        self._prev_location = self._actor.get_location()
        if self._done:
            return self.reset()
        self._control.throttle = max(float(action[0]), 0.0)
        self._control.steer = float(action[1])
        self._control.brake = max(float(action[2]), 0.0)
        self._control.reverse = bool(action[3] > 0.5)
        self._prev_action = action

        return [carla.command.ApplyVehicleControl(self._actor, self._control)]

    def render(self, mode):
        """Render the simulation.

        Args:
            mode (str): one of ['rgb', 'human']
        Returns:
            one of the following:
                - None: if mode is 'human'
                - np.ndarray: the image of shape [height, width, channeles] if
                    mode is 'rgb'
        """
        if mode == 'human':
            import pygame
            if self._display is None:
                pygame.init()
                pygame.font.init()
                self._clock = pygame.time.Clock()
                height, width = self._camera_sensor.observation_spec(
                ).shape[1:3]
                height = max(height, 480)
                width = max(width, 640)
                self._display = pygame.display.set_mode(
                    (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)

            self._clock.tick_busy_loop(1000)
            self._camera_sensor.render(self._display)
            obs = self._current_time_step.observation
            info_text = [
                'FPS: %6.2f' % self._clock.get_fps(),
                'GPS:  (%7.4f, %8.4f, %5.2f)' % tuple(obs['gnss'].tolist()),
                'Goal: (%7.1f, %8.1f, %5.1f)' % tuple(obs['goal'].tolist()),
                'Ahead: (%7.1f, %8.1f, %5.1f)' % tuple(
                    obs['navigation'][2].tolist()),
                'Distance: %7.2f' % np.linalg.norm(obs['goal']),
                'Speed: %5.1f km/h' % (3.6 * obs['speed']),
                'Acceleration: (%4.1f, %4.1f, %4.1f)' % tuple(
                    obs['imu'][0:3].tolist()),
                'Compass: %5.1f' % math.degrees(float(obs['imu'][6])),
                'Throttle: %4.2f' % self._control.throttle,
                'Brake:    %4.2f' % self._control.brake,
                'Steer:    %4.2f' % self._control.steer,
                'Reverse:  %4s' % self._control.reverse,
                'Reward: %.1f' % float(self._current_time_step.reward),
            ]
            self._draw_text(info_text)
            pygame.display.flip()
        elif mode == 'rgb':
            return np.transpose(self._current_time_step.observation['camera'],
                                (1, 2, 0))
        else:
            raise ValueError("Unsupported render mode: %s" % mode)

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
        info_surface = pygame.Surface((240, 220))
        info_surface.set_alpha(100)
        self._display.blit(info_surface, (0, 0))
        v_offset = 4
        for item in texts:
            surface = self._font.render(item, True, (255, 255, 255))
            self._display.blit(surface, (8, v_offset))
            v_offset += 18


def _exec(command):
    stream = os.popen(command)
    ret = stream.read()
    stream.close()
    return ret


@gin.configurable
class CarlaServer(object):
    """CarlaServer for doing the simulation."""

    def __init__(self,
                 rpc_port=2000,
                 streaming_port=2001,
                 docker_image="horizonrobotics/alf:0.0.3-carla",
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
            # server processs so that we can kill it.
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


@gin.configurable
class CarlaEnvironment(AlfEnvironment):
    """Carla simulation environment.

    In order to use it, you need to either download a valid docker image or
    a Carla package.
    """

    def __init__(self,
                 batch_size,
                 map_name,
                 vehicle_filter='vehicle.*',
                 safe=True,
                 step_time=0.05):
        """
        Args:
            safe (bool): avoid spawning vehicles prone to accidents.
            step_time (float): how many seconds does each step of simulation represents.
        """
        super().__init__()

        with _get_unused_port(2000, n=2) as (rpc_port, streaming_port):
            self._server = CarlaServer(rpc_port, streaming_port)

        self._batch_size = batch_size
        self._world = None
        try:
            for i in range(10):
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

        self._client.set_timeout(20)
        self._alf_world = World(self._world)
        self._safe = safe
        self._vehicle_filter = vehicle_filter

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = step_time

        self._world.apply_settings(settings)
        self._map_name = map_name
        self._spawn()

    def _spawn(self):
        blueprints = self._world.get_blueprint_library().filter(
            self._vehicle_filter)
        assert len(
            blueprints) > 0, "Cannot find vehicle '%s'" % self._vehicle_filter
        if self._safe:
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
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        assert len(
            blueprints
        ) > 0, "Cannot find safe vehicle '%s'" % self._vehicle_filter

        spawn_points = self._world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        if self._batch_size < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif self._batch_size > number_of_spawn_points:
            raise ValueError(
                "requested %d vehicles, but could only find %d spawn points" %
                (self._batch_size, number_of_spawn_points))

        SpawnActor = carla.command.SpawnActor

        commands = []
        for transform in spawn_points[:self._batch_size]:
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(
                    blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(
                    blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            commands.append(SpawnActor(blueprint, transform))

        # TODO: add walkers and autopilot vehicles, see carla/PythonAPI/examples/spawn_npc.py

        self._players = []
        for response in self._client.apply_batch_sync(commands, True):
            if response.error:
                logging.error(response.error)
            else:
                vehicle = self._world.get_actor(response.actor_id)
                self._players.append(Player(vehicle, self._alf_world))
                self._alf_world.add_vehicle(vehicle)

        assert len(self._players) == self._batch_size, (
            "Fail to create %s vehicles" % self._batch_size)

        self._observation_spec = self._players[0].observation_spec()
        self._action_spec = self._players[0].action_spec()

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

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._batch_size

    def env_info_spec(self):
        return {}

    def observation_spec(self):
        return self._observation_spec

    def observation_desc(self):
        return self._players[0].observation_desc()

    def action_spec(self):
        return self._action_spec

    def action_desc(self):
        return self._players[0].action_desc()

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
        self._current_frame = self._world.tick()
        return self._get_current_time_step()

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
        return self._get_current_time_step()


@gin.configurable(whitelist=[])
def load(map_name, batch_size):
    """Load CaraEnvironment

    Args:
        map_name (str): name of the map. Currently available maps are:
            'Town01, Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07',
            and 'Town10HD'
        batch_size (int): the number of vehicles in the simulation.
    """
    return CarlaEnvironment(batch_size, map_name)


load.batched = True
