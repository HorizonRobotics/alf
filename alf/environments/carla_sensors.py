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

import abc
from absl import logging
import gin
import math
import numpy as np
import weakref
import threading

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

MINIMUM_RENDER_WIDTH = 640
MINIMUM_RENDER_HEIGHT = 240


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
                to make sure that the data is correctly interpretted. Note that
                if the sensor receives event in the most recent frame,
                event.frame should be equal to current_frame - 1.
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
        self._prev_cached_frame = -1
        self._cached_impulse = None
        self._empty_impulse = np.zeros([max_num_collisions, 3],
                                       dtype=np.float32)
        self._collisions = []
        self._lock = threading.Lock()

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        self._frame = event.frame
        with self._lock:
            self._collisions.append([impulse.x, impulse.y, impulse.z])

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
        if current_frame == self._prev_cached_frame:
            return self._cached_impulse

        assert current_frame > self._prev_cached_frame, (
            "Cannot get frames %d older than previously cached one %d!" %
            (current_frame, self._prev_cached_frame))

        with self._lock:
            impulses = np.array(self._collisions, dtype=np.float32)
            self._collisions = []

        n = impulses.shape[0]
        if n == 0:
            impulses = self._empty_impulse
        elif n < self._max_num_collisions:
            impulses = np.concatenate([
                np.zeros([self._max_num_collisions - n, 3], dtype=np.float32),
                impulses
            ],
                                      axis=0)
        elif n > self._max_num_collisions:
            impulses = impulses[-self._max_num_collisions:]

        self._cached_impulse = impulses
        self._prev_cached_frame = current_frame
        return impulses


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
        self._prev_cached_frame = -1
        self._cached_points = None

        self._lock = threading.Lock()

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return

        self._frame = radar_data.frame

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

        with self._lock:
            self._detected_points = points

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
        if current_frame == self._prev_cached_frame:
            return self._cached_points

        assert current_frame > self._prev_cached_frame, (
            "Cannot get frames %d older than previously cached one %d!" %
            (current_frame, self._prev_cached_frame))

        with self._lock:
            self._cached_points = self._detected_points
            self._detected_points = self._empty_points

        self._prev_cached_frame = current_frame
        return self._cached_points


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
            image_size_y (int): image height in pixels.
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
            import cv2
            import pygame
            height, width = self._image.shape[1:3]
            image = np.transpose(self._image, (2, 1, 0))
            if width < MINIMUM_RENDER_WIDTH:
                height = height * MINIMUM_RENDER_WIDTH // width
                image = cv2.resize(
                    image,
                    dsize=(height, MINIMUM_RENDER_WIDTH),
                    interpolation=cv2.INTER_NEAREST)
            surface = pygame.surfarray.make_surface(image)
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


class World(object):
    """Keeping data for the world."""

    def __init__(self, world: carla.World, route_resolution=1.0):
        """
        Args:
            world (carla.World): the carla world instance
            route_resolution (float): the resolution in meters for planned route
        """
        self._world = world
        self._map = world.get_map()
        self._waypoints = self._map.generate_waypoints(2.0)
        self._actors = []  # including vehicles and walkers
        self._actor_dict = {}
        self._actor_locations = {}
        self._route_resolution = route_resolution

        dao = GlobalRoutePlannerDAO(
            world.get_map(), sampling_resolution=route_resolution)
        self._global_route_planner = GlobalRoutePlanner(dao)
        self._global_route_planner.setup()

    @property
    def route_resolution(self):
        """The sampling resolution of route."""
        return self._route_resolution

    def trace_route(self, origin, destination):
        """Find the route from ``origin`` to ``destination``.

        Args:
            origin (carla.Location):
            destination (carla.Location):
        Returns:
            list[tuple(carla.Waypoint, RoadOption)]
        """
        return self._global_route_planner.trace_route(origin, destination)

    def add_actor(self, actor: carla.Actor):
        self._actors.append(actor)
        self._actor_dict[actor.id] = actor

    def get_actors(self):
        return self._actors

    def update_actor_location(self, aid, loc):
        """Update the next location of the actor.

        Args:
            aid (int): actor id
            loc (carla.Location): location of the actor
        """
        self._actor_locations[aid] = loc

    def get_actor_location(self, aid):
        """Get the latest location of the actor.

        The reason of using this instead of calling ``carla.Actor.get_location()``
        directly is that the location of actors may not have been updated before
        world.tick().

        Args:
            aid (int): actor id
        Returns:
            carla.Location:
        """
        loc = self._actor_locations.get(aid, None)
        if loc is None:
            loc = self._actor_dict[aid].get_location()
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

    Note that the route is fixed (not change based on current vehicle location).
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
        Returns:
            The total length of the route in meters
        """
        start = self._alf_world.get_actor_location(self._parent.id)
        self._route = self._alf_world.trace_route(start, destination)
        self._waypoints = np.array([[
            wp.transform.location.x, wp.transform.location.y,
            wp.transform.location.z
        ] for wp, _ in self._route])
        self._road_option = np.array(
            [road_option for _, road_option in self._route])
        self._nearest_index = 0
        d = self._waypoints[:-1] - self._waypoints[1:]
        self._num_waypoints = self._waypoints.shape[0]
        return np.sum(np.sqrt(d * d))

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
            np.ndarray: 8 3-D positions of future waypoints on the route. Note
            that the positions are absolution coordinates. However, the ``Player``
            will transform them to egocentric coordinates as the observation for
            ``Player``
        """
        loc = self._alf_world.get_actor_location(self._parent.id)
        loc = np.array([loc.x, loc.y, loc.y])
        nearby_waypoints = self._waypoints[self._nearest_index:self.
                                           _nearest_index + self.WINDOW]
        dist = np.linalg.norm(nearby_waypoints - loc, axis=1)
        self._nearest_index = self._nearest_index + np.argmin(dist)
        indices = np.minimum(self._nearest_index + self._future_indices,
                             self._num_waypoints - 1)
        return self._waypoints[indices]

    @property
    def num_waypoints(self):
        """The number of waypoints in the route."""
        return self._num_waypoints

    def get_waypoint(self, i):
        """Get the coordinate of waypoint ``i``.

        Args:
            i (int): waypoint index
        Returns:
            numpy.ndarray: 3-D vector of location
        """
        return self._waypoints[i]

    def get_next_waypoint_index(self):
        """Get the index next waypoint.

        The next waypoint is the waypoint after the nearest waypoint to the car.

        Returns:
            int: index of the next waypoint
        """
        return min(self._nearest_index + 1, self._num_waypoints - 1)
