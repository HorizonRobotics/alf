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

import abc
from absl import logging
import math
import numpy as np
import weakref
import threading
from unittest.mock import Mock

try:
    import carla
except ImportError:
    # create 'carla' as a mock to not break python argument type hints
    carla = Mock()

if not isinstance(carla, Mock):
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
        from agents.navigation.local_planner import RoadOption
    except ImportError:
        logging.fatal("Cannot import carla agents package. Please add "
                      "$CARLA_ROOT/PythonAPI/carla to your PYTHONPATH")
        carla = Mock()

import alf
from alf.data_structures import namedtuple

MINIMUM_RENDER_WIDTH = 640
MINIMUM_RENDER_HEIGHT = 240

MAXIMUM_RENDER_WIDTH = 1280
MAXIMUM_RENDER_HEIGHT = 640


def get_scaled_image_size(height, width):
    """Compute properly scaled image size.

    The scaled image height and width are calculated based on the minimum and
    maximum allowed sizes for rendering, while keeping the aspect ratio of the
    image unchanged.
    If both the height and width are within the bound, no scaling is applied.

    Returns:
        tuple:
        - scaled_height (int): scaled image height
        - scaled_width (int): scaled image width
    """
    min_scaling_factor = max(
        float(MINIMUM_RENDER_HEIGHT) / height,
        float(MINIMUM_RENDER_WIDTH) / width)

    max_scaling_factor = min(
        float(MAXIMUM_RENDER_HEIGHT) / height,
        float(MAXIMUM_RENDER_WIDTH) / width)

    if min_scaling_factor > 1:
        scaling_factor = min(min_scaling_factor, max_scaling_factor)
    elif max_scaling_factor < 1:
        scaling_factor = max(min_scaling_factor, max_scaling_factor)
    else:
        scaling_factor = 1

    scaled_height = int(height * scaling_factor)
    scaled_width = int(width * scaling_factor)
    return scaled_height, scaled_width


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


@alf.configurable
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
@alf.configurable
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
            xyz (tuple[float]): the attachment position (x, y, z) relative to
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
# -- CameraSensor --------------------------------------------------------------
# ==============================================================================
@alf.configurable
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
            xyz (tuple[float]): the attachment position (x, y, z) relative to
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
        self._fov = fov

    def render(self, display):
        """Render the camera image to a pygame display.

        Args:
            display (pygame.Surface): the display surface to draw the image
        """
        if self._image is not None:
            import cv2
            import pygame
            height, width = self._image.shape[1:3]
            # (c, y, x) => (x, y, c)
            image = np.transpose(self._image, (2, 1, 0))

            if self._sensor_type.startswith(
                    'sensor.camera.semantic_segmentation'):
                image = image * 10  # scale the label map for better viewing

            scaled_height, scaled_width = get_scaled_image_size(height, width)

            if scaled_height != height or scaled_width != width:
                image = cv2.resize(
                    image,
                    dsize=(scaled_height, scaled_width),
                    interpolation=cv2.INTER_NEAREST)
            surface = pygame.surfarray.make_surface(image)
            display.blit(surface, (0, 0))

    def _draw_world_points_on_image(self,
                                    world_point,
                                    rgb_img,
                                    color=(255, 0, 0),
                                    size=3):
        """Render points with world coordinated onto the camera image.

        Args:
            world_point (np.ndarray): point to be rendered in the camera image,
                represented in world coordinate. The shape is [N, 3]
            rgb_img (np.ndarray): with the meaning of axis as (y, x, c)
            color (tuple[int]): color values for the [R, G, B] channels
                of the rendered points
            size (int): size of the rendered point in terms of pixels
        """

        assert len(color) == 3, ("the color code should contain values for "
                                 "[R, G, B] channels respectively")

        point_cam = self._world_to_camera_image(world_point, rgb_img.shape[1],
                                                rgb_img.shape[0])

        half_size = size // 2
        for i in range(point_cam.shape[0]):
            pt = point_cam[i]

            xi = int(np.asscalar(pt[0]))
            yi = int(np.asscalar(pt[1]))

            if xi >= 0 and xi < rgb_img.shape[
                    1] and yi >= 0 and yi < rgb_img.shape[0]:

                xst = xi - half_size
                yst = yi - half_size

                xed = xst + size
                yed = yst + size

                xst = np.clip(xst, 0, rgb_img.shape[1])
                xed = np.clip(xed, 0, rgb_img.shape[1])
                yst = np.clip(yst, 0, rgb_img.shape[0])
                yed = np.clip(yed, 0, rgb_img.shape[0])

                rgb_img[yst:yed, xst:xed] = color

    def _world_to_camera_image(self, world_points, image_width, image_height):
        """Project points in world coordinates to camera image plane.

        Adapted from https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/lidar_to_camera.py
        Additional reference:
        https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf

        Args:
            world_points (np.ndarray): [N, 3] in world coordinate
            image_width (int): the width of the image to be rendered on
            image_height (int): the height of the image to be rendered on
        Output:
            np.ndarray: representing the image plane coordinates for the set
                of points that are visible in the camera. Its shape is [N', 2],
                with N' <= N.
        """

        # Build the projection matrix K:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]

        focal = image_width / (2.0 * np.tan(self._fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_width / 2.0
        K[1, 2] = image_height / 2.0

        # [4, 4]
        world_2_camera = np.linalg.inv(
            _get_transform_matrix(self._sensor.get_transform()))

        # transform the points from world space to camera space through the
        # following several steps

        # [N, 3] -> [3, N]
        world_points = world_points.transpose()
        # [3, N] -> [4, N]
        world_points = np.concatenate(
            (world_points, np.ones_like(world_points[0:1, ...])), axis=0)

        sensor_points = np.matmul(world_2_camera, world_points)

        # change from UE4's left handed coordinate system to a typical
        # right handed camera coordinate system, which is equivalent to
        # axis swapping: (x, y, z) -> (y, -z, x)
        point_in_camera_coords = np.array(
            [sensor_points[1], -sensor_points[2], sensor_points[0]])

        # remove points that are behind the camera as they are invisible
        cam_z = point_in_camera_coords[2]
        valid_ind = cam_z >= 0
        point_in_camera_coords = point_in_camera_coords[:, valid_ind]

        # use projection matrix K to do the mapping from 3D to 2D in
        # homogeneous coordinates
        points_2d_homogeneous = np.matmul(K, point_in_camera_coords)

        # normalize the x, y values by z value
        points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

        # [2, N] -> [N, 2]
        points_2d = points_2d.transpose()
        return points_2d

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

        # Need to slice the multi-channel image according to the number of
        # channels specified in observation_spec.
        # For raw data from the semantic segmentation camera, the tag information
        # is encoded in the red channel.
        # For logarithmic depth from depth camera, the scalar depth is the same
        # for all three channels and therefore we can do a similar slicing.
        array = array[:, :, 0:self._observation_spec.shape[0]]

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


NumpyWaypoint = namedtuple(
    "NumpyWaypoint",
    [
        'id',  # int
        'location',  # [3] (x, y, z)
        'rotation',  # [3] (pitch, yaw, rolll)
        'road_id',  # int
        'section_id',  # int
        'lane_id',  # int
        'is_junction',  # bool
        'lane_width',  # float
        'lane_change',  # int (carla.LaneChange) whether lane change is allowed. 0: None, 1: Right, 2: Left, 3: Both
        'lane_type',  # int (carla.LaneType)
        'right_lane_marking',  # int (carla.LaneMarking)
        'left_lane_marking',  # int (carla.LaneMarking)
    ])


def _to_numpy_loc(loc: carla.Location):
    return np.array([loc.x, loc.y, loc.z], dtype=np.float)


def _to_carla_loc(loc):
    return carla.Location(float(loc[0]), float(loc[1]), float(loc[2]))


def _to_numpy_rot(rot: carla.Rotation):
    """
    Returns:
        3D np.array [pitch, yaw, roll] in radians
    """
    return np.radians(np.array([rot.pitch, rot.yaw, rot.roll], dtype=np.float))


NumpyLaneMarking = namedtuple("NumpyLaneMarking",
                              ['color', 'lane_change', 'type', 'width'])

dummy_lane_marking = NumpyLaneMarking(
    color=np.int(-1),
    lane_change=np.int(-1),
    type=np.int(-1),
    width=np.float(0.),
)


def _to_numpy_lane_marking(lane_marking: carla.LaneMarking):
    return NumpyLaneMarking(
        color=np.int(lane_marking.color),
        lane_change=np.int(lane_marking.lane_change),
        type=np.int(lane_marking.type),
        width=np.float(lane_marking.width))


def _to_numpy_waypoint(wp: carla.Waypoint):
    return NumpyWaypoint(
        id=np.int(wp.id),
        location=_to_numpy_loc(wp.transform.location),
        rotation=_to_numpy_rot(wp.transform.rotation),
        road_id=np.int(wp.road_id),
        section_id=np.int(wp.section_id),
        lane_id=np.int(wp.lane_id),
        is_junction=np.bool(wp.is_junction),
        lane_width=np.float(wp.lane_width),
        lane_change=np.int(wp.lane_change),
        lane_type=np.int(wp.lane_type),
        right_lane_marking=_to_numpy_lane_marking(wp.right_lane_marking),
        left_lane_marking=_to_numpy_lane_marking(wp.left_lane_marking))


def _get_transform_matrix(transform):
    """Computes the 4 by 4 transformation matrix representation of the
    3D transform using homogeneous coordinates.

    Adapted from Math::GetMatrix() in https://github.com/carla-simulator/carla/blob/5bd3dab1013df554c0198662e0ceb50b7857feba/LibCarla/source/carla/geom/Transform.h

    Args:
        ransform (carla.Transform):
    Returns:
        np.ndarray: with the shape of [4, 4]
    """
    loc = transform.location

    yaw = math.radians(transform.rotation.yaw)
    cy, sy = np.cos(yaw), np.sin(yaw)

    roll = math.radians(transform.rotation.roll)
    cr, sr = np.cos(roll), np.sin(roll)

    pitch = math.radians(transform.rotation.pitch)
    cp, sp = np.cos(pitch), np.sin(pitch)

    x, y, z = loc.x, loc.y, loc.z

    mat = np.array(
        [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
         [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
         [sp, -cp * sr, cp * cr, z], [0., 0., 0., 1.]])
    return mat


def _get_forward_vector(rotation):
    """Computes the vector pointing forward according to the orientation of each axis.

    Adapted from Math::GetForwardVector() in https://github.com/carla-simulator/carla/blob/dev/LibCarla/source/carla/geom/Math.cpp

    Args:
        rotation (np.array): rotation[:, 0], rotation[:, 1], rotation[:, 2] are
            pitch, yaw, roll in radians
    Returns:
        np.array: forward vector(s)
    """
    c = np.cos(rotation)
    s = np.sin(rotation)
    return np.stack([c[:, 1] * c[:, 0], s[:, 1] * c[:, 0], s[:, 0]], axis=-1)


def _rotate_point(point: carla.Vector3D, angle):
    """Rotate a given point by a given angle

    Args:
        point (carla.Vector3D):
        angle (float): in degrees
    Returns:
        carla.Vector3D
    """
    x_ = math.cos(math.radians(angle)) * point.x - math.sin(
        math.radians(angle)) * point.y
    y_ = math.sin(math.radians(angle)) * point.x + math.cos(
        math.radians(angle)) * point.y
    return carla.Vector3D(x_, y_, point.z)


def _rotate_np_point(point, angle):
    """Rotate a given point by a given angle

    Args:
        point (np.ndarray): batch of points of shape [n, 3]
        angle (np.ndarray): in radians of shape [n]
    Returns:
        np.ndarray of shape [n, 3]
    """
    c = np.cos(angle)
    s = np.sin(angle)
    x = c * point[:, 0] - s * point[:, 1]
    y = s * point[:, 0] + c * point[:, 1]
    return np.stack([x, y, point[:, 2]], axis=-1)


def _is_segments_intersecting(seg1, seg2):
    """Check whether two batched segments intersect.

    Based on the answer by Norbu Tsering at
    https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    Args:
        seg1 (tuple[ndarray]): ``seg1[0]`` and ``seg1[1]`` are [1, 3] or [n,3] arrays.
        seg2 (tuple[ndarray]): ``seg2[0]`` and ``seg2[1]`` are [1, 3] or [n,3] arrays
    Returns:
        bool ndarray of shape [n] indicating wether each segment in ``seg1``
            intersects with the corresponding segment in ``seg2``
    """
    seg1 = np.copy(seg1[0]), np.copy(seg1[1])
    seg2 = np.copy(seg2[0]), np.copy(seg2[1])
    seg1[0][:, 2] = 1
    seg1[1][:, 2] = 1
    seg2[0][:, 2] = 1
    seg2[1][:, 2] = 1
    l1 = np.cross(seg1[0], seg1[1])
    l2 = np.cross(seg2[0], seg2[1])
    xyz = np.cross(l1, l2)
    z = np.expand_dims(xyz[:, 2], -1)
    parallel = np.abs(z) < 1e-10
    z = np.where(parallel, 1e-10, z)

    # xyz is the intersection of the two lines
    xyz = xyz / z

    # If seg1 intersects with seg2, xyz should be inside each segments.
    inside_seg1 = np.sum((xyz - seg1[0]) * (xyz - seg1[1]), axis=-1) < 0
    inside_seg2 = np.sum((xyz - seg2[0]) * (xyz - seg2[1]), axis=-1) < 0
    return ~parallel & inside_seg1 & inside_seg2


dummy_waypoint = NumpyWaypoint(
    id=np.int(-1),
    location=np.zeros(3),
    rotation=np.zeros(3),
    road_id=np.int(-1),
    section_id=np.int(-1),
    lane_id=np.int(-1),
    is_junction=np.bool(False),
    lane_width=np.float(0),
    lane_change=np.int(0),
    lane_type=np.int(0),
    right_lane_marking=dummy_lane_marking,
    left_lane_marking=dummy_lane_marking,
)


class World(object):
    """Keeping data for the world."""

    # only consider a car for running red light if it is within such distance
    RED_LIGHT_ENFORCE_DISTANCE = 15  # m

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
        self._prepare_traffic_light_data()
        self._prepare_speed_limit_data()

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
        """Get the coordinates of waypoints

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

    def _prepare_speed_limit_data(self):
        actor_list = self._world.get_actors()
        speed_limit_locations = []
        speed_limit_values = []
        for speed_limit in actor_list.filter('traffic.speed_limit.*'):
            loc = _to_numpy_loc(speed_limit.get_location())
            value = float(speed_limit.type_id.split('.')[-1]) / 3.6
            speed_limit_locations.append(loc)
            speed_limit_values.append(value)

        self._speed_limit_locations = np.stack(speed_limit_locations)
        self._speed_limit_values = np.stack(speed_limit_values)

        logging.info(
            "Found %d speed limit signs" % len(self._speed_limit_locations))

    def get_active_speed_limit(self, actor, dis_threshold=1.0):
        """Get active speed limit for the actor.

        Args:
            actor (carla.Actor): the vehicle actor
            dis_threshold (float): the distance within which to consider the
                speed limit sign as active. The one closest to the actor in the
                active set will be used as the current speed limit.
                If a negative value is provided, all speed limit signs are
                taken into considerations for determining the closest one.
        Returns:
            - the value of the speed limit in m/s is there is a speed limit sign
                within the distance of ``dis_threshold``
            - None if there is no active speed limit sign
        """

        veh_transform = actor.get_transform()
        veh_location = veh_transform.location
        dist = self._speed_limit_locations - _to_numpy_loc(veh_location)
        dist = np.linalg.norm(dist, axis=-1)
        min_ind = np.argmin(dist)
        min_dis = dist[min_ind]
        if min_dis <= dis_threshold or dis_threshold < 0:
            return self._speed_limit_values[min_ind]
        else:
            return None

    def _prepare_traffic_light_data(self):
        # Adapted from RunningRedLightTest.__init__() in
        # https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenariomanager/scenarioatomics/atomic_criteria.py

        actors = self._world.get_actors()
        self._traffic_light_actors = []
        traffic_light_centers = []
        # traffic_light_waypoints are the waypoints which enter into the intersection
        # covered by the traffic light
        traffic_light_waypoints = []
        max_num_traffic_light_waypoints = 0
        for actor in actors:
            if 'traffic_light' in actor.type_id:
                center, waypoints = self._get_traffic_light_waypoints(actor)
                self._traffic_light_actors.append(actor)
                traffic_light_centers.append(_to_numpy_loc(center))
                waypoints = [_to_numpy_waypoint(wp) for wp in waypoints]
                traffic_light_waypoints.append(waypoints)
                if len(waypoints) > max_num_traffic_light_waypoints:
                    max_num_traffic_light_waypoints = len(waypoints)

        logging.info(
            "Found %d traffic lights" % len(self._traffic_light_actors))

        self._traffic_light_centers = np.array(traffic_light_centers,
                                               np.float32)
        np_traffic_light_waypoints = []
        for waypoints in traffic_light_waypoints:
            pad = max_num_traffic_light_waypoints - len(waypoints)
            if pad > 0:
                waypoints.extend([dummy_waypoint] * pad)
            np_traffic_light_waypoints.append(
                alf.nest.map_structure(lambda *x: np.stack(x), *waypoints))

        self._traffic_light_waypoints = alf.nest.map_structure(
            lambda *x: np.stack(x), *np_traffic_light_waypoints)

    def on_tick(self):
        """Should be called after every world tick() to update data."""
        self._traffic_light_states = np.array(
            [a.state for a in self._traffic_light_actors], dtype=np.int)
        self._actor_locations = {}

    def _get_traffic_light_waypoints(self, traffic_light):
        # Copied from RunningRedLightTest.get_traffic_light_waypoints() in
        # https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenariomanager/scenarioatomics/atomic_criteria.py
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(
            traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x,
                             1.0)  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = _rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[
                    -1].lane_id != wpx.lane_id:
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            if wpx.transform.location.distance(
                    area_loc) >= self.RED_LIGHT_ENFORCE_DISTANCE - 2:
                # if area_loc is too far from wpx, when the vehicle is going over
                # wpx, it is already too far from area_loc. And red light will not
                # be detected.
                logging.fatal(
                    "traffic light center is too far from traffic light "
                    "waypoint: %s. Need to increase RED_LIGHT_ENFORCE_DISTANCE"
                    % wpx.transform.location.distance(area_loc))
            wps.append(wpx)

        # self._draw_waypoints(wps, vertical_shift=1.0, persistency=50000.0)

        return area_loc, wps

    def is_running_red_light(self, actor):
        """Whether actor is running red light.

        Adapted from RunningRedLightTest.update() in https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenariomanager/scenarioatomics/atomic_criteria.py

        Args:
            actor (carla.Actor): the vehicle actor
        Returns:
            red light id if running red light, None otherwise
        """
        veh_transform = actor.get_transform()
        veh_location = veh_transform.location

        veh_extent = actor.bounding_box.extent.x
        tail_close_pt = _rotate_point(
            carla.Vector3D(-0.8 * veh_extent, 0.0, veh_location.z),
            veh_transform.rotation.yaw)
        tail_close_pt = veh_location + carla.Location(tail_close_pt)

        tail_far_pt = _rotate_point(
            carla.Vector3D(-veh_extent - 1, 0.0, veh_location.z),
            veh_transform.rotation.yaw)
        tail_far_pt = veh_location + carla.Location(tail_far_pt)
        tail_far_wp = self._map.get_waypoint(tail_far_pt)

        veh_seg = (np.expand_dims(_to_numpy_loc(tail_close_pt), axis=0),
                   np.expand_dims(_to_numpy_loc(tail_far_pt), axis=0))

        is_red = self._traffic_light_states == carla.TrafficLightState.Red
        dist = self._traffic_light_centers - _to_numpy_loc(veh_location)
        dist = np.linalg.norm(dist, axis=-1)

        candidate_light_index = np.nonzero(
            is_red & (dist <= self.RED_LIGHT_ENFORCE_DISTANCE))[0]
        ve_dir = _to_numpy_loc(veh_transform.get_forward_vector())

        waypoints = self._traffic_light_waypoints
        for index in candidate_light_index:
            wp_dir = _get_forward_vector(waypoints.rotation[index])
            dot_ve_wp = (ve_dir * wp_dir).sum(axis=-1)

            same_lane = ((tail_far_wp.road_id == waypoints.road_id[index])
                         & (tail_far_wp.lane_id == waypoints.lane_id[index])
                         & (dot_ve_wp > 0))

            yaw_wp = waypoints.rotation[index][:, 1]
            lane_width = waypoints.lane_width[index]
            location_wp = waypoints.location[index]

            d = np.stack([
                0.4 * lane_width,
                np.zeros_like(lane_width), location_wp[:, 2]
            ],
                         axis=-1)
            left_lane_wp = _rotate_np_point(d, yaw_wp + 0.5 * math.pi)
            left_lane_wp = location_wp + left_lane_wp
            right_lane_wp = _rotate_np_point(d, yaw_wp - 0.5 * math.pi)
            right_lane_wp = location_wp + right_lane_wp
            if np.any(same_lane & _is_segments_intersecting(
                    veh_seg, (left_lane_wp, right_lane_wp))):
                # If veh_seg intersects with (left_lane_wp, right_lane_wp), that
                # means the vehicle is crossing the line dividing intersection
                # and the outside area.
                return self._traffic_light_actors[index].id
        return None

    def _draw_waypoints(self, waypoints, vertical_shift, persistency=-1):
        """Draw a list of waypoints at a certain height given in vertical_shift."""
        for wp in waypoints:
            loc = wp.transform.location + carla.Location(z=vertical_shift)

            size = 0.2
            color = carla.Color(255, 0, 0)
            self._world.debug.draw_point(
                loc, size=size, color=color, life_time=persistency)


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
            The total length of the route in meters, starting from the current
            vehicle location to the destination.
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
        waypoints_with_start_end = np.concatenate((
            np.array([[start.x, start.y, start.z]]),
            self._waypoints,
            np.array([[destination.x, destination.y, destination.z]]),
        ),
                                                  axis=0)
        d = waypoints_with_start_end[:-1] - waypoints_with_start_end[1:]
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
        loc = np.array([loc.x, loc.y, loc.z])
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
