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

from typing import NamedTuple, Tuple

import torch
import numpy as np
import gym

try:
    import pygame
    import metadrive
    from metadrive.obs.observation_base import ObservationBase
    from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
except ImportError:
    from unittest.mock import Mock
    # create 'metadrive' as a mock to not break python argument type hints
    metadrive = Mock()
    pygame = Mock()

import alf
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from .geometry import FieldOfView, Polyline
from .map_perception import MapPolylinePerception
from .agent_perception import AgentPerception


@alf.configurable
class VectorizedObservation(ObservationBase):
    """This implements a customized observation for the MetaDrive environment that
    produces vectorized inputs (as opposed to raster inputs such as BEV).

    The main API observe() is designed to produces a dictionary of vectorized
    inputs. Depending on the configuration the dictionary may consist of:

    1. 'map': polyline based features of road network (lanes) and navigations
    2. 'ego': polyline based feature of ego car history
    3. 'agents': polyline based feature of dynamic road users for interaction

    All polylines are within a specified field of view, and transformed so that
    they are in the ego car's body frame, where x-axis points to the heading
    direction of the ego car.

    """

    def __init__(self,
                 vehicle_config: metadrive.utils.Config,
                 fov: FieldOfView = FieldOfView(),
                 segment_resolution: float = 2.0,
                 polyline_size: int = 4,
                 polyline_limit: int = 64,
                 history_window_size: int = 8,
                 agent_limit: int = 16):
        """Construct a VectorizedObservation instance.

        Args:

            vehicle_config: A MetaDrive global configuration of the MetaDrive
                environment that this observation is attached to.
            fov: Defines the field of view with respect to the ego car. Map object
                and agents outside of the field of view will not appear in the
                observation. Usually the bigger the field of view, the more expensive
                the computation of the obsrevation (and the training) will be.
            segment_resolution: The length of each line segment in the polylines
                during sampling. The smaller the value, the more segments and
                polylines. As a result, it also implies more expensive training.
            polyline_size: Specify the number of segments in one polyline. Putting
                more segments in a polyline can reduce the number of polylines for
                each observation.
            polyline_limit: Specify the maximum number of polylines in the
                observation for map and navigation. If the actual number of polylines
                goes beyond this limit, farthest polylines (from the ego car) will be
                filtered out until the limit is satisfied. If the actual number of
                polylines is below this limit, zero padding will be employed to bring
                fill the vacancies.
            history_window_size: The past positions of the most recent this number of
                frames will be recorded and used for the ego history feature.
            agent_limit: the maximum number of agents appears in the agent feature.
        """
        super().__init__(vehicle_config)
        self._map_perception = MapPolylinePerception(
            fov=fov,
            segment_resolution=segment_resolution,
            polyline_size=polyline_size,
            polyline_limit=polyline_limit)

        self._agent_perception = AgentPerception(
            fov=fov,
            history_window_size=history_window_size,
            agent_limit=agent_limit)

        self._position_history = Polyline(
            point=np.zeros((history_window_size, 2), dtype=np.float32))

        self.fov = fov
        self.history_window_size = history_window_size
        self.polyline_size = polyline_size
        self.polyline_limit = polyline_limit
        self.agent_limit = agent_limit

    @property
    def observation_space(self):
        """The base class ObservationBase requires that we have observation_space()
        implemented and returns a gym.spaces.Box object.

        THIS IS A HACK.

        This is just a dummy function and it does not return the actual observation
        spec (because the actual observation spec is a dictionary, which defies the
        requirement of the base class. We do this nonetheless because in our
        implementation we are actually using ``observation_spec()`` below.

        """
        return gym.spaces.Box(low=-1000.0, high=1000.0, shape=())

    @property
    def observation_spec(self):
        return {
            'map':
                self._map_perception.observation_spec,
            'map_mask':
                TensorSpec(shape=(self.polyline_limit, ), dtype=torch.bool),
            'ego':
                TensorSpec(
                    shape=((self._position_history.point.shape[0] - 1) * 6, ),
                    dtype=torch.float32),
            'agents':
                self._agent_perception.observation_spec,
            'agent_mask':
                TensorSpec(shape=(self.agent_limit, ), dtype=torch.bool)
        }

    def observe(self, vehicle: BaseVehicle):
        """The main API to generate the vectorized input, given the current ego state.

        All the static polylines are generated during pre-compuation in ``reset()``
        (see below). The call to ``observe()`` basically crops the pre-computed
        polylines within the field of view, and does the coordinate frame
        transformation into the ego car's body frame.

        Args:

            vehicle: The MetaDrive vehicle object is the container for all the ego
                car related information.

        """
        map_feature, polyline_count = self._map_perception.observe(
            vehicle.position, vehicle.heading_theta)
        map_mask = np.ones(map_feature.shape[0], dtype=bool)
        map_mask[polyline_count:] = False

        agent_feature, agent_count = self._agent_perception.observe()
        agent_mask = np.ones(agent_feature.shape[0], dtype=bool)
        agent_mask[agent_count:] = False

        self._position_history.point[:-1, :] = self._position_history.point[
            1:, :]
        self._position_history.point[-1, :] = vehicle.position

        return {
            'map':
                map_feature,
            'map_mask':
                map_mask,
            'ego':
                self._position_history.transformed(
                    vehicle.position, vehicle.heading_theta).to_feature(),
            'agents':
                agent_feature,
            'agent_mask':
                agent_mask,
        }

    def reset(self, env, vehicle=None):
        # Initialize by generating all the polylines of map and navigation via the
        # MapPolylinePerception object.
        self._map_perception.reset(env.current_map.road_network,
                                   vehicle.navigation)
        self._agent_perception.reset(env.engine, vehicle)
        # Initialize the vehicle history buffer.
        self._position_history.point[:, :] = vehicle.position


@alf.configurable
class BirdEyeObservation(TopDownMultiChannel):
    """This implements a MetaDrive Observation that produces BEV, together
    with historical velocities.

    The implementation is simply adding velocity alongside the original BEV
    produced by MetaDrive's TopDownMultiChannel.

    Note that the velocity histories are not encoded as images. The model will
    have the flexibility to choose to how to encode it.

    """

    def __init__(self,
                 env_config: metadrive.utils.Config,
                 velocity_steps: int = 1,
                 velocity_normalization: float = 100.0):
        """Construct a BirdEyeObservation instance.

        Args:
            
            env_config: MetaDrive's environment configuration.
            velocity_steps: The number of historical steps for the velocity.
            velocity_normalization: The velocities (in m/s) will be normalized
                by this factor before producing the feature.

        """
        super().__init__(
            env_config["vehicle_config"],
            env_config["use_render"],
            env_config["rgb_clip"],
            frame_stack=env_config["frame_stack"],
            post_stack=env_config["post_stack"],
            frame_skip=env_config["frame_skip"],
            resolution=(env_config["resolution_size"],
                        env_config["resolution_size"]),
            max_distance=env_config["distance"])

        self._velocity_steps = velocity_steps
        self._velocity_normalization = velocity_normalization
        self._velocity_history = np.zeros(
            self._velocity_steps, dtype=np.float32)

    @property
    def observation_spec(self):
        h, w, c = self.observation_space.shape

        return {
            'bev':
                BoundedTensorSpec(
                    shape=(c, h, w),
                    dtype=torch.float32,
                    minimum=0.0,
                    maximum=1.0),
            'vel':
                BoundedTensorSpec(
                    shape=(self._velocity_steps, ),
                    dtype=torch.float32,
                    minimum=0.0,
                    maximum=30.0)
        }

    def observe(self, vehicle: BaseVehicle):
        """The main API to generate the feature given the current ego car status.

        Args:

            vehicle: The MetaDrive vehicle object is the container for all the ego
                car related information.

        """
        img = super().observe(vehicle)

        self._velocity_history[:-1] = self._velocity_history[1:]
        self._velocity_history[-1] = vehicle.speed * 1000.0 / 3600.0

        return {
            'bev': np.transpose(img, (2, 0, 1)),
            'vel': self._velocity_history / self._velocity_normalization
        }
