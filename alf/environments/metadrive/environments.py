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

from typing import Optional, Union

import numpy as np
import gym

from alf.tensor_specs import TensorSpec

try:
    import pygame
    import metadrive
    from metadrive.obs.observation_base import ObservationBase
except ImportError:
    from unittest.mock import Mock
    # create 'metadrive' as a mock to not break python argument type hints
    metadrive = Mock()
    pygame = Mock()

from .geometry import FieldOfView
from .sensors import VectorizedObservation, BirdEyeObservation
from .renderer import Renderer, make_vectorized_observation_renderer, make_bird_eye_observation_renderer


class VectorizedTopDownEnv(metadrive.MetaDriveEnv):
    """This is the counterpart of the TopDownEnv from MetaDrive with vectorized
    input instead of raster input (BEV).

    """

    @classmethod
    def default_config(cls) -> metadrive.utils.Config:
        """The default config is identical to that of the raster TopDownEnv.

        """
        config = metadrive.MetaDriveEnv.default_config()
        config["vehicle_config"]["lidar"] = {"num_lasers": 0, "distance": 0}
        config.update({
            "frame_skip": 5,
            "frame_stack": 3,
            "post_stack": 5,
        })
        return config

    def get_single_observation(self, _=None) -> ObservationBase:
        """Implements the get_single_observation for the base class MetaDriveEnv.

        The base class is calling this function to acquire the sensor (typed
        ObservationBase) that is used for generating observations. Unlike the
        name may suggest, it is

        1. actually only called once per environment
        2. returning a sensor object instead of the actual observation

        The sensor object is then used to produce the actual observation of each
        frame.

        """
        return VectorizedObservation(self.config["vehicle_config"])

    def render(self, observation=None) -> Optional[np.ndarray]:
        if self._top_down_renderer is None:
            self._top_down_renderer = Renderer(
                observation_renderer=make_vectorized_observation_renderer(
                    sensor=self.get_single_observation()))
        return self._top_down_renderer.render(observation)

    @property
    def observation_spec(self):
        return self.get_single_observation().observation_spec


class BirdEyeTopDownEnv(metadrive.MetaDriveEnv):
    """This is the counterpart of the TopDownEnv from MetaDrive with vectorized
    input instead of raster input (BEV).

    """

    @classmethod
    def default_config(cls) -> metadrive.utils.Config:
        """The default config is identical to that of the raster TopDownEnv.

        """
        config = metadrive.MetaDriveEnv.default_config()
        config["vehicle_config"]["lidar"] = {"num_lasers": 0, "distance": 0}
        config.update({
            "frame_skip": 5,
            "frame_stack": 3,
            "post_stack": 5,
            "rgb_clip": True,
            "resolution_size": 84,
            "distance": 30
        })
        return config

    def get_single_observation(self, _=None) -> ObservationBase:
        """Implements the get_single_observation for the base class MetaDriveEnv.

        The base class is calling this function to acquire the sensor (typed
        ObservationBase) that is used for generating observations. Unlike the
        name may suggest, it is

        1. actually only called once per environment
        2. returning a sensor object instead of the actual observation

        The sensor object is then used to produce the actual observation of each
        frame.

        """
        return BirdEyeObservation(self.config)

    @property
    def observation_spec(self):
        return self.get_single_observation().observation_spec

    def render(self, observation=None) -> Optional[np.ndarray]:
        if self._top_down_renderer is None:
            self._top_down_renderer = Renderer(
                observation_renderer=make_bird_eye_observation_renderer())
        return self._top_down_renderer.render(observation)
