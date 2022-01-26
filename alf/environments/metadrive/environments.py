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

import numpy as np
import gym
import torch

from alf.tensor_specs import TensorSpec

try:
    import metadrive
except ImportError:
    from unittest.mock import Mock
    # create 'metadrive' as a mock to not break python argument type hints
    metadrive = Mock()

from .geometry import FieldOfView
from .sensors import VectorizedObservation


class VectorizedTopDownEnv(metadrive.MetaDriveEnv):
    """This is the counterpart of the TopDownEnv from MetaDrive with vectorized
    input insead of raster input (BEV).

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

    def get_single_observation(self, _=None):
        return VectorizedObservation(self.config["vehicle_config"])

    @property
    def observation_spec(self):
        return self.get_single_observation().observation_spec
