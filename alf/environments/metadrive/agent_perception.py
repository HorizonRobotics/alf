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

from typing import Tuple, Optional, NamedTuple
import functools

import torch
import numpy as np

from alf.tensor_specs import TensorSpec
from .geometry import FieldOfView, Polyline

try:
    import metadrive
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    from metadrive.engine.base_engine import BaseEngine
except ImportError:
    from unittest.mock import Mock
    # create 'metadrive' as a mock to not break python argument type hints
    metadrive = Mock()


class AgentPerception(object):
    """A perception module that once initialized can produced the vectorized feature
    of the dynamic road users (agents) that are visible to the ego car in the
    driving scenario.

    The essential method of AgentPerception is observe(), which is called upon
    every observation to generate the observation of the dynamic agents.

    Useful Notations:

        B - the batch size, a.k.a. the agent limit (see below)
        H - history_window_size, a.k.a. the number of historical steps
        F - the size of the feature for each of the agent at each step

    The final feature is a 3D tensor of shape [B, H, F].

    The feature for each agent at each step is a vector of 7 elements:

        * The distance between the centers of ego and the agent (1)
        * The unit vector point from the center of ego to that of the agent (2)
        * The width and length of the agent (2)
        * The heading of agent (w.r.t. ego's heading) as an unit vector (2)

    """

    def __init__(self,
                 fov: FieldOfView,
                 history_window_size: int,
                 history_frame_skip: int = 4,
                 agent_limit: int = 16):
        """Construct an AgentPerception instance.

        Args:

            fov: Describe the field of view (FOV) of the ego car. When
                generating the agent features, only those agent who is within
                the FOV result in the feature.
            history_window_size: The feature only tracks this number of
                historical frames (current frame included).
            history_frame_skip: Pick every this number of frames to form the historical
                trajectories for the agents.
            agent_limit: The maximum number of agents shown in the feature. If
                the number of visible agents exceeds this limit, the farthest
                ones are filtered out until this limit is satisfied.

        """
        self._history_window_size = history_window_size
        self._fov = fov
        self._agent_limit = agent_limit
        self._unit_feature_size = 7
        self._history_frame_skip = history_frame_skip

        self._engine = None
        self._ego = None

        # Static Information
        self._num_agents = 0
        self._dimension = None
        self._agent_to_index = None

        # Dynamic Buffers. Please refer to documentation in reset() to
        # understand them if needed.
        self._visible = None  # 1 = visible, 0 = invisible
        self._history_position = None
        self._history_heading = None

        # Handle Frame Skip
        H = self._history_window_size
        self._sampled_index = np.arange((H - 1) % self._history_frame_skip, H,
                                        self._history_frame_skip)
        self._spec = TensorSpec(
            shape=(self._agent_limit, self._sampled_index.shape[0],
                   self._unit_feature_size),
            dtype=torch.float32)

    @property
    def observation_spec(self):
        return self._spec

    def reset(self, engine: BaseEngine, ego: BaseVehicle):
        """Initialize by creating the buffers for holding the dynamic agents
        information.

        This is internally used by MetaDrive related Observation objects, called
        once when a new MetaDrive environment (which is required to produce
        agent related observations) is constructed.

        """

        # NOTE that the buffers track ALL the agents in the MetaDrive
        # environment. Later when the buffers are actually queried for
        # observation, only those who is within the field of view of the ego car
        # are retrieved.

        self._engine = engine
        self._ego = ego

        # We are going to use A to denote the number of agents below. Also H
        # will be used to denote the history window size.
        self._num_agents = len(self._engine.traffic_manager.vehicles) - 1

        self._agent_to_index = {}

        # Shape is [A, 2]. The variable self._dimension stores the width and
        # length of each agents, which holds constant throughout the MetaDrive
        # environment's lifetime.
        self._dimension = np.zeros((self._num_agents, 2), dtype=np.float32)
        i = 0
        for agent in self._engine.traffic_manager.vehicles:
            # Ego car is excluded.
            if agent is self._ego:
                continue
            self._dimension[i] = (agent.LENGTH, agent.WIDTH)
            self._agent_to_index[agent] = i
            i += 1

        # Shape is [A, H]. Stores whether the agent is visible (True for
        # visible) for each agent, at each historical step (including the
        # current step).
        self._visible = np.zeros((self._num_agents, self._history_window_size),
                                 dtype=bool)
        # Shape is [A, H, 2]. Stores the WORLD FRAME position of each agent, at
        # each historical step (including the current step).
        self._history_position = Polyline(
            point=np.zeros((self._num_agents, self._history_window_size, 2),
                           dtype=np.float32))
        # Shape is [A, H]. Stores the WORLD FRAME heading orientation of each
        # agent, at each historical step (including the current step).
        self._history_heading = np.zeros(
            (self._num_agents, self._history_window_size), dtype=np.float32)

    def observe(self) -> Tuple[np.ndarray, int]:
        """Called upon every observation to produce the feature vectors describing the
        dynamic agents that are visible to the ego car. The vectors are
        transformed so that they are in ego car's body frame.

        Returns:

            A tuple of 2:

            1. A 3D feature tensor of shape [B, H, F]. See class docstring for
               the meaning of B, H and F.

            2. An integer indicating how many agents are actually filled.

        """

        # Shift the buffer so that slot -1 is available for insertion.
        self._history_position.point[:, :-1] = self._history_position.point[:,
                                                                            1:]
        self._history_heading[:, :-1] = self._history_heading[:, 1:]
        self._visible[:, :-1] = self._visible[:, 1:]

        alive = np.zeros_like(self._visible[:, -1])
        # Insert the new positions and headings
        for agent in self._engine.traffic_manager.vehicles:
            if agent is self._ego:
                continue
            i = self._agent_to_index[agent]
            alive[i] = True
            self._history_heading[i, -1] = agent.heading_theta
            self._history_position.point[i, -1] = agent.position
            i += 1

        # Transform the position so that we can test whether it is in the field
        # of view of the ego car. The test result is stored in self._visible.
        transformed_position = self._history_position.transformed(
            self._ego.position, self._ego.heading_theta)
        transformed_heading = self._history_heading - self._ego.heading_theta
        self._visible[:, -1] = self._fov.within(
            transformed_position.point[:, -1])
        self._visible[~alive, -1] = False
        sampled_visible = self._visible[:, self._sampled_index]
        sampled_position = transformed_position.point[:, self._sampled_index]
        sampled_heading = transformed_heading[:, self._sampled_index]

        # Shape is [B,]. Denote whether a car is picked to show in the final
        # feature tensor or not. The criterion is that the car has been visible
        # in at least 1 step within the latest H steps.
        picked = np.any(sampled_visible, axis=-1)
        picked_position = sampled_position[picked]
        picked_heading = sampled_heading[picked]
        picked_dimension = self._dimension[picked]
        picked_visible = sampled_visible[picked]

        # Filter out the farthest agents in case the total number of visible
        # agents exceeds the limit.
        if np.count_nonzero(picked) > self._agent_limit:
            distances = np.linalg.norm(picked_position[:, -1], axis=-1)
            closest = np.argpartition(distances,
                                      self._agent_limit)[:self._agent_limit]
            picked_position = picked_position[closest]
            picked_heading = picked_heading[closest]
            picked_dimension = picked_dimension[closest]
            picked_visible = picked_visible[closest]

        size = picked_dimension.shape[0]

        # [B, H, 2]
        center = picked_position
        # [B, H, 1]
        r = np.linalg.norm(center, axis=-1, keepdims=True) + 1e-5
        # [B, H]
        cos = np.cos(picked_heading)
        sin = np.sin(picked_heading)

        feature = np.zeros(self._spec.shape, dtype=np.float32)
        feature_view = feature[:size]  # [B, H, 7]

        feature_view[:, :, 0] = r.squeeze(axis=-1)
        feature_view[:, :, 1:3] = center / r
        feature_view[:, :, 3:5] = np.expand_dims(picked_dimension, axis=1)
        feature_view[:, :, 5] = cos
        feature_view[:, :, 6] = sin
        feature_view[~picked_visible] = 0.0

        return feature, size
