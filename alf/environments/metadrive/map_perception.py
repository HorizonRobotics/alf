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

import torch
import numpy as np

from alf.tensor_specs import TensorSpec
from .geometry import FieldOfView, Polyline, CategoryEncoder

try:
    import metadrive
    from metadrive.constants import LineType
except ImportError:
    from unittest.mock import Mock
    # create 'metadrive' as a mock to not break python argument type hints
    metadrive = Mock()


class MapPolylinePerception(object):
    """A perception module that once initialized can produced the polyline-based
    vectorized feature of the semantic map and navigation of a MetaDrive
    scenario.

    The essential method of MapPolylinePerception is observe(), which is called
    upon every observation to generate the part of the observation from the map.

    Under the hood, upon initialization we will create polylines for all of the
    map objects. Whenever observe() is called, it will crop and rotate the
    polylines (with batch vectorized operations) to produce the feature vectors
    in a fast manner.

    """

    def __init__(self,
                 fov: FieldOfView,
                 segment_resolution: float = 1.0,
                 polyline_size: int = 1,
                 polyline_limit: int = 256):
        """Construct a MapPolylinePerception instance.

        Args:
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

        """
        self._fov = fov
        self._segment_resolution = segment_resolution
        self._polyline_size = polyline_size
        self._polyline_length = segment_resolution * polyline_size
        self._polyline_limit = polyline_limit

        self._category_encoder = CategoryEncoder()

        # Please refer to Polyline.to_feature to understand the details about
        # the shape of the features.
        self._segment_feature_size = 6
        self._feature_size = self._segment_feature_size * polyline_size
        # Add the size of one hot type encoding
        self._feature_size += self._category_encoder.size

        self._spec = TensorSpec(
            shape=(self._polyline_limit, self._feature_size),
            dtype=torch.float32)

        self._polylines = None

    @property
    def observation_spec(self):
        return self._spec

    def reset(self, road_network, navigation):
        """Initialize by creating the polylines for all the map objects and navigation.

        Args:
            road_network: The road network (map) of the current MetaDrive
                environment.
            navigation: The navigatiton of the current MetaDrive environment.

        """
        polylines = []

        # 1. Collect all the lane boundaries. Each of the lane boundaries will
        # be divided into a set of polylines. Note that ``Polyline.from_lane()``
        # returns a Polyline instance that represents a batch of polylines.
        for _from in road_network.graph.keys():
            for _to in road_network.graph[_from].keys():
                # Between two road network nodes _from and _to, there can be
                # several parallel lanes.
                lanes = road_network.graph[_from][_to]

                # Now extracting polylines from the lane boundaries.
                for lane in lanes:
                    # Since adjacent lanes share boundaries, we are extracting 1
                    # lane boundaries from each lane, unless it is the last
                    # lane, where we extract its boundaries of both sides.
                    num_sides = 2 if lane is lanes[-1] else 1
                    for side in range(num_sides):
                        # Ignore lane boundaries with a type of NONE, because
                        # that means in the real world there will be no line
                        # drawn on the ground for them, even though logically
                        # they exist.
                        if lane.line_types[side] == LineType.NONE:
                            continue
                        category = self._category_encoder.encode_line_type(
                            lane.line_types[side])
                        offset = side - 0.5
                        polylines.append(
                            Polyline.from_lane(lane, offset, category,
                                               self._segment_resolution,
                                               self._polyline_size))

        # 2. Collect all the navigation lines. They are actually center lines of
        # selected lanes.
        for i in range(len(navigation.checkpoints) - 1):
            _from = navigation.checkpoints[i]
            _to = navigation.checkpoints[i + 1]
            lanes = road_network.graph[_from][_to]
            for lane in lanes:
                category = self._category_encoder.encode_navigation()
                polylines.append(
                    Polyline.from_lane(lane, 0.0, category,
                                       self._segment_resolution,
                                       self._polyline_size))

        # 3. Now polylines is a list of Polyline instances, where each Polyline
        # instance corresponds to a batch of polylines. Here we flatten it so
        # that it becomes a big batch of polylines within one single Polyline
        # instance, which gets stored in self._polylines.
        self._polylines = Polyline(
            point=np.concatenate([pl.point for pl in polylines], axis=0),
            category=np.concatenate([pl.category for pl in polylines], axis=0))

    def observe(self, position: tuple,
                heading: float) -> Tuple[np.ndarray, int]:
        """Called upon every observation to get a rotated and cropped view of the map
        objects and navigation. Returns the feature vector of the observation.

        The position and heading of the observer car needs to be provided to
        define the coordinate frame of the resulting features.

        Args:

            position: A 2D vector denoting the current world frame position of
                the ego car.
            heading: A radian denoting the current world frame heading of the
                ego car.

        Returns:

            A Tuple of 2:

            1. A feature tensor of shape [polyline_limit, feature_size], where
               feature size is determined by the number of segments in each
               polyline (polyline_size).

            2. An integer indicating among the polyline_limit of polylines, how
               many of them are actually filled. If the feature has 128
               polylines and only 120 are filled, feature's [120:] will be all
               zeros.

        """
        # 1. Filter the polylines to keep only the ones that are within FOV
        polylines = self._polylines.transformed_within_fov(
            position, heading, self._fov)

        # 2. Filter the polylines to make the population below the limit
        polylines = polylines.keep_closest_n(self._polyline_limit)
        polyline_count = polylines.point.shape[0]

        # 3. Fill in the features
        return polylines.to_feature(
            required_batch_size=self._polyline_limit,
            category_encoder=self._category_encoder), polyline_count
