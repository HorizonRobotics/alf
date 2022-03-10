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

from __future__ import annotations
from typing import NamedTuple, Optional

import numpy as np

try:
    import metadrive
    from metadrive.constants import LineType
    from metadrive.component.lane.metadrive_lane import MetaDriveLane
except ImportError:
    from unittest.mock import Mock
    # create 'metadrive' as a mock to not break python argument type hints
    metadrive = Mock()


class FieldOfView(object):
    """Describe the area with respect to the origin (0, 0) that are visible.

    Under the hood the FieldOfView object is a rectangular bounding box.

    """

    def __init__(self,
                 front: float = 60.0,
                 rear: float = 40.0,
                 lateral: float = 30.0):
        """Construct a FieldOfView object by specifying the relative metrics. Note that
        this is in car-body coordinate frame, where (1, 0) points to the car's
        orientation direction.

        Args:

            front: Defines how far away are visible to the front of the car.
            rear: Defines how far away are visible to the back of the car.
            lateral: Defines how far away are visible to the left and right of the car.

        """
        self._bbox = np.array([[-rear, -lateral], [front, -lateral],
                               [front, lateral], [-rear, lateral]])

    @property
    def bbox(self):
        return self._bbox

    def within(self, points: np.ndarray) -> np.ndarray:
        """Returns for each of the input points, whether they are within the
        field of view or not.

        Args:

            points: A n-d tensor with shape of [..., 2] describing a
                batch of input 2D points.

        Returns:
            
            A (n-1)-d tensor with shape ``points.shape[:-1]``. Each
            cell in the result is True (the point is within the FOV)
            or False (the point is not within the FOV).

        """
        assert points.shape[-1] == 2
        return np.all(
            np.logical_and(points >= self._bbox[0], points <= self._bbox[2]),
            axis=-1)


class CategoryEncoder(object):
    """A category encoder can 
       
    1. Convert integer categories into their corresponding one-hot encoding.
    2. Translate the type of driving scenario objects into its integer category.

    """

    def __init__(self):
        self._num_types = 4
        self._one_hots = np.identity(self._num_types, dtype=np.float32)

    @property
    def size(self) -> int:
        """Get the number of categories.

        Returns:
            An integer representing the total number of categories.
        """
        return self._num_types

    def encode_line_type(self, line_type: LineType) -> int:
        """Translate the line type into its corresponding category index.

        Args:
            line_type: the line type specifying the category of the
                line, e.g. broken line, continuous line.

        Returns:
            An integer within [0, self.size - 1] denoting the index of
            the category corresponding to the line type.

        """
        if line_type == LineType.BROKEN:
            return 1
        elif line_type == LineType.CONTINUOUS:
            return 2
        elif line_type == LineType.SIDE:
            return 3

    def encode_navigation(self) -> int:
        """Get the category index of the navigation.

        Returns:
            An integer within [0, self.size - 1] denoting the index of
            the category representing the navigation category.

        """
        return 0

    def get_codes(self, indices: np.ndarray) -> np.ndarray:
        """Convert category indices to the onehot encoding vectors.

        Args:
            indices: A tensor of category indices.

        Returns:
            A tensor with one extra dimension compared to the input, with each
            integer in the input replaced by its corresponding onehot encoding
            vector.

        """
        return self._one_hots[indices]


class Polyline(NamedTuple):
    """Hold a single polyline or a batch of polylines.

    A single 2D polyline of S segments can be represented as an ordered sequence
    of (S + 1) 2D points. A batch of B 2D polylines of S segments can be
    represented as B x such point sequences. Each polyline can optionally have a
    category attached to them, which is denoted as an integer.

    When the Polyline instance is used to store a batch of polylines
        ``point`` will be of shape [B, S + 1, 2], and
        ``category`` will be of shape [B,] if it is not None

    When the Polyline instance is used to store a single polyline
        ``point`` will be of shape [S + 1, 2], and
        ``category`` will be a single integer if it is not None

    The Polyline class provides useful utility methods to transform the
    polyline(s) within it, and to extract features from it.

    """
    point: np.ndarray  # [B, S + 1, 2] or [S + 1, 2], float32
    category: Optional[np.ndarray] = None  # when not None, [B,] or [], int32

    @staticmethod
    def from_lane(lane: MetaDriveLane, lateral: float, category: int,
                  segment_resolution: float, polyline_size: int) -> Polyline:
        """Constructs a Polyline instance from a MetaDrive lane.

        The constructed Polyline instance will contain a set of polylines. The
        input MetaDrive lane (tegother with ``lateral``) describes a curve. It
        will be divided into a set of polylines where each polyline will have
        ``polyline_size`` segments, and each segment will be targeting a length
        of ``segment_resolution``.

        This means that the length of each polyline will be the product of
        ``segment_resolution`` and ``polyline_size`` unless it is the last
        polyline. The last polyline is usually longer or smaller, depending on
        the length of curve described by the lane.

        Args:

            lane: a MetaDrive lane as the reference for the target curve.
            lateral: describes the lateral offset (ratio) of the target curve
                with respect to the lane. It should be within [-0.5, 0.5]. For
                example, if it is set to -0.5, it means that the target curve is
                the right side boundary of the lane. Similarly, 0.5 is for the
                left side boungdary, and 0.0 is for the cetner line of the lane.
            category: an integer representing the category of the curve. For
                example, whether it is a broken line or a solid line. Not all
                polylines need to have a category, but it is necessary here
                since the polyline represents an object from the map.
            segment_resolution: together with ``polyline_size`` it describes how
                to divided the target curve into polylines. See method docstring
                for details. Unit in meters.
            polyline_size: together with ``segment_resolution`` it describes how
                to divided the target curve into polylines. See method docstring
                for details. Unit in meters.

        Returns:
            A Polyline instance containing a batch of polylines extracted from
            the input lane.

        """
        polyline_length = segment_resolution * polyline_size

        # Trying to figure out how many polylines we can get from the target
        # curve, whose length is ``lane.length``. In the usual case when the
        # length is not perfectly divisible by the polyline's length, we will
        # have to adjust the segment length ``seg_len`` a bit so that we have an
        # integer number of segments.
        num_polylines = int(np.ceil(lane.length / polyline_length))
        seg_len = lane.length / (num_polylines * polyline_size)

        result = Polyline(
            point=np.zeros((num_polylines, polyline_size + 1, 2),
                           dtype=np.float32),
            category=np.full((num_polylines, ), category, dtype=np.int32))

        # Here is the distance of the sampled point along the curve, from the
        # starting point of the curve.
        s = 0.0
        sample_point = lane.position(s, lateral * lane.width_at(s))
        for i in range(num_polylines):
            result.point[i, 0] = sample_point

            for j in range(polyline_size):
                s += seg_len
                sample_point = lane.position(s, lateral * lane.width_at(s))
                result.point[i, j + 1] = sample_point

        return result

    @property
    def batched(self) -> bool:
        """Returns true if the instance represents a batch of polylines, as opposed to a
        single polyline.

        """
        return self.point.ndim == 3

    def transformed(self, center: np.ndarray, orientation: float) -> Polyline:
        """Returns a transformed (set of) polyline(s) so that the resulting points are
        in the coordinate frame defined by the pose (center and the orientation).

        Args:

            center: A 2D point denoting the origin of the new coordinate frame.
            orientation: orientation (x-axis direction) of the new coordinate in radian.

        Returns:

            A NEW Polyline instance where all the polylines within are transformed.
        """
        # Construct the 2D rotation matrix.
        cos = np.cos(orientation)
        sin = np.sin(orientation)
        rotation = np.array([[cos, -sin], [sin, cos]])

        transformed = np.matmul(self.point - center, rotation)

        return Polyline(point=transformed, category=self.category)

    def transformed_within_fov(self, position: np.ndarray, heading: float,
                               fov: FieldOfView) -> Polyline:
        """Transform the polylines into the car-body coordinate frame defined by the
        car's position and heading, and filtered out the polylines that are not
        within the field of view.

        Args:

            position: the position of the car serving as the observer.
            heading: heading of the car serving as the observer.
            fov: the field of view of the car defining the area that are visible.

        Returns:

            A NEW Polyline instance where only the polylines within the
            specified FOV are kept.

        """
        transformed = self.transformed(position, heading)

        within_bbox = fov.within(transformed.point)  # Shape is [B, S]
        within_bbox = np.any(within_bbox, axis=1)  # Shape is now [B,]

        return Polyline(
            point=transformed.point[within_bbox],
            category=self.category[within_bbox])

    def keep_closest_n(self, n: int) -> Polyline:
        """Filter the polylines so that only the closest ``n`` polylines are kept. The
        distances are measured as L2 distance with respect to (0, 0).

        """
        if self.point.shape[0] > n:
            distances = np.min(np.linalg.norm(self.point, axis=-1), axis=-1)
            closest = np.argpartition(distances, n)[:n]
            return Polyline(
                point=self.point[closest], category=self.category[closest])
        return self

    def to_feature(
            self,
            required_batch_size: Optional[int] = None,
            category_encoder: Optional[CategoryEncoder] = None) -> np.ndarray:
        """Convert the polyline(s) to their corresponding feature vectors.

        Suppose there are B polylines, each with S segments (and therefore S + 1
        points). The feature construction is described below.

        1. The feature vector of a single segment is of 6d
           - the direction of the middle point of the segment (unit vector)
           - the direction along the segment (unit vector)
           - the distance between (0, 0) and the middle point
           - the length of the segment

        2. The feature vector of a polyline is a concatenation (ordered) of all
           the feature vectors of its segments plus one-hot encoding of the
           polyline's category if present. Each polyline feature vector is of
           size F = (S * 6 + numumber of categories).

        3. The feature vector of the batch of polylines (i.e. the Polyline
           instance itself) is a stack of all the polyline feature vectors. The
           shape is therefore [B, F].

        Note that if the Polyline instance is NOT batched, ignore No. 3 above.

        Args:

            required_batch_size: The returned feature should have a batch size
                of this. If the number of polylines does not match this number,
                employ zero padding to make it so. ONLY USEFUL for batched case.
            category_encoder: an encoder that can convert integer category to
                its corresponding one-hot encoding. When not provided, category
                one-hot encoding WILL NOT be appended to the polyline feature.

        Returns:
            A tensor as the feature representation of the polylines in the instance.
        """
        size = self.point.shape[0] if self.batched else None

        assert (size is None) == (required_batch_size is None)
        if required_batch_size is not None:
            assert size <= required_batch_size, (
                f'batch size ({size}) should not exceed '
                f'required_batch_size ({required_batch_size})')

        # S = polyline size
        S = self.point.shape[-2] - 1
        mid_points = (self.point[..., 1:, :] + self.point[..., :-1, :]) * 0.5
        r = np.linalg.norm(mid_points, axis=-1, keepdims=True) + 1e-5
        mid_points = mid_points / r
        vecs = self.point[..., 1:, :] - self.point[..., :-1, :]
        d = np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-5
        vecs = vecs / d

        category_size = category_encoder.size if category_encoder else 0

        # TODO(breakds): Merge those almost identical duplicate code below if
        # there is a good way that does not hurt readability.
        if size is not None:
            feature = np.zeros((required_batch_size, 6 * S + category_size),
                               dtype=np.float32)
            feature[:size, :(S * 2)] = mid_points.reshape(-1, S * 2)
            feature[:size, (S * 2):(S * 4)] = vecs.reshape(-1, S * 2)
            feature[:size, (S * 4):(S * 5)] = r.squeeze(axis=-1)
            feature[:size, (S * 5):(S * 6)] = d.squeeze(axis=-1)
            if category_encoder is not None and self.category is not None:
                feature[:size, (S * 6):] = category_encoder.get_codes(
                    self.category)
        else:
            feature = np.zeros(6 * S, dtype=np.float32)
            feature[:(S * 2)] = mid_points.reshape(S * 2)
            feature[(S * 2):(S * 4)] = vecs.reshape(S * 2)
            feature[(S * 4):(S * 5)] = r.squeeze(axis=-1)
            feature[(S * 5):(S * 6)] = d.squeeze(axis=-1)
            if category_encoder is not None and self.category is not None:
                feature[(S * 6):] = category_encoder.get_codes(self.category)

        return feature
