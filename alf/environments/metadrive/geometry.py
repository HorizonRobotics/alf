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


class CategoryEncoder(object):
    """A category encoder can 
       
    1. Converts integer categories into their corresponding one-hot encoding.
    2. Translate the type of driving scenario objects into its integer category.

    """

    def __init__(self):
        self._num_types = 4
        self._one_hots = np.identity(self._num_types, dtype=np.float32)

    @property
    def size(self):
        return self._num_types

    def encode_line_type(self, line_type: LineType) -> int:
        if line_type == LineType.BROKEN:
            return 1
        elif line_type == LineType.CONTINUOUS:
            return 2
        elif line_type == LineType.SIDE:
            return 3

    def encode_navigation(self) -> int:
        return 0

    def get_codes(self, indices: np.ndarray) -> int:
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

        """
        polyline_length = segment_resolution * polyline_size

        # Trying to figure out how many polylines we can get from the target
        # curve, whose length is ``lane.length``. In the usual case when the
        # length is not perfectly divisible by the polyline's length, we will
        # have to make it up by making the last polyline longer or shorter (in
        # the shorter case, number of polylines is increased by 1).
        num_polylines = int(lane.length / polyline_length)
        if lane.length % polyline_length > 0.5:
            num_polylines += 1

        result = Polyline(
            point=np.zeros((num_polylines, polyline_size + 1, 2),
                           dtype=np.float32),
            category=np.full((num_polylines, ), category, dtype=np.int32))

        # Here s is the distance of the sampled point along the curve, from the
        # starting point of the curve.
        s = 0.0
        sample_point = lane.position(s, lateral * lane.width_at(s))
        for i in range(num_polylines):
            result.point[i, 0] = sample_point

            # Normally it samples a point every ``segment_resolution`` meters.
            # However for the last segment since it does not have the standard
            # length, we will have to improvise.
            if i == num_polylines - 1:
                seg_len = (lane.length - s) / polyline_size
            else:
                seg_len = segment_resolution

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

        """
        size = self.point.shape[0] if self.batched else None

        assert (size is None) == (required_batch_size is None)

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
