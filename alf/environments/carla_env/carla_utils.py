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
"""Utility functions for Carla.
"""

import cv2
from enum import IntEnum
import numpy as np
from pathlib import Path
from typing import NamedTuple
from unittest.mock import Mock

import alf

try:
    import carla
except ImportError:
    # create 'carla' as a mock to not break python argument type hints
    carla = Mock()


# ==============================================================================
# -- Traffic Light Utilities ---------------------------------------------------
# ==============================================================================
class TrafficLightHandler(object):
    """
    This class provides utility functions related to traffic lights.
    Currently the main functionality is to retrieve the stoplines
    for all the traffic lights, organized according to their current color
    states, i.e., the stopline of a traffic light is recorded into the list
    for the current color state of the traffic light.

    Adapted from https://github.com/zhejz/carla-roach/blob/5654984748b64d79f0dafbe0e02a01bff4337eb4/carla_gym/utils/traffic_light.py
    """

    num_tl = 0
    list_tl_actor = []  # traffic light actor
    list_tv_loc = []  # traffic light trigger volume location
    list_stopline_wps = []  # stop line waypoint
    list_stopline_vtx = []  # stop line vertex

    @staticmethod
    def reset(alf_world):
        """
        Args:
            alf_world (World): an instance of the World class defined in
                alf.environments.carla_sensors.
        """
        TrafficLightHandler.num_tl = 0
        TrafficLightHandler.list_tl_actor = []
        TrafficLightHandler.list_tv_loc = []
        TrafficLightHandler.list_stopline_wps = []
        TrafficLightHandler.list_stopline_vtx = []

        # get all actors including traffic lights
        all_actors = alf_world._world.get_actors()
        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                tv_loc, stopline_wps, stopline_vtx = \
                    alf_world._get_traffic_light_waypoints(_actor)

                TrafficLightHandler.list_tl_actor.append(_actor)
                TrafficLightHandler.list_tv_loc.append(tv_loc)
                TrafficLightHandler.list_stopline_wps.append(stopline_wps)
                TrafficLightHandler.list_stopline_vtx.append(stopline_vtx)

                TrafficLightHandler.num_tl += 1

    @staticmethod
    def get_stopline_vtx(query_location: carla.Location, dist_threshold=50.0):
        """Get the list of stop line vertices satisfying both the color and
        distance conditions for a given query location.
        Args:
            query_location (carla.Location): the location we want to query with
            dist_threshold (float): unit: meter. Only the traffic lights with a
                distance to query_location smaller than `dist_threshold` will
                be considered as valid.
        Return:
            return the list of stop line vertices for green, yellow and red
            lights respectively.
        """
        TRAFFIC_LIGHT_STATES = [
            carla.TrafficLightState.Green, carla.TrafficLightState.Yellow,
            carla.TrafficLightState.Red
        ]

        stopline_vtx = [[] for _ in range(3)]
        for i in range(TrafficLightHandler.num_tl):
            traffic_light = TrafficLightHandler.list_tl_actor[i]
            tv_loc = TrafficLightHandler.list_tv_loc[i]
            if tv_loc.distance(query_location) > dist_threshold:
                continue
            stopline_vtx[TRAFFIC_LIGHT_STATES.index(
                traffic_light.state)].extend(
                    TrafficLightHandler.list_stopline_vtx[i])

        return stopline_vtx[0], stopline_vtx[1], stopline_vtx[2]


# ==============================================================================
# -- Map utilities -------------------------------------------------------------
# ==============================================================================

# unit: meter
MAP_BOUNDARY_MARGIN = 100


class MapBoundaries(NamedTuple):
    """Distances in carla.World coordinates (unit: meter)"""
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@alf.configurable
class MapHandler(object):
    """This class provides a number of utility functions related to the map,
    including the structured representation generation and mask generation
    for map related elements, e.g. lane, road etc.
    See the Carla documentation about details on map related concepts at
    `<https://carla.readthedocs.io/en/latest/core_map/>`_.
    Current implementation will generate masks for road, lane, broken lane
    Adapted from https://github.com/deepsense-ai/carla-birdeye-view/blob/master/carla_birdeye_view/mask.py
    """

    def __init__(self,
                 world,
                 pixels_per_meter,
                 render_lanes_on_junctions=False,
                 fill_road_mask=True):
        """
        Args:
            world (carla.World): an instance of carla.World which provides interface
                to access to map related information.
            pixels_per_meter (float): how many pixels are used to represent one meter
            render_lanes_on_junctions (bool): whether to render the lanes on
                junctions
            fill_road_mask (bool): fill the road polygon if True. Otherwise,
                only the polylines are used to represent the road.
        """
        self._pixels_per_meter = pixels_per_meter
        self._world = world
        self._render_lanes_on_junctions = render_lanes_on_junctions
        self._fill_road_mask = fill_road_mask

        # map and road
        self._map = self._world.get_map()
        # topology contains a list of road segments. Each road segment is a
        # simplified representation of a road using a start and an end waypoint
        self._topology = self._map.get_topology()
        # generate waypoints that are about 2 meters apart
        self._waypoints = self._map.generate_waypoints(2)
        self._map_boundaries = self._find_map_boundaries()

        self._waypoints_by_road = self._generate_road_waypoints()
        # mask size in pixels
        self._mask_height_in_pixels, self._mask_width_in_pixels = \
                self._calculate_mask_size()

    def get_masks(self):
        """Return the masks for all map elements.
        """
        mask_dict = {
            "road": self.get_road_mask(self._fill_road_mask),
            "lane": self.get_lanes_mask()
        }
        return mask_dict

    def _generate_road_waypoints(self, precision=0.5):
        """Return all, precisely located waypoints from the map.
        Topology contains simplified representation (a start and an end
        waypoint for each road segment). By expanding each until another
        road segment is found, we explore all possible waypoints on the map.
        Returns a list of waypoints for each road segment.

        Args:
            precision (float): unit: meter, the precision in representing the
                road with waypoint, i.e., the distance between two adjacent
                road waypoints.
        """

        road_segments_starts: carla.Waypoint = [
            road_start for road_start, road_end in self._topology
        ]

        each_road_waypoints = []
        for road_start_waypoint in road_segments_starts:
            road_waypoints = [road_start_waypoint]

            # Generate as long as it's the same road
            next_waypoints = road_start_waypoint.next(precision)

            if len(next_waypoints) > 0:
                # Always take first (may be at intersection)
                next_waypoint = next_waypoints[0]
                while next_waypoint.road_id == road_start_waypoint.road_id:
                    road_waypoints.append(next_waypoint)
                    next_waypoint = next_waypoint.next(precision)

                    if len(next_waypoint) > 0:
                        next_waypoint = next_waypoint[0]
                    else:
                        # Reached the end of road segment
                        break
            each_road_waypoints.append(road_waypoints)
        return each_road_waypoints

    def _find_map_boundaries(self):
        """Find extreme locations on a map.
        It adds a decent margin because waypoints lie on the road, which means
        that anything that is slightly further than the boundary
        could cause out-of-range exceptions (e.g. pavements, walkers, etc.)
        """
        return MapBoundaries(
            min_x=min(self._waypoints, key=lambda x: x.transform.location.
                      x).transform.location.x - MAP_BOUNDARY_MARGIN,
            min_y=min(self._waypoints, key=lambda x: x.transform.location.
                      y).transform.location.y - MAP_BOUNDARY_MARGIN,
            max_x=max(self._waypoints, key=lambda x: x.transform.location.
                      x).transform.location.x + MAP_BOUNDARY_MARGIN,
            max_y=max(self._waypoints, key=lambda x: x.transform.location.y).
            transform.location.y + MAP_BOUNDARY_MARGIN,
        )

    def _calculate_mask_size(self):
        """Convert map boundaries to size in term of number of pixel."""
        width_in_meters = self._map_boundaries.max_x - self._map_boundaries.min_x
        height_in_meters = self._map_boundaries.max_y - self._map_boundaries.min_y
        width_in_pixels = int(width_in_meters * self._pixels_per_meter)
        height_in_pixels = int(height_in_meters * self._pixels_per_meter)
        return height_in_pixels, width_in_pixels

    def make_empty_mask(self):
        shape = (self._mask_height_in_pixels, self._mask_width_in_pixels)
        return np.zeros(shape, np.uint8)

    def world_to_pixel(self, location: carla.Location, projective=False):
        """Convert the world coordinates to pixel coordinates.
        For example: top leftmost location will be a pixel at (0, 0).
        """
        min_x = self._map_boundaries.min_x
        min_y = self._map_boundaries.min_y
        x = self._pixels_per_meter * (location.x - min_x)
        y = self._pixels_per_meter * (location.y - min_y)

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def location_to_array(self, locations):
        """Convert the format of input positions from carla.Location to
            np.array of the shape [L, 2], with L the number of elements
            in `locations`.
        """
        locs = []
        for loc in locations:
            locs.append(np.array([loc.x, loc.y]))

        loc_np = np.stack(locs, axis=0)
        return loc_np

    def world_to_pixel_np(self, location: np.array):
        """Convert the world coordinates to pixel coordinates.
        `location` is of the shape [L, 2]
        """
        min_x = self._map_boundaries.min_x
        min_y = self._map_boundaries.min_y
        loc = self._pixels_per_meter * (
            location - np.array([[min_x, min_y]], dtype=np.float32))
        return loc

    def get_road_mask(self, fill_road_mask=False):
        """Get the road mask.
        Args:
            fill_road_mask (bool): fill the road polygon if True. Otherwise,
                only the polylines are used to represent the road.
        Return:
            the road mask
        """
        mask = self.make_empty_mask()
        for road_waypoints in self._waypoints_by_road:
            road_left_side = [
                lateral_shift(w.transform, -w.lane_width * 0.5)
                for w in road_waypoints
            ]
            road_right_side = [
                lateral_shift(w.transform, w.lane_width * 0.5)
                for w in road_waypoints
            ]

            polygon_in_world = [*road_left_side, *reversed(road_right_side)]

            polygon = [self.world_to_pixel(x) for x in polygon_in_world]
            if len(polygon) > 2:
                polygon = np.array([polygon], dtype=np.int32)
                cv2.polylines(
                    img=mask, pts=polygon, isClosed=True, color=1, thickness=5)
                if fill_road_mask:
                    cv2.fillPoly(img=mask, pts=polygon, color=1)

        return mask

    def get_lanes_mask(self):
        mask = self.make_empty_mask()
        for road_waypoints in self._waypoints_by_road:
            if self._render_lanes_on_junctions or not road_waypoints[
                    0].is_junction:
                # Left Side
                draw_lane_marking_single_side(
                    mask,
                    road_waypoints,
                    side=LaneSide.LEFT,
                    location_to_pixel_func=self.world_to_pixel,
                    lane_marking_color=1,
                )

                # Right Side
                draw_lane_marking_single_side(
                    mask,
                    road_waypoints,
                    side=LaneSide.RIGHT,
                    location_to_pixel_func=self.world_to_pixel,
                    lane_marking_color=1,
                )
        return mask


# ==============================================================================
# -- Lane utilities ------------------------------------------------------------
# ==============================================================================
class LaneSide(IntEnum):
    LEFT = -1
    RIGHT = 1


def lateral_shift(transform: carla.Transform, shift):
    """Make a lateral shift of the forward vector of a transform.
    Args:
        transform (carla.Transform): an input transform to apply the lateral
            shift to.
        shift (float): the amout of lateral shift to apply. It has the same unit
            as the transform.location.
            A positive value for lateral shift to the right and a negative value
            for lateral shift to the left.
    Return:
        shifted_location (carla.Location): the shifted location
    """
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


def draw_solid_line(canvas, color, closed, points, thickness):
    """Draw solid lines on a canvas given a set of points, width and color.
    Args:
        canvas (np.ndarray): the canvas on which to draw the lines.
        color (float|tuple): a scalar value or a tuple representing the color
            for the line.
        closed (bool): whether the polylines to be drawn are closed or not.
            If True, the function draws a line from the last vertex to its first
            vertex to make it closed.
        points (list): list of points representing the line to be drawn
        thickness (float): the thickness of the line
    """
    if len(points) >= 2:
        cv2.polylines(
            img=canvas,
            pts=np.int32([points]),
            isClosed=closed,
            color=color,
            thickness=thickness,
        )


def draw_broken_line(canvas, color, closed, points, thickness):
    """Draw solid lines on a canvas given a set of points, width and color.
    Args:
        canvas (np.ndarray): the canvas on which to draw the lines.
        color (float|tuple): a scalar value or a tuple representing the color
            for the line.
        closed (bool): whether the polylines to be drawn are closed or not.
            If True, the function draws a line from the last vertex to its first
            vertex to make it closed.
        points (list): list of points representing the line to be drawn
        thickness (float): the thickness of the line
    """

    # Select which lines are going to be rendered from the set of lines
    broken_lines = [
        x for n, x in enumerate(zip(*(iter(points), ) * 20)) if n % 3 == 0
    ]

    # Draw selected lines
    for line in broken_lines:
        cv2.polylines(
            img=canvas,
            pts=np.int32([line]),
            isClosed=closed,
            color=color,
            thickness=thickness,
        )


def get_lane_markings(
        lane_marking_type,
        lane_marking_color,
        waypoints,
        side: LaneSide,
        location_to_pixel_func,
):
    """There are several lane marking types in Carla (SolidSolid, BrokenSolid,
        SolidBroken and BrokenBroken). This function converts them to the
        combination of two basic types: Broken and Solid lines. This makes it
        easier to render different lane marking types by implementing only the
        rendering function of the two basic line types.
        For more information on Carla lane types, please refer to
        `<https://carla.readthedocs.io/en/latest/python_api/#carlalanemarkingtype>`_.
    Args:
        lane_marking_type (carla.LaneMarkingType): the input lane marking type
        lane_marking_color (float|tuple): the color for representing the lane.
        waypoints (list[carla.Waypoint]): the list of waypoints representing
            the lane
        side (LaneSide): the side of the lane to draw the lane marking
        location_to_pixel_func (Callable): the function to convert location
            from world coordinate representation to pixel space representation
    """
    margin = 0.25
    sign = side.value
    marking_1 = [
        location_to_pixel_func(
            lateral_shift(wp.transform, sign * wp.lane_width * 0.5))
        for wp in waypoints
    ]
    if lane_marking_type == carla.LaneMarkingType.Broken or (
            lane_marking_type == carla.LaneMarkingType.Solid):
        return [(lane_marking_type, lane_marking_color, marking_1)]
    else:
        marking_2 = [
            location_to_pixel_func(
                lateral_shift(wp.transform,
                              sign * (wp.lane_width * 0.5 + margin * 2)))
            for wp in waypoints
        ]
        if lane_marking_type == carla.LaneMarkingType.SolidBroken:
            return [
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_2),
            ]
        elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
            return [
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_2),
            ]
        elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
            return [
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_2),
            ]
        elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
            return [
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_2),
            ]
    return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]


def draw_lane_marking_single_side(canvas, waypoints, side: LaneSide,
                                  location_to_pixel_func, lane_marking_color):
    """Draw the lane marking given a set of waypoints and decides
        whether drawing the right or left side of the waypoint based
        on the sign parameter
    Args:
        canvas (np.ndarray): the canvas to draw the lane marking
        waypoints (list[carla.Waypoint]): the list of waypoints representing
            the lane
        side (LaneSide): the side of the lane to draw the lane marking
        location_to_pixel_func (Callable): the function to convert location
            from world coordinate representation to pixel space representation
        lane_marking_color (float|tuple): the color for representing the lane.
    """
    previous_marking_type = carla.LaneMarkingType.NONE
    markings_list = []
    temp_waypoints = []
    current_lane_marking = carla.LaneMarkingType.NONE
    for sample in waypoints:
        lane_marking = (sample.left_lane_marking if side is LaneSide.LEFT else
                        sample.right_lane_marking)

        if lane_marking is None:
            continue

        marking_type = lane_marking.type

        if current_lane_marking != marking_type:
            # Get the list of lane markings to draw
            markings = get_lane_markings(
                previous_marking_type,
                lane_marking_color,
                temp_waypoints,
                side,
                location_to_pixel_func,
            )
            current_lane_marking = marking_type

            # Append each lane marking in the list
            for marking in markings:
                markings_list.append(marking)

            temp_waypoints = temp_waypoints[-1:]

        else:
            temp_waypoints.append((sample))
            previous_marking_type = marking_type

    # add last marking
    last_markings = get_lane_markings(
        previous_marking_type,
        lane_marking_color,
        temp_waypoints,
        side,
        location_to_pixel_func,
    )

    for marking in last_markings:
        markings_list.append(marking)

    # draw the lines once the lane markings have been simplified to Solid or
    # Broken lines
    for markings in markings_list:
        if markings[0] == carla.LaneMarkingType.Solid:
            draw_solid_line(canvas, markings[1], False, markings[2], 1)
        elif markings[0] == carla.LaneMarkingType.Broken:
            draw_broken_line(canvas, markings[1], False, markings[2], 1)
