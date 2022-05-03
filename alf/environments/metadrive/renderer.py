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

from typing import Optional, Callable, Any

import numpy as np

try:
    import pygame
    import metadrive
    from metadrive.obs.top_down_renderer import TopDownRenderer, history_object
    from metadrive.utils.map_utils import is_map_related_instance
except ImportError:
    from unittest.mock import Mock
    # create 'metadrive' as a mock to not break python argument type hints
    metadrive = Mock()
    pygame = Mock()

from .sensors import VectorizedObservation

# Fix the color for the ego car to makes it stand out.
EGO_COLOR = (255, 127, 80)


class Renderer(TopDownRenderer):
    """Specialized TopDownRenderer for MetaDrive that adds extra information by

    1. rendering actions and some other internal info
    2. rendering observations

    The original MetaDrive top-down renderer renders on an 1000 x 1000 canvas.
    This specialized renderer extends the area to be 1000 x 1200 so that the
    bottom part of size 1000 x 200 can be used for extra information.

    """

    def __init__(self,
                 observation_renderer: Optional[
                     Callable[[pygame.Surface, Any], None]] = None):
        """Construct a Renderer instance.

        Please refer to make_vectorized_observation_renderer nad
        make_bird_eye_observation_renderer below for available observaion
        renderers.

        Args:

            observation_renderer: A function that takes in a canvas (typed as
                pygame.Surface) and an observation, and renders the observation
                on the canvas. When specified as None, this Renderer will not
                render observation.

        """
        super().__init__()
        self._render_size = (1000, 1200)
        self._render_canvas = pygame.display.set_mode(self._render_size)
        self._render_canvas.set_alpha(None)
        self._render_canvas.fill((255, 255, 120))
        self._observation_renderer = observation_renderer
        pygame.init()

    def _append_frame_objects(self, objects):
        ego = self.engine.agent_manager.active_agents['default_agent']
        frame_objects = []
        for name, obj in objects.items():
            color = obj.top_down_color
            if obj is ego:
                color = EGO_COLOR
            frame_objects.append(
                history_object(
                    name=name,
                    heading_theta=obj.heading_theta,
                    WIDTH=obj.top_down_width,
                    LENGTH=obj.top_down_length,
                    position=obj.position,
                    color=color,
                    done=False))
        return frame_objects

    def _draw_info(self):
        """Draws extra info on the screen, including

        1. The longitudinal control (throttle/brake), in [-1, 1]
        2. The lateral control (steering), in [-1, 1]
        3. The current velocity of ego

        """
        if self.pygame_font is None:
            self.pygame_font = pygame.font.SysFont("Arial.ttf", 20)
        ego = self.engine.agent_manager.active_agents['default_agent']
        text = self.pygame_font.render(f'Lon: {ego.throttle_brake:.3f}', True,
                                       (0, 0, 255))
        self.canvas.blit(text, (40, 40))
        text = self.pygame_font.render(f'Lat: {ego.steering:.3f}', True,
                                       (0, 0, 255))
        self.canvas.blit(text, (40, 60))
        speed = ego.speed * 1000.0 / 3600.0
        text = self.pygame_font.render(f'Vel: {speed:.3f} m/s', True,
                                       (0, 0, 255))
        self.canvas.blit(text, (40, 80))

    def render(self, observation=None):
        """Renders the current frame.

        This function is designed to be called once per frame. It dras the map,
        the dynamic objects (ego and other cars), while also visualizing the
        observation given

        1. An observation renderer is specified upon construction
        2. An observation is passed in

        Args:

            observation: The observation of the current frame. If None, nothing
                will be rendered about the observation.

        Returns:

            The canvas with everything rendered. The return value is provided
            for recording purposes, and the render on screen does not rely on
            having this return value.

        """
        # Record current target vehicle
        objects = self.engine.get_objects(lambda obj:
                                          not is_map_related_instance(obj))
        this_frame_objects = self._append_frame_objects(objects)
        self.history_objects.append(this_frame_objects)

        self._handle_event()
        self.refresh()
        self._draw()
        self._draw_info()
        if observation is not None and self._observation_renderer is not None:
            self._observation_renderer(self.canvas, observation)
        self.blit()

        frame = self._render_canvas.copy()
        frame = frame.convert(24)
        return frame


def make_vectorized_observation_renderer(sensor: VectorizedObservation):
    """Create a renderer for the vectorized observation.

    The created renderer is a closure that draws vectorized observation on a
    pygame Surface. The parameters about the observation is retrieved from the
    input sensor.

    Args:

        sensor: A vectorized observation sensor providing properties about the
            observation, such as the number of polylines and the number of
            segments within the polylines.

    """
    # The display area of the visualization will be a scaled rectangle
    # corresponding to the field of view of the sensor.
    fov = sensor.fov
    actual_height = fov.bbox[2][1] - fov.bbox[0][1]  # display height in meters
    actual_width = fov.bbox[1][0] - fov.bbox[0][0]  # display width in meters

    # The display height in pixels is fixed at 200. Multiplying by scale will
    # convert distance in meters to distance in pixels.
    scale = 200.0 / actual_height
    height = 200.0
    width = actual_width * scale

    # Draw a rectangular background indicating the field of view.
    background = pygame.Rect((0.0, 1000.0, width, 200.0))

    # Extract the centers of all segments.
    origin = np.array(
        [-fov.bbox[0][0] * scale, 1000.0 + fov.bbox[2][1] * scale])
    # Number of segments per polyline.
    k = sensor.polyline_size

    # Helper function that draws polyline features from the map on the canvas.
    def draw_map(canvas, map_feature):
        r = (map_feature[:, :(k * 2)].reshape(-1, k, 2) * np.expand_dims(
            map_feature[:, (k * 4):(k * 5)], -1))
        ab = (map_feature[:, (k * 2):(k * 4)].reshape(-1, k, 2) *
              np.expand_dims(map_feature[:, (k * 5):(k * 6)], -1)) * 0.5
        points = np.zeros((map_feature.shape[0], k + 1, 2))
        points[:, :-1] = r - ab
        points[:, -1] = r[:, -1] + ab[:, -1]
        points = points * scale + origin
        colors = (map_feature[:, (k * 6 + 1):(k * 6 + 4)] * 255.0).astype(
            np.int32)

        for i in range(map_feature.shape[0]):
            pygame.draw.lines(canvas, colors[i], False, points[i])

    # Helper function that draws agent features from the map on the canvas.
    def draw_agents(canvas, agent_feature):
        n, h = agent_feature.shape[:2]
        # n * h * 2
        cg = agent_feature[:, :, 1:3] * np.expand_dims(agent_feature[:, :, 0],
                                                       -1)
        lon = agent_feature[:, :, 5:7]
        lat = np.matmul(lon, np.array([[0.0, -1.0], [1.0, 0.0]]))
        lon = lon * np.expand_dims(agent_feature[:, :, 3], -1) * 0.5
        lat = lat * np.expand_dims(agent_feature[:, :, 4], -1) * 0.5
        contour = np.zeros((n, h, 4, 2), dtype=np.float32)
        contour[:, :, 0] = cg - lon - lat
        contour[:, :, 1] = cg - lon + lat
        contour[:, :, 2] = cg + lon + lat
        contour[:, :, 3] = cg + lon - lat
        contour = contour * scale + origin

        for i in range(n):
            for j in range(h):
                length = agent_feature[i, j, 3]
                if length < 1e-2:
                    continue
                pygame.draw.lines(canvas, (50, 50, 50), True, contour[i, j])

    # The actual render closure that will be returned.
    def render(canvas: pygame.Surface, observation):
        pygame.draw.rect(canvas, (240, 240, 240), background)
        pygame.draw.circle(canvas, EGO_COLOR, center=origin, radius=4.0)
        draw_map(canvas, observation['map'])
        draw_agents(canvas, observation['agents'])

    return render


def make_bird_eye_observation_renderer():
    """Create a renderer for the BEV observation.

    Each channel from the BEV will be drawn on the canvas in a row.

    """

    def render(canvas: pygame.Surface, observation):
        bevs = observation['bev'] * 255.0
        bevs = bevs.astype(int)

        # Draw each channel in a row.
        for i in range(observation['bev'].shape[0]):
            observation_surface = pygame.surfarray.make_surface(bevs[i, :, :])
            canvas.blit(observation_surface, (120 + i * 100, 1020))

    return render
