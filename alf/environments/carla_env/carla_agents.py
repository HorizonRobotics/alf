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
"""Implementation of Agents for Carla.
"""

from agents.navigation.basic_agent import BasicAgent


class SimpleNavigationAgent(BasicAgent):
    """
    SimpleNavigationAgent is derived from BasicAgent, which is an agent that
    navigates the scene and respects traffic lights and other vehicles, but
    ignores stop signs.
    Here we adapt it to follow the navigation route from the navigation sensor.
    TODO: Implemnet more advanced control logics.
    """

    def __init__(self, vehicle, navigation_sensor, alf_world, target_speed=20):
        """
        Args:
            vehicle (carla.Actor): the vehicle actor to apply the control onto
            navigation_sensor (NavigationSensor): the navigation sensor which
                will provide the navigation route for the agent to follow
            alf_world (World): an instance of World which keeps all the data
                of the world.
            target_speed (float): speed (in Km/h) at which the vehicle will move
        """
        super().__init__(vehicle)

        self._vehicle = vehicle
        self._navigation_sensor = navigation_sensor
        self._alf_world = alf_world
        self._target_speed = target_speed
        self._global_planner = self._alf_world._global_route_planner

    def set_destination(self):
        """
        Set navigation destination for the agent. It uses the same destination
        as the navigation sensor to enable the data collection is along a valid
        route. It then retrieves the route from the navigation sensor and set it
        as the global plan of the local planner.
        """
        route_trace = self._navigation_sensor._route
        self._local_planner.set_global_plan(route_trace)
