# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from collections import deque
import math

import carla

import alf


class PIDController(object):
    """PID controller.

    See https://en.wikipedia.org/wiki/PID_controller for reference
    """

    def __init__(self, K_P, K_I, K_D, dt, integration_time_window=0.5):
        """
        Args:
            K_P (float): coefficient for the proportional term
            K_I (float): coefficient for the integral term
            K_D (float): coefficient for the derivative term
            dt (float): time interval in seconds for each step
            integration_time_window (float): the window for the integral in terms
                of seconds. The integration is implemented as an exponentially
                weighted sum over the past errors where the weight is decayed by
                1 - dt/integration_time_window every step.
        """
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._dt = dt
        self._integral = 0.
        self._prev_error = 0.
        self._int_decay = 1 - dt / integration_time_window

    def step(self, current, target):
        """Calculate control for one step.

        Args:
            current (float): the current value
            target (float): the desired value
        Returns:
            float: control
        """
        error = target - current
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / self._dt
            self._integral = self._int_decay * self._integral + error * self._dt
        else:
            derivative = 0.
        self._prev_error = error
        return self._K_P * error + self._K_I * self._integral + self._K_D * derivative

    def reset(self):
        """Reset the controller."""
        self._prev_error = None
        self._integral = 0.


@alf.configurable(blacklist=['vehicle', 'step_time'])
class VehicleController(object):
    """A simple vehicle controller using PID controller."""

    def __init__(self,
                 vehicle,
                 step_time,
                 max_speed=5.56,
                 max_throttle=0.75,
                 max_steering=0.8,
                 max_brake=0.3,
                 s_P=3.6,
                 s_I=0.18,
                 s_D=0,
                 d_P=1.95,
                 d_I=0.07,
                 d_D=0.2):
        """
        The defaults are from https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/local_planner.py.
        Note that the max_speed and gain parameters for speed are originally
        specified for speed in the unit of km/h. Since here we use m/s, we have
        converted them as follows as our default values:
            max_speed = (20 km/h) / 3.6 =  5.56 m/s
            s_P = (1.0 h/km) * 3.6 = 3.6 s/m
            s_I = (0.05 h/km) * 3.6 = 0.18 s/m
            s_D = (0 h/km) * 3.6 = 0 s/m

        Args:
            vehicle (carla.Actor): the actor for vehicle
            step_time (float): time interval in seconds for each step
            max_speed (float): maximal speed in m/s. Default to 5.6 m/s which
                is about 20 km/h.
            max_throttle (float): maximal throttle
            max_steering (float): maximal steering
            max_brake (float): maximal brake
            s_P (float): coefficient of the proportional term for the speed
                controller, with the unit as s/m
            s_I (float): coefficient of the integral term for the speed
                controller, with the unit as s/m
            s_D (float): coefficient of the derivative term for the speed
                controller, with the unit as s/m
            d_P (float): coefficient of the proportional term for the direction controller
            d_I (float): coefficient of the integral term for the direction controller
            d_D (float): coefficient of the derivative term for the direction controller
        """
        self._vehicle = vehicle
        self._speed_controller = PIDController(s_P, s_I, s_D, step_time)
        self._direction_controller = PIDController(d_P, d_I, d_D, step_time)
        self._prev_steering = 0.
        self._max_speed = max_speed
        self._max_throttle = max_throttle
        self._max_steering = max_steering
        self._max_brake = max_brake

    def action_spec(self):
        """Get the action spec.

        The action is a 3-D vector of [speed, direction, reverse], where speed is in
        [-1.0, 1.0] with negative value meaning zero speed and 1.0 corresponding
        to maximally allowed speed as provided by the ``max_speed`` argument for
        ``__init__()``, and direction is the relative direction that the vehicle
        is facing, with 0 being front, -0.5 being left and 0.5 being right, and
        reverse is interpreted as a boolean value with values greater than 0.5
        corresponding to True to indicate going backward.

        Returns:
            alf.BoundedTensorSpec
        """
        return alf.BoundedTensorSpec([3],
                                     minimum=[-1., -1., 0.],
                                     maximum=[1., 1., 1.])

    def action_desc(self):
        """Get the description about the action.

        Returns:
            str: the description about the action
        """
        return (
            "3-D vector of [speed, direction, reverse], where speed is in "
            "[-1.0, 1.0] with negative value meaning zero speed and 1.0 corresponding to "
            "maximally allowed speed as provided by max_speed argument for __init__(), "
            "direction is the relative direction that the vehicle is facing, with "
            "0 being front, -0.5 being left and 0.5 being right, and reverse is "
            "interpreted as a boolean value with values greater than 0.5 "
            "corresponding to True to indicate going backward.")

    def act(self, action):
        """Generate carla.VehicleControl based on ``action``

        Args:
            action (np.ndarray): 3-D vector representing action
        Returns:
            carla.VehicleControl
        """
        target_speed, target_direction, reverse = action
        target_speed = max(0., target_speed) * self._max_speed
        target_direction = min(1., max(-1., target_direction))
        target_direction = math.pi * target_direction
        v = self._vehicle.get_velocity()
        speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

        control = carla.VehicleControl()

        acceleration = self._speed_controller.step(speed, target_speed)
        if acceleration >= 0.:
            control.throttle = min(acceleration, self._max_throttle)
            control.brake = 0.
        else:
            control.brake = min(-acceleration, self._max_brake)
            control.throttle = 0.

        reverse = bool(reverse > 0.5)
        if reverse:
            target_direction = -target_direction
        steering = self._direction_controller.step(0., target_direction)
        steering = max(self._prev_steering - 0.1,
                       min(self._prev_steering + 0.1, steering))

        steering = max(-self._max_steering, min(self._max_steering, steering))
        control.steer = steering

        self._prev_steering = steering
        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        control.reverse = reverse

        return control

    def reset(self):
        """Reset the controller."""
        self._speed_controller.reset()
        self._direction_controller.reset()
        self._prev_steering = 0.
