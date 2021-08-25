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
"""Wrapper for gym-based Gym3 environments
"""

import gym
import gym3

from alf.environments.alf_wrappers import AlfEnvironmentBaseWrapper


class AlfGym3Wrapper(AlfEnvironmentBaseWrapper):
    """Wraps gym-based Gym3 environments to enable rendering

    The set of Gym3 environemtns provide a gym wrapper which is mostly
    compatible with existing gym interfaces except for rendering.

    The render is enabled upon constructing the environments instead of by
    calling ``render()``. In fact for Gym3 environments ``render()`` does
    nothing.

    This wrapper works that around by re-constructing the gym-based Gym3
    environment upon the first call to ``render()``, with specified mode. This
    makes it transparent for alf's play.

    Note that in our use case ``render()`` is almost always called at
    the very beginning of a play. This makes reconstructing the
    environment a plausible approach that does not interrupt ongoing
    episodes.

    """

    def __init__(self, gym_env: gym3.interop.ToGymEnv):
        """
        Args:
            gym_env (gym3.interop.ToGymEnv): An instance of OpenAI gym environment.

        """
        assert isinstance(gym_env, gym3.interop.ToGymEnv), 'Wrong type!'

        super().__init__(gym_env)

        # Once enabled (set to True), it will be True forever.
        self._render_enabled = False

    def render(self, mode: str):
        """Enables rendering by reconstructing the environment
        """
        if not self._render_enabled:
            self.replace_wrapped_env(self.wrapped_env().spec.make(
                render=(mode == 'human'), render_mode=mode))
            self._render_enabled = True

        # For rgb_array rendering, return the image frame
        if len(self.wrapped_env().env.get_info()) > 0:
            if 'rgb' in self.wrapped_env().env.get_info()[0]:
                return self.wrapped_env().env.get_info()[0]['rgb']
        return None
