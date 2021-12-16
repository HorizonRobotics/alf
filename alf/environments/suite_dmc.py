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
import gym

import alf
from alf.environments.dmc_gym_wrapper import DMCGYMWrapper, dm_control
from alf.environments.suite_gym import wrap_env


def is_available():
    """
    Check if the required environment is installed.
    """
    return dm_control is not None


class VideoRenderWrapper(gym.Wrapper):
    """A wrapper to enable 'rgb_array' rendering for the gym wrapper of
    ``dm_control``. To do this, the ``metadata`` field needs to be updated.
    """
    _metadata = {'render.modes': ["rgb_array"]}

    def __init__(self, env):
        super().__init__(env)
        self.metadata.update(self._metadata)


@alf.configurable
def load(environment_name='cheetah:run',
         from_pixels=True,
         image_size=100,
         env_id=None,
         discount=1.0,
         visualize_reward=False,
         max_episode_steps=1000,
         gym_env_wrappers=(),
         alf_env_wrappers=()):
    """ Load a MuJoCo environment.

    For installation of DMControl, see https://github.com/deepmind/dm_control.
    For installation of MuJoCo210, see https://mujoco.org.

    Args:
        environment_name (str): this string must have the format
            "domain_name:task_name", where "domain_name" is defined by DM control as
            the physical model name, and "task_name" is an instance of the model
            with a parcular MDP structure.
        from_pixels (boolean): Output image if set to True.
        image_size (int): The height and width of the output
            image from the environment.
        env_id (int): (optional) ID of the environment.
        discount (float): Discount to use for the environment.
        visualize_reward: if True, then the rendered frame will have
            a highlighted color when the agent achieves a reward.
        max_episode_steps (int): The maximum episode step in the environment.
            Note that the episode length in the alf will be max_episode_steps/action_repeat.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers
            classes to use directly on the gym environment.
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment. There will be an
            AlfEnvironmentDMC2GYMWrapper added before any alf_wrappers.

    Returns:
        A wrapped AlfEnvironment
    """
    names = environment_name.split(":")
    assert len(names) == 2, (
        "environment_name must be in the format 'domain_name:task_name'!"
        f" Provided environment_name: {environment_name}")

    domain_name, task_name = names
    gym_env = DMCGYMWrapper(
        domain_name=domain_name,
        task_name=task_name,
        visualize_reward=visualize_reward,
        from_pixels=from_pixels,
        # The episode_length passed to DMCGYMWrapper is doubled because we want to
        # make sure that the reset is triggered only in alf_wrappers.TimeLimit.
        episode_length=max_episode_steps * 2,
        height=image_size,
        width=image_size)
    gym_env = VideoRenderWrapper(gym_env)
    return wrap_env(
        gym_env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        image_channel_first=False)
