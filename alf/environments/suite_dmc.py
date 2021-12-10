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

import alf
from alf.environments.alf_wrappers import AlfEnvironmentBaseWrapper
from alf.environments.suite_gym import wrap_env
from unittest.mock import Mock

try:
    import dmc2gym
except ImportError:
    # create 'dmc2gym' as a mock to not break python argument type hints
    dmc2gym = Mock()


def is_available():
    """
    Check if the required environment is installed.
    """
    return not isinstance(dmc2gym, Mock)


class AlfEnvironmentDMC2GYMWrapper(AlfEnvironmentBaseWrapper):
    """
    AlfEnvironment wrapper forwards calls to the given environment.
    This wrapper is added to remove env_info, which is useless and 
    will cause some bug in the training process, from the TimeStep.
    """

    @property
    def base_env(self):
        return self._env

    def _reset(self):
        reset_result = self._env.reset()
        reset_result = reset_result._replace(env_info={})
        return reset_result

    def _step(self, action):
        step_result = self._env.step(action)
        step_result = step_result._replace(env_info={})
        return step_result

    def get_info(self):
        return {}

    def env_info_spec(self):
        return {}


@alf.configurable
def dmc_loader(environment_name='dmc2gym',
               domain_name='cheetah',
               task_name='run',
               seed=1,
               from_pixels=True,
               image_size=100,
               action_repeat=1,
               env_id=None,
               discount=1.0,
               max_episode_steps=1000,
               gym_env_wrappers=(),
               alf_env_wrappers=()):
    """ Load the MuJoCo environment through dmc2gym

    This loader will not take environment_name, instead please use domain_name and tesk_name.
    For installation of dmc2gym, see https://github.com/denisyarats/dmc2gym.
    For installation of DMControl, see https://github.com/deepmind/mujoco.
    For installation of MuJoCo200, see https://roboti.us.

    Args:
        environment_name (str): Do not use this arg, this arg is here to
            match up with create_environment.
        domain_name (str): The name of MuJoCo domain that is used.
        task_name (str): The name of task we want the agent to do in the
            current MuJoCo domain.
        seed (int): Random seed for the environment.
        from_pixels (boolean): Output image if set to True.
        image_size (int): The height and width of the output 
            image from the environment.
        action_repeat (int): Action repeat of gym environment.
        env_id (int): (optional) ID of the environment.
        discount (float): Discount to use for the environment.
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

    gym_env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=False,
        from_pixels=from_pixels,
        # The episode_length passed to dmc2gym is doubled because we want to
        # make sure that the reset is triggered only in alf_wrappers.TimeLimit.
        episode_length=max_episode_steps * 2,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat)
    alf_env_wrappers = (AlfEnvironmentDMC2GYMWrapper, ) + alf_env_wrappers
    return wrap_env(
        gym_env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=(max_episode_steps + action_repeat - 1) //
        action_repeat,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        image_channel_first=False,
    )
