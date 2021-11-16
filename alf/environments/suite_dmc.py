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
from alf.environments.alf_environment import AlfEnvironment
from alf.environments.suite_gym import wrap_env
from unittest.mock import Mock

try:
    import dmc2gym
except ImportError:
    # create 'carla' as a mock to not break python argument type hints
    dmc2gym = Mock()


def is_available():
    return not isinstance(dmc2gym, Mock)


class AlfEnvironmentDMC2GYMWrapper(AlfEnvironment):
    """AlfEnvironment wrapper forwards calls to the given environment."""

    def __init__(self, env):
        """Create an ALF environment base wrapper.

        Args:
            env (AlfEnvironment): An AlfEnvironment instance to wrap.

        Returns:
            A wrapped AlfEnvironment
        """
        super(AlfEnvironmentDMC2GYMWrapper, self).__init__()
        self._env = env

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self._env, name)

    @property
    def batched(self):
        return self._env.batched

    @property
    def batch_size(self):
        return self._env.batch_size

    @property
    def num_tasks(self):
        return self._env.num_tasks

    @property
    def task_names(self):
        return self._env.task_names

    def _reset(self):
        reset_result = self._env.reset()
        reset_result = reset_result._replace(env_info={})
        return reset_result

    def _step(self, action):
        step_result = self._env.step(action)
        step_result = step_result._replace(env_info={})
        return step_result

    def get_info(self):
        return self._env.get_info()

    def env_info_spec(self):
        return self._env.env_info_spec()

    def time_step_spec(self):
        return self._env.time_step_spec()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reward_spec(self):
        return self._env.reward_spec()

    def close(self):
        return self._env.close()

    def render(self, mode='rgb_array'):
        return self._env.render(mode)

    def seed(self, seed):
        return self._env.seed(seed)

    def wrapped_env(self):
        return self._env


@alf.configurable
def dmc_loader(environment_name,
               domain_name='cheetah',
               task_name='run',
               seed=1,
               from_pixels=True,
               pre_transform_image_size=100,
               action_repeat=1,
               env_id=None,
               discount=1.0,
               max_episode_steps=1000,
               gym_env_wrappers=(),
               alf_env_wrappers=[AlfEnvironmentDMC2GYMWrapper]):
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
        pre_transform_inage_size (int): The height and width of the output 
            image from the environment.
        action_repeat (int): Action repeat of gym environment.
        env_id (str): The environment id generated form domain_name, task_name
            and seed.
        discount (float): Discount to use for the environment.
        max_episode_steps (int): The maximum episode step in the environment.
            Note that the episode length in the alf will be max_episode_steps/action_repeat.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers
            classes to use directly on the gym environment.
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment.
        image_channel_first (bool): whether transpose image channels to first dimension.

    Returns:
        A wrapped AlfEnvironment
    """

    gym_env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=False,
        from_pixels=from_pixels,
        episode_length=max_episode_steps,
        height=pre_transform_image_size,
        width=pre_transform_image_size,
        frame_skip=action_repeat)
    return wrap_env(
        gym_env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps / action_repeat,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers)
