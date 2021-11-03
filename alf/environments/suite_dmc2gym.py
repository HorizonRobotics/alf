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
import dmc2gym
from alf.environments.alf_environment import AlfEnvironment
from alf.environments import gym_wrappers, alf_wrappers, alf_gym_wrapper


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
def dmc2gym_loader(environment_name,
                   domain_name='cheetah',
                   task_name='run',
                   seed=1,
                   pre_transform_image_size=100,
                   action_repeat=4,
                   env_id=None,
                   discount=1.0,
                   max_episode_steps=1000,
                   gym_env_wrappers=(),
                   alf_env_wrappers=[AlfEnvironmentDMC2GYMWrapper],
                   image_channel_first=False):
    """ Load the MuJoCo environment through dmc2gym

    This loader will not take environment_name, instead please use domain_name and tesk_name.
    For installation of dmc2gym, see https://github.com/denisyarats/dmc2gym.
    For installation of DMControl, see https://github.com/deepmind/mujoco.
    For installation of MuJoCo200, see https://roboti.us.
    Args:
        environment_name (str): Do not use this arg, this arg is here to
            metch up with create_environment.
        domain_name (str): The name of MuJoCo domain that is used.
        task_name (str): The name of task we want the agent to do in the
            current MuJoCo domain.
        seed (int): Random seed for the environment.
        pre_transform_inage_size (int): The height and width of the output 
            image from the environment.
        action_repeat (int): Action repeat of gym environment.
        env_id (str): The environment id generated form domain_name, task_name
            and seed.
        discount (float): Discount to use for the environment.
        max_episode_steps (int): If None the max_episode_steps will be set to the
            default step limit defined in the environment's spec. No limit is applied
            if set to 0 or if there is no max_episode_steps set in the environment's
            spec.
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
        from_pixels=True,
        episode_length=max_episode_steps,
        height=pre_transform_image_size,
        width=pre_transform_image_size,
        frame_skip=action_repeat)
    return wrap_env(
        gym_env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        image_channel_first=image_channel_first)


@alf.configurable
def wrap_env(gym_env,
             env_id=None,
             discount=1.0,
             max_episode_steps=0,
             gym_env_wrappers=(),
             time_limit_wrapper=alf_wrappers.TimeLimit,
             normalize_action=True,
             clip_action=True,
             alf_env_wrappers=(),
             image_channel_first=True,
             auto_reset=True):
    """Wraps given gym environment with AlfGymWrapper.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Also note that all gym wrappers assume images are 'channel_last' by default,
    while PyTorch only supports 'channel_first' image inputs. To enable this
    transpose, 'image_channel_first' is set as True by default. ``gym_wrappers.ImageChannelFirst``
    is applied after all gym_env_wrappers and before the AlfGymWrapper.

    Args:
        gym_env (gym.Env): An instance of OpenAI gym environment.
        env_id (int): (optional) ID of the environment.
        discount (float): Discount to use for the environment.
        max_episode_steps (int): Used to create a TimeLimitWrapper. No limit is applied
            if set to 0. Usually set to `gym_spec.max_episode_steps` as done in `load.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers,
            classes to use directly on the gym environment.
        time_limit_wrapper (AlfEnvironmentBaseWrapper): Wrapper that accepts
            (env, max_episode_steps) params to enforce a TimeLimit. Usually this
            should be left as the default, alf_wrappers.TimeLimit.
        normalize_action (bool): if True, will scale continuous actions to
            ``[-1, 1]`` to be better used by algorithms that compute entropies.
        clip_action (bool): If True, will clip continuous action to its bound specified
            by ``action_spec``. If ``normalize_action`` is also ``True``, this
            clipping happens after the normalization (i.e., clips to ``[-1, 1]``).
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment.
        image_channel_first (bool): whether transpose image channels to first dimension.
            PyTorch only supports channgel_first image inputs.
        auto_reset (bool): If True (default), reset the environment automatically after a
            terminal state is reached.

    Returns:
        An AlfEnvironment instance.
    """

    for wrapper in gym_env_wrappers:
        gym_env = wrapper(gym_env)

    # To apply channel_first transpose on gym (py) env
    if image_channel_first:
        gym_env = gym_wrappers.ImageChannelFirst(gym_env)

    if normalize_action:
        # normalize continuous actions to [-1, 1]
        gym_env = gym_wrappers.NormalizedAction(gym_env)

    if clip_action:
        # clip continuous actions according to gym_env.action_space
        gym_env = gym_wrappers.ContinuousActionClip(gym_env)

    env = alf_gym_wrapper.AlfGymWrapper(
        gym_env=gym_env,
        env_id=env_id,
        discount=discount,
        auto_reset=auto_reset,
    )

    if max_episode_steps > 0:
        env = time_limit_wrapper(env, max_episode_steps)

    for wrapper in alf_env_wrappers:
        env = wrapper(env)

    return env
