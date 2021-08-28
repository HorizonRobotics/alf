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

import os
import gym
import numpy as np
import pybullet as p

from alf.environments import suite_gym
from alf.environments.gym_wrappers import BaseObservationWrapper, ImageChannelFirst, FrameGrayScale
import alf

from igibson.objects.visual_marker import VisualMarker
from alf.environments.igibson.igibson_tasks import VisualPointNavFixedTask, \
    VisualPointNavRandomTask, VisualObjectNavTask
from alf.environments.igibson.igibson_env import iGibsonCustomEnv


@alf.configurable()
class FlattenWrapper(BaseObservationWrapper):
    """Flatten selected obversation fields of the environment"""

    def __init__(self, env, fields=None):
        super().__init__(env, fields=fields)

    def transform_space(self, observation_space):
        if isinstance(observation_space, gym.spaces.Box):
            if self._need_flatten(observation_space.shape):
                low = observation_space.low
                high = observation_space.high
                if np.isscalar(low) and np.isscalar(high):
                    shape = observation_space.shape[:-1]
                    return gym.spaces.Box(
                        low=low,
                        high=high,
                        shape=shape,
                        dtype=observation_space.dtype)
                else:
                    low = self._flatten(observation_space.low, flatten=True)
                    high = self._flatten(observation_space.high, flatten=True)
                    return gym.spaces.Box(
                        low=low, high=high, dtype=observation_space.dtype)
        else:
            raise TypeError('Unsupported observation space type')
        return observation_space

    def transform_observation(self, observation):
        flatten = self._need_flatten(observation.shape)
        return self._flatten(observation, flatten)

    def _need_flatten(self, shape):
        if len(shape) == 2:
            return True
        return False

    def _flatten(self, np_array, flatten=False):
        """
        Note that the extra copy() after np.transpose is crucial to pickle dump speed
        when called by subprocesses. An explanation of the non-contiguous memory caused
        by numpy tranpose can be found in the following:

        https://stackoverflow.com/questions/26998223/
        """

        if flatten:
            np_array = np_array.flatten()
        return np_array.copy()


@alf.configurable()
class StackRGBDWrapper(gym.ObservationWrapper):
    """This wrapper stacks rgb and depth observation field together into an array of shape (H, W, 4)

    Note that the observation space must include `rgb` and `depth`, otherwise an error will be raised.
    """

    def __init__(self, env):
        super().__init__(env)
        observation_space = env.observation_space
        self.observation_space = self.transform_space(observation_space)

    def transform_space(self, observation_space):
        assert isinstance(observation_space, gym.spaces.dict.Dict)
        rgb_space = observation_space.spaces['rgb']
        depth_space = observation_space.spaces['depth']
        try:
            assert not np.isscalar(rgb_space.low) and not np.isscalar(
                rgb_space.low)
            assert not np.isscalar(depth_space.low) and not np.isscalar(
                depth_space.low)
        except AssertionError as e:
            print(e)

        rgbd_low = np.vstack((rgb_space.low, depth_space.low))
        rgbd_high = np.vstack((rgb_space.high, depth_space.high))
        assert rgbd_low.shape == rgbd_high.shape
        rgbd_space = gym.spaces.Box(
            low=rgbd_low, high=rgbd_high, dtype=np.uint8)

        observation_space.spaces.pop('rgb')
        observation_space.spaces.pop('depth')
        observation_space.spaces['rgbd'] = rgbd_space
        return observation_space

    def observation(self, observation):
        return self.transform_observation(observation)

    def transform_observation(self, observation):
        rgbd_obs = np.vstack((observation['rgb'], observation['depth']))
        observation.pop('rgb')
        observation.pop('depth')
        observation['rgbd'] = rgbd_obs
        return observation


@alf.configurable()
class OneHotIntWrapper(BaseObservationWrapper):
    """Transforms the data type of one-hot input from float to int"""

    def __init__(self, env, fields=None):
        super().__init__(env, fields=fields)

    def transform_space(self, observation_space):
        obs_low = np.full_like(
            observation_space.low, fill_value=0, dtype=np.uint8)
        obs_high = np.full_like(
            observation_space.high, fill_value=255, dtype=np.uint8)
        assert obs_low.shape == obs_high.shape
        observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.uint8)
        return observation_space

    def transform_observation(self, observation):
        return (observation * 255).astype(np.uint8)


@alf.configurable()
class ImageIntWrapper(BaseObservationWrapper):
    """Transforms the data type of image input from float to int"""

    def __init__(self, env, fields=None):
        super().__init__(env, fields=fields)

    def transform_space(self, observation_space):
        obs_low = np.full_like(
            observation_space.low, fill_value=0, dtype=np.uint8)
        obs_high = np.full_like(
            observation_space.high, fill_value=255, dtype=np.uint8)
        assert obs_low.shape == obs_high.shape
        observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.uint8)
        return observation_space

    def transform_observation(self, observation):
        return (observation * 255).astype(np.uint8)


@alf.configurable()
class RenderWrapper(gym.Wrapper):
    """Implements render() for iGibson env for visualization purposes"""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.metadata['render.modes'] = ['rgb_array', 'human']

    def render(self, mode='human', **kwargs):
        if 'vision' in self.env.sensors:
            vision_obs = self.sensors['vision'].get_obs(self)
            if 'rgb' in vision_obs:
                rgb_obs = vision_obs['rgb']
                rgb_obs = (rgb_obs * 255).astype(np.uint8)
                if 'depth' not in vision_obs:
                    return rgb_obs
                else:
                    depth_obs = vision_obs['depth']
                    depth_obs = (depth_obs * 255).astype(np.uint8)
                    depth_obs_rescaled = np.tile(depth_obs, (1, 1, 3))
                    rgbd_obs = np.concatenate((rgb_obs, depth_obs_rescaled),
                                              axis=1)
                    return rgbd_obs
            else:
                raise AttributeError("robot doesn't have modality: rgb")
        else:
            raise AttributeError("robot doesn't have sensor: vision")


@alf.configurable()
class GFTTupleInputWrapper(gym.ObservationWrapper):
    """Transforms observation space into tuple for GFT"""

    def __init__(self, env):
        super().__init__(env)
        assert 'rgbd' in self.observation_space.spaces \
               and 'task_obs' in self.observation_space.spaces
        self.observation_space = self.transform_space(self.observation_space)

    def transform_space(self, observation_space):
        assert isinstance(observation_space, gym.spaces.dict.Dict)
        rgbd_space = observation_space.spaces['rgbd']
        task_obs_space = observation_space.spaces['task_obs']
        tuple_space = gym.spaces.Tuple([rgbd_space, task_obs_space])

        observation_space.spaces.pop('task_obs')
        observation_space.spaces['obs'] = tuple_space
        return observation_space

    def observation(self, observation):
        return self.transform_observation(observation)

    def transform_observation(self, observation):
        rgbd_obs = observation['rgbd']
        task_obs = observation['task_obs']
        observation.pop('task_obs')
        observation['obs'] = (rgbd_obs, task_obs)
        return observation


@alf.configurable()
class PyBulletRecorderWrapper(gym.Wrapper):
    """Enables pybullet video logging"""

    def __init__(self, env, record_file):
        super().__init__(env)
        self.p_logging = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                             record_file)

    def close(self):
        p.stopStateLogging(self.p_logging)
        super().close()


@alf.configurable
def load(env_name=None,
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         config_file=None,
         scene_id=None,
         env_mode='headless',
         task=None,
         target_pos=[5, 5, 0],
         fov=45,
         action_timestep=1.0 / 10.0,
         physics_timestep=1.0 / 120.0,
         device_idx=0,
         record_pybullet=False,
         pybullet_record_file=None,
         use_gray_scale=False,
         use_gft=False,
         gym_env_wrappers=(),
         alf_env_wrappers=()):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a ``TimeLimit`` wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.
    Args:
        env_name (str): Name for the environment to load.
        env_id (int): A scalar ``Tensor`` of the environment ID of the time step.
        discount (float): This argument is not used; discount is defined in the
            iGibson config file.
        max_episode_steps: If None or 0 the ``max_episode_steps`` will be set to
            the default step limit -1 defined in the environment. Otherwise
            ``max_episode_steps`` will be set to the smaller value of the two.
        config_file (str): Name for the iGibson config file. Only the file name
            is needed, not the full path.
        scene_id (str): Overrides scene_id in config file.
        env_mode (str): Three options for rendering: headless, gui, iggui.
        task (str): A string that specifies the iGibson task.
        target_pos (list[float]): [x, y, z] used for "visual_point_nav_fixed".
        fov (int): Field of view of the robot in degrees.
        num_objects: (int) number of objects in the environment, for object_nav
            tasks only
        action_timestep (float): Environment executes action per action_timestep
            second
        physics_timestep (float): Physics timestep for pybullet
        device_idx (int): Which GPU to run the simulation and rendering on
        record_pybullet (bool): Whether to record pybullet rendering video
        pybullet_record_file (str): Video record file for storing pybullet recording
        use_gray_scale (bool): Whether to use alf.environments.gym_wrappers.FrameGrayScale
            to convert RGB to grey scale
        use_gft (bool): Whether to use alf.layers.GFT in conf file
        render_to_tensor (bool): Whether to render directly to pytorch tensors
        automatic_reset (bool): Whether to automatic reset after an episode finishes
        gym_env_wrappers (gym.wrappers): Iterable with references to wrapper classes to use
            directly on the gym environment.
        alf_env_wrappers (alf.environments.gym_wrappers): Iterable with references to wrapper classes to use on
            the torch environment.
    Returns:
        An AlfEnvironment instance.
    """
    env = iGibsonCustomEnv(
        config_file=config_file,
        scene_id=scene_id,
        mode=env_mode,
        action_timestep=action_timestep,
        physics_timestep=physics_timestep,
        device_idx=device_idx)

    # object params (for point navigation tasks only)
    visual_shape = p.GEOM_CYLINDER
    cyl_length = 1.5
    cyl_radius = 0.2
    rgba_color = [0, 0, 1, 1.0]
    initial_offset = [0, 0, cyl_length / 2.0]
    vis_obj = VisualMarker(
        visual_shape=visual_shape,
        rgba_color=rgba_color,
        radius=cyl_radius,
        length=cyl_length,
        initial_offset=initial_offset)
    if task == 'visual_point_nav_fixed':
        env_task = VisualPointNavFixedTask(
            env=env, target_pos=target_pos, target_pos_vis_obj=vis_obj)
        env.task = env_task
    elif task == 'visual_point_nav_random':
        env_task = VisualPointNavRandomTask(
            env=env, target_pos_vis_obj=vis_obj)
        env.task = env_task
    elif task == 'visual_object_nav':
        pass
    else:
        raise ValueError(f'Unrecoganized task: {task}')
    env.simulator.renderer.set_fov(fov)

    discount = env.config.get('discount_factor')
    max_episode_steps = env.config.get('max_step') - 1

    # apply wrappers
    sensors = env.observation_space.spaces.keys()
    image_int_wrapper_fields = [
        field for field in ['rgb', 'depth'] if field in sensors
    ]
    one_hot_int_wrapper_fields = [
        field for field in ['task_obs'] if field in sensors
    ]
    change_channel_fields = [
        field for field in ['rgb', 'depth'] if field in sensors
    ]
    flatten_fields = [field for field in ['scan'] if field in sensors]
    stack_rgbd = 'rgb' in sensors and 'depth' in sensors
    apply_render_wrapper = 'rgb' in sensors

    if image_int_wrapper_fields:
        env = ImageIntWrapper(env, fields=image_int_wrapper_fields)
    if one_hot_int_wrapper_fields:
        env = OneHotIntWrapper(env, fields=one_hot_int_wrapper_fields)
    if use_gray_scale:
        assert 'rgb' in sensors
        env = FrameGrayScale(env, fields=['rgb'])
    if change_channel_fields:
        env = ImageChannelFirst(env, fields=change_channel_fields)
    if stack_rgbd:
        env = StackRGBDWrapper(env)
    if flatten_fields:
        env = FlattenWrapper(env, fields=flatten_fields)
    if use_gft:
        env = GFTTupleInputWrapper(env)
    if apply_render_wrapper:
        env = RenderWrapper(env)
    if record_pybullet:
        assert pybullet_record_file is not None
        env = PyBulletRecorderWrapper(env, pybullet_record_file)

    return suite_gym.wrap_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        # instead of using suite_gym's wrapper, for which we cannot define fields, we apply the wrapper within load()
        image_channel_first=False)
