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

import numpy as np
import gym
import torch

import alf
from alf.environments.alf_environment import AlfEnvironment
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
import alf.data_structures as ds
import alf.nest as nest

try:
    import metadrive
except ImportError:
    from unittest.mock import Mock
    # create 'metadrive' as a mock to not break python argument type hints
    metadrive = Mock()


def _space_to_spec(space: gym.spaces.box.Box):
    # NOTE: this is meta drive specific conversion function as it
    # assumes low and high are uniform.
    return BoundedTensorSpec(
        shape=space.shape,
        dtype=space.dtype.name,
        minimum=space.low.flat[0],
        maximum=space.high.flat[0])


class AlfMetaDriveWrapper(AlfEnvironment):
    """Wrapper over the MetaDrive autonomous driving environment.
    You will need to have metadrive installed as a dependency to use this.
    """

    def __init__(self,
                 metadrive_env: metadrive.MetaDriveEnv,
                 env_id: int = 0,
                 image_channel_first: bool = True):
        """Constructor of AlfMetaDriveWrapper.
        Args:

            metadrive_env: the original meta drive environment being wrapped.
                The meta drive environment should be properly configured on its
                own before being wrapped.
            env_id: the ID of this environment when appear as part of a batched
                environment.
            image_channel_first: when set to True, the returned image-based
                observation will have a shape of (channel, height, width). Note
                that the raw observation from meta drive environment place
                channel as the last dimension.
        """

        self._env = metadrive_env
        self._env_id = env_id

        self._image_channel_first = image_channel_first
        self._observation_spec = _space_to_spec(self._env.observation_space)

        if self._image_channel_first:
            w, h, c = self._observation_spec.shape
            self._observation_spec = BoundedTensorSpec(
                shape=(c, w, h),
                dtype=self._observation_spec.dtype,
                minimum=self._observation_spec.minimum,
                maximum=self._observation_spec.maximum)

        self._action_spec = _space_to_spec(self._env.action_space)
        self._env_info_spec = {
            # Add "@step" postfix to fields such as ``velocity`` and
            # ``abs_steering`` so that when being reported as metrics they are
            # averaged instead of summed over the episode steps.
            'velocity@step': TensorSpec(shape=(), dtype=torch.float32),
            'abs_steering@step': TensorSpec(shape=(), dtype=torch.float32),
            'reach_goal': TensorSpec(shape=(), dtype=torch.float32),
        }

        # Stateful member indicating whether the last ``step()`` call returns a
        # step that marks the end of an episode. If it is True, the next call to
        # ``step()`` will perform a ``reset()``.
        self._last_step_is_done = True

    @property
    def batched(self):
        # TODO(breakds): Add support for multiple algorithm controlled agents in
        # the future. This environment should be batched in that case.
        return False

    def env_info_spec(self):
        return self._env_info_spec

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode):
        return self._env.render(mode='top_down')

    def _acquire_next_frame(self, action):
        """Returns the TimeStep give the input action.
        This is the underlying implementation of both _step() and _reset()
        1. In _step(), normally it just delegates to this method unless a reset
           needs to be performed.
        2. In _reset(), it just delegates to this method after the wrapped
           environment is reset. The action for _reset() is a simple all-zero
           action, which makes sense for driving environments.
        """
        observation, reward, done, info = self._env.step(action)

        if self._image_channel_first:
            observation = nest.map_structure(lambda x: x.transpose(2, 0, 1),
                                             observation)

        discount = [0.0 if done else 1.0]

        self._last_step_is_done = done

        return ds.TimeStep(
            step_type=ds.StepType.LAST if done else ds.StepType.MID,
            reward=reward,
            discount=discount,
            observation=observation,
            env_id=self._env_id,
            prev_action=action,
            env_info={
                'velocity@step': info['velocity'],
                'abs_steering@step': abs(info['steering']),
                'reach_goal': 1.0 if info['arrive_dest'] else 0.0,
            })

    def _step(self, action) -> ds.TimeStep:
        if self._last_step_is_done:
            return self._reset()

        return self._acquire_next_frame(action)

    def _reset(self) -> ds.TimeStep:
        _ = self._env.reset()

        # Zero action means do nothing in both longitudinal and lateral
        first_time_step = self._acquire_next_frame(
            self._action_spec.zeros().cpu().numpy())

        return first_time_step._replace(step_type=ds.StepType.FIRST)

    def seed(self, seed):
        pass

    def close(self):
        self._env.close()


@alf.configurable
def load(
        map_name,
        env_id: int = 0,
        batch_size: int = 1,
        traffic_density: float = 0.1,
        environment_num: int = 200,
        start_seed: int = np.random.randint(10000),
        decision_repeat: int = 5,  # 0.02 * 5 = 0.1 seconds per action
        map_size: int = 4):
    """The loader for metadrive environment.

    See https://metadrive-simulator.readthedocs.io/en/latest/config_system.html

    for details about the configurations.
    """
    # TODO(breakds): Extend this to support more customization other than the
    # default meta drive environment.
    env = metadrive.TopDownMetaDrive(
        config={
            'use_render': False,
            'traffic_density': traffic_density,
            'environment_num': environment_num,
            'random_agent_model': False,
            'random_lane_width': False,
            'random_lane_num': True,
            'map': map_size,
            'decision_repeat': decision_repeat,
            'start_seed': start_seed,
        })

    return AlfMetaDriveWrapper(env, env_id=env_id)


load.no_thread_env = True
