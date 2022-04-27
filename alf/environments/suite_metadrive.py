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

from typing import Tuple, Optional, Union

import numpy as np
import gym
import torch

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import alf
from alf.environments.alf_environment import AlfEnvironment
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
import alf.data_structures as ds
import alf.nest as nest

from alf.environments.metadrive import VectorizedTopDownEnv, BirdEyeTopDownEnv

try:
    import metadrive
    import pygame
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

    def __init__(self, metadrive_env: metadrive.MetaDriveEnv, env_id: int = 0):
        """Constructor of AlfMetaDriveWrapper.
        Args:
            metadrive_env: the original meta drive environment being wrapped.
                The meta drive environment should be properly configured on its
                own before being wrapped.
            env_id: the ID of this environment when appear as part of a batched
                environment.

        """

        self._env = metadrive_env
        self._env_id = env_id

        self._observation_spec = self._env.observation_spec

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

        # Support video recording
        self.metadata = {'render.modes': ['rgb_array']}

        self._current_observation = None

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
        # The type of frame is pygame.Surface of 1000 x 1000
        frame = self._env.render(self._current_observation)

        if mode != 'rgb_array':
            return None

        # Now canvas is a numpy H x W x C image (ndarray)
        return pygame.surfarray.array3d(frame).swapaxes(0, 1)

    def _acquire_next_frame(self, action):
        """Returns the TimeStep given the input action.

        This is the underlying implementation of both _step() and _reset()
        1. In _step(), normally it just delegates to this method unless a reset
           needs to be performed.
        2. In _reset(), it just delegates to this method after the wrapped
           environment is reset. The action for _reset() is a simple all-zero
           action, which makes sense for driving environments.
        """
        observation, reward, done, info = self._env.step(action)

        self._current_observation = observation

        discount = 0.0 if done else 1.0

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

        # Zero actin means do nothing in both longitudinal and lateral
        first_time_step = self._acquire_next_frame(
            self._action_spec.zeros().cpu().numpy())

        return first_time_step._replace(step_type=ds.StepType.FIRST)

    def seed(self, seed: Optional[int] = None):
        """Reset the underlying MetaDrive environment with a specified seed.

        MetaDrive uses a slightly different mechanism for seeds. Upon
        construction of a MetaDrive environment, the user needs to
        specify a seed range [start_seed, start_seed + scenario_num].
        When being forced to reset with a specific seed, that seed
        must be within the predefined range.

        Args:
            seed: the seed that the environment will be reset with. If it is
                specified as None, a random seed within the range will be
                selected by the underlying MetaDrive environment.
        """
        if seed is not None:
            # Ensure the seed is within the range
            scenario_num = self._env.config['environment_num']
            start_seed = self._env.config['start_seed']
            seed = seed % scenario_num + start_seed
        self._env.reset(force_seed=seed)

    def close(self):
        self._env.close()


@alf.configurable
def load(
        env_name: str = 'Vectorized',
        env_id: int = 0,
        traffic_density: float = 0.1,
        start_seed: int = np.random.randint(10000),
        scenario_num: int = 5000,
        decision_repeat: int = 5,  # 0.02 * 5 = 0.1 seconds per action
        map_spec: Union[int, str] = 4,
        crash_penalty: float = 5.0,
        speed_reward_weight: float = 0.1,
        success_reward: float = 10.0):
    """Load the MetaDrive environment and wraps it with AlfMetaDriveWrapper.
    Args:

        env_name: Used to specify whether the environment produces observation
            in vectorized form or raster (Bird Eye View) form. The user is only
            allowed to specify "Vectorized" or "BirdEye".
        env_id: (optional) ID of the environment.
        traffic_density: number of traffic vehicles per 10 meter per lane.
        start_seed: random seed of the first map.
        scenario_num: specifies the range of the scenario seeds together with
            ``start_seed``. When being reset, a seed will be picked randomly
            from [start_seed, start_seed + scenario_num]. Note that even with
            the same seed, the generated map can vary as there are other
            randomness such as "random lane number".
        decision_repeat: how many times for the simulation engine to repeat the
            applied action to the vehicles. The minimal simulation interval
            physics_world_step_size is 0.02 s. Therefore each RL step will last
            decision_repeat * 0.02 s in the simulation world.
        map_spec: User can set a string or int as the key to generate map in an
            easy way. For example, config["map"] = 3 means generating a map
            containing 3 blocks, while config["map"] = "SCrRX" means the first
            block is Straight, and the following blocks are Circular, InRamp,
            OutRamp and Intersection. The character here are the unique ID of
            different types of blocks as shown in the next table. Therefore
            using a string can determine the block type sequence. Detailed list
            of block types can be found at
            https://metadrive-simulator.readthedocs.io/en/latest/config_system.html
        crash_penalty: the immediate penalty when the car hits the road
            boundary, cars or other objects. It should be a positive number.
        speed_reward_weight: at each step, the incentive reward for being at a
            high speed is this weight * the speed in km/h.
        success_reward: the amount of reward will be given (at most 1 time per
            episode) when the ego car reaches the destination.

    """
    assert env_name in [
        'BirdEye', 'Vectorized'
    ], (f'"{env_name}" is not a valid ALF MetaDrive env name')

    env_ctor = {
        'Vectorized': VectorizedTopDownEnv,
        'BirdEye': BirdEyeTopDownEnv,
    }[env_name]

    env = env_ctor(
        config={
            # This means that the environment is not required to
            # render in 3D photo-realistic mode.
            'use_render': False,
            'traffic_density': traffic_density,
            'environment_num': scenario_num,
            'random_agent_model': False,
            'random_lane_width': False,
            'random_lane_num': True,
            'map': map_spec,
            'decision_repeat': decision_repeat,
            'start_seed': start_seed,
            # Reward
            'out_of_road_penalty': crash_penalty,
            'crash_vehicle_penalty': crash_penalty,
            'crash_object_penalty': crash_penalty,
            'speed_reward': speed_reward_weight,
            'success_reward': success_reward,
        })

    return AlfMetaDriveWrapper(env, env_id=env_id)


# Set no_thread_env to True so that when being created for evaluation or play,
# the environment is not wrapped with ThreadEnvironment. MetaDrive requires
# being accessed from the main thread of a process.
load.no_thread_env = True
