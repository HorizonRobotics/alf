# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
"""Wrappers for torch environments.

Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/wrappers.py
"""

import abc
import cProfile
import gin
import numpy as np
import random
import six
import torch

from alf.data_structures import StepType, TimeStep
from alf.environments import torch_environment
import alf.nest as nest
import alf.tensor_specs as ts


class TorchEnvironmentBaseWrapper(torch_environment.TorchEnvironment):
    """TorchEnvironment wrapper forwards calls to the given environment."""

    def __init__(self, env):
        """Create a torch environment base wrapper.

        Args:
            env (TorchEnvironment): A TorchEnvironment instance to wrap.

        Returns:
            A wrapped TorchEnvironment
        """
        super(TorchEnvironmentBaseWrapper, self).__init__()
        self._env = env

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self._env, name)

    @property
    def batched(self):
        return getattr(self._env, 'batched', False)

    @property
    def batch_size(self):
        return getattr(self._env, 'batch_size', None)

    def _reset(self):
        return self._env.reset()

    def _step(self, action):
        return self._env.step(action)

    def get_info(self):
        return self._env.get_info()

    def time_step_spec(self):
        return self._env.time_step_spec()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def close(self):
        return self._env.close()

    def render(self, mode='rgb_array'):
        return self._env.render(mode)

    def seed(self, seed):
        return self._env.seed(seed)

    def wrapped_env(self):
        return self._env


# Used in ALF
@gin.configurable
class TimeLimit(TorchEnvironmentBaseWrapper):
    """End episodes after specified number of steps."""

    def __init__(self, env, duration):
        """Create a TimeLimit torch environment.

        Args:
            env (TorchEnvironment): An TorchEnvironment instance to wrap.
            duration (int): time limit, usually set to be the max_eposode_steps
                of the environment.
        """
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._num_steps = None

    def _reset(self):
        self._num_steps = 0
        return self._env.reset()

    def _step(self, action):
        if self._num_steps is None:
            return self.reset()

        time_step = self._env.step(action)

        self._num_steps += 1
        if self._num_steps >= self._duration:
            time_step = time_step._replace(step_type=StepType.LAST)

        if time_step.is_last():
            self._num_steps = None

        return time_step

    @property
    def duration(self):
        return self._duration


def _spec_channel_transpose(spec):
    """Transpose the third (channel) dimension of the spec to the first.
    """
    assert isinstance(spec, ts.TensorSpec)
    if len(spec.shape) == 3:
        if spec.is_bounded():
            return ts.BoundedTensorSpec(spec.shape[-1:] + spec.shape[:-1],
                                        spec.dtype, spec.minimum, spec.maximum)
        else:
            return ts.TensorSpec(spec.shape[-1:] + spec.shape[:-1])
    return spec


def _observation_channel_transpose(time_step):
    """Transpose the third (channel) dimension of observation to the first.
    """

    def _channel_transpose(tensor):
        if tensor.dim() == 3:
            return tensor.transpose(0, 2)
        return tensor

    observation = nest.map_structure(_channel_transpose, time_step.observation)
    return time_step._replace(observation=observation)


@gin.configurable
class ImageChannelFirst(TorchEnvironmentBaseWrapper):
    """Make images in observations channel_first. """

    def __init__(self, env):
        """Create a TimeLimit torch environment.

        Args:
            env (TorchEnvironment): An TorchEnvironment instance to wrap.
        """
        super(ImageChannelFirst, self).__init__(env)

    def time_step_spec(self):
        time_step_spec = self._env.time_step_spec()
        return time_step_spec._replace(observation=self.observation_spec())

    def observation_spec(self):
        return nest.map_structure(_spec_channel_transpose,
                                  self._env.observation_spec())

    def _reset(self):
        return _observation_channel_transpose(self._env.reset())

    def _step(self, action):
        return _observation_channel_transpose(self._env.step(action))


@gin.configurable
class PerformanceProfiler(TorchEnvironmentBaseWrapper):
    """End episodes after specified number of steps."""

    def __init__(self, env, process_profile_fn, process_steps):
        """Create a PerformanceProfiler that uses cProfile to profile env execution.

        Args:
            env (TorchEnvironment): A TorchEnvironment instance to wrap.
            process_profile_fn (Callable): A callback that accepts a `Profile` object.
                After `process_profile_fn` is called, profile information is reset.
            process_steps (int): The frequency with which `process_profile_fn` is
                called.  The counter is incremented each time `step` is called
                (not `reset`); every `process_steps` steps, `process_profile_fn`
                is called and the profiler is reset.
        """
        super(PerformanceProfiler, self).__init__(env)
        self._started = False
        self._num_steps = 0
        self._process_steps = process_steps
        self._process_profile_fn = process_profile_fn
        self._profile = cProfile.Profile()

    def _reset(self):
        self._profile.enable()
        try:
            return self._env.reset()
        finally:
            self._profile.disable()

    def _step(self, action):
        if not self._started:
            self._started = True
            self._num_steps += 1
            return self.reset()

        self._profile.enable()
        try:
            time_step = self._env.step(action)
        finally:
            self._profile.disable()

        self._num_steps += 1
        if self._num_steps >= self._process_steps:
            self._process_profile_fn(self._profile)
            self._profile = cProfile.Profile()
            self._num_steps = 0

        if time_step.is_last():
            self._started = False

        return time_step

    @property
    def duration(self):
        return self._duration


# TODO: trajectory is not a data structure in alf.
@six.add_metaclass(abc.ABCMeta)
class GoalReplayEnvWrapper(TorchEnvironmentBaseWrapper):
    """Adds a goal to the observation, used for HER (Hindsight Experience Replay).

    Sources:
        [1] Hindsight Experience Replay. https://arxiv.org/abs/1707.01495.

    To use this wrapper, create an environment-specific version by inheriting this
    class.
    """

    def __init__(self, env):
        """Create a wrapper to add a goal to the observation.

        Args:
            env (TorchEnvironment): A TorchEnvironment isinstance to wrap.

        Raises:
            ValueError: If environment observation is not a dict
        """
        super(GoalReplayEnvWrapper, self).__init__(env)
        self._env = env
        self._goal = None

    @abc.abstractmethod
    def get_trajectory_with_goal(self, trajectory, goal):
        """Generates a new trajectory assuming the given goal was the actual target.

        One example is updating a "distance-to-goal" field in the observation. Note
        that relevant state information must be recovered or re-calculated from the
        given trajectory.

        Args:
            trajectory: An instance of `Trajectory`.
            goal: Environment specific goal

        Returns:
            Updated instance of `Trajectory`

        Raises:
            NotImplementedError: function should be implemented in child class.
        """
        pass

    @abc.abstractmethod
    def get_goal_from_trajectory(self, trajectory):
        """Extracts the goal from a given trajectory.

        Args:
            trajectory: An instance of `Trajectory`.

        Returns:
            Environment specific goal

        Raises:
            NotImplementedError: function should be implemented in child class.
        """
        pass

    def _reset(self, *args, **kwargs):
        """Resets the environment, updating the trajectory with goal."""
        trajectory = self._env.reset(*args, **kwargs)
        self._goal = self.get_goal_from_trajectory(trajectory)
        return self.get_trajectory_with_goal(trajectory, self._goal)

    def _step(self, *args, **kwargs):
        """Execute a step in the environment, updating the trajectory with goal."""
        trajectory = self._env.step(*args, **kwargs)
        return self.get_trajectory_with_goal(trajectory, self._goal)


# Used in ALF
@gin.configurable
class NonEpisodicAgent(TorchEnvironmentBaseWrapper):
    """
    Make the agent non-episodic by replacing all termination time steps with
    a non-zero discount (essentially the same type as returned by the TimeLimit
    wrapper).

    This wrapper could be useful for pure intrinsic-motivated agent, as
    suggested in the following paper:

        EXPLORATION BY RANDOM NETWORK DISTILLATION, Burda et al. 2019,

    "... We argue that this is a natural way to do exploration in simulated
    environments, since the agent’s intrinsic return should be related to all
    the novel states that it could find in the future, regardless of whether
    they all occur in one episode or are spread over several.

    ... If Alice is modelled as an episodic reinforcement learning agent, then
    her future return will be exactly zero if she gets a game over, which might
    make her overly risk averse. The real cost of a game over to Alice is the
    opportunity cost incurred by having to play through the game from the
    beginning."

    NOTE: For PURE intrinsic-motivated agents only. If you use both extrinsic
    and intrinsic rewards, then DO NOT use this wrapper! Because without
    episodic setting, the agent could exploit extrinsic rewards by intentionally
    die to get easy early rewards in the game.

    Example usage:
        suite_mario.load.env_wrappers=(@NonEpisodicAgent, )
        suite_gym.load.env_wrappers=(@NonEpisodicAgent, )
    """

    def __init__(self, env, discount=1.0):
        """Create a NonEpisodicAgent wrapper.

        Args:
            env (TorchEnvironment): A TorchEnvironment instance to wrap.
            discount (float): discount of the environment.
        """
        super().__init__(env)
        self._discount = discount

    def _step(self, action):
        time_step = self._env.step(action)
        if time_step.step_type == StepType.LAST:
            # We set a non-zero discount so that the target value would not be
            # zero (non-episodic).
            time_step = time_step._replace(
                discount=torch.tensor(self._discount, torch.float32))
        return time_step


# Used in ALF
@gin.configurable
class RandomFirstEpisodeLength(TorchEnvironmentBaseWrapper):
    """Randomize the length of the first episode.

    The motivation is to make the observations less correlated for the
    environments that have fixed episode length.

    Example usage:
        RandomFirstEpisodeLength.random_length_range=200
        suite_gym.load.env_wrappers=(@RandomFirstEpisodeLength, )
    """

    def __init__(self, env, random_length_range, num_episodes=1):
        """Create a RandomFirstEpisodeLength wrapper.

        Args:
            env (TorchEnvironment): A TorchEnvironment isinstance to wrap.
            random_length_range (int): [1, random_length_range]
            num_episodes (int): randomize the episode length for the first so
                many episodes.
        """
        super().__init__(env)
        self._random_length_range = random_length_range
        self._num_episodes = num_episodes
        self._episode = 0
        self._num_steps = 0
        self._max_length = random.randint(1, self._random_length_range)

    def _reset(self):
        self._num_steps = 0
        return self._env.reset()

    def _step(self, action):
        if self._num_steps is None:
            return self.reset()

        time_step = self._env.step(action)

        self._num_steps += 1
        if (self._episode < self._num_episodes
                and self._num_steps >= self._max_length):
            time_step = time_step._replace(step_type=StepType.LAST)
            self._max_length = random.randint(1, self._random_length_range)
            self._episode += 1

        if time_step.is_last():
            self._num_steps = None

        return time_step
