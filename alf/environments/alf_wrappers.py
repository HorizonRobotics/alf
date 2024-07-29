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
"""Wrappers for ALF environments.

Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/wrappers.py
"""

import abc
from collections import OrderedDict
import copy
import cProfile
import math
import numpy as np
import random
import six
from typing import List

import torch
import torch.nn.functional as F

import alf
from alf.data_structures import StepType, TimeStep, _is_numpy_array
from alf.environments.alf_environment import AlfEnvironment
from alf.environments.parallel_environment import ParallelAlfEnvironment
import alf.nest as nest
import alf.tensor_specs as ts
from alf.utils import spec_utils
from alf.utils.tensor_utils import to_tensor


class AlfEnvironmentBaseWrapper(AlfEnvironment):
    """AlfEnvironment wrapper forwards calls to the given environment."""

    def __init__(self, env):
        """Create an ALF environment base wrapper.

        Args:
            env (AlfEnvironment): An AlfEnvironment instance to wrap.

        Returns:
            A wrapped AlfEnvironment
        """
        super(AlfEnvironmentBaseWrapper, self).__init__()
        self._env = env

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self._env, name)

    @property
    def is_tensor_based(self):
        return self._env.is_tensor_based

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
        return self._env.reset()

    def _step(self, action):
        return self._env.step(action)

    def get_info(self):
        return self._env.get_info()

    def env_info_spec(self):
        return self._env.env_info_spec()

    def time_step_spec(self):
        # By default, returns a timestep spec that respect the overrides of
        # observation, action and rewards spec.
        return self._env.time_step_spec()._replace(
            observation=self.observation_spec(),
            prev_action=self.action_spec(),
            reward=self.reward_spec(),
        )

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

    def sync_progress(self):
        return self._env.sync_progress()


# Used in ALF
@alf.configurable
class TimeLimit(AlfEnvironmentBaseWrapper):
    """End episodes after specified number of steps."""

    def __init__(self, env, duration):
        """Create a TimeLimit ALF environment.

        Args:
            env (AlfEnvironment): An AlfEnvironment instance to wrap.
            duration (int): time limit, usually set to be the max_eposode_steps
                of the environment.
        """
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._num_steps = None
        assert self.batch_size is None or self.batch_size == 1, (
            "does not support batched environment with batch size larger than one"
        )

    def _reset(self):
        self._num_steps = 0
        return self._env.reset()

    def _step(self, action):
        if self._num_steps is None:
            return self.reset()

        time_step = self._env.step(action)

        self._num_steps += 1
        if self._num_steps >= self._duration:
            if _is_numpy_array(time_step.step_type):
                time_step = time_step._replace(step_type=StepType.LAST)
            else:
                time_step = time_step._replace(
                    step_type=torch.full_like(time_step.step_type, StepType.
                                              LAST))

        if time_step.is_last():
            self._num_steps = None

        return time_step

    @property
    def duration(self):
        return self._duration


@alf.configurable
class PerformanceProfiler(AlfEnvironmentBaseWrapper):
    """Use cProfile to profile env execution."""

    def __init__(self, env, process_profile_fn, process_steps):
        """Create a PerformanceProfiler that uses cProfile to profile env execution.

        Args:
            env (AlfEnvironment): An AlfEnvironment instance to wrap.
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
class GoalReplayEnvWrapper(AlfEnvironmentBaseWrapper):
    """Adds a goal to the observation, used for HER (Hindsight Experience Replay).

    Sources:
        [1] Hindsight Experience Replay. https://arxiv.org/abs/1707.01495.

    To use this wrapper, create an environment-specific version by inheriting this
    class.
    """

    def __init__(self, env):
        """Create a wrapper to add a goal to the observation.

        Args:
            env (AlfEnvironment): An AlfEnvironment isinstance to wrap.

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
@alf.configurable
class NonEpisodicAgent(AlfEnvironmentBaseWrapper):
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

    This wrapper could also be used for approximating the average reward objective
    in an episodic setting, even if average reward always assumes an infinite
    horizon. By setting the last discount=1, the value learning essentially
    continues although an episode finishes.

    Example usage:
        suite_mario.load.env_wrappers=(@NonEpisodicAgent, )
        suite_gym.load.env_wrappers=(@NonEpisodicAgent, )
    """

    def __init__(self, env, discount=1.0):
        """Create a NonEpisodicAgent wrapper.

        Args:
            env (AlfEnvironment): An AlfEnvironment instance to wrap.
            discount (float): discount of the environment.
        """
        super().__init__(env)
        self._discount = discount

    def _step(self, action):
        time_step = self._env.step(action)
        if time_step.is_last():
            # We set a non-zero discount so that the target value would not be
            # zero (non-episodic).
            time_step = time_step._replace(discount=np.float32(self._discount))
        return time_step


# Used in ALF
@alf.configurable
class RandomFirstEpisodeLength(AlfEnvironmentBaseWrapper):
    """Randomize the length of the first episode.

    The motivation is to make the observations less correlated for the
    environments that have fixed episode length.

    Example usage:
        RandomFirstEpisodeLength.random_length_range=200
        suite_gym.load.alf_env_wrappers=(@RandomFirstEpisodeLength, )
    """

    def __init__(self, env, random_length_range, num_episodes=1):
        """Create a RandomFirstEpisodeLength wrapper.

        Args:
            env (AlfEnvironment): An AlfEnvironment isinstance to wrap.
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


@alf.configurable(whitelist=[])
class ActionObservationWrapper(AlfEnvironmentBaseWrapper):
    def __init__(self, env):
        """Add prev_action to observation.

        The new observation is:

        .. code-block:: python

            {
                'observation': original_observation,
                'prev_action': prev_action
            }

        Args:
            env (AlfEnvironment): An AlfEnvironment isinstance to wrap.
        """
        super().__init__(env)
        self._time_step_spec = self._add_action(env.time_step_spec())
        self._observation_spec = self._time_step_spec.observation

    def observation_spec(self):
        return self._observation_spec

    def time_step_spec(self):
        return self._time_step_spec

    def _reset(self):
        return self._add_action(self._env.reset())

    def _step(self, action):
        return self._add_action(self._env.step(action))

    def _add_action(self, time_step):
        return time_step._replace(
            observation=dict(
                observation=time_step.observation,
                prev_action=time_step.prev_action))


@alf.configurable
class ScalarRewardWrapper(AlfEnvironmentBaseWrapper):
    """A wrapper that converts a vector reward to a scalar reward by averaging
    reward dims with a weight vector."""

    def __init__(self, env, reward_weights=None):
        """
        Args:
            env (AlfEnvironment): An AlfEnvironment instance to be wrapped.
            reward_weights (list[float] | tuple[float]): a list/tuple of weights
                for the rewards; if None, then the first dimension will be 1 and
                the other dimensions will be 0s.
        """
        super(ScalarRewardWrapper, self).__init__(env)
        reward_spec = env.reward_spec()
        assert reward_spec.ndim == 1, (
            "This wrapper only supports vector rewards! Reward tensor rank: %d"
            % reward_spec.ndim)

        rewards_n = reward_spec.shape[0]
        if reward_weights is None:
            reward_weights = [1.] + [0.] * (rewards_n - 1)
        assert (isinstance(reward_weights, (list, tuple))
                and len(reward_weights) == rewards_n)
        self._np_reward_weights = np.array(reward_weights)
        self._tensor_reward_weights = to_tensor(reward_weights)

    def _average_rewards(self, time_step):
        if _is_numpy_array(time_step.reward):
            reward = np.tensordot(
                time_step.reward, self._np_reward_weights, axes=1)
        else:
            reward = torch.tensordot(
                time_step.reward, self._tensor_reward_weights, dims=1)
        return time_step._replace(reward=reward)

    def _step(self, action):
        time_step = self._env._step(action)
        return self._average_rewards(time_step)

    def _reset(self):
        time_step = self._env._reset()
        return self._average_rewards(time_step)

    def reward_spec(self):
        return ts.TensorSpec(())

    def time_step_spec(self):
        spec = self._env.time_step_spec()
        return spec._replace(reward=self.reward_spec())


class MultitaskWrapper(AlfEnvironment):
    """Multitask environment based on a list of environments.

    All the environments need to have same observation_spec, action_spec, reward_spec
    and info_spec. The action_spec of the new environment becomes:

    .. code-block:: python

        {
            'task_id': TensorSpec((), maximum=num_envs - 1, dtype='int64'),
            'action': original_action_spec
        }

    'task_id' is used to specify which task to run for the current step. Note
    that current implementation does not prevent switching task in the middle of
    one episode.
    """

    def __init__(self, envs, task_names, env_id=None):
        """
        Args:
            envs (list[AlfEnvironment]): a list of environments. Each one
                represents a different task.
            task_names (list[str]): the names of each task.
            env_id (int): (optional) ID of the environment.
        """
        assert len(envs) > 0, "`envs should not be empty"
        assert len(set(task_names)) == len(task_names), (
            "task_names should "
            "not contain duplicated names: %s" % str(task_names))
        self._envs = envs
        self._observation_spec = envs[0].observation_spec()
        self._action_spec = envs[0].action_spec()
        self._reward_spec = envs[0].reward_spec()
        self._env_info_spec = envs[0].env_info_spec()
        self._task_names = task_names
        if env_id is None:
            env_id = 0
        self._env_id = np.int32(env_id)

        def _nested_eq(a, b):
            return all(
                alf.nest.flatten(
                    alf.nest.map_structure(lambda x, y: x == y, a, b)))

        for env in envs:
            assert _nested_eq(
                env.observation_spec(), self._observation_spec), (
                    "All environment should have same observation spec. "
                    "Got %s vs %s" % (self._observation_spec,
                                      env.observation_spec()))
            assert _nested_eq(env.action_spec(), self._action_spec), (
                "All environment should have same action spec. "
                "Got %s vs %s" % (self._action_spec, env.action_spec()))
            assert _nested_eq(env.reward_spec(), self._reward_spec), (
                "All environment should have same reward spec. "
                "Got %s vs %s" % (self._reward_spec, env.reward_spec()))
            assert _nested_eq(env.env_info_spec(), self._env_info_spec), (
                "All environment should have same env_info spec. "
                "Got %s vs %s" % (self._env_info_spec, env.env_info_spec()))
            env.reset()

        self._current_env_id = np.int64(0)
        self._action_spec = OrderedDict(
            task_id=alf.BoundedTensorSpec((),
                                          maximum=len(envs) - 1,
                                          dtype='int64'),
            action=self._action_spec)

    @staticmethod
    def load(load_fn, environment_name, env_id=None, **kwargs):
        """
        Args:
            load_fn (Callable): function used to construct the environment for
                each tasks. It will be called as ``load_fn(env_name, **kwargs)``
            environment_name (list[str]): list of environment names
            env_id (int): (optional) ID of the environment.
            kwargs (**): arguments passed to load_fn
        """
        # TODO: may need to add the option of using ProcessEnvironment to wrap
        # the underlying environment
        envs = []
        for name in environment_name:
            envs.append(load_fn(name, **kwargs))
        return MultitaskWrapper(envs, environment_name, env_id)

    @property
    def num_tasks(self):
        return len(self._envs)

    @property
    def task_names(self):
        return self._task_names

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def env_info_spec(self):
        return self._env_info_spec

    def get_num_tasks(self):
        return len(self._envs)

    def _reset(self):
        time_step = self._envs[self._current_env_id].reset()
        return time_step._replace(
            env_id=self._env_id,
            prev_action=OrderedDict(
                task_id=self._current_env_id, action=time_step.prev_action))

    def _step(self, action):
        self._current_env_id = action['task_id']
        action = action['action']
        assert self._current_env_id < len(self._envs)
        time_step = self._envs[self._current_env_id].step(action)
        return time_step._replace(
            env_id=self._env_id,
            prev_action=OrderedDict(
                task_id=self._current_env_id, action=time_step.prev_action))

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self._envs[self._current_env_id], name)

    def seed(self, seed):
        for env in self._envs:
            env.seed(seed)


@alf.configurable(blacklist=['env'])
class CurriculumWrapper(AlfEnvironmentBaseWrapper):
    """A wrapper to provide automatic curriculum task selection.

    The probability of a task being chosen is based on its recent progress in
    terms of episode reward. A task will be chosen more often if its episode
    reward increases faster than other tasks.

    The progress of a task is defined as the difference between its current score
    and its past score divided by the average episode length for that task.
    """

    def __init__(self,
                 env,
                 progress_favor=10.0,
                 current_score_update_rate=1e-3,
                 past_score_update_rate=5e-4,
                 warmup_period=100):
        """
        env (AlfEnvironment): environment to be wrapped. It needs to be batched.
        progress_favor (float): how much more likely to choose the environment with the
            fastest progress than the ones with no progress. If ``progress_favor``
            is 1, all tasks are sampled uniformly.
        current_score_update_rate (float): the rate for updating the current score
        past_score_update_rate (float): the rate for updating the past score
        warmup_period (int): gradually increase ``progress_favor`` from 1 to
            ``progress_favor`` during the first ``num_tasks * warmup_period``
            episodes
        """
        self._env = env
        assert env.batched, "Only batched env is supported"
        num_tasks = env.num_tasks
        task_names = env.task_names
        batch_size = env.batch_size
        self._episode_rewards = torch.zeros(batch_size, device='cpu')
        self._episode_lengths = torch.zeros(batch_size, device='cpu')
        assert (
            len(env.action_spec()) == 2 and 'action' in env.action_spec()
            and 'task_id' in env.action_spec()
        ), ("The action_spec in the wrapped "
            "environment should have exactly two keys: 'task_id' and 'action'")
        self._action_spec = env.action_spec()['action']
        self._num_tasks = num_tasks
        self._task_names = task_names
        self._env_info_spec = copy.copy(env.env_info_spec())
        self._env_info_spec.update(
            self._add_task_names({
                'curriculum_task_count': [alf.TensorSpec(())] * num_tasks,
                'curriculum_task_score': [alf.TensorSpec(())] * num_tasks,
                'curriculum_task_prob': [alf.TensorSpec(())] * num_tasks
            }))
        self._zero_curriculum_info = self._add_task_names({
            'curriculum_task_count': [torch.zeros(batch_size, device='cpu')] *
                                     num_tasks,
            'curriculum_task_score': [torch.zeros(batch_size, device='cpu')] *
                                     num_tasks,
            'curriculum_task_prob': [torch.zeros(batch_size, device='cpu')] *
                                    num_tasks
        })
        self._progress_favor = progress_favor
        self._current_score_update_rate = current_score_update_rate
        self._past_score_update_rate = past_score_update_rate
        self._warmup_period = warmup_period * num_tasks
        self._scale = math.log(progress_favor)
        self._total_count = 0
        self._current_episode_lengths = torch.zeros(num_tasks, device='cpu')
        self._current_scores = torch.zeros(num_tasks, device='cpu')
        self._past_scores = torch.zeros(num_tasks, device='cpu')
        self._task_probs = torch.ones(num_tasks, device='cpu') / num_tasks
        self._task_counts = torch.zeros(num_tasks, device='cpu')

        self._current_task_ids = self._sample_tasks(batch_size)

    def _add_task_names(self, info):
        for k, v in info.items():
            info[k] = dict(zip(self._task_names, v))
        return info

    def _sample_tasks(self, num_samples):
        return torch.multinomial(
            self._task_probs, num_samples=num_samples, replacement=True)

    def _update_curriculum(self, task_ids, task_scores, task_episode_lengths):
        for task_id, task_score in zip(task_ids, task_scores):
            self._total_count += 1
            self._current_episode_lengths[
                task_ids] += self._current_score_update_rate * (
                    task_episode_lengths -
                    self._current_episode_lengths[task_ids])
            self._task_counts[task_id] += 1
            self._current_scores[
                task_id] += self._current_score_update_rate * (
                    task_score - self._current_scores[task_id])
            self._past_scores[task_id] += self._past_score_update_rate * (
                task_score - self._past_scores[task_id])

        # obtain the unbiased estimate of current scores and past scores
        current_scores = self._current_scores / (1 - (
            1 - self._current_score_update_rate)**self._task_counts + 1e-30)
        past_scores = self._past_scores / (
            1 - (1 - self._past_score_update_rate)**self._task_counts + 1e-30)
        current_episode_lengths = self._current_episode_lengths / (1 - (
            1 - self._current_score_update_rate)**self._task_counts + 1e-30)
        current_episode_lengths += 1e-30
        progresses = (
            current_scores - past_scores).relu() / current_episode_lengths
        max_progress = progresses.max()
        progresses = progresses / (max_progress + 1e-30)
        # Gradually increase scale from 0 to self._scale so that we tend to do
        # random sampling of the environments initially
        scale = self._scale * min(1, self._total_count / self._warmup_period)
        self._task_probs = F.softmax(scale * progresses, dim=0)

    def env_info_spec(self):
        return self._env_info_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        time_step = self._env.reset()
        info = copy.copy(time_step.env_info)
        info.update(self._zero_curriculum_info)
        return time_step._replace(
            env_info=info, prev_action=time_step.prev_action['action'])

    def _step(self, action):
        time_step = self._env.step(
            OrderedDict(task_id=self._current_task_ids, action=action))
        task_ids = self._current_task_ids
        time_step_cpu = time_step.cpu()
        info = time_step_cpu.env_info

        is_first_step = time_step_cpu.is_first()
        self._episode_rewards[is_first_step] = 0
        self._episode_lengths[is_first_step] = 0
        self._episode_rewards += alf.math.sum_to_leftmost(
            time_step_cpu.reward, 1)
        self._episode_lengths += 1
        is_last_step = time_step.cpu().is_last()
        last_env_ids = is_last_step.nonzero(as_tuple=True)[0]
        if last_env_ids.numel() > 0:
            self._update_curriculum(task_ids[last_env_ids],
                                    self._episode_rewards[last_env_ids],
                                    self._episode_lengths[last_env_ids])
            new_task_ids = self._sample_tasks(last_env_ids.numel())
            self._current_task_ids[last_env_ids] = new_task_ids

            num_envs = self._env.batch_size
            # Tensors in time_step need to have a batch dimension
            # [num_tasks, num_envs]
            task_counts = self._task_counts.unsqueeze(1).expand(-1, num_envs)
            current_scores = self._current_scores.unsqueeze(1).expand(
                -1, num_envs)
            task_probs = self._task_probs.unsqueeze(1).expand(-1, num_envs)
            # [1, num_envs]
            not_last = (~is_last_step).unsqueeze(0)
            # [num_tasks, num_envs]
            task_counts = task_counts.masked_fill(not_last, 0).cpu()
            current_scores = current_scores.masked_fill(not_last, 0).cpu()
            task_probs = task_probs.masked_fill(not_last, 0).cpu()
            # These info is for the purpose of generating summary by
            # ``alf.metrics.metrics.AverageEnvInfoMetric``, which calculates the
            # average of epsodic sum of the values of info. So we only provide
            # the info as LAST steps.
            info.update(
                self._add_task_names({
                    'curriculum_task_count': list(task_counts),
                    'curriculum_task_score': list(current_scores),
                    'curriculum_task_prob': list(task_probs)
                }))
        else:
            info.update(self._zero_curriculum_info)

        time_step = time_step._replace(
            prev_action=time_step.prev_action['action'], env_info=info)
        time_step._cpu = time_step_cpu._replace(
            prev_action=time_step_cpu.prev_action['action'], env_info=info)
        return time_step


class BatchedTensorWrapper(AlfEnvironmentBaseWrapper):
    """Wrapper that converts non-batched numpy-based I/O to batched tensors.
    """

    def __init__(self, env):
        assert not env.batched, (
            'BatchedTensorWrapper can only be used to wrap non-batched env')
        super().__init__(env)

    @property
    def is_tensor_based(self):
        return True

    @property
    def batched(self):
        return True

    @staticmethod
    def _to_batched_tensor(raw):
        """Convert the structured input into batched (batch_size = 1) tensors
        of the same structure.
        """
        return nest.map_structure(
            lambda x: (torch.as_tensor(x).unsqueeze(dim=0) if isinstance(
                x, (np.ndarray, np.number, float, int)) else x), raw)

    def _step(self, action):
        numpy_action = nest.map_structure(
            lambda x: x.squeeze(dim=0).cpu().numpy(), action)
        return BatchedTensorWrapper._to_batched_tensor(
            super()._step(numpy_action))

    def _reset(self):
        return BatchedTensorWrapper._to_batched_tensor(super()._reset())


class TensorWrapper(AlfEnvironmentBaseWrapper):
    """Wrapper that converts numpy-based I/O to tensors.
    """

    def __init__(self, env):
        assert env.batched, (
            'TensorWrapper can only be used to wrap batched env')
        super().__init__(env)

    @property
    def is_tensor_based(self):
        return True

    @staticmethod
    def _to_tensor(raw):
        """Convert the structured input into batched (batch_size = 1) tensors
        of the same structure.
        """
        return nest.map_structure(
            lambda x: (torch.as_tensor(x) if isinstance(
                x, (np.ndarray, np.number, float, int)) else x), raw)

    def _step(self, action):
        numpy_action = nest.map_structure(lambda x: x.cpu().numpy(), action)
        return TensorWrapper._to_tensor(super()._step(numpy_action))

    def _reset(self):
        return TensorWrapper._to_tensor(super()._reset())


@alf.configurable
class DiscreteActionWrapper(AlfEnvironmentBaseWrapper):
    """Discretize each continuous action dim into several evenly distributed
    values. Currently only support unnested action spec with a rank-1 shape.

    This wrapper can be used in both batch env mode (tensors) and individual env
    mode (numpy array).
    """

    def __init__(self, env: AlfEnvironment, actions_num: int):
        """
        Args:
            env: ALF env to be wrapped
            actions_num: number of values to discretize each action dim into
        """
        super().__init__(env)
        action_spec = env.action_spec()
        assert not alf.nest.is_nested(action_spec), (
            "This wrapper doesn't support nested action spec!")
        assert (
            isinstance(action_spec, ts.BoundedTensorSpec)
            and action_spec.is_continuous), (
                "This wrapper only supports bounded continuous action spec!")
        assert action_spec.ndim == 1, (
            "This wrapper only supports rank-1 action!")
        assert actions_num > 1, "Should define at least 2 discrete actions!"
        self._actions_num = actions_num
        self._action_delta = (
            (action_spec.maximum - action_spec.minimum) / (actions_num - 1))
        self._N = action_spec.numel
        self._dtype = action_spec.dtype
        self._minimum = action_spec.minimum
        # create the new discrete action spec
        self._action_spec = ts.BoundedTensorSpec(
            shape=(), dtype=torch.int64, maximum=actions_num**self._N - 1)
        self._time_step_spec = env.time_step_spec()._replace(
            prev_action=self._action_spec)

    def action_spec(self):
        return self._action_spec

    def time_step_spec(self):
        return self._time_step_spec

    def _reset(self):
        time_step = self._env.reset()
        if _is_numpy_array(time_step.prev_action):
            prev_action = np.zeros_like(time_step.step_type, dtype=np.int64)
        else:
            prev_action = torch.zeros_like(
                time_step.step_type, dtype=torch.int64)
        return time_step._replace(prev_action=prev_action)

    def _step(self, action):
        # convert the discrete action to a multi-dim continuous action
        # action shape: [B] or []
        idx = []
        base = self._actions_num**(self._N - 1)
        prev_action = action
        # convert to an idx number with base ``actions_num``
        for i in range(self._N):
            idx.append(action // base)
            action = action % base
            base //= self._actions_num
        if _is_numpy_array(action):
            idx = np.stack(
                idx, axis=-1).astype(ts.torch_dtype_to_str(self._dtype))
            action = idx * self._action_delta + self._minimum
        else:
            idx = torch.stack(idx, dim=-1).to(self._dtype)
            action = (idx * torch.as_tensor(self._action_delta) +
                      torch.as_tensor(self._minimum))
        # action: [B, action_dim] or [action_dim]
        time_step = self._env.step(action)
        return time_step._replace(prev_action=prev_action)


@alf.configurable
class AtariTerminalOnLifeLossWrapper(AlfEnvironmentBaseWrapper):
    """Wrapper to change discount to 0 upon life loss for Atari.

    This can potentially make it easier for the learning agent to recognize the
    significance of losing a life.

    Some papers report the results with this enabled (e.g. arXiv:2111.00210)
    """

    def __init__(self, env):
        """
        Args:
            env: ALF env to be wrapped
            actions_num: number of values to discretize each action dim into
        """
        super().__init__(env)
        self._prev_lives = 0

    def _reset(self):
        time_step = self._env.reset()
        self._prev_lives = time_step.env_info['ale.lives']
        return time_step

    def _step(self, action):
        time_step = self._env.step(action)
        lives = time_step.env_info['ale.lives']
        if lives < self._prev_lives:
            time_step = time_step._replace(discount=np.float32(0))
        self._prev_lives = lives
        return time_step


@alf.configurable
class TemporallyCorrelatedNoiseWrapper(AlfEnvironmentBaseWrapper):
    """Adding temporally correlated noise to actions.
    Reference:
    ::
        Swamy et al. Causal Imitation Learning under Temporally Correlated Noise, arXiv:2202.01312
    """

    def __init__(self, env, sigma=0.5, past_noise_weight=1.0):
        """Create a Temporally Correlated Noise wrapper, which adds temporally
        correlated noise to the action before interacting with the environment:

        noisy_action = action + past_noise_weight * past_noise + current_noise

        Args:
            sigma (float): standard deviation of the noise.
            past_noise_weight (float): the weight for the noise from the past
                when adding into the action for the current time step.
        """
        super().__init__(env)
        self._action_spec = env.action_spec()
        self._past_noise_weight = max(past_noise_weight, 0)
        self._past_noise = None
        self._sigma = sigma

    def _reset(self):
        self._past_noise = None
        return self._env.reset()

    def _step(self, action):
        if self._past_noise is None:
            self._past_noise = np.random.randn(*action.shape) * self._sigma

        current_noise = np.random.randn(*action.shape) * self._sigma
        noisy_action = action + self._past_noise_weight * self._past_noise + current_noise
        self._past_noise = current_noise

        noisy_action = np.clip(
            noisy_action,
            a_min=self._action_spec.minimum,
            a_max=self._action_spec.maximum).astype(np.float32)

        time_step = self._env.step(noisy_action)

        return time_step


class NormalizedActionWrapper(AlfEnvironmentBaseWrapper):
    """Normalize actions into [-1,1].

    The reason why we'd like to normalize the actions, even though our action
    distribution networks can do this, is because we want to set target entropy
    independent of action ranges for algorithms like SAC.

    This wrapper can be used only for individual envs (numpy array) or a batched
    env (tensor).
    """

    def __init__(self, env: AlfEnvironment):
        """
        Args:
            env: ALF env to be wrapped
        """
        super().__init__(env)
        action_spec = env.action_spec()
        assert all([
            isinstance(s, alf.BoundedTensorSpec)
            for s in nest.flatten(action_spec)
        ]), ("All action specs must be bounded! Got %s" % action_spec)

        def _action_affine_paras(spec):
            assert np.all(np.isfinite(spec.minimum))
            assert np.all(np.isfinite(spec.maximum))
            b0, b1 = spec.minimum, spec.maximum
            b = 0.5 * (b1 - b0)
            c = b0 + b
            return b, c

        self._affine_paras = nest.map_structure(_action_affine_paras,
                                                action_spec)
        # overwrite all action bounds to [-1,1]
        self._action_spec = nest.map_structure(
            lambda spec: alf.BoundedTensorSpec(
                minimum=-1., maximum=1., shape=spec.shape, dtype=spec.dtype),
            action_spec)
        self._time_step_spec = env.time_step_spec()._replace(
            prev_action=self._action_spec)

    def action_spec(self):
        return self._action_spec

    def time_step_spec(self):
        return self._time_step_spec

    def _step(self, action):
        def _scale_back(a, paras):
            b, c = paras
            return a * b + c

        scaled_action = nest.map_structure_up_to(action, _scale_back, action,
                                                 self._affine_paras)
        time_step = self._env.step(scaled_action)
        return time_step._replace(prev_action=action)


class BatchEnvironmentWrapper(AlfEnvironment):
    """Wrapper to make a list of non-batched environment into a batched environment.

    Note the individual environments in ``envs`` are executed sequentially doring
    one ``step()`` of ``reset()``.

    Args:
        envs: a list of unbatched ``AlfEnvironment``.
    """

    def __init__(self, envs: List[AlfEnvironment]):
        self._envs = envs
        super().__init__()
        self._action_spec = self._envs[0].action_spec()
        self._observation_spec = self._envs[0].observation_spec()
        self._reward_spec = self._envs[0].reward_spec()
        self._time_step_spec = self._envs[0].time_step_spec()
        self._env_info_spec = self._envs[0].env_info_spec()
        self._num_tasks = self._envs[0].num_tasks
        self._task_names = self._envs[0].task_names
        if any(env.is_tensor_based for env in self._envs):
            raise ValueError('All environments must be array-based.')
        if any(env.action_spec() != self._action_spec for env in self._envs):
            raise ValueError(
                'All environments must have the same action spec.')
        if any(env.time_step_spec() != self._time_step_spec
               for env in self._envs):
            raise ValueError(
                'All environments must have the same time_step_spec.')
        if any(env.env_info_spec() != self._env_info_spec
               for env in self._envs):
            raise ValueError(
                'All environments must have the same env_info_spec.')
        if any(env.batched for env in self._envs):
            raise ValueError('All environments must be non-batched.')

    @property
    def metadata(self):
        return self._envs[0].metadata

    @property
    def is_tensor_based(self):
        return False

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return len(self._envs)

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def task_names(self):
        return self._task_names

    def env_info_spec(self):
        return self._env_info_spec

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def time_step_spec(self):
        return self._time_step_spec

    def close(self):
        for env in self._envs:
            env.close()

    def render(self, mode):
        return self._envs[0].render(mode)

    def seed(self, seed):
        for i, env in enumerate(self._envs):
            env.seed(seed * len(self._envs) + i)

    def _step(self, action):
        def _ith(i):
            return alf.nest.map_structure(lambda x: x[i], action)

        time_steps = [env.step(_ith(i)) for i, env in enumerate(self._envs)]
        time_step = alf.nest.map_structure(lambda *arrays: np.stack(arrays),
                                           *time_steps)
        return time_step

    def _reset(self):
        time_steps = [env.reset() for env in self._envs]
        time_step = alf.nest.map_structure(lambda *arrays: np.stack(arrays),
                                           *time_steps)
        return time_step
