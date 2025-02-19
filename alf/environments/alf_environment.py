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
"""ALF RL Environment API.

Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/py_environment.py
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/tf_environment.py
"""

import abc
import six

import alf
from alf.data_structures import time_step_spec


@six.add_metaclass(abc.ABCMeta)
class AlfEnvironment(object):
    """Abstract base class for ALF RL environments.

    Observations and valid actions are described with ``TensorSpec``, defined in
    the ``specs`` module.

    The ``current_time_step()`` method returns current ``time_step``, resetting the
    environment if necessary.

    The ``step(action)`` method applies the action and returns the new ``time_step``.
    This method will also reset the environment if needed and ignore the action in
    that case.

    The ``reset()`` method returns ``time_step`` that results from an environment
    reset and is guaranteed to have ``step_type=ts.FIRST``.

    The ``reset()`` method is only needed for explicit resets. In general, the
    environment will reset automatically when needed, for example, when no
    episode was started or when it reaches a step after the end of the episode
    (i.e. ``step_type=ts.LAST``).

    If the environment can run multiple steps at the same time and take a batched
    set of actions and return a batched set of observations, it should overwrite
    the property batched to True.

    Example for collecting an episode:

    .. code-block:: python

        env = AlfEnvironment()

        # reset() creates the initial time_step and resets the environment.
        time_step = env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
    """

    def __init__(self):
        self._current_time_step = None

    @property
    def num_tasks(self):
        """Number of tasks supported by this environment."""
        return 1

    @property
    def task_names(self):
        """The name of each tasks."""
        return [str(i) for i in range(self.num_tasks)]

    @property
    def is_tensor_based(self):
        """Whether the environment is tensor-based or not.

        Tensor-based environment means that the observations and actions are
        represented as tensors. Otherwise, they are represented as numpy arrays.
        """
        return False

    @property
    def batched(self):
        """Whether the environment is batched or not.

        If the environment supports batched observations and actions, then overwrite
        this property to True.

        A batched environment takes in a batched set of actions and returns a
        batched set of observations. This means for all numpy arrays in the input
        and output nested structures, the first dimension is the batch size.

        When batched, the left-most dimension is not part of the action_spec
        or the observation_spec and corresponds to the batch dimension.

        Returns:
            A boolean indicating whether the environment is batched or not.
        """
        return False

    @property
    def batch_size(self):
        """The batch size of the environment.

        Returns:
            The batch size of the environment, or 1 if the environment is not
            batched.

        Raises:
          RuntimeError: If a subclass overrode batched to return True but did not
              override the ``batch_size`` property.
        """
        if self.batched:
            raise RuntimeError(
                'Environment %s marked itself as batched but did not override the '
                'batch_size property' % type(self))
        return 1

    @abc.abstractmethod
    def env_info_spec(self):
        """Defines the env_info provided by the environment."""

    @abc.abstractmethod
    def observation_spec(self):
        """Defines the observations provided by the environment.

        May use a subclass of ``TensorSpec`` that specifies additional properties such
        as min and max bounds on the values.

        Returns:
            nested TensorSpec
        """

    @abc.abstractmethod
    def action_spec(self):
        """Defines the actions that should be provided to ``step()``.

        May use a subclass of ``TensorSpec`` that specifies additional properties such
        as min and max bounds on the values.

        Returns:
            nested TensorSpec
        """

    def reward_spec(self):
        """Defines the reward provided by the environment.

        The reward of the most environments is a scalar. So we provide a default
        implementation which returns a scalar spec.

        Returns:
            alf.TensorSpec
        """
        return alf.TensorSpec(())

    def time_step_spec(self):
        """Describes the ``TimeStep`` fields returned by ``step()``.

        Override this method to define an environment that uses non-standard values
        for any of the items returned by ``step()``. For example, an environment with
        tensor-valued rewards.

        Returns:
            A ``TimeStep`` namedtuple containing (possibly nested) ``TensorSpec`` defining
            the step_type, reward, discount, observation, prev_action, and end_id.
        """
        return time_step_spec(self.observation_spec(), self.action_spec(),
                              self.reward_spec())

    def current_time_step(self):
        """Returns the current timestep."""
        return self._current_time_step

    def reset(self):
        """Starts a new sequence and returns the first ``TimeStep`` of this sequence.

        Note: Subclasses cannot override this directly. Subclasses implement
        ``_reset()`` which will be called by this method. The output of ``_reset()``
        will be cached and made available through ``current_time_step()``.

        Returns:
            TimeStep:
        """
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, action):
        """Updates the environment according to the action and returns a ``TimeStep``.

        If the environment returned a ``TimeStep`` with ``StepType.LAST`` at the
        previous step the implementation of ``_step`` in the environment should call
        ``reset`` to start a new sequence and ignore ``action``.

        This method will start a new sequence if called after the environment
        has been constructed and ``reset`` has not been called. In this case
        ``action`` will be ignored.

        Note: Subclasses cannot override this directly. Subclasses implement
        ``_step()`` which will be called by this method. The output of ``_step()``
        will be cached and made available through ``current_time_step()``.

        Args:
            action (nested Tensor): input actions.

        Returns:
            TimeStep:
        """
        if self._current_time_step is None:
            return self.reset()

        self._current_time_step = self._step(action)
        return self._current_time_step

    def sync_progress(self):
        """A placeholder interface to make it consistent with some derived classes
        such as ``ParallelAlfEnvironment`` and ``FastParallelAlfEnvironment``.
        """
        pass

    def close(self):
        """Frees any resources used by the environment.

        Implement this method for an environment backed by an external process.

        This method can be used directly:

        .. code-block:: python

            env = Env(...)
            # Use env.
            env.close()


        or via a context manager:

        .. code-block:: python

            with Env(...) as env:
            # Use env.
        """
        pass

    def __enter__(self):
        """Allows the environment to be used in a with-statement context."""
        return self

    def __exit__(self, unused_exception_type, unused_exc_value,
                 unused_traceback):
        """Allows the environment to be used in a with-statement context."""
        self.close()

    def render(self, mode='rgb_array'):
        """Renders the environment.

        Args:
            mode: One of ['rgb_array', 'human', 'online']. Renders to an numpy array, or brings
                up a window where the environment can be visualized.
                The ``online`` mode can be used to collect and return a list of frames 
                (e.g., generated by ``render(self, mode='rgb_array')``) of a running environment.

        Returns:
            An ndarray of shape ``[width, height, 3]`` denoting an RGB image if mode is
            ``rgb_array``. Otherwise return nothing and render directly to a display
            window.
        Raises:
            NotImplementedError: If the environment does not support rendering.
        """
        del mode  # unused
        raise NotImplementedError('No rendering support.')

    def seed(self, seed):
        """Seeds the environment.

        Args:
            seed (int): Value to use as seed for the environment.
        """
        del seed  # unused
        raise NotImplementedError('No seed support for this environment.')

    def get_info(self):
        """Returns the environment info returned on the last step.

        Returns:
            Info returned by last call to ``step()``. None by default.

        Raises:
            NotImplementedError: If the environment does not use info.
        """
        raise NotImplementedError(
            'No support of get_info for this environment.')

    #  These methods are to be implemented by subclasses:

    @abc.abstractmethod
    def _step(self, action):
        """Updates the environment according to action and returns a ``TimeStep``.

        See ``step(self, action)`` docstring for more details.

        Args:
            action: A tensor, or a nested dict, list or tuple of tensors
                corresponding to ``action_spec()``.
        """

    @abc.abstractmethod
    def _reset(self):
        """Starts a new sequence, returns the first ``TimeStep`` of this sequence.

        See ``reset(self)`` docstring for more details
        """
