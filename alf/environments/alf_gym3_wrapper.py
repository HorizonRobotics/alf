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
"""Wrapper providing an AlfEnvironment adapter for Gym3 environments

Gym3 provides an unified interface for reinforcement leraning environments that
improves upon the gym interface and includes vectorization (i.e. natively
supported batched environments).

Gym3 has a different set of considerations which lead to different design
choices compared to gym. See the following links to learn about those design
choices.

https://github.com/openai/gym3/blob/master/docs/design.md

"""
from typing import List, Callable, Optional, Any

import torch
import numpy as np
import gym3
from absl import logging

from alf.environments.alf_environment import AlfEnvironment
from alf import TensorSpec, BoundedTensorSpec
import alf.data_structures as ds
import alf.nest as nest


def _gym3_space_to_tensor_spec(space, force_int64: bool = False):
    """Convert Gym3 tensor specifications (a.k.a. spaces) to TensorSpec

    This is a helper function to form obesrvation spec and action specs for the
    AlfGym3Wrapper.

    Gym3 defines its own tensor specifications and as an adapting layer we use
    this function to convert that to Alf's BoundedTensorSpec.

    The code logic here follows the gym3's implementation:

    https://github.com/openai/gym3/blob/4c3824680eaf9dd04dce224ee3d4856429878226/gym3/interop.py#L74

    Args:

        space: the Gym3 space that describes the tensor specification of
            observation or action.

        force_int64: If set to True, all the discrete type will be converted to
            torch.int64. This is useful for action spec where we expect all
            discrete action to be converted to int64. The main reason for this
            is that actions are usually generated from action distributions,
            whose sample always produce int64 tensors.

    Returns:

        A nested BoundedTensorSpec with the same structure.

    """

    def __convert(gym3_space: gym3.types.TensorType) -> BoundedTensorSpec:
        # Gym3 space's eltype is the counterpart of the dtype
        eltype = gym3_space.eltype

        if isinstance(eltype, gym3.types.Discrete):
            eltype = gym3_space.eltype
            return BoundedTensorSpec(
                shape=gym3_space.shape,
                dtype=torch.int64 if force_int64 else eltype.dtype_name,
                minimum=0,
                maximum=eltype.n - 1)
        elif isinstance(eltype, gym3.types.Real):
            # Currently this follows gym3's logic to convert it to unbounded
            # tesnor as gym3.types.Real is not bounded.
            eltype = gym3_space.eltype
            return BoundedTensorSpec(
                shape=gym3_space.shape,
                dtype=eltype.dtype_name,
                minimum=float('-inf'),
                maximum=float('inf'))
        else:
            raise NotImplementedError(
                f'AlfGym3Wrapper does not support space element type {eltype} yet'
            )

    if isinstance(space, gym3.types.DictType):
        return {
            key: nest.map_structure(lambda x: __convert(x), space[key])
            for key in space.keys()
        }
    return nest.map_structure(lambda x: __convert(x), space)


def _extract_env_info_spec(sampe_env_info, ignored_info_keys: List[str] = []):
    """Extracts the environment info spec from a sample

    Args:

        sample_env_info (nested numpy array): A sample environment info instance
            whose array specification will be extracted and converted to nested
            TesnorSpec.

        ignored_info_keys: a list of keys that should be ignored from the
            environment info. Only the top level keys in the nested structure
            obey this.

    Returns:

        A nested TensorSpec that shares the same structure as the sample instance.

    """

    def __to_tensor_spec(entry):
        x = entry
        if np.isscalar(x):
            x = np.array(x)
        return TensorSpec.from_array(np.zeros_like(x))

    trimmed = {
        key: sampe_env_info[key]
        for key in sampe_env_info if key not in ignored_info_keys
    }

    return nest.map_structure(__to_tensor_spec, trimmed)


class AlfGym3Wrapper(AlfEnvironment):
    """An adapter to make Gym3 environments follow Alf's convention

    Although Gym3 provides an official gym wrapper, we decided to not base the
    Alf adapter upon that gym wrapper because:

    1. Performance and resource-wise, relying the natively supported batch
       (vectorized) environments from Gym3 is much more memory-efficient than
       creating a lot of Gym3 instances in subprocesses in batch mode.

    2. Gym3 has a different interface on indicating the last step and first step
       of an episode compared to gym.

    3. Gym3 has different interfaces to rendering and recording from gym.

    4. Gym3 normally do not provide support for resetting the environment.

    In this adapter, all above are considered and patched to achieve
    compatibility with AlfEnvironment.

    Normally you are not expected to call AlfGym3Wrapper directly. Instead the
    ``load()`` functions for various Gym3-based environments are preferred.

    For example, ``suite_procgen.load()`` is used to construct procgen
    environments which themselves are Gym3-based environments.

    NOTE: TimeLimit is currently not applicable to Gym3 environments
    as it does not offer reset() interface.

    """

    def __init__(self,
                 gym3_env: gym3.Env,
                 image_channel_first: bool = True,
                 ignored_info_keys: List[str] = [],
                 support_force_reset: bool = False,
                 render_activator: Optional[Callable[[], gym3.Env]] = None,
                 frame_extractor: Optional[Callable[[gym3.Env], Any]] = None):
        """Construct an adapted instance for the input Gym3 environment

        Args:

            gym3_env: the input environment which should be an instance of a
                class that derives from gym3.Env
            image_channel_first: when set to True, the image-based (of 3
                channels) observation will be permuted so that the channel
                dimension comes first.
            ignored_info_keys: a list of keys in the env info that should not be
                included in the env info of the TimeStep. This is useful when
                some huge but not useful information are stored in the env info
                of the underlying Gym3 environment, and ignoring them is crucial
                to achieve better performance.
            support_force_reset: Gym3 environments do not support force reset in
                general. However, some of the environments such as procgen
                allows sending action -1 to reset the environments. Set this to
                True to enable such behavior.
            render_activator: when set to None, it indicates that this
                environment does not support rendering. Otherwise it will be a
                function that re-creates a Gym3 environment with render enabled.
                See render() for details.
            frame_extractor: when set to None, it indicates that this
                environment does not support recording. Otherwise it will be a
                function that extracts the rendered frame for recording from the
                environment.

        """
        assert isinstance(gym3_env, gym3.Env), \
            f'AlfGym3Wrapper: {type(gym3_env)} is not derived from gym3.Env'
        super().__init__()

        # The underlying Gym3 environment
        self._gym3_env = gym3_env

        self._support_force_reset = support_force_reset

        # +--------------------------+
        # | Render/Recording Related |
        # +--------------------------+

        # When initially constructed, render is not enabled until the first call
        # to render() is invoked. Use self._render_enabled to make sure render
        # is not enabled for more than once.
        self._render_enabled = False
        self._render_activator = render_activator
        self._frame_extrator = frame_extractor
        # Create metadata with 'render.modes' so that it is compatible with
        # VideoRecorder.
        self.metadata = {'render.modes': []}
        if self._render_activator is not None and self._frame_extrator is not None:
            self.metadata['render.modes'].append('rgb_array')

        # +--------------------------+
        # | Cache the Tensor Specs   |
        # +--------------------------+

        # NOTE(breakds): when needed, expose this and allow an user to set it.
        self._discount = 1.0
        self._observation_spec = _gym3_space_to_tensor_spec(
            self._gym3_env.ob_space)

        self._image_channel_first = image_channel_first

        def _image_channel_first_permute_spec(spec):
            # Only transform the image-based component of the observation. This
            # simply assumes ndim == 3 implies an image.
            if spec.ndim != 3:
                return spec
            return BoundedTensorSpec(
                shape=(spec.shape[2], spec.shape[0], spec.shape[1]),
                dtype=spec.dtype,
                minimum=spec.minimum,
                maximum=spec.maximum)

        if image_channel_first:
            self._observation_spec = nest.map_structure(
                _image_channel_first_permute_spec, self._observation_spec)

        # For discrete action type, always use int64 during the conversion.
        self._action_spec = _gym3_space_to_tensor_spec(
            self._gym3_env.ac_space, force_int64=True)
        self._env_info_spec = _extract_env_info_spec(
            self._gym3_env.get_info()[0], ignored_info_keys=ignored_info_keys)

        # +--------------------------+
        # | Stateful Contexts        |
        # +--------------------------+

        # A list representing whether the corresponding single environment
        # finishes the current episode
        self._prev_first = [False] * self.batch_size

    @property
    def is_tensor_based(self):
        return True

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._gym3_env.num

    # Implement abstract env_info_spec
    def env_info_spec(self):
        return self._env_info_spec

    # Implement abstract observation_spec
    def observation_spec(self):
        return self._observation_spec

    # Implement abstract action_spec
    def action_spec(self):
        return self._action_spec

    def _create_time_step(self, reward, observation, action,
                          step_type: List[ds.StepType]) -> ds.TimeStep:
        """Construct a TimeStep object for Alf algorithms to consume

        This function construct the TimeStep objects based on the information
        observed from the underlying Gym3 environment. It essentially does:

        1. Convert numpy arrays or lists to tensors
        2. Apply image channel first if necessary
        3. Trim ignored keys from the env info

        """
        observation = nest.map_structure(lambda x: torch.as_tensor(x),
                                         observation)
        if self._image_channel_first:
            observation = nest.map_structure(
                lambda x: x.permute(0, 3, 1, 2).contiguous(), observation)

        trimmed_info = [
            nest.prune_nest_like(info, self.env_info_spec())
            for info in self._gym3_env.get_info()
        ]

        # In the case when we assume no timeouts, all episode end will
        # be due to success or failure, where discount is set to 0.0.
        discount = [
            0.0 if s == ds.StepType.LAST else self._discount for s in step_type
        ]

        return ds.TimeStep(
            step_type=torch.as_tensor(step_type),
            reward=torch.as_tensor(reward),
            discount=torch.as_tensor(discount),
            observation=observation,
            env_id=torch.arange(self.batch_size),
            prev_action=torch.as_tensor(action),
            env_info=nest.map_structure(
                lambda *values: torch.as_tensor(values), *trimmed_info))

    # Implement abstract _reset
    def _reset(self) -> ds.TimeStep:
        """Implement ``_reset()`` for compatibility

        Note that by default Gym3 environmenst cannot be reset. However, if
        ``support_force_reset`` is set to True, action -1 will be sent to all
        the sub-environments to reset them if the underlying environment follows
        the convention.

        Otherwise the reset will be ignored.

        """
        if self._support_force_reset:
            self._gym3_env.act(np.array([-1] * self.batch_size))
            self._prev_first = [False] * self.batch_size
        else:
            logging.warning('reset() ignored by AlfGym3Wrapper')

        reward, observation, _ = self._gym3_env.observe()

        time_step = self._create_time_step(
            reward,
            observation,
            step_type=[ds.StepType.FIRST] * self.batch_size,
            # Faking actions
            action=nest.map_structure(
                lambda spec: spec.numpy_zeros(outer_dims=(self.batch_size, )),
                self.action_spec()))

        return time_step

    # Implement abstract _step
    def _step(self, action) -> ds.TimeStep:
        """Implement ``_step()`` for compatibility

        A very important note here is that special treatment is done at the end
        of each episode. Unlike Gym environments, Gym3 environments do not
        return ``done=True`` immediately. Instead, ``first=True`` is returned
        upon the ``observe()`` of the NEXT ``act()``, which is actually the
        FIRST FRAME of a new episode in Gym3 sense.

        With this wrapper we CHANGED that definition, so that the frame with
        ``first=True`` actually becomes the LAST FRAME of the previous episode.
        Because the observation of the new episode can be DRAMATICALLY DIFFERENT
        from the actual last frame of the previous episode, the previous
        observation is returned in this case.

        Therefore to summarize, effectively we will

        1. Repeat the end-of-episode observation twice for each episode.

        2. Throw away the first frame of each episode, and use the second frame
           as if it is the first frame.

        Gym3's official Gym wrapper DID THE SAME.

        """
        _, prev_observation, _ = self._gym3_env.observe()

        np_action = nest.map_structure(lambda x: x.cpu().numpy(), action)
        self._gym3_env.act(np_action)

        reward, observation, first = self._gym3_env.observe()

        # Override the observation with the previous observation if that
        # particular environment has ``first=True``.

        def __override_with_prev_observation(ob_array: np.ndarray,
                                             prev_ob_array: np.ndarray):
            ob_array[first] = prev_ob_array[first]

        nest.map_structure(__override_with_prev_observation, observation,
                           prev_observation)

        # TODO(breakds): More properly deal with this by pre-process the
        # experiences in the replay buffer so that if the next step has first =
        # True, the previous step is considered an END frame. The current
        # implementation will incur an unnecessarily non-zero TD error with the
        # last 2 frames of an episode when TimeLimit is applied, but when the
        # episodes are long enough, the impact will be small.

        # This does the trick of repeating end-of-episode frames and throwing
        # away first-of-episode frames.
        step_type = [
            ds.StepType.FIRST if d else
            (ds.StepType.LAST if f else ds.StepType.MID)
            for d, f in zip(self._prev_first, first)
        ]

        time_step = self._create_time_step(
            reward=reward,
            observation=observation,
            step_type=step_type,
            action=action)

        self._prev_first = first

        return time_step

    def render(self, mode: str):
        """Enables rendering by re-activating the environment

        Args:

            mode: A string indicate the rendering mode. This is to make it
                compatible with Gym environments' rendering interface. For
                AlfGym3Wrapper, it returns the RGB array image if mode is
                specified as `rgb_array`, and None for other modes.

        """
        if not self._render_enabled:
            assert self._render_activator is not None, \
                ('This gym3 environment does not support rendering because '
                 'render_activator is not provided.')
            self._gym3_env = self._render_activator()
            self._render_enabled = True

        if mode == 'rgb_array':
            assert self._frame_extrator is not None, \
                ('This gym3 environment does not support recording because '
                 'frame_extractor is not provided.')
            return self._frame_extrator(self._gym3_env)

        return None
