import numpy as np

import alf.data_structures as ds
import alf.nest as nest
from alf.tensor_specs import torch_dtype_to_str
from alf.environments.alf_gym_wrapper import AlfGymWrapper, tensor_spec_from_gym_space


def _as_array(nested):
    """Convert scalars in ``nested`` to np.ndarray."""

    def __as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x

    return nest.map_structure(__as_array, nested)


class AlfBsuiteWrapper(AlfGymWrapper):
    def __init__(self,
                 gym_env,
                 env_id=None,
                 discount=1.0,
                 auto_reset=True,
                 simplify_box_bounds=True):

        super(AlfBsuiteWrapper, self).__init__(gym_env,
                                               env_id=env_id,
                                               discount=discount,
                                               auto_reset=auto_reset,
                                               simplify_box_bounds=simplify_box_bounds)

        self._observation_spec = tensor_spec_from_gym_space(
            self._gym_env.observation_space, simplify_box_bounds)

    @property
    def gym(self):
        """Return the gym environment. """
        return self._gym_env

    def _obtain_zero_info(self):
        """Get an env info of zeros only once when the env is created.
        This info will be filled in each ``FIRST`` time step as a placeholder.
        """
        self._gym_env.reset()
        action = nest.map_structure(lambda spec: spec.numpy_zeros(),
                                    self._action_spec)
        _, _, _, info = self._gym_env.step(action)
        self._gym_env.reset()
        info = _as_array(info)
        return nest.map_structure(lambda a: np.zeros_like(a), info)

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self._gym_env, name)

    def get_info(self):
        """Returns the gym environment info returned on the last step."""
        return self._info

    def _reset(self):
        # TODO: Upcoming update on gym adds **kwargs on reset. Update this to
        # support that.
        observation = self._gym_env.reset()
        self._info = None
        self._done = False

        observation = self._to_spec_dtype_observation(observation)
        observation = np.reshape(observation, (observation.shape[1],))
        return ds.restart(
            observation=observation,
            action_spec=self._action_spec,
            reward_spec=self._reward_spec,
            env_id=self._env_id,
            env_info=self._zero_info)

    @property
    def done(self):
        return self._done

    def _step(self, action):
        # Automatically reset the environments on step if they need to be reset.
        if self._auto_reset and self._done:
            return self.reset()

        observation, reward, self._done, self._info = self._gym_env.step(
            action)
        observation = self._to_spec_dtype_observation(observation)
        self._info = _as_array(self._info)

        if self._done:
            return ds.termination(
                observation,
                action,
                reward,
                self._reward_spec,
                self._env_id,
                env_info=self._info)
        else:
            return ds.transition(
                observation,
                action,
                reward,
                self._reward_spec,
                self._discount,
                self._env_id,
                env_info=self._info)

    def _to_spec_dtype_observation(self, observation):
        """Make sure observation from env is converted to the correct dtype.

        Args:
            observation (nested arrays or tensors): observations from env.

        Returns:
            A (nested) arrays of observation
        """

        def _as_spec_dtype(arr, spec):
            dtype = torch_dtype_to_str(spec.dtype)
            if str(arr.dtype) == dtype:
                return arr
            else:
                return arr.astype(dtype)

        return nest.map_structure(_as_spec_dtype, observation,
                                  self._observation_spec)
