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
"""Helper functions for config alf training.

The main motivation is to give the access of observation_spec and action_spec,
which are necessary for config some models. observation_spec and action_spec are
only available after the environment is created. So we create an environment
based TrainerConfig in this module.
"""

from alf.environments.utils import create_environment
from alf.algorithms.config import TrainerConfig
from alf.algorithms.data_transformer import create_data_transformer

__all__ = [
    'close_env', 'get_raw_observation_spec', 'get_observation_spec',
    'get_action_spec', 'get_env'
]

_env = None
_transformed_observation_spec = None


def get_raw_observation_spec(field=None):
    """Get the ``TensorSpec`` of observations provided by the global environment.

    Note: you need to finish all the config for environments and
    TrainerConfig.random_seed before using this function.

    Args:
        field (str): a multi-step path denoted by "A.B.C".
    Returns:
        nested TensorSpec: a spec that describes the observation.
    """
    env = get_env()
    specs = env.observation_spec()
    if field:
        for f in field.split('.'):
            specs = specs[f]
    return specs


def get_observation_spec(field=None):
    """Get the spec of observation transformed by data transformers.

    The data transformers are specified by ``TrainerConfig.data_transformer_ctor``.

    Note: you need to finish all the config for environments and
    TrainerConfig.data_transformer_ctor before using this function.

    Args:
        field (str): a multi-step path denoted by "A.B.C".
    Returns:
        nested TensorSpec: a spec that describes the observation.
    """
    global _transformed_observation_spec
    if _transformed_observation_spec is None:
        config = TrainerConfig(root_dir='')
        env = get_env()
        data_transformer = create_data_transformer(
            config.data_transformer_ctor, env.observation_spec())
        observation_spec = data_transformer.transformed_observation_spec
        _transformed_observation_spec = observation_spec

    specs = _transformed_observation_spec
    if field:
        for f in field.split('.'):
            specs = specs[f]
    return specs


def get_action_spec():
    """Get the specs of the tensors expected by ``step(action)`` of the
    environment.

    Note: you need to finish all the config for environments and
    TrainerConfig.random_seed before using this function.

    Returns:
        nested TensorSpec: a spec that describes the shape and dtype of each tensor
        expected by ``step()``.
    """
    env = get_env()
    return env.action_spec()


def get_env():
    """Get the training environment.

    Note: you need to finish all the config for environments and
    TrainerConfig.random_seed before using this function.

    Returns:
        AlfEnvironment
    """
    global _env
    if _env is None:
        trainer_config = TrainerConfig(root_dir='')
        _env = create_environment(seed=trainer_config.random_seed)
    return _env


def close_env():
    """Close the global environment.

    This function will be automatically called by ``RLTrainer``.
    """
    global _env
    global _transformed_observation_spec
    if _env is not None:
        _env.close()
    _env = None
    _transformed_observation_spec = None
