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

import runpy
import math
from alf.algorithms.config import TrainerConfig
from alf.algorithms.data_transformer import create_data_transformer
from alf.config_util import config1, get_config_value, pre_config, validate_pre_configs
from alf.environments.utils import create_environment
from alf.utils.common import set_random_seed
from alf.utils.per_process_context import PerProcessContext

__all__ = [
    'close_env', 'get_raw_observation_spec', 'get_observation_spec',
    'get_action_spec', 'get_reward_spec', 'get_env', 'parse_config'
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
        data_transformer_ctor = get_config_value(
            'TrainerConfig.data_transformer_ctor')
        env = get_env()
        data_transformer = create_data_transformer(data_transformer_ctor,
                                                   env.observation_spec())
        _transformed_observation_spec = data_transformer.transformed_observation_spec

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


def get_reward_spec():
    """Get the spec of the reward returned by the environment.

    Note: you need to finish all the config for environments and
    TrainerConfig.random_seed before using this function.

    Returns:
        TensorSpec: a spec that describes the shape and dtype of reward.
    """
    return get_env().reward_spec()


_is_parsing = False


def adjust_config_by_multi_process_divider(multi_process_divider: int = 1):
    """Adjust specific configuration value in multiple process settings
    Alf assumes all configuration files geared towards single process training.
    This means that in multi-process settings such as DDP some of the
    configuration values needs to be adjusted to achieve parity on number of
    processes.
    For example, if we run 64 environments in parallel for single process
    settings, the value needs to be overriden with 16 if there are 4 identical
    processes running DDP training to achieve parity.
    Args:
        multi_process_divider (int): this is equivalent to number of processes
    """
    if multi_process_divider <= 1:
        return

    # Adjust the num of environments per process. The value for single process
    # (before adjustment) is divided by the multi_process_divider and becomes
    # the per-process value.
    tag = 'create_environment.num_parallel_environments'
    num_parallel_environments = get_config_value(tag)
    config1(
        tag,
        math.ceil(num_parallel_environments / multi_process_divider),
        raise_if_used=False)


def parse_config(conf_file, conf_params):
    """Parse config file and config parameters

    Note: a global environment will be created (which can be obtained by
    alf.get_env()) and random seed will be initialized by this function using
    common.set_random_seed().

    Args:
        conf_file (str): The full path of the config file.
        conf_params (list[str]): the list of config parameters. Each one has a
            format of CONFIG_NAME=VALUE.
    """
    global _is_parsing
    _is_parsing = True

    try:
        if conf_params:
            for conf_param in conf_params:
                pos = conf_param.find('=')
                if pos == -1:
                    raise ValueError("conf_param should have a format of "
                                     "'CONFIG_NAME=VALUE': %s" % conf_param)
                config_name = conf_param[:pos]
                config_value = conf_param[pos + 1:]
                config_value = eval(config_value)
                pre_config({config_name: config_value})

        runpy.run_path(conf_file)
        validate_pre_configs()
    finally:
        _is_parsing = False

    # Create the global environment and initialize random seed
    get_env()


def get_env():
    """Get the global training environment.

    Note: you need to finish all the config for environments and
    TrainerConfig.random_seed before using this function.

    Note: random seed will be initialized in this function.

    Returns:
        AlfEnvironment
    """
    global _env
    if _env is None:
        if _is_parsing:
            random_seed = get_config_value('TrainerConfig.random_seed')
        else:
            # We construct a TrainerConfig object here so that the value
            # configured through gin-config can be properly retrieved.
            train_config = TrainerConfig(root_dir='')
            random_seed = train_config.random_seed
        # We have to call set_random_seed() here because we need the actual
        # random seed to call create_environment.
        seed = set_random_seed(random_seed)
        # We need to re-set 'TrainerConfig.random_seed' to record the actual
        # random seed we are using.
        if random_seed is None:
            config1('TrainerConfig.random_seed', seed, raise_if_used=False)

        # In case when running in multi-process mode, the number of environments
        # per process need to be adjusted (divided by number of processes).
        adjust_config_by_multi_process_divider(
            PerProcessContext().num_processes)
        _env = create_environment(seed=random_seed)
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
