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

import math
import random
from alf.algorithms.config import TrainerConfig
from alf.algorithms.data_transformer import create_data_transformer
from alf.config_util import config1, get_config_value, load_config, pre_config, validate_pre_configs
from alf.environments.utils import create_environment
from alf.utils.common import set_random_seed
from alf.utils.per_process_context import PerProcessContext
from alf.utils.spawned_process_utils import SpawnedProcessContext, get_spawned_process_context

__all__ = [
    'close_env', 'get_raw_observation_spec', 'get_observation_spec',
    'get_action_spec', 'get_reward_spec', 'get_env', 'parse_config'
]

_env = None
_transformed_observation_spec = None


def get_raw_observation_spec(field=None):
    """Get the ``TensorSpec`` of observations provided by the global environment.

    .. note::
        This function can only be called after all gym wrappers and ``TrainerConfig.random_seed``
        have been configured. Otherwise the created environment might have unexpected
        behaviors.

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

    .. note::

        You need to finish all the config for environments and
        ``TrainerConfig.data_transformer_ctor`` before using this function.

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


def adjust_config_by_multi_process_divider(ddp_rank: int,
                                           multi_process_divider: int = 1):
    """Adjust specific configuration value in multiple process settings
    Alf assumes all configuration files geared towards single process training.
    This means that in multi-process settings such as DDP some of the
    configuration values needs to be adjusted to achieve parity on number of
    processes.
    For example, if we run 64 environments in parallel for single process
    settings, the value needs to be overridden with 16 if there are 4 identical
    processes running DDP training to achieve parity.

    The adjusted configs are

    1. TrainerConfig.mini_batch_size: divided by processes
    2. TrainerConfig.num_env_steps: divided by processes if used
    3. TrainerConfig.mini_batch_size: divided by processes if used
    4. TrainerConfig.evaluate: set to False except for process 0

    Args:
        ddp_rank: the rank of device to the process.
        multi_process_divider: this is equivalent to number of processes
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
        raise_if_used=False,
        override_all=True)

    # Adjust the mini_batch_size. If the original configured value is 64 and
    # there are 4 processes, it should mean that "jointly the 4 processes have
    # an effective mini_batch_size of 64", and each process has a
    # mini_batch_size of 16.
    tag = 'TrainerConfig.mini_batch_size'
    mini_batch_size = get_config_value(tag)
    if isinstance(mini_batch_size, int):
        config1(
            tag,
            math.ceil(mini_batch_size / multi_process_divider),
            raise_if_used=False,
            override_all=True)

    # If the termination condition is num_env_steps instead of num_iterations,
    # we need to adjust it as well since each process only sees env steps taking
    # by itself.
    tag = 'TrainerConfig.num_env_steps'
    num_env_steps = get_config_value(tag)
    if num_env_steps > 0:
        config1(
            tag,
            math.ceil(num_env_steps / multi_process_divider),
            raise_if_used=False,
            override_all=True)

    tag = 'TrainerConfig.initial_collect_steps'
    init_collect_steps = get_config_value(tag)
    config1(
        tag,
        math.ceil(init_collect_steps / multi_process_divider),
        raise_if_used=False,
        override_all=True)

    # Only allow process with rank 0 to have evaluate. Enabling evaluation for
    # other parallel processes is a waste as such evaluation does not offer more
    # information.
    if ddp_rank > 0:
        config1(
            'TrainerConfig.evaluate',
            False,
            raise_if_used=False,
            override_all=True)


def parse_config(conf_file, conf_params, create_env=True):
    """Parse config file and config parameters

    Note: a global environment will be created (which can be obtained by
    alf.get_env()) and random seed will be initialized by this function using
    common.set_random_seed().

    Args:
        conf_file (str): The full path of the config file.
        conf_params (list[str]): the list of config parameters. Each one has a
            format of CONFIG_NAME=VALUE.
        create_env (bool): whether to create env. If True, create the env if it is not
            created yet, and the ``ml_type`` is ``rl``.
            For other types (e.g. ``sl``), no env is created.

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

        load_config(conf_file)
        validate_pre_configs()
    finally:
        _is_parsing = False

    ml_type = get_config_value('TrainerConfig.ml_type')

    # only create env for ``ml_type`` with value of ``rl``
    create_env = create_env and (ml_type == 'rl')

    if create_env:
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
        # When ``get_env()`` is called in a spawned process (this is almost
        # always due to a ``ProcessEnvironment`` created with "spawn" method),
        # use the environment constructor from the context to create the
        # environment. This is to avoid creating a parallel environment which
        # leads to infinite recursion.
        ctx = get_spawned_process_context()
        if isinstance(ctx, SpawnedProcessContext):
            _env = ctx.create_env()
            return _env

        if _is_parsing:
            random_seed = get_config_value('TrainerConfig.random_seed')
        else:
            # We construct a TrainerConfig object here so that the value
            # configured through gin-config can be properly retrieved.
            train_config = TrainerConfig(root_dir='')
            random_seed = train_config.random_seed

        # Override the random seed based on ddp rank. With the following logic:
        #
        # - Rank 0 will get the original random seed specified in the config
        # - Rank 1 will get the 1st pseudo random integer as seed (using original seed)
        # - Rank 2 will get the 2nd pseudo random integer as seed (using original seed)
        # - ...
        #
        # The random.seed(random_seed) is temporary and will be overridden by
        # set_random_seed() below.
        random.seed(random_seed)

        if random_seed is not None:
            # If random seed is None, we will have None for other ranks, too.
            # A 'None' random seed won't set a deterministic torch behavior.
            for _ in range(PerProcessContext().ddp_rank):
                random_seed = random.randint(0, 2**32)
        config1(
            "TrainerConfig.random_seed",
            random_seed,
            raise_if_used=False,
            override_sole_init=True)

        # We have to call set_random_seed() here because we need the actual
        # random seed to call create_environment.
        seed = set_random_seed(random_seed)
        # In case when running in multi-process mode, the number of environments
        # per process need to be adjusted (divided by number of processes).
        adjust_config_by_multi_process_divider(
            PerProcessContext().ddp_rank,
            PerProcessContext().num_processes)
        _env = create_environment(seed=seed)
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
