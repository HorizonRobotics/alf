import bsuite
from bsuite import sweep
from bsuite.utils import gym_wrapper

import alf
from alf.environments import gym_wrappers, alf_wrappers, alf_gym_wrapper


@alf.configurable
def load(environment=sweep.CARTPOLE_SWINGUP,
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         image_channel_first=True):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name (str): Name for the environment to load.
        env_id (int): (optional) ID of the environment.
        discount (float): Discount to use for the environment.
        max_episode_steps (int): If None the max_episode_steps will be set to the
            default step limit defined in the environment's spec. No limit is applied
            if set to 0 or if there is no max_episode_steps set in the environment's
            spec.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers
            classes to use directly on the gym environment.
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment.
        image_channel_first (bool): whether transpose image channels to first dimension.

    Returns:
        An AlfEnvironment instance.
    """

    if env_id == None:
        env_id = 0

    env = bsuite.load_from_id('catch/' + str(environment))
    gym_env = gym_wrapper.GymFromDMEnv(env)

    if max_episode_steps is None:
        if gym_env.max_episode_steps is not None:
            max_episode_steps = gym_env.max_episode_steps
        else:
            max_episode_steps = 0

    return wrap_env(
        gym_env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        image_channel_first=image_channel_first)


@alf.configurable
def wrap_env(gym_env,
             env_id=None,
             discount=1.0,
             max_episode_steps=0,
             gym_env_wrappers=(),
             time_limit_wrapper=alf_wrappers.TimeLimit,
             normalize_action=True,
             clip_action=True,
             alf_env_wrappers=(),
             image_channel_first=True,
             auto_reset=True):
    """Wraps given gym environment with AlfGymWrapper.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Also note that all gym wrappers assume images are 'channel_last' by default,
    while PyTorch only supports 'channel_first' image inputs. To enable this
    transpose, 'image_channel_first' is set as True by default. ``gym_wrappers.ImageChannelFirst``
    is applied after all gym_env_wrappers and before the AlfGymWrapper.

    Args:
        gym_env (gym.Env): An instance of OpenAI gym environment.
        env_id (int): (optional) ID of the environment.
        discount (float): Discount to use for the environment.
        max_episode_steps (int): Used to create a TimeLimitWrapper. No limit is applied
            if set to 0. Usually set to `gym_spec.max_episode_steps` as done in `load.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers,
            classes to use directly on the gym environment.
        time_limit_wrapper (AlfEnvironmentBaseWrapper): Wrapper that accepts
            (env, max_episode_steps) params to enforce a TimeLimit. Usually this
            should be left as the default, alf_wrappers.TimeLimit.
        normalize_action (bool): if True, will scale continuous actions to
            ``[-1, 1]`` to be better used by algorithms that compute entropies.
        clip_action (bool): If True, will clip continuous action to its bound specified
            by ``action_spec``. If ``normalize_action`` is also ``True``, this
            clipping happens after the normalization (i.e., clips to ``[-1, 1]``).
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment.
        image_channel_first (bool): whether transpose image channels to first dimension.
            PyTorch only supports channgel_first image inputs.
        auto_reset (bool): If True (default), reset the environment automatically after a
            terminal state is reached.

    Returns:
        An AlfEnvironment instance.
    """

    for wrapper in gym_env_wrappers:
        gym_env = wrapper(gym_env)

    # To apply channel_first transpose on gym (py) env
    if image_channel_first:
        gym_env = gym_wrappers.ImageChannelFirst(gym_env)

    if normalize_action:
        # normalize continuous actions to [-1, 1]
        gym_env = gym_wrappers.NormalizedAction(gym_env)

    if clip_action:
        # clip continuous actions according to gym_env.action_space
        gym_env = gym_wrappers.ContinuousActionClip(gym_env)

    env = alf_gym_wrapper.AlfGymWrapper(
        gym_env=gym_env,
        env_id=env_id,
        discount=discount,
        auto_reset=auto_reset,
    )

    if max_episode_steps > 0:
        env = time_limit_wrapper(env, max_episode_steps)

    for wrapper in alf_env_wrappers:
        env = wrapper(env)

    return env
