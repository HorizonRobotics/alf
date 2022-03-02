# Copyright (c) 2022 Horizon Robotics. All Rights Reserved.
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
"""Suite for loading RLBench::

    "RLBench: The Robot Learning Benchmark \& Learning Environment", James et al., 2020

You need to first follow the `installation instructions <https://github.com/stepjam/RLBench>`_
to install all software dependencies. We use a **forked version** at
`https://github.com/HorizonRobotics/RLBench`_ where some customized changes are made.
So when pip installing RLBench, please use our url.

For headless rendering with VirtualGL if image inputs are trained on a machine
without displays, also follow their instructions on how to set it up.

Qt required by RLBench might have conflicts with ``cv2``. To resolve the this,
we need to switch to a headless version:

.. code-block::python

    pip uninstall opencv-python
    pip install opencv-python-headless

Note that the headless version won't allow us to use GUI with ``cv2``, e.g.,
``cv2.imshow()``.

RLBench also have a conflict with ``matplotlib`` regarding xserver when multiple
envs are run in parallel. Our ``VideoRecorder`` (``alf.summary.render``) uses
``matplotlib`` for rendering, so when recording a video, currently only one
RLBench env can be used.

Tip: when GPU rendering is needed, setting ``CUDA_VISIBLE_DEVICES`` to other gpus
will separate the GPU training and rendering. Also PyRep seems not reactive to
SIGINT (ctrl+c), so to kill the training job, we need SIGQUIT (ctrl+\).
(https://github.com/stepjam/PyRep/issues/12)
"""

import alf

try:
    import rlbench
    import rlbench.gym
    from rlbench.observation_config import ObservationConfig
    import alf.environments.rlbench_custom_tasks
except ImportError:
    rlbench = None
    ObservationConfig = None

from typing import List
import gym

from alf.environments.alf_wrappers import AlfEnvironmentBaseWrapper
from alf.environments.alf_environment import AlfEnvironment
from alf.environments import suite_gym


def is_available():
    return rlbench is not None


@alf.configurable
def load(environment_name: str,
         env_id: int = None,
         discount: float = 1.,
         max_episode_steps: int = None,
         observation_config: ObservationConfig = None,
         gym_env_wrappers: List[gym.Wrapper] = (),
         alf_env_wrappers: List[AlfEnvironmentBaseWrapper] = ()
         ) -> AlfEnvironment:
    """Loads the selected environment and wraps it with the specified wrappers.

    Currently, the fields for the vision-based environment observation are hardcoded.
    Potentially we can customize RLBench to return different vision&depth sensorary
    data.

    Args:
        environment_name: Name for the environment to load. For a complete list of
            100 tasks, take a look at `https://github.com/HorizonRobotics/RLBench/tree/master/rlbench/tasks`_.
            For a particular task file ``<task_name>.py``, we can use one of
            ``["<task_name>-state-v0", "<task_name>-vision-v0", "<task_name>-v0"]`` as
            the environment name. The first two are envs with predefined observation configs,
            where the first returns a low-dim flattend state vector and the second
            returns a dictionary with all five camera rgb images and the state vector.
            When using the third env name, it is required that the user also provides
            ``observation_config`` for customization.
        env_id: A scalar ``Tensor`` of the environment ID of the time step.
        discount: Discount to use for the environment.
        max_episode_steps: If None the ``max_episode_steps`` will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no ``timestep_limit`` set in the environment's spec.
        observation_config: configuration object for observation. Using this config,
            we can easily customize which sensors to turn on in the env observation.
            For all options, please see ``rlbench.observation_config.py``. This arg
            is only used when ``environment_name`` is ``"<task_name>-v0"``.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        alf_env_wrappers: Iterable with references to wrapper classes to use on
            the torch environment.
    """

    gym_spec = gym.spec(environment_name)
    env = gym_spec.make(obs_config=observation_config)

    if max_episode_steps is None:
        if gym_spec.max_episode_steps is not None:
            max_episode_steps = gym_spec.max_episode_steps
        else:
            max_episode_steps = 0

    return suite_gym.wrap_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        # RLBench returns 'channels_last' rgb, so we always need to transpose it
        image_channel_first=True)
