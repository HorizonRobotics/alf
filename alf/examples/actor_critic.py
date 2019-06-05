# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
r"""Train using ActorCriticPolicy

To run actor_critic on gym CartPole:
```bash
python actor_critic.py \
  --root_dir=~/tmp/cart_pole \
  --alsologtostderr \
  --gin_file=ac_cart_pole.gin
```

You can visualize playing of the trained model by running:
```bash
python actor_critic.py \
  --root_dir=~/tmp/cart_pole \
  --alsologtostderr \
  --gin_file=ac_cart_pole.gin \
  --play
```

To run on SocialBot CartPole:
```bash
python actor_critic.py \
  --root_dir=~/tmp/socialbot-cartpole \
  --alsologtostderr \
  --gin_param='create_environment.env_name="SocialBot-CartPole-v0"' \
  --gin_param='create_environment.num_parallel_environments=16' \
  --gin_param='create_environment.env_load_fn=@suite_socialbot.load' \
  --gin_file=ac_cart_pole.gin
```

To Run on SocailBot SimpleNavigation
```bash
python actor_critic.py \
    --root_dir=~/tmp/simple_navigation \
    --alsologtostderr \
    --gin_file=ac_simple_navigation.gin
```

To Run on MountainCar using intrinsic curiosity module:
```bash
python actor_critic.py \
    --root_dir=~/tmp/mountain_car \
    --alsologtostderr \
    --gin_file=icm_mountain_car.gin
```

"""

import os
import random
import shutil
import time

from absl import app
from absl import flags
from absl import logging

import gin.tf.external_configurables
import tensorflow as tf

from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.encoding_network import EncodingNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.environments import atari_wrappers

from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.drivers.on_policy_driver import OnPolicyDriver
from alf.environments import suite_socialbot
from alf.trainers import on_policy_trainer

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
flags.DEFINE_bool('play', False, 'Visualize the playing')

FLAGS = flags.FLAGS

tf.keras.layers.Conv2D = gin.external_configurable(tf.keras.layers.Conv2D,
                                                   'tf.keras.layers.Conv2D')
tf.optimizers.Adam = gin.external_configurable(tf.optimizers.Adam,
                                               'tf.optimizers.Adam')
gin.external_configurable(atari_wrappers.FrameStack4)


@gin.configurable
def load_with_random_max_episode_steps(env_name,
                                       env_load_fn=suite_gym.load,
                                       min_steps=200,
                                       max_steps=250):
    return suite_gym.load(
        env_name, max_episode_steps=random.randint(min_steps, max_steps))


@gin.configurable
def create_algorithm(env,
                     actor_fc_layers=(200, 100),
                     value_fc_layers=(200, 100),
                     encoding_fc_layers=(),
                     use_rnns=False,
                     use_icm=False,
                     num_steps_per_iter=10000,
                     num_iterations=1000,
                     learning_rate=5e-5,
                     debug_summaries=False):
    """Create a simple ActorCriticAlgorithm."""
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    if use_rnns:
        actor_net = ActorDistributionRnnNetwork(
            env.observation_spec(),
            env.action_spec(),
            input_fc_layer_params=actor_fc_layers,
            output_fc_layer_params=None)
        value_net = ValueRnnNetwork(
            env.observation_spec(),
            input_fc_layer_params=value_fc_layers,
            output_fc_layer_params=None)
    else:
        actor_net = ActorDistributionNetwork(
            env.observation_spec(),
            env.action_spec(),
            fc_layer_params=actor_fc_layers)
        value_net = ValueNetwork(
            env.observation_spec(), fc_layer_params=value_fc_layers)

    encoding_net = None
    if encoding_fc_layers:
        encoding_net = EncodingNetwork(
            env.observation_spec(), fc_layer_params=encoding_fc_layers)

    icm = None
    if use_icm:
        feature_spec = env.observation_spec()
        if encoding_net:
            feature_spec = tf.TensorSpec((encoding_fc_layers[-1], ),
                                         dtype=tf.float32)
        icm = ICMAlgorithm(
            env.action_spec(), feature_spec, encoding_net=encoding_net)

    return ActorCriticAlgorithm(
        action_spec=env.action_spec(),
        actor_network=actor_net,
        value_network=value_net,
        intrinsic_curiosity_module=icm,
        optimizer=optimizer,
        debug_summaries=debug_summaries)


@gin.configurable
def create_environment(env_name='CartPole-v0',
                       env_load_fn=suite_gym.load,
                       num_parallel_environments=30):
    """Create environment."""
    if num_parallel_environments == 1:
        py_env = env_load_fn(env_name)
    else:
        if env_load_fn == suite_socialbot.load:
            logging.info("suite_socialbot environment")
            # No need to wrap with process since ParllelPyEnvironment will do it
            env_load_fn = lambda env_name: suite_socialbot.load(
                env_name, wrap_with_process=False)
        py_env = parallel_py_environment.ParallelPyEnvironment(
            [lambda: env_load_fn(env_name)] * num_parallel_environments)
    return tf_py_environment.TFPyEnvironment(py_env)


@gin.configurable
def train_eval(train_dir, debug_summaries=False):
    """A simple train and eval for ActorCriticAlgorithm."""
    env = create_environment()
    algorithm = create_algorithm(env, debug_summaries=debug_summaries)
    on_policy_trainer.train(
        train_dir, env, algorithm, debug_summaries=debug_summaries)


def play(train_dir):
    """A simple train and eval for ActorCriticAlgorithm."""
    env = create_environment(num_parallel_environments=1)
    algorithm = create_algorithm(env)
    on_policy_trainer.play(train_dir, env, algorithm)


def copy_gin_configs(root_dir, gin_files):
    """Copy gin config files to root_dir

    Args:
        root_dir (str): directory path
        gin_files (None|list[str]): list of file paths
    """
    if gin_files is None:
        return
    root_dir = os.path.expanduser(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    for f in gin_files:
        shutil.copyfile(f, os.path.join(root_dir, os.path.basename(f)))


def main(_):
    logging.set_verbosity(logging.INFO)
    copy_gin_configs(FLAGS.root_dir, FLAGS.gin_file)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    if FLAGS.play:
        play(FLAGS.root_dir + "/train")
    else:
        train_eval(FLAGS.root_dir + "/train")


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    flags.mark_flag_as_required('root_dir')
    app.run(main)
