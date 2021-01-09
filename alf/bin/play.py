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
r"""Play a trained model.

You can visualize playing of the trained model by running:
```bash
cd ${PROJECT}/alf/examples;
python -m alf.bin.play \
  --root_dir=~/tmp/cart_pole \
  --gin_file=ac_cart_pole.gin \
  --alsologtostderr
```

"""

from absl import app
from absl import flags
from absl import logging
import gin
import os

import torch

from alf.algorithms.algorithm import Algorithm
from alf.algorithms.data_transformer import create_data_transformer
from alf.environments.utils import create_environment
from alf.trainers import policy_trainer
from alf.utils import common
import alf.utils.external_configurables

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer(
    'checkpoint_step', None, "the number of training steps which is used to "
    "specify the checkpoint to be loaded. If None, the latest checkpoint under "
    "train_dir will be used.")
flags.DEFINE_float('epsilon_greedy', 0., "probability of sampling action.")
flags.DEFINE_integer('random_seed', None, "random seed")
flags.DEFINE_integer('num_episodes', 10, "number of episodes to play")
flags.DEFINE_integer('max_episode_length', 0,
                     "If >0,  each episode is limited "
                     "to so many steps")
flags.DEFINE_integer(
    'future_steps', 0, "If >0, display information from so many "
    "number of future steps in addition to the current step "
    "on the current frame. Otherwise only information from the "
    "current step will be displayed.")
flags.DEFINE_integer(
    'append_blank_frames', 0,
    "If >0, wil append such number of blank frames at the "
    "end of each episode in the rendered video file.")
flags.DEFINE_float('sleep_time_per_step', 0.01,
                   "sleep so many seconds for each step")
flags.DEFINE_string(
    'record_file', None, "If provided, video will be recorded"
    "to a file instead of shown on the screen.")
flags.DEFINE_bool('render', True,
                  "Whether render ('human'|'rgb_array') the frames or not")
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
flags.DEFINE_string(
    'ignored_parameter_prefixes', "",
    "Comma separated strings to ingore the parameters whose name has one of "
    "these prefixes in the checkpoint.")

FLAGS = flags.FLAGS


def main(_):
    seed = common.set_random_seed(FLAGS.random_seed)
    gin_file = common.get_gin_file()
    gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)
    algorithm_ctor = gin.query_parameter(
        'TrainerConfig.algorithm_ctor').scoped_configurable_fn
    env = create_environment(nonparallel=True, seed=seed)
    env.reset()
    common.set_global_env(env)
    config = policy_trainer.TrainerConfig(root_dir="")
    data_transformer = create_data_transformer(config.data_transformer_ctor,
                                               env.observation_spec())
    config.data_transformer = data_transformer
    observation_spec = data_transformer.transformed_observation_spec
    common.set_transformed_observation_spec(observation_spec)
    algorithm = algorithm_ctor(
        observation_spec=observation_spec,
        action_spec=env.action_spec(),
        config=config)
    try:
        policy_trainer.play(
            FLAGS.root_dir,
            env,
            algorithm,
            checkpoint_step=FLAGS.checkpoint_step or "latest",
            epsilon_greedy=FLAGS.epsilon_greedy,
            num_episodes=FLAGS.num_episodes,
            max_episode_length=FLAGS.max_episode_length,
            sleep_time_per_step=FLAGS.sleep_time_per_step,
            record_file=FLAGS.record_file,
            future_steps=FLAGS.future_steps,
            append_blank_frames=FLAGS.append_blank_frames,
            render=FLAGS.render,
            ignored_parameter_prefixes=FLAGS.ignored_parameter_prefixes.split(
                ",") if FLAGS.ignored_parameter_prefixes else [])
    finally:
        env.close()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required('root_dir')
    if torch.cuda.is_available():
        alf.set_default_device("cuda")
    app.run(main)
