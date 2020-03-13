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
# TODO: Fix alf.utils.dist_utils.epsilon_greedy_sample() to handle
# epsilon_greedy < 1.0 and change the default to 0.1
flags.DEFINE_float('epsilon_greedy', 1.0, "probability of sampling action.")
flags.DEFINE_integer('random_seed', None, "random seed")
flags.DEFINE_integer('num_episodes', 10, "number of episodes to play")
flags.DEFINE_float('sleep_time_per_step', 0.01,
                   "sleep so many seconds for each"
                   " step")
flags.DEFINE_string(
    'record_file', None, "If provided, video will be recorded"
    "to a file instead of shown on the screen.")
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

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
    algorithm = algorithm_ctor(
        observation_spec=env.observation_spec(), action_spec=env.action_spec())
    policy_trainer.play(
        FLAGS.root_dir,
        env,
        algorithm,
        checkpoint_step=FLAGS.checkpoint_step or "latest",
        epsilon_greedy=FLAGS.epsilon_greedy,
        num_episodes=FLAGS.num_episodes,
        sleep_time_per_step=FLAGS.sleep_time_per_step,
        record_file=FLAGS.record_file)
    env.close()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required('root_dir')
    app.run(main)
