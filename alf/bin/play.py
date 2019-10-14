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

import os

from absl import app
from absl import flags
from absl import logging

import gin.tf.external_configurables
from alf.environments.utils import create_environment
from alf.utils import common
import alf.utils.external_configurables
from alf.trainers import policy_trainer

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string(
    'checkpoint_name', None, "name of the checkpoint "
    "(e.g. 'ckpt-12800`). If None, the latest checkpoint under train_dir will "
    "be used.")
flags.DEFINE_bool('greedy_predict', True, "use greedy action for evaluation.")
flags.DEFINE_integer('random_seed', 0, "random seed")
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
    gin_file = common.get_gin_file()
    gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)
    algorithm_ctor = gin.query_parameter(
        'TrainerConfig.algorithm_ctor').scoped_configurable_fn
    env = create_environment(nonparallel=True)
    common.set_global_env(env)
    algorithm = algorithm_ctor()
    policy_trainer.play(
        FLAGS.root_dir,
        env,
        algorithm,
        checkpoint_name=FLAGS.checkpoint_name,
        greedy_predict=FLAGS.greedy_predict,
        random_seed=FLAGS.random_seed,
        num_episodes=FLAGS.num_episodes,
        sleep_time_per_step=FLAGS.sleep_time_per_step,
        record_file=FLAGS.record_file)
    env.pyenv.close()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    flags.mark_flag_as_required('root_dir')
    app.run(main)
