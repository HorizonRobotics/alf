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
r"""Train using ActorCriticAlgorithm.

To run actor_critic on gym CartPole:
```bash
python main.py \
  --root_dir=~/tmp/cart_pole \
  --gin_file=ac_cart_pole.gin \
  --gin_param='create_environment.num_parallel_environments=8' \
  --alsologtostderr
```

You can visualize playing of the trained model by running:
```bash
python main.py \
  --root_dir=~/tmp/cart_pole \
  --gin_file=ac_cart_pole.gin \
  --play \
  --alsologtostderr
```

"""

import os

from absl import app
from absl import flags
from absl import logging

import gin.tf.external_configurables

from alf.algorithms.actor_critic_algorithm import create_ac_algorithm
from alf.environments.utils import create_environment
from alf.trainers import on_policy_trainer, off_policy_trainer
from alf.utils import common
import alf.utils.external_configurables

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
flags.DEFINE_bool('play', False, 'Visualize the playing')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(train_dir,
               algorithm_ctor=create_ac_algorithm,
               trainer=on_policy_trainer.train,
               evaluate=True,
               debug_summaries=False):
    env = create_environment()
    if evaluate:
        eval_env = create_environment(num_parallel_environments=1)
    else:
        eval_env = None
    algorithm = algorithm_ctor(env, debug_summaries=debug_summaries)

    trainer(
        train_dir,
        env,
        algorithm,
        eval_env=eval_env,
        debug_summaries=debug_summaries)


@gin.configurable
def play(train_dir, algorithm_ctor=create_ac_algorithm):
    """Play using the latest checkpoint under `train_dir`."""
    env = create_environment(num_parallel_environments=1)
    algorithm = algorithm_ctor(env)
    on_policy_trainer.play(train_dir, env, algorithm)


def main(_):
    logging.set_verbosity(logging.INFO)

    gin_file = common.get_gin_file()

    if FLAGS.gin_file and not FLAGS.play:
        common.copy_gin_configs(FLAGS.root_dir, gin_file)

    gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)

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
