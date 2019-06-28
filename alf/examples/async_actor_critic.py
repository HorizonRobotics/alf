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
"""async training using ActorCriticAlgorithm.

A TEMPORARY main script for running async training. To be merged with
main.py.

To run on gym CartPole:
```bash
python async_actor_critic.py \
  --root_dir=~/tmp/cart_pole_async \
  --alsologtostderr \
  --gin_file=async_ac_cart_pole.gin
```
"""

from absl import app
from absl import flags
from absl import logging

import gin
import os
import tensorflow as tf

from alf.algorithms.actor_critic_algorithm import create_ac_algorithm
from alf.environments.utils import create_environment
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.trainers import async_off_policy_trainer
import alf.utils.external_configurables
from alf.utils import common


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')


FLAGS = flags.FLAGS


@gin.configurable
def train(train_dir,
          algorithm_ctor=create_ac_algorithm,
          debug_summaries=False):
    """A simple train and eval for ActorCriticAlgorithm."""
    def env_f(): return create_environment()
    algorithm = algorithm_ctor(env_f(), debug_summaries=debug_summaries)

    if isinstance(algorithm, OffPolicyAlgorithm):
        async_off_policy_trainer.train(
            train_dir,
            env_f,
            algorithm,
            debug_summaries=debug_summaries)
    else:
        raise ValueError("Algorithm must be off policy", type(algorithm))


def main(_):
    logging.set_verbosity(logging.INFO)

    gin_file = common.get_gin_file()

    if FLAGS.gin_file:
        common.copy_gin_configs(FLAGS.root_dir, gin_file)

    gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)

    train(FLAGS.root_dir + "/train")


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    flags.mark_flag_as_required('root_dir')
    app.run(main)
