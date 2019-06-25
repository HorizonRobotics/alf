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
"""Off-policy training using ActorCriticAlgorithm.
To run actor_critic on gym CartPole:
```bash
python actor_critic.py \
  --root_dir=~/tmp/cart_pole_ppo \
  --alsologtostderr \
  --gin_file=ppo_cart_pole.gin
```
"""

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from alf.examples.actor_critic import create_algorithm, create_environment
from alf.algorithms.on_policy_algorithm import OffPolicyAdapter
from alf.trainers import off_policy_trainer, on_policy_trainer
import alf.utils.external_configurables
from alf.utils import common
import alf.algorithms.ppo_loss

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(train_dir, evaluate=True, debug_summaries=False):
    """A simple train and eval for ActorCriticAlgorithm."""
    env = create_environment()
    if evaluate:
        eval_env = create_environment(num_parallel_environments=1)
    else:
        eval_env = None
    algorithm = create_algorithm(env, debug_summaries=debug_summaries)
    algorithm = OffPolicyAdapter(algorithm)
    off_policy_trainer.train(
        train_dir,
        env,
        algorithm,
        eval_env=eval_env,
        debug_summaries=debug_summaries)


def play(train_dir):
    """Play using the latest checkpoint under `train_dir`."""
    env = create_environment(num_parallel_environments=1)
    algorithm = create_algorithm(env)
    algorithm = OffPolicyAdapter(algorithm)
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
