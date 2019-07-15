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
cd ${PROJECT}/alf/examples;
python -m alf.bin.main \
  --root_dir=~/tmp/cart_pole \
  --gin_file=ac_cart_pole.gin \
  --gin_param='create_environment.num_parallel_environments=8' \
  --alsologtostderr
```

You can view various training curves using Tensorboard by running the follwoing
command in a different terminal:
```bash
tensorboard --logdir=~/tmp/cart_pole
```

You can visualize playing of the trained model by running:
```bash
cd ${PROJECT}/alf/examples;
python -m alf.bin.main \
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
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.utils import common
import alf.utils.external_configurables

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
flags.DEFINE_bool('play', False, 'Visualize the playing')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(root_dir, algorithm_ctor, evaluate=True, debug_summaries=False):
    """Train and evaluate algorithm

    Args:
        root_dir (str): directory for saving summary and checkpoints
        algorithm_ctor (Callable): callable that create an
            `OffPolicyAlgorithm` or `OnPolicyAlgorithm` instance
        evaluate (bool): A bool to evaluate when training.
        debug_summaries (bool): A bool to gather debug summaries.
    """
    if evaluate:
        eval_env = create_environment(num_parallel_environments=1)
    else:
        eval_env = None
    env = create_environment(num_parallel_environments=1)
    algorithm = algorithm_ctor(env, debug_summaries=debug_summaries)
    env.pyenv.close()

    if isinstance(algorithm, OffPolicyAlgorithm):
        trainer = off_policy_trainer.train
        env_or_env_f = create_environment
    elif isinstance(algorithm, OnPolicyAlgorithm):
        trainer = on_policy_trainer.train
        env_or_env_f = create_environment()
    else:
        raise ValueError(
            "Algorithm must be one of `OffPolicyAlgorithm`,"
            " `OnPolicyAlgorithm`. Received:", type(algorithm))
    trainer(
        root_dir,
        env_or_env_f,
        algorithm,
        eval_env=eval_env,
        debug_summaries=debug_summaries)


@gin.configurable
def play(root_dir, algorithm_ctor):
    """Play using the latest checkpoint under `train_dir`.

    Args:
        root_dir (str): directory where checkpoints stores
        algorithm_ctor (Callable): callable that create an algorithm
            parameter value is bind with `__main__.train_eval.algorithm_ctor`,
            just config `__main__.train_eval.algorithm_ctor` when using with gin configuration
    """
    env = create_environment(num_parallel_environments=1)
    algorithm = algorithm_ctor(env)
    on_policy_trainer.play(root_dir, env, algorithm)


def main(_):
    logging.set_verbosity(logging.INFO)

    gin_file = common.get_gin_file()

    if FLAGS.gin_file and not FLAGS.play:
        common.copy_gin_configs(FLAGS.root_dir, gin_file)

    gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)
    if FLAGS.play:
        with gin.unlock_config():
            gin.bind_parameter(
                '__main__.play.algorithm_ctor',
                gin.query_parameter('__main__.train_eval.algorithm_ctor'))
        play(FLAGS.root_dir)
    else:
        train_eval(FLAGS.root_dir)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    flags.mark_flag_as_required('root_dir')
    app.run(main)
