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
r"""Train model.

To run actor-critic on gym `CartPole`:

.. code-block:: bash

    cd ${PROJECT}/alf/examples;
    python -m alf.bin.train \
    --root_dir=~/tmp/cart_pole \
    --gin_file=ac_cart_pole.gin \
    --gin_param='create_environment.num_parallel_environments=8' \
    --alsologtostderr

You can view various training curves using Tensorboard by running the follwoing
command in a different terminal:

.. code-block:: bash

    tensorboard --logdir=~/tmp/cart_pole

You can visualize playing of the trained model by running:

.. code-block:: bash

    cd ${PROJECT}/alf/examples;
    python -m alf.bin.play \
    --root_dir=~/tmp/cart_pole \
    --gin_file=ac_cart_pole.gin \
    --alsologtostderr

If instead of Gin configuration file, you want to use ALF python conf file, then
replace the "--gin_file" option with "--conf", and "--gin_param" with "--conf_param".
"""

from absl import app
from absl import flags
from absl import logging
import gin
import os
import pathlib
import torch

from alf.utils import common
import alf.utils.external_configurables
from alf.trainers import policy_trainer


def _define_flags():
    flags.DEFINE_string(
        'root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
        'Root directory for writing logs/summaries/checkpoints.')
    flags.DEFINE_string('gin_file', None, 'Path to the gin-config file.')
    flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
    flags.DEFINE_string('conf', None, 'Path to the alf config file.')
    flags.DEFINE_multi_string('conf_param', None, 'Config binding parameters.')
    flags.DEFINE_bool('store_snapshot', True,
                      'Whether store an ALF snapshot before training')


FLAGS = flags.FLAGS


@gin.configurable
def train_eval(root_dir):
    """Train and evaluate algorithm

    Args:
        root_dir (str): directory for saving summary and checkpoints
    """
    trainer_conf = policy_trainer.TrainerConfig(root_dir=root_dir)
    if trainer_conf.ml_type == 'rl':
        trainer = policy_trainer.RLTrainer(trainer_conf)
    elif trainer_conf.ml_type == 'sl':
        trainer = policy_trainer.SLTrainer(trainer_conf)
    else:
        raise ValueError("Unsupported ml_type: %s" % trainer_conf.ml_type)

    trainer.train()


def main(_):
    FLAGS.alsologtostderr = True
    root_dir = os.path.expanduser(FLAGS.root_dir)
    os.makedirs(root_dir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(log_dir=root_dir)

    if FLAGS.store_snapshot:
        # ../<ALF_REPO>/alf/bin/train.py
        file_path = os.path.abspath(__file__)
        alf_root = str(pathlib.Path(file_path).parent.parent.parent.absolute())
        # generate a snapshot of ALF repo as ``<root_dir>/alf``
        common.generate_alf_root_snapshot(alf_root, root_dir)

    conf_file = common.get_conf_file()
    try:
        common.parse_conf_file(conf_file)
        train_eval(FLAGS.root_dir)
    finally:
        alf.close_env()


if __name__ == '__main__':
    _define_flags()
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required('root_dir')
    if torch.cuda.is_available():
        alf.set_default_device("cuda")
    app.run(main)
