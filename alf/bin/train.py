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

To run actor_critic on gym CartPole:
```bash
cd ${PROJECT}/alf/examples;
python -m alf.bin.train \
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

from alf.utils import common
import alf.utils.external_configurables
from alf.trainers import policy_trainer

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(root_dir):
    """Train and evaluate algorithm

    Args:
        root_dir (str): directory for saving summary and checkpoints
    """
    trainer_conf = policy_trainer.TrainerConfig(root_dir=root_dir)
    trainer = policy_trainer.Trainer(trainer_conf)
    trainer.train()


def main(_):
    gin_file = common.get_gin_file()
    gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)
    train_eval(FLAGS.root_dir)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required('root_dir')
    if torch.cuda.is_available():
        alf.set_default_device("cuda")
    app.run(main)
