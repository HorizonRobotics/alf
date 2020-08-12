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
r"""Train a supervised learning model.

To train hypernetwork_algorithm on `mnist`:

.. code-block:: bash

    cd ${PROJECT}/alf/examples;
    python -m alf.bin.supervised_train \
    --root_dir=~/tmp/mnist \
    --gin_file=hypernet_mnist.gin \
    --alsologtostderr

You can view various training curves using Tensorboard by running the follwoing
command in a different terminal:

.. code-block:: bash

    tensorboard --logdir=~/tmp/mnist
"""

from absl import app
from absl import flags
from absl import logging
import gin
import os
import sys
import time
import torch
import torch.nn as nn

from alf.algorithms.config import SupervisedTrainerConfig
from alf.tensor_specs import TensorSpec
from alf.utils import common, git_utils, math_ops
import alf.utils.datagen as datagen
import alf.utils.external_configurables
from alf.utils.summary_utils import record_time

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS
