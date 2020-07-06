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


@gin.configurable
class Config(object):
    """Configuration for supervised training."""

    def __init__(self,
                 root_dir,
                 algorithm_ctor=None,
                 random_seed=None,
                 epochs=2e+5,
                 evaluate=True,
                 summary_interval=50,
                 summaries_flush_secs=1,
                 summary_max_queue=100,
                 summarize_grads_and_vars=False):
        """
        Args:
            root_dir (str): directory for saving summary and checkpoints
            algorithm_ctor (Callable): callable that create an
                ``OffPolicyAlgorithm`` or ``OnPolicyAlgorithm`` instance
            random_seed (None|int): random seed, a random seed is used if None
            epochs (int): number of training epoches
            evaluate (bool): whether to evluate after training
            summary_interval (int): write summary every so many training steps
            summaries_flush_secs (int): flush summary to disk every so many seconds
            summary_max_queue (int): flush to disk every so mary summaries
            summarize_grads_and_vars (bool): If True, gradient and network variable
                summaries will be written during training.
        """
        parameters = dict(
            root_dir=root_dir,
            algorithm_ctor=algorithm_ctor,
            random_seed=random_seed,
            epochs=epochs,
            evaluate=evaluate,
            summary_interval=summary_interval,
            summaries_flush_secs=summaries_flush_secs,
            summary_max_queue=summary_max_queue,
            summarize_grads_and_vars=summarize_grads_and_vars)
        for k, v in parameters.items():
            self.__setattr__(k, v)


@gin.configurable
def create_dataset(dataset_name='mnist',
                   dataset_loader=datagen,
                   train_batch_size=100,
                   test_batch_size=100):
    """Create a pytorch data loaders.

    Args:
        dataset_name (str): dataset_name
        dataset_loader (Callable) : callable that create pytorch data
            loaders for both training and testing.
        train_batch_size (int): batch_size for training.
        test_batch_size (int): batch_size for testing.

    Returns:
        trainset (torch.utils.data.DataLoaderr):
        testset (torch.utils.data.DataLoaderr):
    """

    trainset, testset = getattr(dataset_loader,
                                'load_{}'.format(dataset_name))(
                                    train_bs=train_batch_size,
                                    test_bs=test_batch_size)

    return trainset, testset


def train(config: Config):
    root_dir = os.path.expanduser(config.root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    random_seed = common.set_random_seed(config.random_seed)
    assert config.epochs > 0, \
        "Must provide #epochs for training!"

    trainset, testset = create_dataset()
    input_tensor_spec = TensorSpec(shape=testset.dataset[0][0].shape)
    output_dim = len(testset.dataset.classes)

    # print algorithm related gin parameters
    _, inopt_configs = common.get_gin_confg_strs()
    print(inopt_configs)

    algorithm = config.algorithm_ctor(
        input_tensor_spec=input_tensor_spec,
        last_layer_size=output_dim,
        last_activation=math_ops.identity)

    algorithm.set_data_loader(trainset, testset)

    alf.summary.enable_summary()
    summary_dir = os.path.expanduser(train_dir)
    summary_writer = alf.summary.create_summary_writer(
        summary_dir,
        flush_secs=config.summaries_flush_secs,
        max_queue=config.summary_max_queue)

    def _cond():
        # We always write summary in the initial `summary_interval` steps
        # because there might be important changes at the beginning.
        return (alf.summary.is_summary_enabled()
                and (epoch_num < config.summary_interval
                     or epoch_num % config.summary_interval == 0))

    with alf.summary.push_summary_writer(summary_writer):
        with alf.summary.record_if(_cond):
            epoch_num = 0
            print("==> Begin Training")
            while True:
                print("-" * 68)
                print("Epoch: {}".format(epoch_num + 1))
                with record_time("time/train_iter"):
                    train_steps = algorithm.train_iter()

                if epoch_num == 0:
                    # We need to wait for one iteration to get the operative args
                    # Right just give a fixed gin file name to store operative args
                    common.write_gin_configs(root_dir, "configured.gin")

                    with alf.summary.record_if(lambda: True):

                        def _markdownify(paragraph):
                            return "    ".join(
                                (os.linesep + paragraph).splitlines(
                                    keepends=True))

                        common.summarize_gin_config()
                        alf.summary.text('commandline', ' '.join(sys.argv))
                        alf.summary.text(
                            'optimizers',
                            _markdownify(algorithm.get_optimizer_info()))
                        alf.summary.text('revision', git_utils.get_revision())
                        alf.summary.text('diff',
                                         _markdownify(git_utils.get_diff()))
                        alf.summary.text('seed', str(random_seed))

                if config.evaluate:
                    print("==> Begin testing")
                    algorithm.evaluate()

                epoch_num += 1
                if (config.epochs and epoch_num >= config.epochs):
                    break

    summary_writer.close()


def main(_):
    gin_file = common.get_gin_file()
    gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)
    train_config = Config(FLAGS.root_dir)
    train(train_config)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required('root_dir')
    if torch.cuda.is_available():
        alf.set_default_device("cuda")
    app.run(main)
