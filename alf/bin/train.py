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

You can view various training curves using Tensorboard by running the following
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

In case you have multiple GPUs on the machine and you would like to
train with all of them, specify --distributed multi-gpu. This will use
PyTorch's DistributedDataParallel for training.

If instead of Gin configuration file, you want to use ALF python conf file, then
replace the "--gin_file" option with "--conf", and "--gin_param" with "--conf_param".

"""

from absl import app
from absl import flags
from absl import logging
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from alf.utils import common
from alf.utils.per_process_context import PerProcessContext
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
    flags.DEFINE_bool(
        'force_torch_deterministic', True,
        'torch.use_deterministic_algorithms when random_seed is set')
    flags.DEFINE_bool('store_snapshot', True,
                      'Whether store an ALF snapshot before training')
    flags.DEFINE_enum(
        'distributed', 'none', ['none', 'multi-gpu'],
        'Set whether and how to run training in distributed mode.')
    flags.DEFINE_bool('as_remote_trainer', False,
                      'Whether to run in a remote trainer mode.')
    flags.DEFINE_bool('as_remote_unroller', False,
                      'Whether to run in a remote unroller mode.')
    flags.mark_flag_as_required('root_dir')


FLAGS = flags.FLAGS


def _setup_logging(rank: int, log_dir: str):
    """Setup logging for each process

    Args:
        rank (int): The ID of the process among all of the DDP processes
        log_dir (str): path to the directory where log files are written to
    """
    FLAGS.alsologtostderr = True
    logging.set_verbosity(logging.INFO)
    logging.get_absl_handler().use_absl_log_file(log_dir=log_dir)
    logging.use_absl_handler()


def _setup_device(rank: int = 0):
    """Setup the GPU device for each process

    All tensors of the calling process will use the GPU with the
    specified rank by default.

    Args:
        rank (int): The ID of the process among all of the DDP processes

    """
    if torch.cuda.is_available():
        alf.set_default_device('cuda')
        torch.cuda.set_device(rank)


def _setup_remote_configs_if_needed():
    """Preconfig some configurations for remote training and unrolling.

    There will also be a global config flag named 'TrainerConfig.remote_training'
    set in this function, in case the user needs this info in the conf. The flag
    has one of the three values:
        - 'trainer' for remote training
        - 'unroller' for remote unrolling
        - False for local training
    """
    assert not (FLAGS.as_remote_trainer and FLAGS.as_remote_unroller), (
        'Cannot specify both --as_remote_trainer and --as_remote_unroller')
    if FLAGS.as_remote_trainer:
        alf.pre_config({
            'TrainerConfig.unroll_length': -1,
            'TrainerConfig.evaluate': False,
            'TrainerConfig.remote_training': 'trainer'
        })
    elif FLAGS.as_remote_unroller:
        alf.pre_config({
            'TrainerConfig.async_eval': False,
            'TrainerConfig.remote_training': 'unroller'
        })


def _train(root_dir, rank=0, world_size=1):
    """Launch the trainer after the conf file has been parsed. This function
    could be called by grid search after the config has been modified.

    Args:
        root_dir (str): Path to the directory for writing logs/summaries/checkpoints.
        rank (int): The ID of the process among all of the DDP processes. For
            non-distributed training, this id should be 0.
        world_size (int): The number of processes in total. If set to 1, it is
            interpreted as "non distributed mode".
    """
    conf_file = common.get_conf_file()
    trainer_conf = policy_trainer.TrainerConfig(
        root_dir=root_dir, conf_file=conf_file)

    if trainer_conf.ml_type == 'rl':
        ddp_rank = rank if world_size > 1 else -1
        if FLAGS.as_remote_trainer or FLAGS.as_remote_unroller:
            from alf.algorithms.distributed_off_policy_algorithm import (
                DistributedTrainer, DistributedUnroller)
            if FLAGS.as_remote_trainer:
                alg_wrapper_ctor = DistributedTrainer
            else:
                alg_wrapper_ctor = DistributedUnroller
        else:
            alg_wrapper_ctor = None
        trainer = policy_trainer.RLTrainer(trainer_conf, ddp_rank,
                                           alg_wrapper_ctor)
    elif trainer_conf.ml_type == 'sl':
        # NOTE: SLTrainer does not support distributed training yet
        if world_size > 1:
            raise RuntimeError(
                "Multi-GPU DDP training does not support supervised learning")
        if FLAGS.as_remote_trainer or FLAGS.as_remote_unroller:
            raise RuntimeError(
                "Remote training does not support supervised learning")
        trainer = policy_trainer.SLTrainer(trainer_conf)
    else:
        raise ValueError("Unsupported ml_type: %s" % trainer_conf.ml_type)

    trainer.train()


def _training_worker_helper(rank: int, *args, **kwargs):
    # Helper to start the training worker with the correct rank
    # so that rank 0 is from the main process and the rest are
    # from the spawned processes.
    training_worker(rank + 1, *args, **kwargs)


def training_worker(rank: int,
                    world_size: int,
                    conf_file: str,
                    root_dir: str,
                    paras_queue: mp.Queue = None):
    """An executable instance that trains and evaluate the algorithm

    Args:
        rank (int): The ID of the process among all of the DDP processes.
        world_size (int): The number of processes in total. If set to 1, it is
            interpreted as "non distributed mode".
        conf_file (str): Path to the training configuration.
        root_dir (str): Path to the directory for writing logs/summaries/checkpoints.
        paras_queue: a shared Queue for checking the consistency of model parameters
            in different worker processes, if multi-gpu training is used.
    """
    try:
        _setup_logging(log_dir=root_dir, rank=rank)
        _setup_device(rank)
        if world_size > 1:
            # Specialization for distributed mode
            dist.init_process_group('nccl', rank=rank, world_size=world_size)
            # Recover the flags when spawned as a sub process
            if rank > 0:
                _define_flags()
                FLAGS(sys.argv, known_only=True)
                FLAGS.mark_as_parsed()
            # Set the rank and total number of processes for distributed training.
            PerProcessContext().set_distributed(
                rank=rank, num_processes=world_size)
            assert paras_queue is not None
            PerProcessContext().set_paras_queue(paras_queue)

        # Make PerProcessContext read-only.
        PerProcessContext().finalize()

        # Automatically set up some configs for remote unroller and trainer if needed
        _setup_remote_configs_if_needed()

        # Parse the configuration file, which will also implicitly bring up the environments.
        common.parse_conf_file(conf_file)
        _train(root_dir, rank, world_size)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if world_size >= 1:
            # If the training worker is running as a process in multiprocessing
            # environment, this will make sure that the exception raised in this
            # particular process is captured and shown.
            logging.exception(f'{mp.current_process().name} - {e}')
        raise e
    finally:
        # Note that each training worker will have its own child processes
        # running the environments. In the case when training worker process
        # finishes earlier (e.g. when it raises an exception), it will hang
        # instead of quitting unless all child processes are killed.
        alf.close_env()


def main(_):
    root_dir = common.abs_path(FLAGS.root_dir)
    os.makedirs(root_dir, exist_ok=True)

    conf_file = common.get_conf_file()

    if FLAGS.store_snapshot:
        common.generate_alf_snapshot(common.alf_root(), conf_file, root_dir)

    # FLAGS.distributed is guaranteed to be one of the possible values.
    if FLAGS.distributed == 'none':
        training_worker(
            rank=0, world_size=1, conf_file=conf_file, root_dir=root_dir)
    elif FLAGS.distributed == 'multi-gpu':
        world_size = torch.cuda.device_count()

        if world_size == 1:
            logging.warn(
                'Fallback to single GPU mode as there is only one GPU')
            training_worker(
                rank=0, world_size=1, conf_file=conf_file, root_dir=root_dir)
            return

        os.environ['MASTER_ADDR'] = 'localhost'

        try:
            # Create a shared queue for checking the consistency of the parameters
            # in different work processes.
            manager = mp.Manager()
            paras_queue = manager.Queue()
            with common.get_unused_port(12355) as port:
                # The other process will communicate with the authoritative
                # process via network protocol on localhost:port.
                os.environ['MASTER_PORT'] = str(port)
                # We spawn the processes for rank-1 and above and use the main
                # process for rank-0 so that we can request debug session
                # for the main process. We need to do this because the debug
                # session cannot be started in a subprocess.
                context = mp.spawn(
                    _training_worker_helper,
                    args=(world_size, conf_file, root_dir, paras_queue),
                    join=False,
                    nprocs=world_size - 1,
                    start_method='spawn')
                training_worker(0, world_size, conf_file, root_dir,
                                paras_queue)
                context.join()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            # ``e`` has been printed in the subprocess, so here we won't print it
            # again. But we raise another error so that we will have a correct
            # exit code for the program.
            raise ChildProcessError(f'Training failed on subprocess exception')


if __name__ == '__main__':
    __spec__ = None  # see https://github.com/HorizonRobotics/alf/pull/1554 for explanation
    _define_flags()
    app.run(main)
