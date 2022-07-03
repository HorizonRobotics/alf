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
import pathlib
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

from alf.utils import common
from alf.utils.per_process_context import PerProcessContext
import alf.utils.external_configurables
from alf.trainers import policy_trainer
from alf.config_util import get_config_value, get_operative_configs, get_inoperative_configs
from alf.config_helpers import get_env, parse_config_only


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
        'Set whether and how to run trainning in distributed mode.')
    flags.mark_flag_as_required('root_dir')


FLAGS = flags.FLAGS


def _setup_logging(rank: int, log_dir: str):
    """Setup logging for each process

    Args:
        rank (int): The ID of the process among all of the DDP processes
        log_dir (str): path to the direcotry where log files are written to
    """
    FLAGS.alsologtostderr = True
    logging.set_verbosity(logging.INFO)
    logging.get_absl_handler().use_absl_log_file(log_dir=log_dir)


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


def setup_wandb(trainer_conf, name='train'):
    assert name in ['train', 'eval']
    algorithm = get_config_value("Agent.rl_algorithm_cls").__name__
    env_name = get_config_value("create_environment.env_name")
    version = trainer_conf.version
    entity = trainer_conf.entity
    project = trainer_conf.project
    seed = trainer_conf.random_seed
    if trainer_conf.async_eval and name == 'eval':
        # When enabling async evaluation, the seed will be set differently
        # for the eval worker as per Line 187 of alf.trainers.evaluator.py
        seed -= 13579

    wandb_group = f"{algorithm}-{version}-{env_name}"
    wandb_name = f"seed-{seed}-{name}"

    wandb.tensorboard.patch(
        root_logdir=os.path.join(trainer_conf.root_dir, name))

    config = {k: v for k, v in get_operative_configs()}
    inoperative = {k: v for k, v in get_inoperative_configs()}
    config["inoperative"] = inoperative

    wandb.init(
        project=project,
        entity=entity,
        group=wandb_group,
        name=wandb_name,
        config=config,
        reinit=True,
        sync_tensorboard=True,
    )


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
    trainer_conf = policy_trainer.TrainerConfig(root_dir=root_dir)
    if trainer_conf.use_wandb:
        setup_wandb(trainer_conf)

    if trainer_conf.ml_type == 'rl':
        ddp_rank = rank if world_size > 1 else -1
        trainer = policy_trainer.RLTrainer(trainer_conf, ddp_rank)
    elif trainer_conf.ml_type == 'sl':
        # NOTE: SLTrainer does not support distributed training yet
        if world_size > 1:
            raise RuntimeError(
                "Multi-GPU DDP training does not support supervised learning")
        trainer = policy_trainer.SLTrainer(trainer_conf)
    else:
        raise ValueError("Unsupported ml_type: %s" % trainer_conf.ml_type)

    trainer.train()


def training_worker(rank: int, world_size: int, conf_file: str, root_dir: str):
    """An executable instance that trains and evaluate the algorithm

    Args:
        rank (int): The ID of the process among all of the DDP processes.
        world_size (int): The number of processes in total. If set to 1, it is
            interpreted as "non distributed mode".
        conf_file (str): Path to the training configuration.
        root_dir (str): Path to the directory for writing logs/summaries/checkpoints.
    """
    try:
        _setup_logging(log_dir=root_dir, rank=rank)
        _setup_device(rank)
        if world_size > 1:
            # Specialization for distributed mode
            dist.init_process_group('nccl', rank=rank, world_size=world_size)
            # Recover the flags when spawned as a sub process
            _define_flags()
            FLAGS(sys.argv, known_only=True)
            FLAGS.mark_as_parsed()
            # Set the rank and total number of processes for distributed training.
            PerProcessContext().set_distributed(
                rank=rank, num_processes=world_size)
        # Make PerProcessContext read-only.
        PerProcessContext().finalize()

        # Parse the configuration file, which will also implicitly bring up the environments.
        # common.parse_conf_file(conf_file)
        get_env()
        _train(root_dir, rank, world_size)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if world_size == 1:
            # raise it so the debugger can stop at the original place of exception
            raise e
        else:
            # If the training worker is running as a process in multiprocessing
            # environment, this will make sure that the exception raised in this
            # particular process is captured and shown.
            logging.exception(f'{mp.current_process().name} - {e}')
    finally:
        # Note that each training worker will have its own child processes
        # running the environments. In the case when training worker process
        # finishes ealier (e.g. when it raises an exception), it will hang
        # instead of quitting unless all child processes are killed.
        alf.close_env()


def create_log_dir(conf):
    algorithm = get_config_value("Agent.rl_algorithm_cls").__name__
    env_name = get_config_value("create_environment.env_name")
    version = conf.version
    seed = conf.random_seed

    return os.path.join(common.abs_path(FLAGS.root_dir), algorithm, version,
                        env_name, f'seed_{seed}')


def main(_):
    conf_params = getattr(flags.FLAGS, 'conf_param', None)
    conf_file = common.get_conf_file()
    parse_config_only(conf_file, conf_params)
    trainer_conf = policy_trainer.TrainerConfig(root_dir=FLAGS.root_dir)
    root_dir = create_log_dir(trainer_conf)

    os.makedirs(root_dir, exist_ok=True)

    if FLAGS.store_snapshot:
        common.generate_alf_root_snapshot(common.alf_root(), root_dir)

    # FLAGS.distributed is guaranteed to be one of the possible values.
    if FLAGS.distributed == 'none':
        training_worker(
            rank=0, world_size=1, conf_file=conf_file, root_dir=root_dir)
    elif FLAGS.distributed == 'multi-gpu':
        assert False, "Not considered yet!"
        world_size = torch.cuda.device_count()

        if world_size == 1:
            logging.warn(
                'Fallback to single GPU mode as there is only one GPU')
            training_worker(
                rank=0, world_size=1, conf_file=conf_file, root_dir=root_dir)
            return

        # The other process will communicate with the authoritative
        # process via network protocol on localhost:12355.
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        try:
            processes = mp.spawn(
                training_worker,
                args=(world_size, conf_file, root_dir),
                join=True,
                nprocs=world_size,
                start_method='spawn')
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.exception(f'Training failed on exception: {e}')


if __name__ == '__main__':
    _define_flags()
    app.run(main)
