# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import functools
import datetime
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import torch

import alf
from alf.algorithms.hypernetwork_algorithm import HyperNetwork
from alf.algorithms.rl_algorithm_test import MyEnv, MyAlg
from alf.trainers.policy_trainer import (RLTrainer, Trainer, TrainerConfig,
                                         TrainerProgress, play)
from alf.trainers.policy_trainer import SLTrainer
from alf.utils import common, datagen


def env_load(env_name, batch_size):
    return MyEnv(3)


env_load.batched = True


def _checkpoint_request_worker(rank, init_file):
    alf.set_default_device('cpu')
    torch.distributed.init_process_group(
        'gloo', init_method='file://' + init_file, rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=30))
    try:
        trainer = Trainer.__new__(Trainer)
        trainer._algorithm = SimpleNamespace(
            _distributed_strategy='fsdp2', _fsdp2_performer=object())
        # HTTP targets rank 0; SIGUSR2 can target either rank. Both workers
        # must take the save branch together, and a cleared request stays clear.
        for requesting_rank in (None, 0, None, 1, None):
            trainer._checkpoint_requested = rank == requesting_rank
            requested = trainer._sync_checkpoint_request()
            assert requested == (requesting_rank is not None)
            assert not trainer._checkpoint_requested
    finally:
        torch.distributed.destroy_process_group()


class TrainerTest(alf.test.TestCase):

    def test_fsdp2_checkpoint_requests_are_collective(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _checkpoint_request_worker,
                args=(os.path.join(directory, 'init'),), nprocs=2, join=True)

    def test_ddp_checkpoint_request_does_not_add_collectives(self):
        trainer = Trainer.__new__(Trainer)
        trainer._algorithm = SimpleNamespace(_distributed_strategy='ddp')
        trainer._checkpoint_requested = True
        with mock.patch('torch.distributed.all_reduce') as all_reduce:
            self.assertTrue(trainer._sync_checkpoint_request())
        all_reduce.assert_not_called()
        self.assertFalse(trainer._checkpoint_requested)

    def test_fsdp2_late_checkpoint_request_waits_for_next_collective(self):
        trainer = Trainer.__new__(Trainer)
        trainer._algorithm = SimpleNamespace(
            _distributed_strategy='fsdp2', _fsdp2_performer=object())
        trainer._checkpoint_requested = False

        def request_during_collective(*args, **kwargs):
            trainer._request_checkpoint(None, None)

        with mock.patch('torch.distributed.all_reduce',
                        side_effect=request_during_collective):
            self.assertFalse(trainer._sync_checkpoint_request())
        self.assertTrue(trainer._checkpoint_requested)
        with mock.patch('torch.distributed.all_reduce'):
            self.assertTrue(trainer._sync_checkpoint_request())
        self.assertFalse(trainer._checkpoint_requested)

    def test_reset_progress_loads_before_first_train_iter(self):
        progress = TrainerProgress()
        progress.set_termination_criterion(num_iterations=100)
        metric = mock.Mock()
        algorithm = mock.Mock(loaded=False)
        algorithm.is_rl.return_value = True
        algorithm.get_metrics.return_value = [metric]
        algorithm.train_iter.side_effect = lambda: self.assertTrue(
            algorithm.loaded)

        def load(**kwargs):
            algorithm.loaded = True
            progress._iter_num.fill_(50)
            progress._env_steps.fill_(1000)
            return 50

        checkpointer = mock.Mock()
        checkpointer.has_checkpoint.return_value = True
        checkpointer.has_replay_buffer_checkpoint.return_value = False
        checkpointer.load.side_effect = load
        trainer = Trainer.__new__(Trainer)
        trainer._algorithm = algorithm
        trainer._config = mock.Mock(reset_trainer_progress=True)
        trainer._trainer_progress = progress
        trainer._rank = -1
        trainer._root_dir = ""

        trainer._restore_checkpoint(checkpointer)

        self.assertEqual(int(progress._iter_num), 0)
        self.assertEqual(int(progress._env_steps), 0)
        self.assertEqual(alf.summary.get_global_counter(), 0)
        metric.reset.assert_called_once()
        algorithm.train_iter.assert_not_called()

        algorithm.train_iter()
        algorithm.train_iter.assert_called_once()

    def test_rl_trainer(self):
        with tempfile.TemporaryDirectory() as root_dir:
            alf.config("create_environment", env_load_fn=env_load)
            conf = TrainerConfig(algorithm_ctor=MyAlg,
                                 root_dir=root_dir,
                                 unroll_length=5,
                                 num_iterations=100)

            # test train
            trainer = RLTrainer(conf)
            self.assertEqual(RLTrainer.progress(), 0)
            trainer.train()
            self.assertEqual(RLTrainer.progress(), 1)

            alg = trainer._algorithm
            env = common.get_env()
            time_step = common.get_initial_time_step(env)
            state = alg.get_initial_predict_state(env.batch_size)
            policy_step = alg.rollout_step(time_step, state)
            logits = policy_step.info['dist'].logits
            print("logits: ", logits)
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 0]))
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 2]))

            # test checkpoint
            conf.num_iterations = 200
            new_trainer = RLTrainer(conf)
            new_trainer._restore_checkpoint()
            self.assertEqual(RLTrainer.progress(), 0.5)
            time_step = common.get_initial_time_step(env)
            state = alg.get_initial_predict_state(env.batch_size)
            policy_step = alg.rollout_step(time_step, state)
            logits = policy_step.info['dist'].logits
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 0]))
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 2]))

            new_trainer.train()
            self.assertEqual(RLTrainer.progress(), 1)

            # TODO: test play. Need real env to test.

    def test_sl_trainer(self):
        with tempfile.TemporaryDirectory() as root_dir:
            conf = TrainerConfig(algorithm_ctor=functools.partial(
                HyperNetwork,
                data_creator=datagen.load_test,
                hidden_layers=None,
                loss_type='regression',
                num_train_classes=1,
                optimizer=alf.optimizers.Adam(lr=1e-4, weight_decay=1e-4)),
                                 root_dir=root_dir,
                                 num_checkpoints=1,
                                 evaluate=True,
                                 eval_interval=1,
                                 num_iterations=1)

            # test train
            trainer = SLTrainer(conf)
            self.assertEqual(SLTrainer.progress(), 0)
            trainer.train()
            self.assertEqual(SLTrainer.progress(), 1)

            # test checkpoint
            conf2 = TrainerConfig(algorithm_ctor=functools.partial(
                HyperNetwork,
                data_creator=datagen.load_test,
                hidden_layers=None,
                loss_type='regression',
                num_train_classes=1,
                optimizer=alf.optimizers.Adam(lr=1e-4, weight_decay=1e-4)),
                                  root_dir=root_dir,
                                  num_checkpoints=1,
                                  evaluate=True,
                                  eval_interval=1,
                                  num_iterations=2)

            new_trainer = SLTrainer(conf2)
            new_trainer._restore_checkpoint()
            self.assertEqual(SLTrainer.progress(), 0.5)
            new_trainer.train()
            self.assertEqual(SLTrainer.progress(), 1)


if __name__ == "__main__":
    alf.test.main()
