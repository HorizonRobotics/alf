# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for ALF's torchrun launch path."""

import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from absl.testing import flagsaver
import torch
import torch.distributed as dist

import alf
from alf.bin import train
from alf.utils.per_process_context import PerProcessContext


_SMOKE_CONF = '''
import alf
import torch
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.rl_algorithm_test import MyEnv
from alf.data_structures import AlgStep, LossInfo

class SmokeAlgorithm(RLAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(is_on_policy=False, train_state_spec=(),
                         optimizer=alf.optimizers.Adam(lr=0.01,
                                                       grad_accumulation_steps=2),
                         **kwargs)
        self.linear = torch.nn.Linear(2, 1)

    def rollout_step(self, time_step, state):
        return AlgStep(output=torch.zeros_like(time_step.prev_action))

    def train_step(self, time_step, state, rollout_info):
        prediction = self.linear(time_step.observation).squeeze(-1)
        return AlgStep(info=(prediction - time_step.reward).square())

    def calc_loss(self, info):
        return LossInfo(loss=info)

def load(env_name, batch_size):
    return MyEnv(batch_size)

load.batched = True
alf.config('create_environment', env_load_fn=load, num_parallel_environments=4)
alf.config('TrainerConfig',
           algorithm_ctor=SmokeAlgorithm,
           random_seed=123,
           num_iterations=3,
           unroll_length=4,
           initial_collect_steps=8,
           mini_batch_size=8,
           mini_batch_length=2,
           num_updates_per_train_iter=2,
           replay_buffer_length=16,
           whole_replay_buffer_training=False,
           clear_replay_buffer=False,
           num_checkpoints=1,
           summary_interval=1,
           confirm_checkpoint_upon_crash=False)
'''


def _cpu_smoke_worker():
    """Use the real launcher/trainer/DDP with Gloo on two simulated nodes."""
    sys.argv.remove('--cpu-smoke-worker')
    train._define_flags()
    original_init = dist.init_process_group
    original_train = train._train
    original_trainer = train.policy_trainer.RLTrainer
    original_save = train.policy_trainer.Checkpointer.save
    trainers = []

    def init_group(backend, **kwargs):
        return original_init('gloo', **kwargs)

    def make_trainer(*args, **kwargs):
        trainer = original_trainer(*args, **kwargs)
        initial = torch.cat([p.detach().flatten().clone()
                             for p in trainer._algorithm.parameters()])
        trainers.append((trainer, alf.get_env().batch_size, initial))
        return trainer

    def save(checkpointer, *args, **kwargs):
        rank = dist.get_rank()
        Path(train.FLAGS.root_dir, f'checkpoint-writer-{rank}').touch()
        return original_save(checkpointer, *args, **kwargs)

    def run_and_verify(**kwargs):
        original_train(**kwargs)
        trainer, batch_size, initial = trainers[0]
        parameters = torch.cat([
            parameter.detach().flatten()
            for parameter in trainer._algorithm.parameters()
        ])
        gathered = [torch.empty_like(parameters) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, parameters)
        for other in gathered:
            torch.testing.assert_close(parameters, other)
        assert not torch.equal(parameters, initial), 'No parameter updates occurred'
        context = PerProcessContext()
        result = dict(rank=context.ddp_rank, local_rank=context.local_rank,
                      world_size=context.num_processes,
                      strategy=context.distributed_strategy,
                      batch_size=batch_size,
                      mini_batch_size=trainer._config.mini_batch_size,
                      seed=trainer._config.random_seed,
                      iterations=int(trainer._trainer_progress._iter_num))
        Path(train.FLAGS.root_dir, f'rank-{context.ddp_rank}.json').write_text(
            json.dumps(result))

    with mock.patch.object(torch.cuda, 'is_available', return_value=False), \
            mock.patch.object(dist, 'init_process_group', side_effect=init_group), \
            mock.patch.object(train.policy_trainer, 'RLTrainer', side_effect=make_trainer), \
            mock.patch.object(train.policy_trainer.Checkpointer, 'save', new=save), \
            mock.patch.object(train, '_train', side_effect=run_and_verify):
        train.app.run(train.main)


class TrainTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if 'root_dir' not in train.FLAGS:
            train._define_flags()
        if not train.FLAGS.is_parsed():
            train.FLAGS(['train_test', '--root_dir=/tmp/alf-train-test'])

    def test_gpu_binding_restarts_before_using_cuda(self):
        for devices, rank, expected in (
                (None, 3, '3'), ('6,2', 1, '2'),
                ('GPU-first,GPU-second', 1, 'GPU-second'),
                ('MIG-first,MIG-second', 1, 'MIG-second')):
            with self.subTest(devices=devices), mock.patch.dict(os.environ):
                os.environ.pop('_ALF_MULTI_NODE_GPU_BOUND', None)
                os.environ['LOCAL_RANK'] = str(rank)
                if devices is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = devices
                with mock.patch.object(os, 'execv', side_effect=SystemExit) as restart, \
                        mock.patch.object(torch.cuda, 'device_count') as count:
                    with self.assertRaises(SystemExit):
                        train._bind_multi_node_gpu()
                    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], expected)
                    restart.assert_called_once_with(
                        sys.executable,
                        [sys.executable, '-m', 'alf.bin.train', *sys.argv[1:]])
                    count.assert_not_called()
                    # On restart, local_rank still refers to the full node,
                    # but CUDA_VISIBLE_DEVICES now contains only one GPU.
                    restart.reset_mock()
                    train._bind_multi_node_gpu()
                    restart.assert_not_called()

    def test_multi_node_worker_preserves_settings_and_cleans_up(self):
        with flagsaver.flagsaver(nccl_timeout=60, distributed_strategy='fsdp2',
                                 as_remote_trainer=True), \
                mock.patch.object(PerProcessContext, '_instance', None), \
                mock.patch.object(train, '_setup_logging'), \
                mock.patch.object(train, '_setup_device'), \
                mock.patch.object(train.logging, 'exception'), \
                mock.patch.object(train, '_define_flags') as define_flags, \
                mock.patch.object(dist, 'init_process_group') as init, \
                mock.patch.object(dist, 'is_initialized', return_value=True), \
                mock.patch.object(dist, 'destroy_process_group') as destroy, \
                mock.patch.object(alf, 'pre_config') as pre_config, \
                mock.patch.object(alf, 'close_env') as close, \
                mock.patch.object(train.common, 'parse_conf_file'), \
                mock.patch.object(train, '_train', side_effect=RuntimeError('training failed')):
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                train.training_worker(rank=8, local_rank=0, world_size=16,
                                      conf_file='test_conf.py', root_dir='/tmp')
            init.assert_called_once_with('nccl', rank=8, world_size=16,
                                         timeout=datetime.timedelta(minutes=60))
            context = PerProcessContext()
            self.assertEqual((context.ddp_rank, context.local_rank,
                              context.num_processes, context.distributed_strategy),
                             (8, 0, 16, 'fsdp2'))
            self.assertEqual(pre_config.call_args.args[0]['TrainerConfig.remote_training'],
                             'trainer')
            define_flags.assert_not_called()
            close.assert_called_once()
            destroy.assert_called_once()

    def test_two_torchrun_nodes_train_together(self):
        with tempfile.TemporaryDirectory() as directory:
            conf = Path(directory, 'smoke_conf.py')
            conf.write_text(_SMOKE_CONF)
            root = Path(directory, 'run')
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            env = os.environ.copy()
            # CPU transport test: GPU binding is tested separately.
            env.update(CUDA_VISIBLE_DEVICES='', _ALF_MULTI_NODE_GPU_BOUND='1',
                       ALF_USE_GIN='0', OMP_NUM_THREADS='1')
            processes = []
            logs = []
            try:
                for node in range(2):
                    log = open(Path(directory, f'node-{node}.log'), 'w+')
                    logs.append(log)
                    processes.append(subprocess.Popen([
                        sys.executable, '-m', 'torch.distributed.run',
                        '--nnodes=2', '--nproc-per-node=1', f'--node-rank={node}',
                        '--master-addr=127.0.0.1', f'--master-port={port}',
                        '--module', 'alf.bin.train_test', '--cpu-smoke-worker',
                        '--distributed=multi-node-multi-gpu', '--nostore_snapshot',
                        f'--root_dir={root}', f'--conf={conf}', '--port=0'],
                        env=env, stdout=log, stderr=subprocess.STDOUT))
                codes = [process.wait(timeout=90) for process in processes]
                for log in logs:
                    log.seek(0)
                self.assertEqual(codes, [0, 0], '\n'.join(log.read() for log in logs))
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                for log in logs:
                    log.close()
            results = [json.loads(Path(root, f'rank-{rank}.json').read_text())
                       for rank in range(2)]
            for rank, result in enumerate(results):
                self.assertEqual(result['rank'], rank)
                self.assertEqual(result['local_rank'], 0)
                self.assertEqual(result['world_size'], 2)
                self.assertEqual(result['strategy'], 'ddp')
                self.assertEqual(result['batch_size'], 2)
                self.assertEqual(result['mini_batch_size'], 4)
                self.assertEqual(result['iterations'], 3)
            self.assertNotEqual(results[0]['seed'], results[1]['seed'])
            self.assertTrue(Path(root, 'checkpoint-writer-0').exists())
            self.assertFalse(Path(root, 'checkpoint-writer-1').exists())
            self.assertEqual(len(list(root.rglob('events.out.tfevents.*'))), 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires one GPU')
    def test_torchrun_gpu_restart_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            conf = Path(directory, 'smoke_conf.py')
            conf.write_text(_SMOKE_CONF + '''
import os
assert os.environ['_ALF_MULTI_NODE_GPU_BOUND'] == '1'
assert torch.cuda.device_count() == 1
assert torch.cuda.current_device() == 0
''')
            root = Path(directory, 'run')
            env = os.environ.copy()
            env.pop('_ALF_MULTI_NODE_GPU_BOUND', None)
            env.update(CUDA_VISIBLE_DEVICES=str(torch.cuda.get_device_properties(0).uuid),
                       ALF_USE_GIN='0', OMP_NUM_THREADS='1')
            result = subprocess.run([
                sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nnodes=1', '--nproc-per-node=1', '--module', 'alf.bin.train',
                '--distributed=multi-node-multi-gpu', '--nostore_snapshot',
                f'--root_dir={root}', f'--conf={conf}', '--port=0'],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue(list(root.rglob('ckpt-3*')))
            # A torchrun job with one process should retain summaries.
            self.assertEqual(len(list(root.rglob('events.out.tfevents.*'))), 1)


if __name__ == '__main__':
    if '--cpu-smoke-worker' in sys.argv:
        _cpu_smoke_worker()
    else:
        unittest.main()
