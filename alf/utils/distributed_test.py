# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import copy
import datetime
import math
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import alf
from alf.data_structures import StepType, TimeStep
from alf.metrics import AverageReturnMetric
from alf.utils import common, summary_utils
from alf.utils.distributed import (FSDP2_OPTIMIZER_STATE, data_distributed,
                                   fsdp2_full_state_dict,
                                   load_fsdp2_full_state_dict)
from alf.utils.per_process_context import PerProcessContext


class _DistributedModel(torch.nn.Module):

    def __init__(self, rank, strategy):
        super().__init__()
        # Deliberately initialize each rank differently. DDP has always made
        # rank 0 authoritative, and FSDP2 must retain that behavior.
        torch.manual_seed(1234 + rank)
        self.linear = torch.nn.Linear(4, 2, bias=False)
        self._optimizers = []
        self._ddp_activated_rank = rank
        self._distributed_strategy = strategy

    def optimizers(self):
        return self._optimizers

    @data_distributed
    def compute(self, inputs):
        return self.linear(inputs)

    @data_distributed
    def compute_twice(self, inputs):
        return 2 * self.linear(inputs)


class _ShardPlanModel(torch.nn.Module):

    def __init__(self, rank):
        super().__init__()
        torch.manual_seed(4321 + rank)
        self.block0 = torch.nn.Linear(4, 4, bias=False)
        self.block1 = torch.nn.Linear(4, 4, bias=False)
        self.root_scale = torch.nn.Parameter(torch.ones(4))
        self._optimizers = []
        self._ddp_activated_rank = rank
        self._distributed_strategy = 'fsdp2'

    def optimizers(self):
        return self._optimizers

    @data_distributed
    def compute(self, inputs):
        return self.block1(torch.relu(self.block0(inputs))) * self.root_scale


def _test_shard_plan(model):
    return [model.block0, model.block1]


class _ToyTransformerBlock(torch.nn.Module):

    def __init__(self, d_model=8, num_heads=2, hidden_size=16):
        super().__init__()
        assert d_model % num_heads == 0
        self._num_heads = num_heads
        self._head_size = d_model // num_heads
        self.input_norm = torch.nn.LayerNorm(d_model)
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.attention_output = torch.nn.Linear(d_model, d_model)
        self.output_norm = torch.nn.LayerNorm(d_model)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden_size), torch.nn.GELU(),
            torch.nn.Linear(hidden_size, d_model))
        self.saw_unsharded_parameters = False

    def forward(self, inputs):
        # The block's FSDP pre-forward hook must all-gather its parameters
        # before control reaches the block itself.
        from torch.distributed.tensor import DTensor
        self.saw_unsharded_parameters = not isinstance(self.qkv.weight,
                                                       DTensor)
        batch_size, sequence_length, _ = inputs.shape
        qkv = self.qkv(self.input_norm(inputs))
        query, key, value = qkv.chunk(3, dim=-1)

        def _split_heads(tensor):
            return tensor.view(batch_size, sequence_length, self._num_heads,
                               self._head_size).transpose(1, 2)

        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self._head_size)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch_size, sequence_length, -1)
        inputs = inputs + self.attention_output(context)
        return inputs + self.feed_forward(self.output_norm(inputs))


class _ToyTransformer(torch.nn.Module):

    def __init__(self,
                 rank,
                 d_model=8,
                 num_heads=2,
                 hidden_size=16,
                 num_blocks=2,
                 vocab_size=16,
                 sequence_length=4):
        super().__init__()
        torch.manual_seed(2468 + rank)
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Parameter(
            torch.randn(sequence_length, d_model))
        self.blocks = torch.nn.ModuleList([
            _ToyTransformerBlock(d_model, num_heads, hidden_size)
            for _ in range(num_blocks)
        ])
        self.output_norm = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self._optimizers = []
        self._ddp_activated_rank = rank
        self._distributed_strategy = 'fsdp2'
        self.shard_plan_was_called = False

    def optimizers(self):
        return self._optimizers

    def forward(self, tokens):
        hidden = self.token_embedding(tokens) + self.position_embedding
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.output_norm(hidden))

    @data_distributed
    def compute(self, tokens):
        return self(tokens)


def _toy_transformer_shard_plan(model):
    # alf.config() stores this callback before the model exists. This mutation
    # verifies that it is later invoked with the instantiated root model.
    model.shard_plan_was_called = True
    return reversed(model.blocks)


def _distributed_worker(rank, world_size, init_file, strategy):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl',
                            init_method='file://' + init_file,
                            rank=rank,
                            world_size=world_size)
    try:
        device = torch.device('cuda', rank)
        model = _DistributedModel(rank, strategy).to(device)

        # Capture rank 0's initialization as the synchronization reference.
        initial_weight = model.linear.weight.detach().clone()
        dist.broadcast(initial_weight, src=0)

        # Off-policy ALF setup can populate an optimizer before the first
        # distributed call. FSDP2 must rebind those original Parameters to the
        # DTensors installed while sharding the module.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        model._optimizers.append(optimizer)
        inputs = torch.full((3, 4), float(rank + 1), device=device)
        outputs = model.compute(inputs)

        if strategy == 'fsdp2':
            from torch.distributed.tensor import DTensor
            assert isinstance(model.linear.weight, DTensor)
            assert optimizer.param_groups[0]['params'][
                0] is model.linear.weight
            assert model.linear.weight.to_local().numel() * world_size == \
                model.linear.weight.numel()

        outputs.sum().backward()
        if strategy == 'fsdp2':
            gradient = model.linear.weight.grad
            assert isinstance(gradient, DTensor)
            assert gradient.to_local().numel() * world_size == \
                gradient.numel()
        optimizer.step()
        if strategy == 'fsdp2':
            momentum = optimizer.state[model.linear.weight]['momentum_buffer']
            assert isinstance(momentum, DTensor)
            assert momentum.to_local().numel() * world_size == momentum.numel()

        # DDP averages the per-rank gradients. Each weight element has local
        # gradient 3 * (rank + 1), i.e. 4.5 after averaging both ranks.
        expected_weight = initial_weight - 0.45
        if strategy == 'fsdp2':
            performer = model._fsdp2_performer
            performer.unshard()
            torch.testing.assert_close(model.linear.weight, expected_weight)
            performer.reshard()
        else:
            torch.testing.assert_close(model.linear.weight, expected_weight)

        # A second decorated method must reuse the same FSDP root and gather
        # the newly updated parameters correctly.
        second_outputs = model.compute_twice(inputs)
        expected_outputs = 2 * inputs.matmul(expected_weight.t())
        torch.testing.assert_close(second_outputs, expected_outputs)
        if strategy == 'fsdp2':
            assert model._fsdp2_performer is performer
            state = fsdp2_full_state_dict(model)
            if rank == 0:
                saved_weight = state['linear.weight']
                assert not isinstance(saved_weight, DTensor)
                torch.testing.assert_close(saved_weight, expected_weight.cpu())
                assert state[FSDP2_OPTIMIZER_STATE]['state']
            state_object = [state if rank == 0 else None]
            dist.broadcast_object_list(state_object, src=0, device=device)
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
            incompatible = load_fsdp2_full_state_dict(model, state_object[0])
            assert not incompatible.missing_keys
            assert not incompatible.unexpected_keys
    finally:
        dist.destroy_process_group()


def _shard_plan_worker(rank, world_size, init_file):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl',
                            init_method='file://' + init_file,
                            rank=rank,
                            world_size=world_size)
    try:
        alf.config('make_fsdp2_performer', shard_plan=_test_shard_plan)
        device = torch.device('cuda', rank)
        model = _ShardPlanModel(rank).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model._optimizers.append(optimizer)
        output = model.compute(torch.randn(3, 4, device=device))

        from torch.distributed.tensor import DTensor
        for parameter in (model.block0.weight, model.block1.weight,
                          model.root_scale):
            assert isinstance(parameter, DTensor)
            assert parameter.to_local().numel() * world_size == \
                parameter.numel()

        output.sum().backward()
        optimizer.step()

        # Unsharding the root gathers only root-owned parameters. Child block
        # parameters remain sharded because they are independent FSDP groups.
        performer = model._fsdp2_performer
        performer.unshard()
        assert not isinstance(model.root_scale, DTensor)
        assert isinstance(model.block0.weight, DTensor)
        assert isinstance(model.block1.weight, DTensor)
        performer.reshard()

        state = fsdp2_full_state_dict(model)
        if rank == 0:
            assert 'block0.weight' in state
            assert 'block1.weight' in state
            assert 'root_scale' in state
            assert state[FSDP2_OPTIMIZER_STATE]['state']

        # Evaluation metrics produced by independent rank-local environments
        # are merged into the rank-0 metric buffer.
        example_time_step = TimeStep(step_type=torch.tensor([StepType.FIRST]),
                                     reward=torch.zeros(1),
                                     discount=torch.ones(1),
                                     observation=torch.zeros(1, 1),
                                     prev_action=torch.zeros(1),
                                     env_id=torch.zeros(1, dtype=torch.int32),
                                     env_info={})
        metric = AverageReturnMetric(buffer_size=world_size,
                                     example_time_step=example_time_step)
        metric._buffer.append(torch.tensor([float(rank + 1)]))
        from alf.trainers import policy_trainer  # noqa: F401
        from alf.trainers.evaluator import _merge_distributed_metrics
        merged = _merge_distributed_metrics([metric], episode_limit=1)
        if rank == 0:
            torch.testing.assert_close(merged[0].result(), torch.tensor(1.5))
        else:
            assert merged is None
    finally:
        dist.destroy_process_group()


def _toy_transformer_worker(rank, world_size, init_file):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl',
                            init_method='file://' + init_file,
                            rank=rank,
                            world_size=world_size)
    try:
        alf.config('make_fsdp2_performer',
                   shard_plan=_toy_transformer_shard_plan)
        device = torch.device('cuda', rank)
        model = _ToyTransformer(rank).to(device)
        reference = copy.deepcopy(model)

        # FSDP2 makes rank 0's initialization authoritative when it is first
        # wrapped. Make the unsharded reference model use that same state.
        for parameter in reference.parameters():
            dist.broadcast(parameter.detach(), src=0)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        reference_optimizer = torch.optim.SGD(reference.parameters(),
                                              lr=0.05,
                                              momentum=0.9)
        model._optimizers.append(optimizer)

        tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        if rank == 0:
            tokens.random_(16)
        dist.broadcast(tokens, src=0)

        logits = model.compute(tokens)
        reference_logits = reference(tokens)
        assert model.shard_plan_was_called
        torch.testing.assert_close(logits, reference_logits)

        from torch.distributed.tensor import DTensor
        for block in model.blocks:
            assert block.saw_unsharded_parameters
        for parameter in model.parameters():
            assert isinstance(parameter, DTensor)
            assert parameter.to_local().numel() * world_size == \
                parameter.numel()

        loss = logits.square().mean()
        reference_loss = reference_logits.square().mean()
        loss.backward()
        reference_loss.backward()

        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            assert isinstance(parameter.grad, DTensor)
            assert parameter.grad.to_local().numel() * world_size == \
                parameter.grad.numel()
            torch.testing.assert_close(parameter.grad.full_tensor(),
                                       reference_parameters[name].grad,
                                       rtol=2e-5,
                                       atol=2e-6)

        optimizer.step()
        reference_optimizer.step()
        for name, parameter in model.named_parameters():
            momentum = optimizer.state[parameter]['momentum_buffer']
            assert isinstance(momentum, DTensor)
            assert momentum.to_local().numel() * world_size == \
                momentum.numel()
            reference_momentum = reference_optimizer.state[
                reference_parameters[name]]['momentum_buffer']
            torch.testing.assert_close(momentum.full_tensor(),
                                       reference_momentum,
                                       rtol=2e-5,
                                       atol=2e-6)

        # Root unsharding must materialize only root-owned embeddings and the
        # output head, leaving independently sharded Transformer blocks alone.
        performer = model._fsdp2_performer
        performer.unshard()
        assert not isinstance(model.position_embedding, DTensor)
        assert not isinstance(model.token_embedding.weight, DTensor)
        assert isinstance(model.blocks[0].qkv.weight, DTensor)
        assert isinstance(model.blocks[1].qkv.weight, DTensor)
        performer.reshard()

        # Conversely, unsharding one block must not materialize either its
        # sibling or parameters owned by the root group.
        model.blocks[0].unshard()
        assert not isinstance(model.blocks[0].qkv.weight, DTensor)
        assert isinstance(model.blocks[1].qkv.weight, DTensor)
        assert isinstance(model.position_embedding, DTensor)
        model.blocks[0].reshard()

        state = fsdp2_full_state_dict(model)
        if rank == 0:
            reference_state = reference.state_dict()
            for name, value in reference_state.items():
                torch.testing.assert_close(state[name], value.cpu())
            assert state[FSDP2_OPTIMIZER_STATE]['state']
    finally:
        dist.destroy_process_group()


def _toy_transformer_memory_worker(rank, world_size, init_file, shard_blocks,
                                   result_queue):
    """Measure one isolated root-only or block-level FSDP2 training step."""
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl',
                            init_method='file://' + init_file,
                            rank=rank,
                            world_size=world_size)
    try:
        if shard_blocks:
            alf.config('make_fsdp2_performer',
                       shard_plan=_toy_transformer_shard_plan)
        device = torch.device('cuda', rank)
        model = _ToyTransformer(rank,
                                d_model=256,
                                num_heads=8,
                                hidden_size=1024,
                                num_blocks=6,
                                vocab_size=512,
                                sequence_length=16).to(device)
        tokens = torch.empty((2, 16), dtype=torch.int64, device=device)
        if rank == 0:
            tokens.random_(512)
        dist.broadcast(tokens, src=0)

        # Trigger lazy FSDP2 wrapping before measuring. Fresh subprocesses are
        # used for the two strategies so their allocator histories are
        # independent. memory_allocated() excludes cached but inactive blocks.
        with torch.no_grad():
            warmup_logits = model.compute(tokens)
        del warmup_logits
        torch.cuda.synchronize(device)
        baseline_bytes = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        logits = model.compute(tokens)
        logits.square().mean().backward()
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        local_result = {
            'rank': rank,
            'baseline_bytes': baseline_bytes,
            'peak_bytes': peak_bytes,
            'increment_bytes': peak_bytes - baseline_bytes,
        }
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_result)
        if rank == 0:
            result_queue.put(gathered_results)
    finally:
        dist.destroy_process_group()


def _fsdp2_summary_worker(rank, world_size, init_file, summary_dir):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl',
                            init_method='file://' + init_file,
                            rank=rank,
                            world_size=world_size,
                            timeout=datetime.timedelta(seconds=20))
    try:
        PerProcessContext().set_distributed(rank,
                                            rank,
                                            world_size,
                                            strategy='fsdp2')
        alf.config('make_fsdp2_performer', shard_plan=_test_shard_plan)
        alf.summary.enable_summary()
        device = torch.device('cuda', rank)
        model = _ShardPlanModel(rank).to(device)
        inputs = torch.ones((3, 4), device=device)
        model.compute(inputs).sum().backward()

        parameter = model.block0.weight
        full_parameter = parameter.full_tensor().detach()
        full_gradient = parameter.grad.full_tensor().detach()
        expected_parameter_norm = full_parameter.norm().item()
        expected_gradient_norm = full_gradient.norm().item()
        expected_sum = full_parameter.sum().item()
        expected_sum_squares = full_parameter.square().sum().item()
        expected_numel = full_parameter.numel()
        del full_parameter, full_gradient

        # Reduction over a sharded DTensor produces a scalar DTensor with a
        # Partial placement. Generic scalar summaries must reduce this on all
        # ranks without gathering any model parameter.
        partial_norm = parameter.norm()

        def _summarize():
            assert alf.summary.should_compute_summaries()
            assert alf.summary.should_write_summaries() == (rank == 1)
            named_parameter = [('block0.weight', parameter)]
            summary_utils.summarize_variables(named_parameter,
                                              with_histogram=True)
            summary_utils.summarize_gradients(named_parameter,
                                              with_histogram=True)
            alf.summary.scalar('custom_dtensor_norm', partial_norm)

        common.run_under_record_context(_summarize,
                                        summary_dir=summary_dir,
                                        summary_interval=1,
                                        flush_secs=1)
        dist.barrier()
        if rank == 0:
            from tensorboard.backend.event_processing import event_accumulator
            accumulator = event_accumulator.EventAccumulator(summary_dir)
            accumulator.Reload()
            scalar_tags = accumulator.Tags()['scalars']
            parameter_norm_tag = \
                'summarize_vars/block0.weight_value_norm'
            gradient_norm_tag = \
                'summarize_grads/block0.weight_gradient_norm'
            assert parameter_norm_tag in scalar_tags
            assert gradient_norm_tag in scalar_tags
            assert 'custom_dtensor_norm' in scalar_tags
            assert abs(
                accumulator.Scalars(parameter_norm_tag)[0].value -
                expected_parameter_norm) < 1e-5
            assert abs(
                accumulator.Scalars(gradient_norm_tag)[0].value -
                expected_gradient_norm) < 1e-5
            assert abs(
                accumulator.Scalars('custom_dtensor_norm')[0].value -
                expected_parameter_norm) < 1e-5

            histogram_tag = 'summarize_vars/block0.weight_value'
            assert histogram_tag in accumulator.Tags()['histograms']
            histogram = accumulator.Histograms(
                histogram_tag)[0].histogram_value
            assert histogram.num == expected_numel
            assert sum(histogram.bucket) == expected_numel
            assert abs(histogram.sum - expected_sum) < 1e-5
            assert abs(histogram.sum_squares - expected_sum_squares) < 1e-5
            event_files = [
                name for name in os.listdir(summary_dir)
                if name.startswith('events.out.tfevents')
            ]
            assert len(event_files) == 1
        dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2,
                     'two CUDA devices are required')
class DistributedTest(unittest.TestCase):

    def _run_strategy(self, strategy):
        fd, init_file = tempfile.mkstemp()
        os.close(fd)
        os.unlink(init_file)
        try:
            mp.spawn(_distributed_worker,
                     args=(2, init_file, strategy),
                     nprocs=2,
                     join=True)
        finally:
            if os.path.exists(init_file):
                os.unlink(init_file)

    def _measure_toy_transformer_memory(self, shard_blocks):
        fd, init_file = tempfile.mkstemp()
        os.close(fd)
        os.unlink(init_file)
        context = mp.get_context('spawn')
        result_queue = context.SimpleQueue()
        try:
            mp.spawn(_toy_transformer_memory_worker,
                     args=(2, init_file, shard_blocks, result_queue),
                     nprocs=2,
                     join=True)
            return result_queue.get()
        finally:
            if os.path.exists(init_file):
                os.unlink(init_file)

    def test_ddp_correctness(self):
        self._run_strategy('ddp')

    def test_fsdp2_correctness(self):
        self._run_strategy('fsdp2')

    def test_fsdp2_shard_plan(self):
        fd, init_file = tempfile.mkstemp()
        os.close(fd)
        os.unlink(init_file)
        try:
            mp.spawn(_shard_plan_worker,
                     args=(2, init_file),
                     nprocs=2,
                     join=True)
        finally:
            if os.path.exists(init_file):
                os.unlink(init_file)

    def test_fsdp2_toy_transformer_shard_plan(self):
        fd, init_file = tempfile.mkstemp()
        os.close(fd)
        os.unlink(init_file)
        try:
            mp.spawn(_toy_transformer_worker,
                     args=(2, init_file),
                     nprocs=2,
                     join=True)
        finally:
            if os.path.exists(init_file):
                os.unlink(init_file)

    def test_fsdp2_toy_transformer_peak_memory(self):
        root_results = self._measure_toy_transformer_memory(shard_blocks=False)
        block_results = self._measure_toy_transformer_memory(shard_blocks=True)
        root_peaks = [result['peak_bytes'] for result in root_results]
        block_peaks = [result['peak_bytes'] for result in block_results]
        root_peak = max(root_peaks)
        block_peak = max(block_peaks)
        saved_bytes = root_peak - block_peak
        saved_percent = 100 * saved_bytes / root_peak
        to_mib = lambda values: [round(value / 2**20, 2) for value in values]
        print(
            'FSDP2 toy Transformer peak CUDA allocated memory (MiB): '
            f'root-only per rank={to_mib(root_peaks)}, '
            f'block-level per rank={to_mib(block_peaks)}, '
            f'max-rank reduction={saved_bytes / 2**20:.2f} MiB '
            f'({saved_percent:.1f}%)',
            flush=True)
        self.assertLess(
            block_peak, root_peak,
            'block-level sharding should reduce peak allocated CUDA memory')

    def test_fsdp2_distributed_summaries(self):
        fd, init_file = tempfile.mkstemp()
        os.close(fd)
        os.unlink(init_file)
        try:
            with tempfile.TemporaryDirectory() as summary_dir:
                mp.spawn(_fsdp2_summary_worker,
                         args=(2, init_file, summary_dir),
                         nprocs=2,
                         join=True)
        finally:
            if os.path.exists(init_file):
                os.unlink(init_file)


if __name__ == '__main__':
    unittest.main()
