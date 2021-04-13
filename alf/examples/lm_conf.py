# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""A simple language modeling example using Transformer.

You can use the following command to run it:

.. code-block:: bash

    python -m alf.bin.train --conf lm_conf.py --root_dir ~/tmp/lm/exp0

Note: You need to first install torchtext using ``pip install torchtext``

"""
from functools import partial
import math
import torch
import torch.nn as nn
from typing import Callable

import alf
from alf import layers, networks
from alf.algorithms.config import TrainerConfig
from alf.algorithms.algorithm import Algorithm, LossInfo
from alf.utils import common


@alf.configurable
def create_model(ntokens,
                 embedding_dim=200,
                 memory_size=128,
                 num_layers=4,
                 num_heads=4):
    embedding_layer = torch.nn.Embedding(ntokens, embedding_dim)
    embedding_layer.weight.data.uniform_(-0.1, 0.1)
    return networks.Sequential(
        embedding_layer,
        layers.Reshape(1, -1),
        networks.TransformerNetwork(
            input_tensor_spec=alf.TensorSpec((1, embedding_dim)),
            num_prememory_layers=0,
            num_attention_heads=num_heads,
            d_ff=4 * embedding_dim,
            core_size=1,
            use_core_embedding=False,
            memory_size=memory_size,
            num_memory_layers=num_layers,
            centralized_memory=False),
        layers.FC(embedding_dim, ntokens, kernel_init_gain=0.),
        input_tensor_spec=alf.TensorSpec((), dtype=torch.int64),
    )


@alf.configurable
class LMAlgorithm(Algorithm):
    def __init__(self, data_creator, optimizer, config: TrainerConfig):
        """
        Args:
            data_creator (Callable): called as ``data_creator()`` to get a tuple
                of (train_data, val_data, test_data, vocab)
        """
        self._train_data, self._val_data, self._test_data, self._vocab = data_creator(
        )
        ntokens = len(self._vocab.stoi)  # the size of vocabulary
        model = create_model(ntokens)
        super().__init__(
            train_state_spec=model.state_spec,
            optimizer=optimizer,
            config=config,
            debug_summaries=True,
            name="LM")
        self._model = model
        self._lossf = nn.CrossEntropyLoss(reduction='none')

    def train_iter(self):
        self._model.train()
        unroll_length = self._config.unroll_length
        num_batches = (self._train_data.shape[0] - 1) // unroll_length
        state = self.get_initial_train_state(self._train_data.shape[1])

        for batch in range(num_batches):
            alf.summary.increment_global_counter()
            loss = []
            for t in range(batch * unroll_length, (batch + 1) * unroll_length):
                data = self._train_data[t]
                target = self._train_data[t + 1]
                output, state = self._model(data, state)
                l = self._lossf(output, target)
                loss.append(l)

            state = common.detach(state)
            loss = torch.stack(loss)
            ppl = loss.mean().exp()
            loss_info = LossInfo(loss=loss, extra=dict(ppl=ppl))
            loss_info, params = self.update_with_gradient(loss_info)
            self.summarize_train(None, None, loss_info, params)

    @torch.no_grad()
    def evaluate(self):
        self._model.eval()  # Turn on the evaluation mode
        total_loss = 0.
        length = self._val_data.size(0) - 1
        state = self.get_initial_train_state(self._val_data.shape[1])
        for t in range(0, length):
            data = self._val_data[t]
            target = self._val_data[t + 1]
            output, state = self._model(data, state)
            total_loss += self._lossf(output, target).mean().item()
        total_loss /= length
        with alf.summary.scope("validation"):
            alf.summary.scalar("loss", total_loss)
            alf.summary.scalar("ppl", math.exp(total_loss))


alf.config('TransformerBlock', dropout=0.2)
alf.config('TransformerBlock', activation=torch.nn.functional.gelu)
alf.config('load_wikitext2', train_bs=128, test_bs=10)

alf.config(
    'LMAlgorithm',
    data_creator=alf.utils.datagen.load_wikitext2,
    optimizer=alf.optimizers.Adam(lr=1e-3),
)

alf.config(
    'TrainerConfig',
    ml_type='sl',
    algorithm_ctor=LMAlgorithm,
    unroll_length=16,
    evaluate=True,
    eval_interval=1,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=200,
    num_iterations=10,
)
