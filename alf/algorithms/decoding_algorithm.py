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
"""Decoding algorithm."""

import torch

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, LossInfo
from alf.networks import Network
from alf.utils.math_ops import sum_to_leftmost


@alf.configurable
class DecodingAlgorithm(Algorithm):
    """Generic decoding algorithm."""

    def __init__(self,
                 decoder: Network,
                 loss=torch.nn.MSELoss(reduction='none'),
                 loss_weight=1.0,
                 name="DecodingAlgorithm"):
        """

        Args:
            decoder (Network): network for decoding target from input.
            loss (Callable): loss function with signature ``loss(y_pred, y_true)``.
                Note that it should not reduce to a scalar. It should at least
                keep the batch dimension in the returned loss.
            loss_weight (float): weight for the loss.
        """
        super(DecodingAlgorithm, self).__init__(
            train_state_spec=decoder.state_spec, name=name)

        self._decoder = decoder
        self._loss = loss
        self._loss_weight = loss_weight

    def train_step(self, inputs, state=(), rollout_info=None):
        """Train one step.

        Args:
            inputs (tuple): tuple of (input, target)
            state (nested Tensor): network state for ``decoder``

        Returns:
            AlgStep:
            - output: decoding result
            - state: rnn state from ``decoder``
            - info: loss of decoding
        """
        input, target = inputs
        pred, state = self._decoder(input, state=state)
        assert pred.shape == target.shape
        loss = self._loss(pred, target)

        assert loss.ndim > 0, "`loss` should return a tensor with batch dimension"
        # reduce to (B,)
        loss = sum_to_leftmost(loss, 1)
        return AlgStep(
            output=pred,
            state=state,
            info=LossInfo(loss=self._loss_weight * loss))
