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

import tensorflow as tf

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks.network import Network
import tf_agents.specs.tensor_spec as tensor_spec

from alf.algorithms.algorithm import Algorithm, AlgorithmStep
from alf.utils.encoding_network import EncodingNetwork


class DecodingAlgorithm(Algorithm):
    """Generic decoding algorithm for 1-D continous output."""

    def __init__(self,
                 decoder: Network,
                 loss=tf.losses.mean_squared_error,
                 loss_weight=1.0):
        """Create a decoding algorithm.

        Args:
            decoder (Network)
            loss (Callable): loss function with signature loss(y_true, y_pred)
            loss_weight (float): weight for the loss
        """
        super(DecodingAlgorithm, self).__init__(
            train_state_spec=decoder.state_spec, name="DecodingAlgorithm")

        self._decoder = decoder
        self._loss = loss
        self._loss_weight = loss_weight

    def train_step(self, inputs, state=None):
        """Train one step.

        Args:
            inputs (tuple): tuple of (inputs, target)
            state (nested Tensor): network state for `decoder`

        Returns:
            AlgorithmStep with the following fields:
            outputs: decoding result
            state: rnn state
            info: loss of decoding

        """
        input, target = inputs
        pred, state = self._decoder(input, network_state=state)
        assert pred.shape == target.shape
        loss = self._loss(target, pred)
        return AlgorithmStep(
            outputs=pred,
            state=state,
            info=LossInfo(self._loss_weight * loss, extra=()))
