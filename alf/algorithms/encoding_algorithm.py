# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
"""Encoding algorithm."""

import gin

from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, Experience, LossInfo, TimeStep
from alf.networks import EncodingNetwork


@gin.configurable
class EncodingAlgorithm(Algorithm):
    """Basic encoding algorithm.

    This is just a wrapper for ``EncodingNetwork``. It also serves as a standard
    interface for other representation learning algorithms. It does not have any
    loss. So it relies on the downstream tasks to train the model.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 encoder_cls=EncodingNetwork,
                 debug_summaries=False,
                 name="EncodingAlgorithm"):
        """

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): not used
            encoder_cls (type): The class or function to create the encoder. It
                can be called using ``encoder_cls(input_tensor_spec)``.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """
        encoder = encoder_cls(input_tensor_spec=observation_spec)
        super().__init__(
            train_state_spec=encoder.state_spec,
            debug_summaries=debug_summaries,
            name=name)

        self._encoder = encoder
        self._output_spec = encoder.output_spec

    @property
    def output_spec(self):
        return self._output_spec

    def _step(self, time_step, state=()):
        output, state = self._encoder(time_step.observation, state=state)
        return AlgStep(output=output, state=state, info=LossInfo())

    def predict_step(self, time_step: TimeStep, state):
        return self._step(time_step, state)

    def rollout_step(self, time_step, state=()):
        """Train one step.

        Args:
            time_step (TimeStep|Experience): time step structure
            state (nested Tensor): network state for ``encoder``
        Returns:
            AlgStep:
            - output: encoding result
            - state: rnn state from ``encoder``
            - info: LossInfo
        """
        return self._step(time_step, state)

    def train_step(self, exp: Experience, state):
        return self._step(exp, state)
