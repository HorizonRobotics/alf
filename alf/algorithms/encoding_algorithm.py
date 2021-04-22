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
"""Encoding algorithm."""

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, Experience, LossInfo, TimeStep
from alf.networks import EncodingNetwork
from alf.nest import map_structure, flatten
from alf.nest.utils import get_nested_field


@alf.configurable
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
                 output_fields=None,
                 loss_fields=None,
                 loss_weights=None,
                 optimizer=None,
                 debug_summaries=False,
                 name="EncodingAlgorithm"):
        """

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): not used
            encoder_cls (type): The class or function to create the encoder. It
                can be called using ``encoder_cls(input_tensor_spec)``.
            output_fields (None | nested str): if None, all the output from the
                encoder will be used as the output. Otherwise, ``output_fields``
                will be used to retrieve the fields from the encoder output.
            loss_fields (None | nested str): there is not lsss for this is None.
                Otherwise, ``loss_fields`` will be used to retrieve fields from
                encoder output and use them as loss. Note that those fields must
                be scalar.
            loss_weights (None | nested str): if provided, must have the same
                structure as ``loss_fields`` and will be used as weights for
                the corresponding loss values.
            optimizer (torch.optim.Optimizer): if provided, will be used to optimize
                the parameters of encoder.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """
        encoder = encoder_cls(input_tensor_spec=observation_spec)
        super().__init__(
            train_state_spec=encoder.state_spec,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._encoder = encoder
        output_spec = encoder.output_spec
        if output_fields is not None:
            self._output_spec = get_nested_field(output_spec, output_fields)
        if loss_fields is not None:
            # make sure loss_fields can be found in output_spec
            loss_specs = get_nested_field(output_spec, loss_fields)
            assert all(
                flatten(
                    map_structure(lambda spec: spec.shape is (), loss_specs))
            ), ("The losses should be scalars. Got: %s" % str(loss_specs))
        if loss_weights is not None:
            alf.nest.assert_same_structure(loss_weights, loss_fields)
        self._output_fields = output_fields
        self._loss_fields = loss_fields
        self._loss_weights = loss_weights

    @property
    def output_spec(self):
        return self._output_spec

    def _step(self, time_step, state=()):
        output, state = self._encoder(time_step.observation, state=state)
        if self._loss_fields is not None:
            losses = get_nested_field(output, self._loss_fields)
            if self._loss_weights is not None:
                loss = sum(
                    flatten(
                        map_structure(lambda w, l: w * l, self._loss_weights,
                                      losses)))
            else:
                loss = sum(flatten(losses))
            info = LossInfo(loss=loss, extra=losses)
        else:
            info = LossInfo()
        if self._output_fields is not None:
            output = get_nested_field(output, self._output_fields)
        return AlgStep(output=output, state=state, info=info)

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
