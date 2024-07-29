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

from typing import Optional
from alf.algorithms.config import TrainerConfig
import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, LossInfo, TimeStep
from alf.networks import EncodingNetwork
from alf.nest import map_structure, flatten
from alf.nest.utils import get_nested_field


@alf.configurable
class EncodingAlgorithm(Algorithm):
    """Basic encoding algorithm.

    It uses the provided encoding network to computed the representation. It
    also supports the training of the encoding network by using some of its output
    as losses.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=alf.TensorSpec(()),
                 encoder_cls=EncodingNetwork,
                 time_step_as_input=False,
                 output_fields=None,
                 loss_fields=None,
                 loss_weights=None,
                 optimizer=None,
                 config: Optional[TrainerConfig] = None,
                 checkpoint=None,
                 debug_summaries=False,
                 name="EncodingAlgorithm"):
        """

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): not used
            encoder_cls (type): The class or function to create the encoder. It
                can be called using ``encoder_cls(input_tensor_spec)``.
            time_step_as_input (bool): If True, use the whole TimeStep structure
                as the input to the encoder instead of the observation.
            output_fields (None | nested str): if None, all the output from the
                encoder will be used as the output. Otherwise, ``output_fields``
                will be used to retrieve the fields from the encoder output.
            loss_fields (None | nested str): there is not loss if this is None.
                Otherwise, ``loss_fields`` will be used to retrieve fields from
                encoder output and use them as loss. Note that those fields must
                be scalar.
            loss_weights (None | nested str): if provided, must have the same
                structure as ``loss_fields`` and will be used as weights for
                the corresponding loss values.
            config: The trainer config. Present as representation learner
                interface to be used with ``Agent``.
            optimizer (torch.optim.Optimizer): if provided, will be used to optimize
                the parameters of encoder.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """
        if time_step_as_input:
            time_step_spec = alf.data_structures.time_step_spec(
                observation_spec, action_spec, reward_spec)
            encoder = encoder_cls(input_tensor_spec=time_step_spec)
        else:
            encoder = encoder_cls(input_tensor_spec=observation_spec)
        super().__init__(
            train_state_spec=encoder.state_spec,
            optimizer=optimizer,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._time_step_as_input = time_step_as_input
        self._encoder = encoder
        output_spec = encoder.output_spec
        if output_fields is not None:
            output_spec = get_nested_field(output_spec, output_fields)
        self._output_spec = output_spec
        if loss_fields is not None:
            # make sure loss_fields can be found in output_spec
            loss_specs = get_nested_field(output_spec, loss_fields)
            assert all(
                flatten(
                    map_structure(lambda spec: spec.shape == (), loss_specs))
            ), ("The losses should be scalars. Got: %s" % str(loss_specs))
        if loss_weights is not None:
            alf.nest.assert_same_structure(loss_weights, loss_fields)
        self._output_fields = output_fields
        self._loss_fields = loss_fields
        self._loss_weights = loss_weights

    @property
    def output_spec(self):
        return self._output_spec

    def predict_step(self, inputs: TimeStep, state):
        """override predict_step

        Args:
            inputs (TimeStep): time step structure
            state (nested Tensor): network state for ``encoder``
        Returns:
            AlgStep:
            - output: encoding result
            - state: rnn state from ``encoder``
        """
        if self._time_step_as_input:
            output, state = self._encoder(inputs, state=state)
        else:
            output, state = self._encoder(inputs.observation, state=state)
        if self._output_fields is not None:
            output = get_nested_field(output, self._output_fields)
        return AlgStep(output=output, state=state)

    def rollout_step(self, inputs, state):
        """override rollout_step

        Args:
            inputs (TimeStep): time step structure
            state (nested Tensor): network state for ``encoder``
        Returns:
            AlgStep:
            - output: encoding result
            - state: rnn state from ``encoder``
            - info: LossInfo
        """
        if not self.on_policy:
            return self.predict_step(inputs, state)
        else:
            return self.train_step(inputs, state, None)

    def train_step(self, inputs: TimeStep, state, rollout_info=None):
        """override train_step

        Args:
            inputs (TimeStep): time step structure
            state (nested Tensor): network state for ``encoder``
            rollout_info: not used
        Returns:
            AlgStep:
            - output: encoding result
            - state: rnn state from ``encoder``
            - info: LossInfo
        """
        if self._time_step_as_input:
            output, state = self._encoder(inputs, state=state)
        else:
            output, state = self._encoder(inputs.observation, state=state)
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
