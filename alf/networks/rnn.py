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

import copy
import gin
import torch
import torch.nn as nn

import alf
from .network import Network


@gin.configurable(whitelist=['output_layers'])
class StackedLSTMCell(Network):
    def __init__(self,
                 input_size,
                 lstm_size,
                 output_layers=None,
                 name="StackedLSTMCELL"):
        """Stacked LSTM Cell.

        Args:
            input_size (int): input dimension
            lstm_size (int|list[int]): specifying the LSTM cell sizes to use.
            output_layers (None|int|list[int]): -1 means the output from the last
                layer. ``None`` means all layers.
            name (str):
        """
        super().__init__(
            input_tensor_spec=alf.TensorSpec(shape=(input_size, )), name=name)

        self._cells = nn.ModuleList()
        self._state_spec = []
        if output_layers is None:
            output_layers = list(range(len(lstm_size)))
        elif type(output_layers) == int:
            output_layers = [output_layers]
        self._output_layers = copy.copy(output_layers)
        for hidden_size in lstm_size:
            self._cells.append(nn.LSTMCell(input_size, hidden_size))
            spec = alf.TensorSpec(shape=(hidden_size, ))
            self._state_spec.append((spec, spec))
            input_size = hidden_size

    @property
    def state_spec(self):
        return self._state_spec

    def forward(self, inputs, state):
        new_state = []
        outputs = []
        for cell, s in zip(self._cells, state):
            h, c = cell(inputs, s)
            new_state.append((h, c))
            outputs.append(h)
            inputs = h
        output = []
        for l in self._output_layers:
            output.append(outputs[l])
        if len(output) > 1:
            output = torch.cat(output, -1)
        else:
            output = output[0]
        return output, new_state
