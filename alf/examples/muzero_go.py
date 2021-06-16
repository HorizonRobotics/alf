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

import torch
from torch import nn
import torch.nn.functional as F
import alf
from alf.networks import CategoricalProjectionNetwork


def _encode_action(action, height, width):
    action = F.one_hot(
        action, num_classes=height * width + 1).to(torch.float32)
    return action[..., :-1].reshape(action.shape[0], 1, height, width)


@alf.configurable
class RepresentationNet(alf.networks.Network):
    def __init__(self, input_tensor_spec, num_blocks=16, filters=256):
        super().__init__(input_tensor_spec, name="RepresentationNet")
        board_spec = input_tensor_spec['board']
        in_channels = 2 * board_spec.shape[0] + 3

        shape = in_channels, board_spec.shape[1], board_spec.shape[2]
        enc_layers = []
        for _ in range(num_blocks):
            res_block = alf.layers.BottleneckBlock(
                in_channels=in_channels,
                kernel_size=3,
                filters=(filters, filters, filters),
                stride=1)
            shape = res_block.calc_output_shape(shape)
            enc_layers.append(res_block)
            in_channels = filters

        self._model = nn.Sequential(*enc_layers)

    def forward(self, observation, state=()):
        board = observation['board']
        board0 = (board < 0).to(torch.float32)
        board1 = (board > 0).to(torch.float32)
        obs = [board0, board1]

        action = observation['prev_action']
        height, width = board.shape[-2:]
        action = _encode_action(action, height, width)
        obs.append(action)

        to_play = observation['to_play'].to(torch.float32)
        to_play = (to_play * 2 - 1).reshape(*to_play.shape, 1, 1, 1)
        to_play = to_play.expand(-1, 1, height, width)
        obs.append(to_play)

        steps_to_end = 0.9**(2. * height * width - observation['steps'])
        steps_to_end = steps_to_end.reshape(*steps_to_end.shape, 1, 1, 1)
        steps_to_end = steps_to_end.expand(-1, 1, height, width)
        obs.append(steps_to_end)

        input = torch.cat(obs, dim=1)
        out = self._model(input)
        assert torch.isfinite(out).all()
        return out, ()


@alf.configurable
class DynamicsNet(alf.networks.Network):
    def __init__(self, input_tensor_spec, num_blocks=16, filters=256):
        super().__init__(input_tensor_spec, name="DynamicsNet")
        state_spec, action_spec = input_tensor_spec
        self._num_actions = int(action_spec.maximum - action_spec.minimum + 1)
        in_channels = state_spec.shape[0] + 1

        shape = in_channels, state_spec.shape[1], state_spec.shape[2]
        enc_layers = []
        for _ in range(num_blocks):
            res_block = alf.layers.BottleneckBlock(
                in_channels=in_channels,
                kernel_size=3,
                filters=(filters, filters, filters),
                stride=1)
            shape = res_block.calc_output_shape(shape)
            enc_layers.append(res_block)
            in_channels = filters

        self._model = nn.Sequential(*enc_layers)

    def forward(self, inputs, state=()):
        state, action = inputs
        action = _encode_action(action, state.shape[2], state.shape[3])
        input = torch.cat([state, action], dim=1)
        return self._model(input), ()


@alf.configurable
class PredictionNet(alf.networks.Network):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 filters=256,
                 hidden_size=256,
                 initial_game_over_bias=0.):
        super().__init__(observation_spec, name="PredictionNet")
        in_channels, h, w = observation_spec.shape

        output_weight_initializer = torch.nn.init.zeros_

        self._value_head = nn.Sequential(
            alf.layers.Conv2D(in_channels, 1, kernel_size=1),
            alf.layers.Reshape([-1]),
            alf.layers.FC(
                input_size=h * w,
                output_size=hidden_size,
                activation=torch.relu_,
                use_bn=False),
            alf.layers.FC(
                input_size=hidden_size,
                output_size=1,
                activation=torch.tanh,
                kernel_initializer=output_weight_initializer),
            alf.layers.Reshape(()))

        self._reward_head = nn.Sequential(
            alf.layers.Conv2D(in_channels, 1, kernel_size=1),
            alf.layers.Reshape([-1]),
            alf.layers.FC(
                input_size=h * w,
                output_size=hidden_size,
                activation=torch.relu_,
                use_bn=False),
            alf.layers.FC(
                input_size=hidden_size,
                output_size=1,
                activation=torch.tanh,
                kernel_initializer=output_weight_initializer),
            alf.layers.Reshape(()))

        self._game_over_head = nn.Sequential(
            alf.layers.Conv2D(in_channels, 1, kernel_size=1),
            alf.layers.Reshape([-1]),
            alf.layers.FC(
                input_size=h * w,
                output_size=hidden_size,
                activation=torch.relu_,
                use_bn=False),
            alf.layers.FC(
                input_size=hidden_size,
                output_size=1,
                bias_init_value=initial_game_over_bias,
                kernel_initializer=output_weight_initializer),
            alf.layers.Reshape(()))

        self._action_head = nn.Sequential(
            alf.layers.Conv2D(in_channels, filters, kernel_size=3, padding=1),
            alf.layers.Reshape([-1]))

        self._action_proj = CategoricalProjectionNetwork(
            input_size=h * w * filters,
            action_spec=action_spec,
            logits_init_output_factor=1e-10)

    def forward(self, input, state=()):
        value = self._value_head(input)
        reward = self._reward_head(input)
        game_over_logit = self._game_over_head(input)
        action_distribution = self._action_proj(self._action_head(input))[0]
        return (value, reward, action_distribution, game_over_logit), ()
