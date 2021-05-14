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

from absl.testing import parameterized
import torch
import torch.nn as nn

import alf
from alf.algorithms.entropy_target_algorithm import EntropyTargetAlgorithm, EntropyTargetInfo
from alf.algorithms.entropy_target_algorithm import NestedEntropyTargetAlgorithm
from alf.data_structures import TimeStep, StepType
from alf.networks import NormalProjectionNetwork, StableNormalProjectionNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


class EntropyTargetAlgorithmTest(parameterized.TestCase, alf.test.TestCase):
    def setUp(self):
        self._input_tensor_spec = TensorSpec((10, ))
        self._time_step = TimeStep(
            step_type=torch.as_tensor(StepType.MID),
            reward=0,
            discount=1,
            observation=self._input_tensor_spec.zeros(outer_dims=(1, )),
            prev_action=None,
            env_id=None)
        self._hidden_size = 100

    @parameterized.parameters((NormalProjectionNetwork, False),
                              (NormalProjectionNetwork, True),
                              (StableNormalProjectionNetwork, False),
                              (StableNormalProjectionNetwork, True))
    def test_run_entropy_target_algorithm(self, network_ctor, scaled):
        action_spec = BoundedTensorSpec((1, ), minimum=0, maximum=3)
        alg = EntropyTargetAlgorithm(action_spec=action_spec)
        net = network_ctor(
            self._input_tensor_spec.shape[0],
            action_spec,
            projection_output_init_gain=1.0,
            squash_mean=True,
            scale_distribution=scaled)

        embedding = 10 * torch.rand(
            (100, ) + self._input_tensor_spec.shape, dtype=torch.float32)

        dist, _ = net(embedding)

        alg_step = alg.train_step((dist, self._time_step.step_type))

        info = alg_step.info
        for i in range(-3, 1):
            alg._stage = torch.tensor(i, dtype=torch.int32)
            alg.calc_loss(info)

    def test_nested_entropy_target_algorithm(self):
        action_spec = dict(
            a=BoundedTensorSpec((1, ), minimum=0, maximum=3),
            b=BoundedTensorSpec((), minimum=0, maximum=3, dtype='int64'))
        alg = NestedEntropyTargetAlgorithm(
            action_spec=action_spec, target_entropy=dict(a=None, b=None))
        net = alf.nn.Branch(
            a=NormalProjectionNetwork(
                self._input_tensor_spec.shape[0],
                action_spec['a'],
                projection_output_init_gain=1.0,
                squash_mean=True),
            b=alf.nn.CategoricalProjectionNetwork(
                self._input_tensor_spec.shape[0], action_spec['b']))
        embedding = 10 * torch.rand(
            (100, ) + self._input_tensor_spec.shape, dtype=torch.float32)

        dist, _ = net(embedding)

        alg_step = alg.train_step((dist, self._time_step.step_type))

        info = alg_step.info
        for i in range(-3, 1):
            alg._stage = torch.tensor(i, dtype=torch.int32)
            alg.calc_loss(info)


if __name__ == "__main__":
    alf.test.main()
