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
import math

import alf
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.data_structures import TimeStep, StepType
from alf.networks import EncodingNetwork
from alf.algorithms.diayn_algorithm import DIAYNAlgorithm


class DIAYNAlgorithmTest(alf.test.TestCase):
    def setUp(self):
        input_tensor_spec = TensorSpec((10, ))
        self._time_step = TimeStep(
            step_type=torch.tensor(StepType.MID, dtype=torch.int32),
            reward=0,
            discount=1,
            observation=input_tensor_spec.zeros(outer_dims=(1, )),
            prev_action=None,
            env_id=None)
        self._encoding_net = EncodingNetwork(
            input_tensor_spec=input_tensor_spec)

    def test_discrete_skill_loss(self):
        skill_spec = BoundedTensorSpec((),
                                       dtype=torch.int64,
                                       minimum=0,
                                       maximum=3)
        alg = DIAYNAlgorithm(
            skill_spec=skill_spec, encoding_net=self._encoding_net)
        skill = state = torch.nn.functional.one_hot(
            skill_spec.zeros(outer_dims=(1, )),
            int(skill_spec.maximum - skill_spec.minimum + 1)).to(torch.float32)

        alg_step = alg.train_step(
            self._time_step._replace(
                observation=[self._time_step.observation, skill]), state)

        # the discriminator should predict a uniform distribution
        self.assertTensorClose(
            torch.sum(alg_step.info.loss),
            torch.as_tensor(
                math.log(skill_spec.maximum - skill_spec.minimum + 1)),
            epsilon=1e-4)

    def test_continuous_skill_loss(self):
        skill_spec = TensorSpec((4, ))
        alg = DIAYNAlgorithm(
            skill_spec=skill_spec, encoding_net=self._encoding_net)
        skill = state = skill_spec.zeros(outer_dims=(1, ))

        alg_step = alg.train_step(
            self._time_step._replace(
                observation=[self._time_step.observation, skill]), state)

        # the discriminator should predict a zero skill vector
        self.assertTensorClose(
            torch.sum(alg_step.info.loss), torch.as_tensor(0))


if __name__ == "__main__":
    alf.test.main()
