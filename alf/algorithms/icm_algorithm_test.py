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
from alf.algorithms.icm_algorithm import ICMAlgorithm


class ICMAlgorithmTest(alf.test.TestCase):
    def setUp(self):
        self._input_tensor_spec = TensorSpec((10, ))
        self._time_step = TimeStep(
            step_type=StepType.MID,
            reward=0,
            discount=1,
            observation=self._input_tensor_spec.zeros(outer_dims=(1, )),
            prev_action=None,
            env_id=None)
        self._hidden_size = 100

    def test_discrete_action(self):
        action_spec = BoundedTensorSpec((),
                                        dtype=torch.int64,
                                        minimum=0,
                                        maximum=3)
        alg = ICMAlgorithm(
            action_spec=action_spec,
            observation_spec=self._input_tensor_spec,
            hidden_size=self._hidden_size)
        state = self._input_tensor_spec.zeros(outer_dims=(1, ))

        alg_step = alg.train_step(
            self._time_step._replace(
                prev_action=action_spec.zeros(outer_dims=(1, ))), state)

        # the inverse net should predict a uniform distribution
        self.assertTensorClose(
            torch.sum(alg_step.info.inverse_loss),
            torch.as_tensor(
                math.log(action_spec.maximum - action_spec.minimum + 1)),
            epsilon=1e-4)

    def test_continuous_action(self):
        action_spec = TensorSpec((4, ))
        alg = ICMAlgorithm(
            action_spec=action_spec,
            observation_spec=self._input_tensor_spec,
            hidden_size=self._hidden_size)
        state = self._input_tensor_spec.zeros(outer_dims=(1, ))

        alg_step = alg.train_step(
            self._time_step._replace(
                prev_action=action_spec.zeros(outer_dims=(1, ))), state)

        # the inverse net should predict a zero action vector
        self.assertTensorClose(
            torch.sum(alg_step.info.inverse_loss), torch.as_tensor(0))


if __name__ == "__main__":
    alf.test.main()
