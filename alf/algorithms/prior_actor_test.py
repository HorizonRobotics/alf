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

import math
import torch

import alf
from alf.algorithms.prior_actor import SameActionPriorActor, UniformPriorActor
from alf.tensor_specs import BoundedTensorSpec
from alf.data_structures import TimeStep, StepType


class PriorActorTest(alf.test.TestCase):
    def test_same_actin_prior_actor(self):
        action_spec = dict(
            a=BoundedTensorSpec(shape=()),
            b=BoundedTensorSpec((3, ), minimum=(-1, 0, -2), maximum=(2, 2, 3)),
            c=BoundedTensorSpec((2, 3), minimum=-1, maximum=1))
        actor = SameActionPriorActor(
            observation_spec=(), action_spec=action_spec)
        batch = TimeStep(
            step_type=torch.tensor([StepType.FIRST, StepType.MID]),
            prev_action=dict(
                a=torch.tensor([0., 1.]),
                b=torch.tensor([[-1., 0., -2.], [2., 2., 3.]]),
                c=action_spec['c'].sample((2, ))))
        alg_step = actor.predict_step(batch, ())
        self.assertAlmostEqual(
            alg_step.output['a'].log_prob(torch.tensor([0., 0.]))[0],
            alg_step.output['a'].log_prob(torch.tensor([1., 1.]))[0],
            delta=1e-6)
        self.assertAlmostEqual(
            alg_step.output['a'].log_prob(torch.tensor([0., 0.]))[1],
            alg_step.output['a'].log_prob(torch.tensor([0., 0.]))[0] +
            math.log(0.1),
            delta=1e-6)

        self.assertAlmostEqual(
            alg_step.output['b'].log_prob(torch.tensor(
                [[-1., 0., -2.]] * 2))[0],
            alg_step.output['b'].log_prob(torch.tensor([[2., 2., 3.]] * 2))[0],
            delta=1e-6)

        self.assertAlmostEqual(
            alg_step.output['b'].log_prob(torch.tensor(
                [[-1., 0., -2.]] * 2))[1],
            alg_step.output['b'].log_prob(torch.tensor(
                [[-1., 0., -2.]] * 2))[0] + 3 * math.log(0.1),
            delta=1e-6)

    def test_uniform_prior_actor(self):
        action_spec = dict(
            a=BoundedTensorSpec(shape=()),
            b=BoundedTensorSpec((3, ), minimum=(-1, 0, -2), maximum=(2, 2, 3)),
            c=BoundedTensorSpec((2, 3), minimum=-1, maximum=1))
        actor = UniformPriorActor(observation_spec=(), action_spec=action_spec)
        batch = TimeStep(
            step_type=torch.tensor([StepType.FIRST, StepType.MID]),
            prev_action=dict(
                a=torch.tensor([0., 1.]),
                b=torch.tensor([[-1., 0., -2.], [2., 2., 3.]]),
                c=action_spec['c'].sample((2, ))))

        alg_step = actor.predict_step(batch, ())
        self.assertEqual(
            alg_step.output['a'].log_prob(action_spec['a'].sample()),
            torch.tensor((0., ) * 2))
        self.assertEqual(
            alg_step.output['b'].log_prob(action_spec['b'].sample()),
            -torch.tensor((30., ) * 2).log())
        self.assertEqual(
            alg_step.output['c'].log_prob(action_spec['c'].sample()),
            -torch.tensor((64., ) * 2).log())


if __name__ == '__main__':
    alf.test.main()
