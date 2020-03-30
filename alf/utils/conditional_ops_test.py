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
"""Unittests for conditional_ops.py."""

import torch

import alf
from alf.utils.conditional_ops import conditional_update, select_from_mask


class ConditionalOpsTest(alf.test.TestCase):
    def test_conditional_update(self):
        def _func(x, y):
            return x + 1, y - 1

        batch_size = 256
        target = (torch.rand([batch_size, 3]), torch.rand([batch_size]))
        x = torch.rand([batch_size, 3])
        y = torch.rand([batch_size])

        cond = torch.as_tensor([False] * batch_size)
        updated_target = conditional_update(target, cond, _func, x, y)
        self.assertTensorEqual(updated_target[0], target[0])
        self.assertTensorEqual(updated_target[1], target[1])

        cond = torch.as_tensor([True] * batch_size)
        updated_target = conditional_update(target, cond, _func, x, y)
        self.assertTensorEqual(updated_target[0], x + 1)
        self.assertTensorEqual(updated_target[1], y - 1)

        cond = torch.rand((batch_size, )) < 0.5
        updated_target = conditional_update(target, cond, _func, x, y)
        self.assertTensorEqual(
            select_from_mask(updated_target[0], cond),
            select_from_mask(x + 1, cond))
        self.assertTensorEqual(
            select_from_mask(updated_target[1], cond),
            select_from_mask(y - 1, cond))

        vx = torch.zeros(())
        vy = torch.zeros(())

        def _func1(x, y):
            vx.copy_(torch.sum(x))
            vy.copy_(torch.sum(y))
            return ()

        # test empty return
        conditional_update((), cond, _func1, x, y)
        self.assertEqual(vx, torch.sum(select_from_mask(x, cond)))
        self.assertEqual(vy, torch.sum(select_from_mask(y, cond)))

    def test_conditional_update_high_dims(self):
        def _func(x):
            return x**2

        batch_size = 100
        x = torch.rand([batch_size, 3, 4, 5])
        y = torch.rand([batch_size, 3, 4, 5])
        cond = torch.randint(high=2, size=[batch_size]).to(torch.bool)

        updated_y = conditional_update(y, cond, _func, x)
        self.assertTensorEqual(
            select_from_mask(updated_y, cond), select_from_mask(x**2, cond))
        self.assertTensorEqual(
            select_from_mask(updated_y, ~cond), select_from_mask(y, ~cond))

    def test_select_from_mask(self):
        data = torch.as_tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                                [10, 11]])
        cond = torch.as_tensor([False, True, True, False, False, True])
        result = select_from_mask(data, cond)
        self.assertTensorEqual(result,
                               torch.as_tensor([[3, 4], [5, 6], [10, 11]]))


if __name__ == '__main__':
    alf.test.main()
