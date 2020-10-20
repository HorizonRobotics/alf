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

import alf


class MathOpsTest(alf.test.TestCase):
    def test_argmin(self):
        a = torch.tensor([[2, 5, 2], [0, 1, 2], [1, 1, 1], [3, 2, 1],
                          [4, 2, 2]])
        i = alf.math.argmin(a)
        self.assertEqual(i, torch.tensor([0, 0, 0, 2, 1]))

    def test_softsign(self):
        input = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(alf.math.softsign, input)
        # gradcheck cannot handle inplace op directly, so we need to add another
        # op before softsign_
        torch.autograd.gradcheck(lambda x: alf.math.softsign_(x * 1.0), input)


if __name__ == '__main__':
    alf.test.main()
