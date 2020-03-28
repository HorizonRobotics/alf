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

from absl import logging
import torch
import torch.nn as nn

import alf
from alf.optimizers.trusted_updater import TrustedUpdater


class TrustedUpdaterTest(alf.test.TestCase):
    def test_trusted_updater(self):
        v1 = nn.Parameter(torch.tensor([1.0, 2.0]))
        v2 = nn.Parameter(torch.ones(8))
        updater = TrustedUpdater([v1, v2])

        old_sum_v1 = v1.sum()
        old_sum_v2 = v2.sum()
        v1.data.add_(torch.ones(2))
        v2.data.add_(torch.ones(8))

        def _change_f1():
            logging.info('v1=%s v2=%s' % (v1, v2))
            sum_v1 = v1.sum()
            sum_v2 = v2.sum()
            return sum_v1 - old_sum_v1, sum_v2 - old_sum_v2

        # Test for correctly adjusting the variables
        changes, steps = updater.adjust_step(_change_f1, (1., 2.))
        self.assertEqual(changes[0].detach().cpu().numpy(), 2.)
        self.assertEqual(changes[1].detach().cpu().numpy(), 8.)
        self.assertEqual(1, steps)

        changes = _change_f1()
        self.assertLess(changes[0].detach().cpu().numpy(), 1.)
        self.assertLess(changes[1].detach().cpu().numpy(), 2.)

        def _change_f2():
            return (torch.tensor(8.), torch.tensor(8.))

        # Test for detecting that change cannot be reduced
        self.assertRaises(AssertionError, updater.adjust_step, _change_f2,
                          (1., 2.))


if __name__ == '__main__':
    alf.test.main()
