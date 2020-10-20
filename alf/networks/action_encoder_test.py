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
from alf.networks.action_encoder import SimpleActionEncoder


class SimpleActionEncoderTest(alf.test.TestCase):
    def test_simple_action_encoder(self):
        action_spec = [
            alf.BoundedTensorSpec((3, )),
            alf.BoundedTensorSpec((), dtype=torch.int32, minimum=0, maximum=3)
        ]
        encoder = SimpleActionEncoder(action_spec)

        # test scalar
        x = [torch.tensor([0.5, 1.5, 2.5]), torch.tensor(3)]
        y = encoder(x)[0]
        self.assertEqual(y, torch.tensor([0.5, 1.5, 2.5, 0, 0, 0, 1]))

        # test batch
        x = [torch.tensor([[0.5, 1.5, 2.5], [1, 2, 3]]), torch.tensor([3, 2])]
        y = encoder(x)[0]
        self.assertEqual(
            y,
            torch.tensor([[0.5, 1.5, 2.5, 0, 0, 0, 1], [1, 2, 3, 0, 0, 1, 0]]))

        # test unsupported spec
        action_spec = [
            alf.BoundedTensorSpec((3, )),
            alf.BoundedTensorSpec((), dtype=torch.int32, minimum=1, maximum=3)
        ]

        self.assertRaises(AssertionError, SimpleActionEncoder, action_spec)


if __name__ == "__main__":
    alf.test.main()
