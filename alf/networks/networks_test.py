# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from alf.networks import TemporalStack


class NestworksTest(alf.test.TestCase):
    def test_temporal_stack_skip(self):
        batch_size = 2
        dim = 3

        x = torch.randn((batch_size, 20, dim))
        l = TemporalStack(dim, 5, 3)
        self.assertEqual(l.output_spec, alf.TensorSpec((5, dim)))
        state = alf.utils.common.zero_tensor_from_nested_spec(
            l.state_spec, batch_size)

        for i in range(0, 3):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, 4, :], x[:, 0, :])
            self.assertEqual(o[:, :4, :], torch.zeros((batch_size, 4, dim)))

        for i in range(3, 6):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, 3:, :], x[:, 0:6:3, :])
            self.assertEqual(o[:, :3, :], torch.zeros((batch_size, 3, dim)))

        for i in range(6, 9):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, 2:, :], x[:, 0:9:3, :])
            self.assertEqual(o[:, :2, :], torch.zeros((batch_size, 2, dim)))

        for i in range(9, 12):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, 1:, :], x[:, 0:12:3, :])
            self.assertEqual(o[:, :1, :], torch.zeros((batch_size, 1, dim)))

        for i in range(12, 15):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o, x[:, 0:15:3, :])

        for i in range(15, 18):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o, x[:, 3:18:3, :])

    def test_temporal_stack_max(self):
        batch_size = 2
        dim = 3

        x = torch.randn((batch_size, 20, dim))
        l = TemporalStack(dim, 5, 3, mode='max')
        self.assertEqual(l.output_spec, alf.TensorSpec((5, dim)))
        state = alf.utils.common.zero_tensor_from_nested_spec(
            l.state_spec, batch_size)

        for i in range(0, 2):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, :5, :], torch.zeros((batch_size, 5, dim)))

        for i in range(2, 5):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, 4, :], x[:, 0:3, :].max(dim=1)[0])
            self.assertEqual(o[:, :4, :], torch.zeros((batch_size, 4, dim)))

        for i in range(5, 8):
            o, state = l(x[:, i, :], state)
            self.assertEqual(
                o[:, 3:, :], x[:, 0:6, :].reshape(batch_size, 2, 3,
                                                  dim).max(dim=2)[0])
            self.assertEqual(o[:, :3, :], torch.zeros((batch_size, 3, dim)))

        for i in range(8, 11):
            o, state = l(x[:, i, :], state)
            self.assertEqual(
                o[:, 2:, :], x[:, 0:9, :].reshape(batch_size, 3, 3,
                                                  dim).max(dim=2)[0])
            self.assertEqual(o[:, :2:], torch.zeros((batch_size, 2, dim)))

        for i in range(11, 14):
            o, state = l(x[:, i, :], state)
            self.assertEqual(
                o[:, 1:, :], x[:, 0:12, :].reshape(batch_size, 4, 3,
                                                   dim).max(dim=2)[0])
            self.assertEqual(o[:, :1:], torch.zeros((batch_size, 1, dim)))

        for i in range(14, 17):
            o, state = l(x[:, i, :], state)
            self.assertEqual(
                o, x[:, 0:15, :].reshape(batch_size, 5, 3, dim).max(dim=2)[0])

        for i in range(17, 20):
            o, state = l(x[:, i, :], state)
            self.assertEqual(
                o, x[:, 3:18, :].reshape(batch_size, 5, 3, dim).max(dim=2)[0])

    def test_temporal_stack_avg(self):
        batch_size = 2
        dim = 3

        x = torch.randn((batch_size, 20, dim))
        l = TemporalStack(dim, 5, 3, mode='avg')
        self.assertEqual(l.output_spec, alf.TensorSpec((5, dim)))
        state = alf.utils.common.zero_tensor_from_nested_spec(
            l.state_spec, batch_size)

        for i in range(0, 2):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, :5, :], torch.zeros((batch_size, 5, dim)))

        for i in range(2, 5):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(o[:, 4, :], x[:, 0:3, :].mean(dim=1))
            self.assertEqual(o[:, :4, :], torch.zeros((batch_size, 4, dim)))

        for i in range(5, 8):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(
                o[:, 3:, :], x[:, 0:6, :].reshape(batch_size, 2, 3,
                                                  dim).mean(dim=2))
            self.assertEqual(o[:, :3, :], torch.zeros((batch_size, 3, dim)))

        for i in range(8, 11):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(
                o[:, 2:, :], x[:, 0:9, :].reshape(batch_size, 3, 3,
                                                  dim).mean(dim=2))
            self.assertEqual(o[:, :2:], torch.zeros((batch_size, 2, dim)))

        for i in range(11, 14):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(
                o[:, 1:, :], x[:, 0:12, :].reshape(batch_size, 4, 3,
                                                   dim).mean(dim=2))
            self.assertEqual(o[:, :1:], torch.zeros((batch_size, 1, dim)))

        for i in range(14, 17):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(
                o, x[:, 0:15, :].reshape(batch_size, 5, 3, dim).mean(dim=2))

        for i in range(17, 20):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(
                o, x[:, 3:18, :].reshape(batch_size, 5, 3, dim).mean(dim=2))


if __name__ == '__main__':
    alf.test.main()
