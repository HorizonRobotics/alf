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

import torch

import alf
import alf.networks.memory as memory


class MemoryTest(alf.test.TestCase):
    def assertArrayEqual(self, x, y, epsilon=1e-6):
        self.assertEqual(x.shape, y.shape)
        self.assertLess((x - y).abs().max(), epsilon)

    def test_memory(self):
        mem = memory.MemoryWithUsage(2, 3, usage_decay=1., scale=20)
        self.assertEqual(mem.dim, 2)
        self.assertEqual(mem.size, 3)
        self._test_memory(mem)
        mem.reset()
        self._test_memory(mem)

    def test_snapshot_memory(self):
        mem = memory.MemoryWithUsage(
            2, 3, snapshot_only=True, usage_decay=1., scale=20)
        self.assertEqual(mem.dim, 2)
        self.assertEqual(mem.size, 3)
        self._test_memory(mem)
        mem.reset()
        self._test_memory(mem)

    def _test_memory(self, mem):
        v00 = torch.tensor([1., 0])
        v10 = torch.tensor([1., 2])
        v01 = torch.tensor([0., 1])
        v11 = torch.tensor([-2., 1])

        w0 = torch.stack([v00, v10])
        mem.write(w0)
        # The usage of newly written memory should be 1
        self.assertArrayEqual(mem.usage, torch.tensor([[1., 0, 0], [1., 0,
                                                                    0]]))
        r = mem.read(w0)
        self.assertArrayEqual(r, w0)
        self.assertArrayEqual(mem.usage, torch.tensor([[2., 0, 0], [2., 0,
                                                                    0]]))

        # w1 is othorgonal to w0
        w1 = torch.stack([v01, v11])
        mem.write(w1)
        self.assertArrayEqual(mem.usage, torch.tensor([[2., 1, 0], [2., 1,
                                                                    0]]))
        r = mem.read(w1)
        self.assertArrayEqual(r, w1)
        self.assertArrayEqual(mem.usage, torch.tensor([[2., 2, 0], [2., 2,
                                                                    0]]))
        r = mem.read(w0)
        self.assertArrayEqual(r, w0)
        self.assertArrayEqual(mem.usage, torch.tensor([[3., 2, 0], [3., 2,
                                                                    0]]))
        r = mem.read(torch.tensor([[2., 2.], [1, 1]]))
        self.assertArrayEqual(r, torch.tensor([[0.5, 0.5], [1, 2]]))
        self.assertArrayEqual(mem.usage,
                              torch.tensor([[3.5, 2.5, 0], [4., 2, 0]]))

        mem.write(w0)
        # current memory:  [v00 v01 v00] [v10, v11, v10]
        self.assertArrayEqual(mem.usage,
                              torch.tensor([[3.5, 2.5, 1], [4., 2, 1]]))

        rkey = torch.stack([w0[0], w1[1]])
        r = mem.read(rkey)
        self.assertArrayEqual(r, rkey)
        r = mem.read(rkey)
        self.assertArrayEqual(r, rkey)
        r = mem.read(rkey)
        self.assertArrayEqual(r, rkey)
        r = mem.read(rkey)
        self.assertArrayEqual(r, rkey)
        self.assertArrayEqual(mem.usage,
                              torch.tensor([[5.5, 2.5, 3], [4., 6, 1]]))

        mem.write(w0)
        # current memory:  [v00 v00 v00] [v10, v11, v10]
        self.assertArrayEqual(mem.usage, torch.tensor([[5.5, 1, 3], [4., 6,
                                                                     1]]))
        mem.read(w1)
        self.assertArrayEqual(r, torch.stack([v00, v11]))
        self.assertArrayEqual(
            mem.usage,
            torch.tensor([[5.5 + 1 / 3, 1 + 1 / 3, 3 + 1 / 3], [4., 7, 1]]))

        # test for multiple read keys
        r = mem.read(torch.stack([w0, w1], dim=1))
        self.assertArrayEqual(r[:, 0, :], torch.stack([v00, v10]))
        self.assertArrayEqual(r[:, 1, :], torch.stack([v00, v11]))
        self.assertArrayEqual(mem.usage,
                              torch.tensor([[6.5, 2, 4], [4.5, 8, 1.5]]))

        # test for scale
        r = mem.read(w1, scale=torch.tensor([1., 0.]))
        self.assertArrayEqual(r,
                              torch.stack([v00, 2. / 3 * v10 + 1. / 3 * v11]))

    def test_genkey_and_read(self):
        mem = memory.MemoryWithUsage(2, 3, usage_decay=1., scale=20)
        v00 = torch.tensor([1., 0])
        v10 = torch.tensor([1., 2])
        w0 = torch.stack([v00, v10])
        mem.write(w0)

        def keynet(x):
            s = torch.ones((x.shape[0], 3), dtype=torch.float32) * 20.
            return torch.cat([x, x, x, s], dim=-1)

        r = mem.genkey_and_read(keynet, w0, flatten_result=False)
        self.assertEqual(list(r.shape), [2, 3, 2])
        self.assertArrayEqual(r[:, 0, :], w0)
        self.assertArrayEqual(r[:, 1, :], w0)
        self.assertArrayEqual(r[:, 2, :], w0)


if __name__ == '__main__':
    alf.test.main()
