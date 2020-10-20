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

import random
import torch

import alf
from alf.experience_replayers.segment_tree import SumSegmentTree, MaxSegmentTree


class SegmentTreeTest(alf.test.TestCase):
    def test_max_tree(self):
        for size in [1, 2, 3, 4, 7, 8, 9, 15, 16, 128]:
            tree = MaxSegmentTree(size)
            vals = torch.zeros(size, dtype=torch.float32)
            for _ in range(100):
                n = random.randint(1, size)
                i = torch.randint(size, size=(n, ), dtype=torch.int64)
                i, _ = torch.sort(i)
                i = torch.unique(i)
                i = alf.math.shuffle(i)
                v = torch.randint(0, 10000, size=i.shape, dtype=torch.float32)
                vals[i] = v
                tree[i] = v
                self.assertEqual(tree.summary(), vals.max())
            i = torch.arange(size, dtype=torch.int64)
            self.assertEqual(tree[i], vals)

    def test_sum_tree(self):
        for size in [3, 1, 2, 3, 4, 7, 8, 9, 15, 16, 128]:
            tree = SumSegmentTree(size)
            vals = torch.zeros(size, dtype=torch.float32)
            for _ in range(100):
                n = random.randint(1, size)
                i = torch.randint(size, size=(n, ), dtype=torch.int64)
                i, _ = torch.sort(i)
                i = torch.unique(i)
                i = alf.math.shuffle(i)
                v = torch.randint(0, 10000, size=i.shape, dtype=torch.float32)
                vals[i] = v
                tree[i] = v
                self.assertEqual(tree.summary(), vals.sum())

            s = torch.cumsum(vals, 0)
            s = torch.cat([torch.tensor([0.]), s[:-1]])
            for _ in range(100):
                n = random.randint(1, size)
                i = torch.randint(size, size=(n, ), dtype=torch.int64)
                i, _ = torch.sort(i)
                i = torch.unique(i)
                i = alf.math.shuffle(i)
                v = s[i] + vals[i] * torch.rand(i.shape) * 0.999
                self.assertEqual(tree.find_sum_bound(v), i)

            thresh = tree.summary().reshape(1)
            self.assertEqual(tree.find_sum_bound(thresh), size - 1)
            self.assertRaises(ValueError, tree.find_sum_bound, thresh + 1)

    def test_boundary(self):
        tree = SumSegmentTree(10)
        v = torch.tensor([0.1] * 9 + [0.0])
        i = torch.arange(10)
        tree[i] = v
        thresh = tree.summary().reshape(1)
        self.assertEqual(tree.find_sum_bound(thresh), 8)


if __name__ == '__main__':
    alf.test.main()
