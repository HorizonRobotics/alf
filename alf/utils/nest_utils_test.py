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
"""Unittests for nest_utils.py"""

from collections import namedtuple
import unittest

from alf.utils import nest_utils

NTuple = namedtuple('NTuple', ['a', 'b'])


class TestFlatten(unittest.TestCase):
    def test_flatten(self):
        nest = NTuple(a=1, b=NTuple(a=NTuple(a=(2, ), b=[3]), b=dict(x=2)))
        expected_flat_seq = [1, 2, 3, 2]
        self.assertEqual(nest_utils.flatten(nest), expected_flat_seq)
        self.assertEqual(nest_utils.flatten(1), [1])


class TestAssertSameStructure(unittest.TestCase):
    def test_assert_same_structure(self):
        nest1 = NTuple(
            a=1,
            b=NTuple(a=NTuple(a=(2, ), b=[3]), b=dict(x=NTuple(a=[100], b=1))))
        nest2 = NTuple(
            b=NTuple(
                a=NTuple(a=(2, ), b=[300]), b=dict(x=NTuple(a=[1], b=100))),
            a=3.0)
        nest_utils.assert_same_structure(nest1, nest2)
        nest_utils.assert_same_structure(
            dict(x=1, y=NTuple(a=[1], b=3)), dict(y=NTuple(a=[3], b=1), x=1))
        self.assertRaises(AssertionError, nest_utils.assert_same_structure,
                          dict(x=1, y=[2]), dict(x=[2], y=1))
        self.assertRaises(AssertionError, nest_utils.assert_same_structure,
                          dict(y=[2]), dict(x=[2]))
        self.assertRaises(AssertionError, nest_utils.assert_same_structure,
                          dict(x=1, y=[2]), dict(y=(2, ), x=1))
        self.assertRaises(AssertionError, nest_utils.assert_same_structure,
                          dict(x=1, y=[2]), 1)
        self.assertRaises(AssertionError, nest_utils.assert_same_structure,
                          NTuple(a=1, b=[2]), NTuple(b=[2], a=[1]))
        nest_utils.assert_same_structure(1.0, 10)


class TestMapStructure(unittest.TestCase):
    def test_map_structure(self):
        nest1 = NTuple(a=dict(x=3, y=2), b=[100.0, (5, )])
        nest2 = NTuple(a=dict(x=1, y=-2), b=[100.0, (10, )])
        nest3 = NTuple(a=dict(x=1, y=-2), b=[50.0, (6, )])
        expected_result = NTuple(a=dict(x=5, y=-2), b=[250.0, (21, )])
        self.assertEqual(
            nest_utils.map_structure(lambda a, b, c: a + b + c, nest1, nest2,
                                     nest3), expected_result)
        self.assertEqual(
            nest_utils.map_structure(lambda a, b: a + b, [1, 3], [4, 5]),
            [5, 8])
        self.assertEqual(nest_utils.map_structure(lambda a, b: a * b, 1, 3), 3)


class TestPackSequenceAs(unittest.TestCase):
    def test_pack_sequence_as(self):
        nest = NTuple(a=dict(x=3, y=2), b=[100.0, (5, )])
        flat_seq = [30, 20, -1, 4]
        expected_nest = NTuple(a=dict(x=30, y=20), b=[-1, (4, )])
        self.assertEqual(
            nest_utils.pack_sequence_as(nest, flat_seq), expected_nest)
        self.assertEqual(nest_utils.pack_sequence_as(1, [1]), 1)


class TestFindField(unittest.TestCase):
    def test_find_field(self):
        nest = NTuple(a=1, b=NTuple(a=NTuple(a=2, b=3), b=2))
        ret = nest_utils.find_field(nest, 'a')
        self.assertEqual(len(ret), 2)
        self.assertEqual(ret[0], nest.a)
        self.assertEqual(ret[1], nest.b.a)

        nest = (1, NTuple(a=NTuple(a=2, b=3), b=2))
        ret = nest_utils.find_field(nest, 'a')
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0], nest[1].a)

        nest = NTuple(a=1, b=[NTuple(a=2, b=3), 2])
        ret = nest_utils.find_field(nest, 'a')
        self.assertEqual(len(ret), 2)
        self.assertEqual(ret[0], nest.a)
        self.assertEqual(ret[1], nest.b[0].a)


if __name__ == '__main__':
    unittest.main()
