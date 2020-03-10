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
"""Unittests for nest.py"""

import torch

from collections import namedtuple
import unittest

import alf
import alf.nest as nest
from alf.tensor_specs import TensorSpec
from alf.nest.utils import NestConcat, NestSum

NTuple = namedtuple('NTuple', ['a', 'b'])


class TestIsNested(unittest.TestCase):
    def test_is_nested(self):
        self.assertFalse(nest.is_nested(1))
        self.assertFalse(nest.is_nested(None))
        self.assertTrue(nest.is_nested(dict(x=1)))
        self.assertTrue(nest.is_nested([1]))
        ntuple = NTuple(a=1, b=NTuple(a=NTuple(a=(2, ), b=[3]), b=dict(x=2)))
        self.assertTrue(nest.is_nested(ntuple))


class TestFlatten(unittest.TestCase):
    def test_flatten(self):
        ntuple = NTuple(a=1, b=NTuple(a=NTuple(a=(2, ), b=[3]), b=dict(x=2)))
        expected_flat_seq = [1, 2, 3, 2]
        self.assertEqual(nest.flatten(ntuple), expected_flat_seq)
        self.assertEqual(nest.flatten(1), [1])


class TestAssertSameStructure(unittest.TestCase):
    def test_assert_same_structure(self):
        nest.assert_same_structure(1.0, 10)
        nest1 = NTuple(
            a=1,
            b=NTuple(a=NTuple(a=(2, ), b=[3]), b=dict(x=NTuple(a=[100], b=1))))
        nest2 = NTuple(
            b=NTuple(
                a=NTuple(a=(2, ), b=[300]), b=dict(x=NTuple(a=[1], b=100))),
            a=3.0)
        nest.assert_same_structure(nest1, nest2)
        nest.assert_same_structure(
            dict(x=1, y=NTuple(a=[1], b=3)), dict(y=NTuple(a=[3], b=1), x=1))
        self.assertRaises(AssertionError, nest.assert_same_structure,
                          dict(x=1, y=[2]), dict(x=[2], y=1))
        self.assertRaises(AssertionError, nest.assert_same_structure,
                          dict(y=[2]), dict(x=[2]))
        self.assertRaises(AssertionError, nest.assert_same_structure,
                          dict(x=1, y=[2]), dict(y=(2, ), x=1))
        self.assertRaises(AssertionError, nest.assert_same_structure,
                          dict(x=1, y=[2]), 1)
        self.assertRaises(AssertionError, nest.assert_same_structure,
                          NTuple(a=1, b=[2]), NTuple(b=[2], a=[1]))
        self.assertRaises(AssertionError, nest.assert_same_structure, [1, 2],
                          [[1, 2]])
        self.assertRaises(AssertionError, nest.assert_same_structure, [1, [2]],
                          [1, [2, 2]])
        self.assertRaises(AssertionError, nest.assert_same_structure,
                          dict(x=1, y=dict(x=2)), dict(x=1, y=dict(x=3, y=1)))


class TestMapStructure(unittest.TestCase):
    def test_map_structure(self):
        nest1 = NTuple(a=dict(x=3, y=2), b=[100.0, (5, )])
        nest2 = NTuple(a=dict(x=1, y=-2), b=[100.0, (10, )])
        nest3 = NTuple(a=dict(x=1, y=-2), b=[50.0, (6, )])
        expected_result = NTuple(a=dict(x=5, y=-2), b=[250.0, (21, )])
        self.assertEqual(
            nest.map_structure(lambda a, b, c: a + b + c, nest1, nest2, nest3),
            expected_result)
        self.assertEqual(
            nest.map_structure(lambda a, b: a + b, [1, 3], [4, 5]), [5, 8])
        self.assertEqual(nest.map_structure(lambda a, b: a * b, 1, 3), 3)


class TestFastMapStructure(unittest.TestCase):
    def test_fast_map_structure(self):
        nest1 = NTuple(a=dict(x=3, y=2), b=[100.0, (5, )])
        nest2 = NTuple(a=dict(x=1, y=-2), b=[100.0, (10, )])
        nest3 = NTuple(a=dict(x=1, y=-2), b=[50.0, (6, )])
        expected_result = NTuple(a=dict(x=5, y=-2), b=[250.0, (21, )])
        self.assertEqual(
            nest.fast_map_structure(lambda a, b, c: a + b + c, nest1, nest2,
                                    nest3), expected_result)
        self.assertEqual(
            nest.fast_map_structure(lambda a, b: a + b, [1, 3], [4, 5]),
            [5, 8])
        self.assertEqual(nest.fast_map_structure(lambda a, b: a * b, 1, 3), 3)


class TestMapStructureUpTo(unittest.TestCase):
    def test_different_keys(self):
        self.assertRaises(AssertionError, nest.map_structure_up_to,
                          dict(x=1, z=2), lambda a, b: a * b, dict(x=1, y=2),
                          dict(y=1, x=2))
        self.assertRaises(AssertionError, nest.map_structure_up_to,
                          dict(x=1, y=2), lambda a, b: a * b, dict(x=1, y=2),
                          dict(y=1, z=2))

    def test_different_lengths(self):
        self.assertRaises(AssertionError, nest.map_structure_up_to,
                          [1, 2, 3], lambda x: x * 2, [1, 2])
        self.assertRaises(AssertionError, nest.map_structure_up_to,
                          [1, 2], lambda x, y: x * y, [1, 2, 3], [4, 5])
        self.assertRaises(AssertionError, nest.map_structure_up_to,
                          [1, [2, 3]], lambda x: x * 2,
                          [[1], [[2, 4], [3, 5], 3]])

    def test_map_structure_to(self):
        shallow_nest = [[None], None]
        inp_val = [[1], 2]
        out = nest.map_structure_up_to(shallow_nest, lambda x: 2 * x, inp_val)
        self.assertEqual(out, [[2], 4])

        shallow_nest = [None, None]
        inp_val = [[1], 2]
        out = nest.map_structure_up_to(shallow_nest, lambda x: 2 * x, inp_val)
        self.assertEqual(out, [[1, 1], 4])

        data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
        name_list = ['evens', ['odds', 'primes']]
        out = nest.map_structure_up_to(
            name_list, lambda name, sec: "first_{}_{}".format(len(sec), name),
            name_list, data_list)
        self.assertEqual(out,
                         ['first_4_evens', ['first_5_odds', 'first_3_primes']])

        ab_tuple = namedtuple("ab_tuple", "a, b")
        op_tuple = namedtuple("op_tuple", "add, mul")
        inp_val = ab_tuple(a=2, b=3)
        inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
        out = nest.map_structure_up_to(
            inp_val, lambda val, ops: (val + ops.add) * ops.mul, inp_val,
            inp_ops)
        self.assertEqual(out, ab_tuple(a=6, b=15))


class TestPackSequenceAs(unittest.TestCase):
    def test_pack_sequence_as(self):
        ntuple = NTuple(a=dict(x=3, y=2), b=[100.0, (5, )])
        flat_seq = [30, 20, -1, 4]
        expected_nest = NTuple(a=dict(x=30, y=20), b=[-1, (4, )])
        self.assertEqual(
            nest.pack_sequence_as(ntuple, flat_seq), expected_nest)
        self.assertEqual(nest.pack_sequence_as(1, [1]), 1)


class TestFindField(unittest.TestCase):
    def test_find_field(self):
        ntuple = NTuple(a=1, b=NTuple(a=NTuple(a=2, b=3), b=2))
        ret = nest.find_field(ntuple, 'a')
        self.assertEqual(len(ret), 2)
        self.assertEqual(ret[0], ntuple.a)
        self.assertEqual(ret[1], ntuple.b.a)

        ntuple = (1, NTuple(a=NTuple(a=2, b=3), b=2))
        ret = nest.find_field(ntuple, 'a')
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0], ntuple[1].a)

        ntuple = NTuple(a=1, b=[NTuple(a=2, b=3), 2])
        ret = nest.find_field(ntuple, 'a')
        self.assertEqual(len(ret), 2)
        self.assertEqual(ret[0], ntuple.a)
        self.assertEqual(ret[1], ntuple.b[0].a)


class TestNestConcat(alf.test.TestCase):
    def test_nest_concat_tensors(self):
        ntuple = NTuple(
            a=dict(x=torch.zeros((2, 3)), y=torch.zeros((2, 4))),
            b=torch.zeros((2, 10)))
        ret = NestConcat()(ntuple)
        self.assertTensorEqual(ret, torch.zeros((2, 17)))

    def test_nest_concat_specs(self):
        ntuple = NTuple(
            a=dict(x=TensorSpec((2, 3)), y=TensorSpec((2, 4))),
            b=TensorSpec((2, 10)))
        ret = NestConcat()(ntuple)
        self.assertEqual(ret, TensorSpec((2, 17)))


class TestNestSum(alf.test.TestCase):
    def test_nest_sum_tensors(self):
        ntuple = NTuple(
            a=dict(x=torch.zeros(()), y=torch.zeros((2, 4))),
            b=torch.zeros((4, )))
        ret = NestSum()(ntuple)  # broadcasting
        self.assertTensorEqual(ret, torch.zeros((2, 4)))

    def test_nest_sum_specs(self):
        ntuple = NTuple(
            a=dict(x=TensorSpec(()), y=TensorSpec((2, 4))),
            b=TensorSpec((4, )))
        ret = NestSum()(ntuple)  # broadcasting
        self.assertEqual(ret, TensorSpec((2, 4)))


if __name__ == '__main__':
    alf.test.main()
