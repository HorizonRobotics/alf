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

from absl.testing import parameterized

import alf
import alf.nest as nest
import cnest
from alf.data_structures import namedtuple
from alf.tensor_specs import TensorSpec
from alf.nest.utils import NestConcat, NestSum, NestMultiply, transform_nest

NTuple = namedtuple('NTuple', ['a', 'b'])  # default value will be None


class TestIsNested(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(nest.is_nested, cnest._is_nested)
    def test_is_nested(self, is_nested):
        self.assertFalse(is_nested(1))
        self.assertFalse(is_nested(None))
        self.assertTrue(is_nested(dict(x=1)))
        self.assertTrue(is_nested([1]))
        ntuple = NTuple(a=1, b=NTuple(a=NTuple(a=(2, ), b=[3]), b=dict(x=2)))
        self.assertTrue(is_nested(ntuple))


class TestFlatten(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(nest.py_flatten, cnest.flatten)
    def test_flatten(self, flatten):
        ntuple = NTuple(a=1, b=NTuple(a=NTuple(a=(2, ), b=[3]), b=dict(x=2)))
        expected_flat_seq = [1, 2, 3, 2]
        self.assertEqual(flatten(ntuple), expected_flat_seq)
        self.assertEqual(flatten(1), [1])
        x = NTuple(a=dict(x=1), b=dict(a=dict(x=2), y=NTuple(a='x', b='y')))
        self.assertTrue(flatten(x), [1, 2, 'x', 'y'])


class TestFlattenUpTo(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((nest.py_flatten_up_to, AssertionError),
                              (nest.flatten_up_to, RuntimeError),
                              (cnest.flatten_up_to, RuntimeError))
    def test_flatten_up_to(self, flatten_up_to, error):
        ntuple = NTuple(a=1, b=NTuple(a=NTuple(a=(2, ), b=[3]), b=dict(x=2)))

        shallow_nest = 1
        self.assertEqual(flatten_up_to(shallow_nest, ntuple), [ntuple])
        shallow_nest = NTuple(a=1, b=2)
        self.assertEqual(flatten_up_to(shallow_nest, ntuple), [1, ntuple.b])
        shallow_nest = NTuple(a=1, b=NTuple(a=1, b=dict(x=3)))
        self.assertEqual(
            flatten_up_to(shallow_nest, ntuple), [1, ntuple.b.a, 2])

        shallow_nest = NTuple(a=dict(x=1), b=1)
        self.assertRaises(error, flatten_up_to, shallow_nest, ntuple)
        shallow_nest = NTuple(a=1, b=NTuple(a=1, b=dict(y=3)))
        self.assertRaises(error, flatten_up_to, shallow_nest, ntuple)


class TestAssertSameStructure(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((nest.py_assert_same_structure, AssertionError),
                              (nest.assert_same_structure, RuntimeError),
                              (cnest.assert_same_structure, RuntimeError))
    def test_assert_same_structure(self, assert_same_structure, error):
        assert_same_structure(1.0, 10)
        nest1 = NTuple(
            a=1,
            b=NTuple(a=NTuple(a=(2, ), b=[3]), b=dict(x=NTuple(a=[100], b=1))))
        nest2 = NTuple(
            b=NTuple(
                a=NTuple(a=(2, ), b=[300]), b=dict(x=NTuple(a=[1], b=100))),
            a=3.0)
        assert_same_structure(nest1, nest2)
        assert_same_structure(
            dict(x=1, y=NTuple(a=[1], b=3)), dict(y=NTuple(a=[3], b=1), x=1))
        self.assertRaises(error, assert_same_structure, dict(x=1, y=[2]),
                          dict(x=[2], y=1))
        self.assertRaises(error, assert_same_structure, dict(y=[2]),
                          dict(x=[2]))
        self.assertRaises(error, assert_same_structure, dict(x=1, y=[2]),
                          dict(y=(2, ), x=1))
        self.assertRaises(error, assert_same_structure, dict(x=1, y=[2]), 1)
        self.assertRaises(error, assert_same_structure, NTuple(a=1, b=[2]),
                          NTuple(b=[2], a=[1]))
        self.assertRaises(error, assert_same_structure, [1, 2], [[1, 2]])
        self.assertRaises(error, assert_same_structure, [1, [2]], [1, [2, 2]])
        self.assertRaises(error, assert_same_structure, dict(x=1, y=dict(x=2)),
                          dict(x=1, y=dict(x=3, y=1)))


class TestMapStructure(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((nest.py_map_structure, AssertionError),
                              (cnest.map_structure, RuntimeError))
    def test_map_structure(self, map_structure, error):
        nest1 = NTuple(a=dict(x=3, y=2), b=[100.0, (5, )])
        nest2 = NTuple(a=dict(x=1, y=-2), b=[100.0, (10, )])
        nest3 = NTuple(a=dict(x=1, y=-2), b=[50.0, (6, )])
        expected_result = NTuple(a=dict(x=5, y=-2), b=[250.0, (21, )])
        self.assertEqual(
            map_structure(lambda a, b, c: a + b + c, nest1, nest2, nest3),
            expected_result)
        self.assertEqual(
            map_structure(lambda a, b: a + b, [1, 3], [4, 5]), [5, 8])
        self.assertEqual(map_structure(lambda a, b: a * b, 1, 3), 3)

        add = lambda a, b: a + b
        self.assertRaises(error, map_structure, add, [1], 2)
        self.assertRaises(error, map_structure, add, [1], (2, ))
        self.assertRaises(error, map_structure, add, [1, 2, 3],
                          [1, dict(x=1), 3])
        self.assertRaises(error, map_structure, add, dict(a=0, x=1),
                          dict(y=1, a=0))


class TestFastMapStructure(alf.test.TestCase):
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


class TestMapStructureUpTo(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((nest.py_map_structure_up_to, AssertionError),
                              (nest.map_structure_up_to, RuntimeError),
                              (cnest.map_structure_up_to, RuntimeError))
    def test_different_keys(self, map_structure_up_to, error):
        self.assertRaises(error, map_structure_up_to,
                          dict(x=1, z=2), lambda a, b: a * b, dict(x=1, y=2),
                          dict(y=1, x=2))
        self.assertRaises(error, map_structure_up_to,
                          dict(x=1, y=2), lambda a, b: a * b, dict(x=1, y=2),
                          dict(y=1, z=2))

    @parameterized.parameters((nest.py_map_structure_up_to, AssertionError),
                              (nest.map_structure_up_to, RuntimeError),
                              (cnest.map_structure_up_to, RuntimeError))
    def test_different_lengths(self, map_structure_up_to, error):
        self.assertRaises(error, map_structure_up_to,
                          [1, 2, 3], lambda x: x * 2, [1, 2])
        self.assertRaises(error, map_structure_up_to,
                          [1, 2], lambda x, y: x * y, [1, 2, 3], [4, 5])
        self.assertRaises(error, map_structure_up_to,
                          [1, [2, 3]], lambda x: x * 2,
                          [[1], [[2, 4], [3, 5], 3]])

    @parameterized.parameters(nest.py_map_structure_up_to,
                              cnest.map_structure_up_to)
    def test_map_structure_to(self, map_structure_up_to):
        shallow_nest = [[None], None]
        inp_val = [[1], 2]
        out = map_structure_up_to(shallow_nest, lambda x: 2 * x, inp_val)
        self.assertEqual(out, [[2], 4])

        shallow_nest = [None, None]
        inp_val = [[1], 2]
        out = map_structure_up_to(shallow_nest, lambda x: 2 * x, inp_val)
        self.assertEqual(out, [[1, 1], 4])

        data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
        name_list = ['evens', ['odds', 'primes']]
        out = map_structure_up_to(
            name_list, lambda name, sec: "first_{}_{}".format(len(sec), name),
            name_list, data_list)
        self.assertEqual(out,
                         ['first_4_evens', ['first_5_odds', 'first_3_primes']])

        ab_tuple = namedtuple("ab_tuple", "a, b")
        op_tuple = namedtuple("op_tuple", "add, mul")
        inp_val = ab_tuple(a=2, b=3)
        inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
        out = map_structure_up_to(
            inp_val, lambda val, ops: (val + ops.add) * ops.mul, inp_val,
            inp_ops)
        self.assertEqual(out, ab_tuple(a=6, b=15))


class TestPackSequenceAs(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(nest.py_pack_sequence_as, cnest.pack_sequence_as)
    def test_pack_sequence_as(self, pack_sequence_as):
        ntuple = NTuple(a=dict(x=3, y=2), b=[100.0, (5, )])
        flat_seq = [30, 20, -1, 4]
        expected_nest = NTuple(a=dict(x=30, y=20), b=[-1, (4, )])
        self.assertEqual(pack_sequence_as(ntuple, flat_seq), expected_nest)
        self.assertEqual(len(flat_seq), 4)  # no side effect on ``flat_seq``
        self.assertEqual(pack_sequence_as(1, [1]), 1)


class TestFindField(alf.test.TestCase):
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


class TestNestMultiply(alf.test.TestCase):
    def test_nest_multiply_tensors(self):
        ntuple = NTuple(
            a=dict(x=torch.zeros(()), y=torch.ones((2, 4))),
            b=torch.ones((4, )))
        ret = NestMultiply()(ntuple)  # broadcasting
        self.assertTensorEqual(ret, torch.zeros((2, 4)))

    def test_nest_multiply_specs(self):
        ntuple = NTuple(
            a=dict(x=TensorSpec(()), y=TensorSpec((2, 4))),
            b=TensorSpec((4, )))
        ret = NestMultiply()(ntuple)  # broadcasting
        self.assertEqual(ret, TensorSpec((2, 4)))


class TestPruneNestLike(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((nest.py_prune_nest_like, ValueError),
                              (nest.prune_nest_like, RuntimeError),
                              (cnest.prune_nest_like, RuntimeError))
    def test_prune_nest_like(self, prune_nest_like, error):
        ntuple = NTuple(
            a=dict(x=torch.zeros(()), y=torch.zeros((2, 4))),
            b=NTuple(a=torch.zeros((4, )), b=[1]))
        spec = NTuple(a=dict(y=TensorSpec(())), b=NTuple(b=[TensorSpec(())]))
        pruned_ntuple = prune_nest_like(ntuple, spec)

        nest.map_structure(
            self.assertEqual, pruned_ntuple,
            NTuple(a=dict(y=torch.zeros((2, 4))), b=NTuple(b=[1])))

        lst1 = [1, 3]
        lst2 = [None, 1]
        pruned_lst = prune_nest_like(lst1, lst2)
        self.assertEqual(pruned_lst, [None, 3])

        tuple1 = NTuple(a=1, b=2)
        tuple2 = NTuple(b=1, a=())
        pruned_lst = prune_nest_like(tuple1, tuple2, value_to_match=())
        self.assertEqual(pruned_lst, NTuple(a=(), b=2))

        d1 = dict(x=1, y=2)
        d2 = dict(x=1, z=2)
        self.assertRaises(error, prune_nest_like, d1, d2)


class TestTransformNest(alf.test.TestCase):
    def test_transform_nest(self):
        ntuple = NTuple(
            a=dict(x=torch.zeros(()), y=torch.zeros((2, 4))),
            b=torch.zeros((4, )))
        transformed_ntuple = transform_nest(
            ntuple, field='a.x', func=lambda x: x + 1.0)
        ntuple.a.update({'x': torch.ones(())})
        nest.map_structure(self.assertEqual, transformed_ntuple, ntuple)

        ntuple = NTuple(
            a=dict(x=torch.zeros(()), y=torch.zeros((2, 4))),
            b=NTuple(a=torch.zeros((4, )), b=NTuple(a=[1], b=[1])))
        transformed_ntuple = transform_nest(
            ntuple, field='b.b.b', func=lambda _: [2])
        ntuple = ntuple._replace(
            b=ntuple.b._replace(b=ntuple.b.b._replace(b=[2])))
        nest.map_structure(self.assertEqual, transformed_ntuple, ntuple)

        ntuple = NTuple(a=1, b=2)
        transformed_ntuple = transform_nest(ntuple, None, NestSum())
        self.assertEqual(transformed_ntuple, 3)


class TestExtractAnyLeaf(alf.test.TestCase):
    def test_extract_any_leaf(self):
        nested = NTuple(a=dict(x=3, y=1), b=2)
        self.assertTrue(
            isinstance(nest.extract_any_leaf_from_nest(nested), int))
        self.assertEqual(nest.extract_any_leaf_from_nest([]), None)
        self.assertEqual(nest.extract_any_leaf_from_nest(2), 2)


if __name__ == '__main__':
    alf.test.main()
