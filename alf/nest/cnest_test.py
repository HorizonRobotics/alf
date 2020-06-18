# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
"""Unittest for nest.cpp"""

import time
from collections import OrderedDict
import sys
sys.path.append("./cnest")

import alf
import cnest
from alf.nest.nest_test import NTuple


def _generate_deep_nest(depth=10, keys=['a', 'b']):
    if depth == 1:
        return dict(zip(keys, range(len(keys))))
    x = _generate_deep_nest(depth - 1, keys)
    return dict(zip(keys, [x] * len(keys)))


def _time(func, msg, *args):
    t0 = time.time()
    ret = func(*args)
    print("%s: %s" % (msg, time.time() - t0))
    return ret


class TestIsNamedtuple(alf.test.TestCase):
    def test_is_namedtupe(self):
        x = dict(a=1)
        y = NTuple(a=1, b='x')
        z = (1, 2, '3')

        self.assertFalse(cnest._is_namedtuple(x))
        self.assertFalse(cnest._is_namedtuple(z))
        self.assertTrue(cnest._is_namedtuple(y))


class TestIsUnnamedtuple(alf.test.TestCase):
    def test_is_unnamedtupe(self):
        x = dict(a=1)
        y = NTuple(a=1, b='x')
        z = (1, 2, '3')

        self.assertFalse(cnest._is_unnamedtuple(x))
        self.assertFalse(cnest._is_unnamedtuple(y))
        self.assertTrue(cnest._is_unnamedtuple(z))


class TestAssertSameType(alf.test.TestCase):
    def test_assert_same_type(self):
        x = NTuple(a=1, b=2)
        y = NTuple(a=1, b=1)
        a = [1, 2]
        b = [3]
        c = (1, 2)
        cnest._assert_same_type(x, y)
        cnest._assert_same_type(a, b)
        self.assertRaises(RuntimeError, cnest._assert_same_type, a, c)

        a = dict(x=1)
        b = OrderedDict([('x', 1), ('y', 2)])
        cnest._assert_same_type(a, b)

        self.assertRaises(RuntimeError, cnest._assert_same_type, x, a)

        z = (1, 2)
        self.assertRaises(RuntimeError, cnest._assert_same_type, x, z)


class TestAssertSameLength(alf.test.TestCase):
    def test_assert_same_length(self):
        x = NTuple(a=1, b=2)
        a = [1, 2]
        c = dict(x=3, y=dict(x=1, y=2))

        cnest._assert_same_length(x, a)
        cnest._assert_same_length(a, c)

        self.assertRaises(RuntimeError, cnest._assert_same_length, a, 1)


class TestExtractFieldsFromNest(alf.test.TestCase):
    def test_extract_fields(self):
        x = NTuple(a=dict(x=1), b=1)
        y = dict(aa=1, bb=2)
        z = (1, 2)
        a = {1: 'x', 2: 'y'}

        self.assertTrue(isinstance(cnest._extract_fields_from_nest(x), list))
        self.assertEqual(
            cnest._extract_fields_from_nest(y), [('aa', 1), ('bb', 2)])
        self.assertRaises(RuntimeError, cnest._extract_fields_from_nest, z)
        # only support string keys
        self.assertRaises(RuntimeError, cnest._extract_fields_from_nest, a)


class TestFlatten(alf.test.TestCase):
    def test_flatten_time(self):
        nested = _generate_deep_nest(depth=6, keys=list(map(str, range(10))))
        flat1 = _time(cnest.flatten, "cnest flatten", nested)
        flat2 = _time(alf.nest.py_flatten, "nest flatten", nested)
        self.assertEqual(flat1, flat2)


class TestAssertSameStructure(alf.test.TestCase):
    def test_assert_same_structure_time(self):
        nested = _generate_deep_nest(depth=6, keys=list(map(str, range(10))))
        _time(cnest.assert_same_structure, "cnest assert_same_structure",
              nested, nested)
        _time(alf.nest.py_assert_same_structure, "nest assert_same_structure",
              nested, nested)


class TestMapStructure(alf.test.TestCase):
    def test_map_structure_time(self):
        nested = _generate_deep_nest(depth=6, keys=list(map(str, range(10))))

        ret1 = _time(cnest.map_structure,
                     "cnest map_structure", lambda x, y: x + y, nested, nested)
        ret2 = _time(alf.nest.py_map_structure,
                     "nest map_structure", lambda x, y: x + y, nested, nested)
        self.assertEqual(cnest.flatten(ret1), cnest.flatten(ret2))


class TestPackSequenceAs(alf.test.TestCase):
    def test_pack_sequence_as_time(self):
        nested = _generate_deep_nest(depth=5, keys=list(map(str, range(10))))
        flat = cnest.flatten(nested)

        nest1 = _time(cnest.pack_sequence_as, "cnest pack_sequence_as", nested,
                      flat)
        nest2 = _time(alf.nest.py_pack_sequence_as, "nest pack_sequence_as",
                      nested, flat)
        cnest.assert_same_structure(nest1, nest2)


if __name__ == "__main__":
    alf.test.main()
