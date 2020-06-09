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

import random
import time

import alf
from alf import cnest
from alf.data_structures import namedtuple
from alf.nest.nest_test import NTuple


class TestIsNamedtuple(alf.test.TestCase):
    def test_is_namedtupe(self):
        x = dict(a=1)
        y = NTuple(a=1, b='x')
        z = (1, 2, '3')

        self.assertFalse(cnest.is_namedtuple(x))
        self.assertFalse(cnest.is_namedtuple(z))
        self.assertTrue(cnest.is_namedtuple(y))


class TestIsUnnamedtuple(alf.test.TestCase):
    def test_is_unnamedtupe(self):
        x = dict(a=1)
        y = NTuple(a=1, b='x')
        z = (1, 2, '3')

        self.assertFalse(cnest.is_unnamedtuple(x))
        self.assertFalse(cnest.is_unnamedtuple(y))
        self.assertTrue(cnest.is_unnamedtuple(z))


def _generate_deep_nest(depth=10, keys=['a', 'b']):
    if depth == 1:
        return dict(zip(keys, range(len(keys))))
    x = _generate_deep_nest(depth - 1, keys)
    return dict(zip(keys, [x] * len(keys)))


class TestExtractFieldsFromNest(alf.test.TestCase):
    def test_extract_fields(self):
        x = NTuple(a=dict(x=1), b=1)
        y = dict(aa=1, bb=2)
        z = (1, 2)

        self.assertTrue(isinstance(cnest.extract_fields_from_nest(x), list))
        self.assertEqual(
            cnest.extract_fields_from_nest(y), [('aa', 1), ('bb', 2)])
        self.assertRaises(ValueError, cnest.extract_fields_from_nest, z)

    def test_flatten(self):
        x = NTuple(a=dict(x=1), b=dict(a=dict(x=2), y=NTuple(a='x', b='y')))
        self.assertTrue(cnest.flatten(x), [1, 2, 'x', 'y'])

    def test_extract_fields_time(self):
        N = 1000
        keys = [''.join(random.choices('dfsfcxve', k=5)) for _ in range(N)]
        values = range(N)
        x = dict(zip(keys, values))

        t0 = time.time()
        for i in range(10):
            alf.nest.extract_fields_from_nest(x)
        print("nest extract_fields_from_nest: %s" % (time.time() - t0))

        t0 = time.time()
        for i in range(10):
            cnest.extract_fields_from_nest(x)
        print("cnest extract_fields_from_nest: %s" % (time.time() - t0))

    def test_flatten_time(self):
        nested = _generate_deep_nest(depth=5, keys=list(map(str, range(10))))

        t0 = time.time()
        flat1 = alf.nest.flatten(nested)
        print("nest flatten: %s" % (time.time() - t0))

        t0 = time.time()
        flat2 = cnest.flatten(nested)
        print("cnest flatten: %s" % (time.time() - t0))

        self.assertEqual(flat1, flat2)


if __name__ == "__main__":
    alf.test.main()
