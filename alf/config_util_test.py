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

from absl import logging
import pprint
import alf


@alf.configurable
def test(a, b=123):
    return a, b


@alf.configurable(blacklist=['b'])
def test_func(a, b=100, c=200):
    return a, b, c


@alf.configurable(blacklist=['b'])
def test_func2(a, b=100, c=200):
    return a, b, c


@alf.configurable(blacklist=['b'])
def test_func3(a, b=100, c=200):
    return a, b, c


@alf.configurable("Test.FancyTest")
def test_func4(arg=10):
    return arg


def test_func5(arg=10):
    return arg


def test_func6(arg=10):
    return arg


def test_func7(arg=10):
    return arg


@alf.configurable
class Test(object):
    def __init__(self, a, b, c=10):
        self._a = a
        self._b = b
        self._c = c

    @alf.configurable(whitelist=['c'])
    def func(self, a, b=10, c=100):
        return a, b, c

    def __call__(self):
        return self._a, self._b, self._c


@alf.configurable
class Test2(Test):
    def __init__(self, a, b):
        super().__init__(a, b, 5)

    def func(self, a):
        return a

    def __call__(self):
        return self._a + 1, self._b + 2, self._c + 3


@alf.configurable
class Test3(Test):
    def func(self, a):
        return a

    def __call__(self):
        return self._a - 1, self._b - 2, self._c - 3


class ConfigTest(alf.test.TestCase):
    def test_config1(self):

        # Test simple function
        self.assertEqual(test(1), (1, 123))
        alf.config({'test.a': 3})
        self.assertEqual(test(), (3, 123))
        # Test config after the value has been used
        self.assertRaisesRegex(ValueError, "test.b' has already been used",
                               alf.config, {'test.b': None})

        # Test class
        obj = Test(1, 2)
        self.assertEqual(obj(), (1, 2, 10))
        alf.config({'Test.b': 5})
        obj = Test(1)
        self.assertEqual(obj(), (1, 5, 10))

        self.assertRaises(ValueError, alf.config1, 'c', 32)

        # Test class member function
        # Test whitelist
        alf.config({'Test.func.c': 30})
        # Test whitelist
        self.assertRaises(ValueError, alf.config1, 'Test.func.b', 32)
        self.assertEqual(obj.func(1, 2, 3), (1, 2, 3))
        self.assertEqual(obj.func(3, 5), (3, 5, 30))
        self.assertRaisesRegex(TypeError, "missing 1 required positional",
                               obj.func)

        # Test blacklist
        self.assertRaises(ValueError, alf.config1, 'test_func.b', 30)
        alf.config1('test_func.c', 15)
        self.assertEqual(test_func(10, 20), (10, 20, 15))

        # Test explicit name for function
        alf.config("FancyTest", arg=3)
        self.assertEqual(test_func4(), 3)

        # Test name conflict: long vs. short
        self.assertRaisesRegex(
            ValueError, "'A.Test.FancyTest.arg' conflicts "
            "with existing config name 'Test.FancyTest.arg'",
            alf.configurable("A.Test.FancyTest"), test_func5)

        # Test name conflict: short vs. long
        alf.configurable("A.B.C.D.test")(test_func6)
        self.assertRaisesRegex(
            ValueError, "'B.C.D.test.arg' conflicts "
            "with existing config name 'A.B.C.D.test.arg'",
            alf.configurable("B.C.D.test"), test_func7)
        alf.config('D.test', arg=5)

        # Test name conflict: same
        # Note: this exception is raised by gin instead of alf. Need to change
        # it if gin is removed in the future.
        self.assertRaisesRegex(
            ValueError, "A configurable matching "
            "'A.B.C.D.test' already exists.", alf.configurable("A.B.C.D.test"),
            test_func5)

        # Test duplicated config
        with self.assertLogs() as ctx:
            alf.config1('test_func2.c', 15)
            alf.config1('test_func2.c', 16)
            warning_message = ctx.records[0]
            self.assertTrue("replaced" in str(warning_message))
        self.assertEqual(test_func2(1), (1, 100, 16))

        # Test mutable
        alf.config1('test_func3.c', 15, mutable=False)
        with self.assertLogs() as ctx:
            alf.config1('test_func3.c', 16)
            warning_message = ctx.records[0]
            self.assertTrue("ignored" in str(warning_message))
        self.assertRaisesRegex(
            ValueError,
            "cannot change it",
            alf.config,
            'test_func3',
            mutable=False,
            c=17)
        self.assertEqual(test_func3(1), (1, 100, 15))

        # Test the right constructor is used for subclass
        obj2 = Test2(7, 9)
        self.assertEqual(obj2(), (8, 11, 8))

        # Test subclass using constructor from base
        obj3 = Test3(1, 2)
        self.assertEqual(obj3(), (0, 0, 7))

        operative_configs = alf.get_operative_configs()
        logging.info("get_operative_configs(): \n%s" %
                     pprint.pformat(operative_configs))
        self.assertTrue('Test.FancyTest.arg' in dict(operative_configs))
        inoperative_configs = alf.get_inoperative_configs()
        logging.info("get_inoperative_configs(): \n%s" %
                     pprint.pformat(inoperative_configs))
        self.assertTrue('A.B.C.D.test.arg' in dict(inoperative_configs))


if __name__ == '__main__':
    alf.test.main()
