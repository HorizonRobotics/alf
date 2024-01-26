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
import os
import pprint
import tempfile
import alf
from alf.test_configs.source_code import *


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
        self.assertRaisesRegex(ValueError,
                               "'A.B.C.D.test.arg' has already been defined.",
                               alf.configurable("A.B.C.D.test"), test_func5)

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
        self.assertEqual(test_func3(1), (1, 100, 15))

        # Test the right constructor is used for subclass
        obj2 = Test2(7, 9)
        self.assertEqual(obj2(), (8, 11, 8))

        # Test subclass using constructor from base
        obj3 = Test3(1, 2)
        self.assertEqual(obj3(), (0, 0, 7))

        # Test class with __new__ and __init__
        # A singleton class is such a class.
        alf.config('Test4', c=12)
        obj4 = Test4(1, 2)
        self.assertEqual(obj4(), (1, 2, 12))

        # test pre_config for config not defined yet
        alf.pre_config({'test_func8.c': 10})
        self.assertRaisesRegex(ValueError,
                               "A pre-config 'test_func8.c' was not handled",
                               alf.validate_pre_configs)
        func11 = alf.configurable(test_func11)
        # test pre_config for config already defined
        alf.pre_config({'test_func11.a': 5})
        func8 = alf.configurable(test_func8)
        alf.validate_pre_configs()
        self.assertEqual(func8(), (1, 2, 10))
        self.assertEqual(func11(), (5, 2, 3))

        # test ambiguous pre_config
        alf.pre_config({'test_f.a': 10})
        func9 = alf.configurable("ModuleA.test_f")(test_func9)
        func10 = alf.configurable("ModuleB.test_f")(test_func10)
        self.assertRaisesRegex(ValueError,
                               "config name 'test_f.a' is ambiguous",
                               alf.validate_pre_configs)

        operative_configs = alf.get_operative_configs()
        logging.info("get_operative_configs(): \n%s" %
                     pprint.pformat(operative_configs))
        self.assertTrue('Test.FancyTest.arg' in dict(operative_configs))
        inoperative_configs = alf.get_inoperative_configs()
        logging.info("get_inoperative_configs(): \n%s" %
                     pprint.pformat(inoperative_configs))
        self.assertTrue('A.B.C.D.test.arg' in dict(inoperative_configs))

    def test_repr_wrapper(self):
        a = MyClass(1, 2)
        self.assertEqual(repr(a), "MyClass(1, 2)")
        a = MyClass(3, 5, d=300)
        self.assertEqual(repr(a), "MyClass(3, 5, d=300)")
        b = MySubClass(6)
        self.assertEqual(repr(b), 'MySubClass(6)')

    def test_load_config(self):
        alf.reset_configs()
        dir = os.path.dirname(__file__)
        conf_file = os.path.join(dir, "test_configs/conf_dir/test_conf.py")
        self.assertRaisesRegex(ValueError, "Cannot find conf file",
                               alf.load_config, conf_file)

        alf.reset_configs()
        alf.pre_config({"test_func.a": 12345})
        os.environ['ALF_CONFIG_PATH'] = os.path.join(dir, "test_configs")
        alf.load_config(conf_file)
        self.assertEqual(alf.get_config_value("test_func.a"), 12345)
        self.assertEqual(alf.get_config_value("test_func2.a"), 21)
        self.assertEqual(alf.get_config_value("test_func3.a"), 31)
        self.assertEqual(alf.get_config_value("Test.FancyTest.arg"), 81)
        self.assertEqual(alf.get_config_value("test_func.c"), 101)

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_conf_file = os.path.join(temp_dir, "alf_config.py")
            alf.save_config(saved_conf_file)
            alf.reset_configs()
            alf.load_config(saved_conf_file)
            self.assertEqual(alf.get_config_value("test_func.a"), 12345)
            self.assertEqual(alf.get_config_value("test_func2.a"), 21)
            self.assertEqual(alf.get_config_value("test_func3.a"), 31)
            self.assertEqual(alf.get_config_value("Test.FancyTest.arg"), 81)
            self.assertEqual(alf.get_config_value("test_func.c"), 101)
            os.path.exists(os.path.join(temp_dir, "alf_config.py"))
            os.path.exists(os.path.join(temp_dir, "configs", "test_conf.py"))
            os.path.exists(os.path.join(temp_dir, "configs", "base_conf.py"))
            os.path.exists(
                os.path.join(temp_dir, "configs", "base", "base_conf.py"))


if __name__ == '__main__':
    alf.test.main()
