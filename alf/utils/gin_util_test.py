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

import gin
import unittest
import numpy as np
import alf.utils.external_configurables


class GinUtilsTest(unittest.TestCase):
    """Tests for alf.utils.gin_utils
    """

    def test_refer_to_unregistered_func(self):
        """Test refer to unregister func by eval it with registered helper function
        `gin_eval`.

        builtin and 3rd-party functions are not registered in gin by default,
        we can not refer to it directly and it's also not unrealistic to register
        it explict by `gin.external_configurable(func)` for all functions we might
        refer to
        """

        @gin.configurable
        def _test(func):
            func()

        with self.assertRaises(ValueError):
            gin.parse_config([
                "_test.func=@list",
            ])

        gin.parse_config(
            ["list/gin_eval.source='list'", "_test.func=@list/gin_eval()"])
        _test()

    def test_refer_to_local_values(self):
        """Test refer to local values

        Passing expression as parameter value
        """

        @gin.configurable
        def _add(a, b):
            return a + b

        def _test_add(a, b):
            return _add()

        gin.parse_config([
            "a/gin_eval.source='a'", "_add.a=@a/gin_eval()",
            "b/gin_eval.source='b'", "_add.b=@b/gin_eval()"
        ])

        self.assertEqual(1 + 2, _test_add(1, 2))
        self.assertEqual(2 + 3, _test_add(2, 3))

        @gin.configurable
        def _value(value):
            return value

        gin.parse_config([
            "_value/gin_eval.source='np.prod([a,b])'",
            "_value.value=@_value/gin_eval()"
        ])
        a, b = 1, 2
        self.assertEqual(a * b, _value())
        a, b = 3, 4
        self.assertEqual(a * b, _value())

    def tearDown(self):
        gin.clear_config()


if __name__ == '__main__':
    unittest.main()
