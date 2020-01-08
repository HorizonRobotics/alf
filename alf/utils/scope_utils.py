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

import tensorflow as tf


def get_current_scope():
    """Returns the current name scope in the default_graph.

    For example:
    ```python
    with tf.name_scope('scope1'):
        with tf.name_scope('scope2'):
            print(get_current_scope())
    ```
    would print the string `scope1/scope2/`.

    Returns:
        A string representing the current name scope.
    """
    with tf.name_scope("foo") as scope:
        # With the above example, scope is "scope1/scope2/foo_1/".
        # We want to return "scope1/scope2/".
        return scope[:scope.rfind('/', 0, -1) + 1]
