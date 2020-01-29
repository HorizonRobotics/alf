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
"""TrustedUpdater."""

from typing import Callable

import tensorflow as tf

from alf.utils import math_ops

nest_map = tf.nest.map_structure


class TrustedUpdater(object):
    """Adjust variables based on the change calculated by `change_f()`

    The motivation is that if some quatity changes too much after an SGD update,
    the SGD step might be too big. We want to shink that step so that the
    concerned quatity does not change too much. We can also monitor multiple
    quantities to make sure none of them has sudden big jump.

    It adjusts variables provided at `__init__` if the change calculated by
    `change_f` is too big:
    ```
    change = change_f()
    if change > max_change:
        var <= old_var + 0.9 * (max_change/change) * (var - old_var)
    ```
    The above procedure is repeated until `change` is not bigger than
    `max_change`. Note that `change` and `max_change` can be nests of
    scalars. In this case, the inequality is understood as if any one of the
    changes is greater than its corresponding max_change.
    """

    def __init__(self, variables):
        """Create a TrustedUpdater instance.

        Args:
            varialbes (list[Variables]): variables to be monitored.
        """
        self._variables = variables
        assert len(self._variables) > 0
        self._prev_variables = [
            tf.Variable(initial_value=v, dtype=v.dtype) for v in variables
        ]

    @tf.function
    def adjust_step(self, change_f: Callable, max_change):
        """Adjust `variables` based change calculated by change_f

        This function will copy the new values of the variables to
        a backup to be used for the next call of adjust_step.
        Args:
            change_f (Callable): a function calculate a (nested) change based on
                current variable.
            max_change (float): (nested) max change allowed.
        Returns:
            the initial change before variables are adjusted
            the number of steps to adjust variables. 0 for no adjustment
        """

        def _adjust_step(ratio):
            # 0.9 is to prevent infinite loop when `actual_change` is close to
            # `max_change`
            r = 0.9 / ratio
            for var, prev_var in zip(self._variables, self._prev_variables):
                var.assign(prev_var + r * (var - prev_var))

        steps = tf.zeros((), tf.int32)
        change0 = change_f()
        change = change0
        ratio = nest_map(lambda c, m: tf.abs(c) / m, change, max_change)
        ratio = math_ops.max_n(tf.nest.flatten(ratio))
        while ratio > 1. and steps < 100:
            _adjust_step(ratio)
            change = change_f()
            ratio = nest_map(lambda c, m: tf.abs(c) / m, change, max_change)
            ratio = math_ops.max_n(tf.nest.flatten(ratio))
            steps += 1
        # This suggests something wrong. change cannot be reduced by making
        # the step smaller.
        tf.Assert(steps < 100, [steps])

        for var, prev_var in zip(self._variables, self._prev_variables):
            prev_var.assign(var)

        return change0, steps
