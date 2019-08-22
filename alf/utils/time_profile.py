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

import os
import functools
import tensorflow as tf
from tf_agents.utils import common as tfa_common
from alf.utils import common


def _get_func_qualname(func):
    return getattr(func, 'python_function', func).__qualname__


def timer(
        record=os.environ.get('TIMER_RECORD', False),
        interval=int(os.environ.get('TIMER_INTERVAL', 101)),
        name=None):
    """Decorator to record time cost for a function

    Args:
        record (bool): A bool whether to record time cost for function, we can specify
            it explicitly or config through environment variable `TIMER_RECORD`. It's used
            as a tool function for profiling, set it to `True` only in debugging case
        interval (int): record time cost every so many interval
        name (str): name for this timer
    """

    def wrapper(func):
        if not record:
            return func

        func_name = name or _get_func_qualname(func)
        counter = tfa_common.create_variable("counter", unique_name=False)

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            time_start = tf.timestamp()
            ret = func(*args, **kwargs)
            time_cost = tf.timestamp() - time_start

            count = counter.assign_add(1, use_locking=True)

            def _summary():
                tf.print('Time/%s:' % func_name, time_cost, 'step:', counter)

                # todo Investigate why tf.summary can not work here when decorated function
                # was called inside a tf.while_loop

                # with tf.summary.record_if(True):
                #     tf.summary.scalar(
                #         'time/%s' % func_name, data=time_cost, step=counter)

            common.run_if(tf.equal(tf.math.mod(count, interval), 0), _summary)

            return ret

        return _wrapper

    return wrapper
