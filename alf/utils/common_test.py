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

from absl import logging
from contextlib import redirect_stderr
from io import StringIO

import alf
import alf.utils.common as common


class WraningOnceTest(alf.test.TestCase):
    def test_warning_once(self):
        warning_messages = ["warning message 1", "warning message 2"]

        # omit non-customized logging messages
        logging._warn_preinit_stderr = 0

        with StringIO() as log_stream, redirect_stderr(log_stream):
            for _ in range(10):
                common.warning_once(warning_messages[0])
                common.warning_once(warning_messages[1])
            generated_warning_messages = log_stream.getvalue()

        generated_warning_messages = generated_warning_messages.rstrip().split(
            '\n')

        # previouly we only get one warining message here, although
        # warning once has been called multiple times at difference places
        assert len(warning_messages) == len(generated_warning_messages)
        for msg, gen_msg in zip(warning_messages, generated_warning_messages):
            assert msg in gen_msg


if __name__ == '__main__':
    alf.test.main()
