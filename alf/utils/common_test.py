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
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn

import alf
import alf.utils.common as common


class WraningOnceTest(alf.test.TestCase):
    def setUp(self):
        logging.use_absl_handler()

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

        # previously we only get one warning message here, although
        # warning once has been called multiple times at difference places
        assert len(warning_messages) == len(generated_warning_messages)
        for msg, gen_msg in zip(warning_messages, generated_warning_messages):
            assert msg in gen_msg


class MyObject(object):
    def __init__(self):
        self._list = [alf.layers.FC(3, 4), alf.layers.FC(4, 5)]
        self._dict = {
            "a": nn.Parameter(torch.zeros(3, 4)),
            4: nn.Parameter(torch.zeros(4, 5)),
        }
        self._a = nn.Parameter(torch.zeros(3))
        self._list2 = self._list
        self._dict2 = self._dict


class GetAllParametersTest(alf.test.TestCase):
    def test_get_all_parameters(self):
        obj = MyObject()
        names = set([
            '_a', '_dict.a', '_dict.4', '_list.0._weight', '_list.0._bias',
            '_list.1._weight', '_list.1._bias'
        ])
        params = common.get_all_parameters(obj)
        for name, p in params:
            self.assertTrue(name in names)
        self.assertEqual(len(names), len(params))


class _TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # create a CPU tensor
        self.x = torch.zeros([2]).to('cpu')
        self.y = np.zeros([2])
        if torch.cuda.is_available():
            self.z = torch.zeros([2]).cuda()


def _test_worker(m_):
    m_.x[:] = 1.
    m_.y[:] = 1.
    if torch.cuda.is_available():
        m_.z[:] = 1.


def _test_tensor_sharing():
    """This function is for testing whether tensors are automatically moved
    to shared memory, even when ``Module.share_memory()`` or ``Tensor.share_memory_()``
    has not been called, especially for CPU tensors.

    The official documentation seems to hint that CPU tensors should be shared via
    the explicit call ``Module.share_memory()`` or ``Tensor.share_memory_()``,
    before being passed to a child process. However, our finding is that they will
    be automatically moved to shared memory. This is an undocumented behavior, and
    we want to test this for different ALF users.
    """
    m = _TestModule()
    assert not m.x.is_shared()
    if torch.cuda.is_available():
        # CUDA tensor is always shared
        assert m.z.is_shared()

    start_method = mp.get_start_method()
    mp.set_start_method('spawn', force=True)
    # Change ``m`` in the child process
    process = mp.Process(target=_test_worker, args=(m, ))
    process.start()
    process.join()

    # numpy array should not be modified
    assert np.all(m.y == np.zeros([2]))
    # cuda tensor should be modified
    if torch.cuda.is_available():
        assert torch.all(m.z.cpu() == torch.ones([2]).cpu())
    # check that ``m``'s tensor also been modified in the parent process
    assert m.x.is_shared() and torch.all(m.x == torch.ones([2]).cpu()), (
        "Your pytorch version has a different behavior of sharing CPU tensors "
        "between processes. Please report the version to the ALF team.")

    mp.set_start_method(start_method, force=True)


class TensorSharingTest(alf.test.TestCase):
    def test_tensor_sharing(self):
        _test_tensor_sharing()


if __name__ == '__main__':
    alf.test.main()
