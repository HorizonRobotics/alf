# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import torch

_devece_ddtype_tensor_map = {
    'cpu': {
        torch.float32: torch.FloatTensor,
        torch.float64: torch.DoubleTensor,
        torch.float16: torch.HalfTensor,
        torch.uint8: torch.ByteTensor,
        torch.int8: torch.CharTensor,
        torch.int16: torch.ShortTensor,
        torch.int32: torch.IntTensor,
        torch.int64: torch.LongTensor,
        torch.bool: torch.BoolTensor,
    },
    'cuda': {
        torch.float32: torch.cuda.FloatTensor,
        torch.float64: torch.cuda.DoubleTensor,
        torch.float16: torch.cuda.HalfTensor,
        torch.uint8: torch.cuda.ByteTensor,
        torch.int8: torch.cuda.CharTensor,
        torch.int16: torch.cuda.ShortTensor,
        torch.int32: torch.cuda.IntTensor,
        torch.int64: torch.cuda.LongTensor,
        torch.bool: torch.cuda.BoolTensor,
    }
}


def set_default_device(device_name):
    """Set the default device.

    Cannot find a native torch function for setting default device. We have to
    hack our own.

    Args:
        device_name (str): one of ("cpu", "cuda")
    """
    torch.set_default_tensor_type(
        _devece_ddtype_tensor_map[device_name][torch.get_default_dtype()])


def get_default_device():
    return torch._C._get_default_device()


class device(object):
    """Specifies the device for tensors created in this context."""

    def __init__(self, device_name):
        """Create the context with default device with name `device_name`

        Args:
            device_name (str): one of ("cpu", "cuda")
        """
        self._device_name = device_name

    def __enter__(self):
        self._prev_device_name = get_default_device()
        if self._prev_device_name != self._device_name:
            set_default_device(self._device_name)

    def __exit__(self, type, value, traceback):
        if self._prev_device_name != self._device_name:
            set_default_device(self._prev_device_name)
