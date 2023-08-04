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
"""Patch torch.nn.Module for better performance.

``torch.nn.Module.__getattr__`` is frequently used by all class derived from
``nn.Module``. It can introduce too much unnecessary overhead. So we patch
``nn.Module`` class to explicitly store Parameter/Module as attribute so
that `__getattr__` won't be triggered. This patch can speed up the access
of ``Module`` or ``Parameter`` attributes by more than 10x times.
"""

import torch
from torch.nn import Module

_old_Module__setattr__ = torch.nn.Module.__setattr__


def _new_Module__setattr__(self, name, value):
    _old_Module__setattr__(self, name, value)
    object.__setattr__(self, name, value)


Module.__setattr__ = _new_Module__setattr__

old_register_parameter = torch.nn.Module.register_parameter


def _new_register_parameter(self, name, param):
    old_register_parameter(self, name, param)
    object.__setattr__(self, name, param)


Module.register_parameter = _new_register_parameter

old_register_buffer = torch.nn.Module.register_buffer


def _new_register_buffer(self, name, param):
    old_register_buffer(self, name, param)
    object.__setattr__(self, name, param)


Module.register_buffer = _new_register_buffer

old_add_module = torch.nn.Module.add_module


def _new_add_module(self, name, module):
    old_add_module(self, name, module)
    object.__setattr__(self, name, module)


Module.add_module = _new_add_module
