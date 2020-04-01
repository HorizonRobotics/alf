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

import copy
import gin
import torch

from alf.utils import common


def get_optimizer_with_empty_params(cls):
    """A helper function to construct torch optimizers with
    params as [{'params': []}]. After construction, new parameter
    groups can be adde by using the add_param_group() method.
    """
    NewClsName = cls.__name__ + "_"
    NewCls = type(NewClsName, (cls, ), {})

    @common.add_method(NewCls)
    def __init__(self, **kwargs):
        super(NewCls, self).__init__([{'params': []}], **kwargs)

    return NewCls


Adam = get_optimizer_with_empty_params(torch.optim.Adam)
AdamW = get_optimizer_with_empty_params(torch.optim.AdamW)
SGD = get_optimizer_with_empty_params(torch.optim.SGD)
