# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import torch.nn as nn
from typing import Any


def get_opt_arg(p: nn.Parameter, argname: str, default: Any = None):
    """Get parameter specific optimizer arguments.

    Args:
        p: the parameter
        argname: name of the argument
        default: the default value
    Returns:
        The parameter specific value if it is found, otherwise default
    """
    opt_args = getattr(p, 'opt_args', None)
    if opt_args is None:
        return default
    value = opt_args.get(argname, None)
    return default if value is None else value
