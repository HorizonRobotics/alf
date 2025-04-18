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

from contextlib import contextmanager
from torch.nn import Module

# ALF Algorithm will overwrite these functions, we save the original ones.
old_state_dict = Module.state_dict
old_load_state_dict = Module.load_state_dict
old__save_to_state_dict = Module._save_to_state_dict
old__load_from_state_dict = Module._load_from_state_dict


@contextmanager
def original_torch_module_functions():
    """A context manager for restoring some key original nn.Module functions that
    have been overwritten by ALF.

    This can be used when we are trying to load a pretrained huggingface model which
    require a newer and original ``torch.nn.Module``.

    Example:

    .. code-block:: python

        with original_torch_module_functions():
            model = hf_model_load()
    """
    keys = [
        'state_dict', 'load_state_dict', '_save_to_state_dict',
        '_load_from_state_dict'
    ]
    current_funcs = {k: getattr(Module, k) for k in keys}
    old_funcs = {k: globals()['old_' + k] for k in keys}
    for k in keys:
        setattr(Module, k, old_funcs[k])
    yield
    for k in keys:
        setattr(Module, k, current_funcs[k])
