# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import torch.nn as nn


def _avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    return torch.lerp(model_parameter, averaged_model_parameter,
                      1 / (num_averaged + 1))


class DoubleAveragedModel(nn.Module):
    """
    Use double average to compensate the limited float precision.
    """

    def __init__(self, model, device=None, use_buffers=False):
        super().__init__()
        self._averaged_model = torch.optim.swa_utils.AveragedModel(
            model, device, _avg_fn, use_buffers)
        self._double_averaged_model = torch.optim.swa_utils.AveragedModel(
            self._averaged_model, device, _avg_fn, use_buffers)
        self.register_buffer("_double_average_period",
                             torch.tensor(8, dtype=torch.int64))

    def forward(self, *args, **kwargs):
        if self._double_averaged_model.n_averaged == 0:
            return self._averaged_model(*args, **kwargs)
        else:
            return self._double_averaged_model(*args, **kwargs)

    def update_parameters(self, model):
        self._averaged_model.update_parameters(model)
        if self._averaged_model.n_averaged == self._double_average_period:
            self._double_averaged_model.update_parameters(self._averaged_model)
            self._averaged_model.n_averaged.copy_(0)
            if self._double_averaged_model.n_averaged == self._double_average_period:
                self._double_average_period *= 2
                self._double_averaged_model.n_averaged //= 2


def create_averaged_model(model, average_type, device=None, use_buffers=False):
    if average_type == "simple":
        return torch.optim.swa_utils.AveragedModel(
            model, device=device, avg_fn=_avg_fn, use_buffers=use_buffers)
    elif average_type == "double":
        return DoubleAveragedModel(
            model, device=device, use_buffers=use_buffers)
    else:
        assert average_type == "none"
        return model
