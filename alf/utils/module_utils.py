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

import torch


def set_trainable_flag(modules, flag):
    for model in modules:
        model.requires_grad_(flag)


# def set_trainable_flag(model: torch.nn.Module, flag):
#     for param in model.parameters():
#         param.requires_grad = flag


class no_grad(object):
    def __init__(self, *modules):
        self._modules = modules

    def __enter__(self):
        self._parameters = []
        for m in self._modules:
            for p in m.parameters():
                self._parameters.append((p, p.requires_grad))
                p.required_grad = False

    def __exit__(self, type, value, traceback):
        for p, flag in self._parameters:
            p.requires_grad = flag
