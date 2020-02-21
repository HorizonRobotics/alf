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

import unittest
import torch
import torch.nn as nn
import numpy as np
import alf.utils.checkpoint_utils as ckpt_utils


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3)
        self.fc = nn.Linear(20, 10)


def weights_init_zero(m):
    torch.nn.init.zeros_(m.weight.data)
    torch.nn.init.zeros_(m.bias.data)


net = Net()
net.apply(weights_init_zero)

net = Net()
ckpt_dir = "/tmp/models/"
ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, net=net)
ckpt_mngr.save(0)
