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
import warnings
import torch
import torch.nn as nn
import numpy as np
import alf.utils.checkpoint_utils as ckpt_utils


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(1, 6, 3)
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(20, 10)


def weights_init_zeros(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)


def weights_init_ones(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.ones_(m.weight.data)
        torch.nn.init.ones_(m.bias.data)


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    return lrs


class TestNetAndOptimizer(unittest.TestCase):
    def test_net_and_optimizer(self):
        net = Net()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

        ckpt_dir = "/tmp/models/"
        ckpt_mngr = ckpt_utils.Checkpointer(
            ckpt_dir, net=net, optimizer=optimizer)

        # training-step-0, all parameters are zeros
        step_num = 0
        net.apply(weights_init_zeros)
        set_learning_rate(optimizer, 0.1)
        ckpt_mngr.save(step_num)

        # training-step-0, all parameters are zeros
        step_num = 1
        net.apply(weights_init_ones)
        set_learning_rate(optimizer, 0.01)
        ckpt_mngr.save(step_num)

        # load ckpt-1
        ckpt_mngr.load(global_step=1)
        self.assertTrue(get_learning_rate(optimizer)[0] == 0.01)
        for para in list(net.parameters()):
            self.assertTrue((para == 1).all())

        # load ckpt-0
        ckpt_mngr.load(global_step=0)
        self.assertTrue(get_learning_rate(optimizer)[0] == 0.1)
        for para in list(net.parameters()):
            self.assertTrue((para == 0).all())

        # load latest
        next_step_num = ckpt_mngr.load(global_step='latest')
        self.assertTrue(next_step_num == step_num + 1)
        self.assertTrue(get_learning_rate(optimizer)[0] == 0.01)
        for para in list(net.parameters()):
            self.assertTrue((para == 1).all())

        # load a non-existing ckpt won't change current values
        # but will trigger a UserWarning
        self.assertWarns(UserWarning, ckpt_mngr.load, 2)
        self.assertTrue(get_learning_rate(optimizer)[0] == 0.01)
        for para in list(net.parameters()):
            self.assertTrue((para == 1).all())


if __name__ == '__main__':
    unittest.main()
