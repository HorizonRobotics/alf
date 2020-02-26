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

import json
import unittest
import warnings
import shutil
import os
import numpy as np
import torch
import torch.nn as nn

from alf.data_structures import LossInfo, TrainingInfo
from alf.algorithms.algorithm import Algorithm
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


def set_learning_rate(optimizers, lr):
    if isinstance(optimizers, torch.optim.Optimizer):
        optimizers = [optimizers]
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def decay_learning_rate(optimizers, decay_rate):
    if isinstance(optimizers, torch.optim.Optimizer):
        optimizers = [optimizers]
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate


def get_learning_rate(optimizers):
    lrs = []
    if isinstance(optimizers, torch.optim.Optimizer):
        optimizers = [optimizers]
    for optimizer in optimizers:
        lrs.append(optimizer.param_groups[0]['lr'])
    return lrs


class SimpleAlg(Algorithm):
    def __init__(self,
                 optimizer=None,
                 sub_algs=[],
                 params=[],
                 name="SimpleAlg"):
        super().__init__(optimizer=optimizer, name=name)
        self._module_list = nn.ModuleList(sub_algs)
        self._param_list = nn.ParameterList(params)

    def calc_loss(self, training_info):
        loss = torch.tensor(0.)
        for p in self.parameters():
            loss = loss + torch.sum(p)
        return LossInfo(loss=loss)

    def _trainable_attributes_to_ignore(self):
        return ['ignored_param']


class ComposedAlg(Algorithm):
    def __init__(self,
                 optimizer=None,
                 sub_alg1=None,
                 sub_alg2=None,
                 params=[],
                 name="SimpleAlg"):
        super().__init__(optimizer=optimizer, name=name)
        self._sub_alg1 = sub_alg1
        self._sub_alg2 = sub_alg2
        self._param_list = nn.ParameterList(params)

    def calc_loss(self, training_info):
        loss = torch.tensor(0.)
        for p in self.parameters():
            loss = loss + torch.sum(p)
        return LossInfo(loss=loss)


class ComposedAlgWithIgnore(Algorithm):
    def __init__(self,
                 optimizer=None,
                 sub_alg1=None,
                 sub_alg2=None,
                 params=[],
                 name="SimpleAlg"):
        super().__init__(optimizer=optimizer, name=name)
        self._sub_alg1 = sub_alg1
        self._sub_alg2 = sub_alg2
        self._param_list = nn.ParameterList(params)

    def calc_loss(self, training_info):
        loss = torch.tensor(0.)
        for p in self.parameters():
            loss = loss + torch.sum(p)
        return LossInfo(loss=loss)

    def _trainable_attributes_to_ignore(self):
        return ['_sub_alg2']


class TestNetAndOptimizer(unittest.TestCase):
    def test_net_and_optimizer(self):
        net = Net()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

        ckpt_dir = "/tmp/ckpt_data/net/"

        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        ckpt_mngr = ckpt_utils.Checkpointer(
            ckpt_dir, net=net, optimizer=optimizer)

        # test the case loading from 'latest' which does not exist
        self.assertWarns(UserWarning, ckpt_mngr.load, 'latest')

        # training-step-0, all parameters are zeros
        step_num = 0
        net.apply(weights_init_zeros)
        set_learning_rate(optimizer, 0.1)
        ckpt_mngr.save(step_num)

        # training-step-1, all parameters are ones
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

        # load 'latest'
        step_num_from_ckpt = ckpt_mngr.load(global_step='latest')
        self.assertTrue(step_num_from_ckpt == step_num)
        self.assertTrue(get_learning_rate(optimizer)[0] == 0.01)
        for para in list(net.parameters()):
            self.assertTrue((para == 1).all())

        # load a non-existing ckpt won't change current values
        # but will trigger a UserWarning
        self.assertWarns(UserWarning, ckpt_mngr.load, 2)
        self.assertTrue(get_learning_rate(optimizer)[0] == 0.01)
        for para in list(net.parameters()):
            self.assertTrue((para == 1).all())


class TestMultiAlgSingleOpt(unittest.TestCase):
    def test_multi_algo_single_opt(self):

        ckpt_dir = "/tmp/ckpt_data/multi_alg_single_opt/"

        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        # construct algorithms
        param_1 = nn.Parameter(torch.Tensor([1.0]))
        alg_1 = SimpleAlg(params=[param_1], name="alg_1")

        param_2_1 = nn.Parameter(torch.Tensor([2.1]))
        alg_2_1 = SimpleAlg(params=[param_2_1], name="alg_2_1")

        param_2 = nn.Parameter(torch.Tensor([2]))
        alg_2 = SimpleAlg(params=[param_2], sub_algs=[alg_2_1], name="alg_2")

        optimizer_root = torch.optim.Adam(lr=0.1)
        param_root = nn.Parameter(torch.Tensor([0]))
        alg_root = SimpleAlg(
            params=[param_root],
            optimizer=optimizer_root,
            sub_algs=[alg_1, alg_2],
            name="root")

        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root)

        all_optimizers = alg_root.optimizers()
        # a number of training steps
        step_num = 0
        ckpt_mngr.save(step_num)
        step_num = 1
        set_learning_rate(all_optimizers, 0.01)
        alg_root.apply(weights_init_ones)
        ckpt_mngr.save(step_num)

        self.assertTrue(get_learning_rate(all_optimizers) == [0.01])

        # load checkpoints
        ckpt_mngr.load(0)

        # check the recovered optimizers
        self.assertTrue(get_learning_rate(all_optimizers) == [0.1])

        # check the recovered paramerter values for all modules
        sd = alg_root.state_dict()
        self.assertTrue((list(sd.values())[0:4] == [
            torch.tensor([1]),
            torch.tensor([2.1]),
            torch.tensor([2.0]),
            torch.tensor([0.0])
        ]))


class TestMultiAlgMultiOpt(unittest.TestCase):
    def test_multi_alg_multi_opt(self):

        ckpt_dir = "/tmp/ckpt_data/multi_alg_multi_opt/"

        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        # construct algorithms
        param_1 = nn.Parameter(torch.Tensor([1]))
        alg_1 = SimpleAlg(params=[param_1], name="alg_1")

        param_2 = nn.Parameter(torch.Tensor([2]))
        optimizer_2 = torch.optim.Adam(lr=0.1)
        alg_2 = SimpleAlg(
            params=[param_2], optimizer=optimizer_2, name="alg_2")

        optimizer_root = torch.optim.Adam(lr=0.1)
        param_root = nn.Parameter(torch.Tensor([0]))
        alg_root = ComposedAlg(
            params=[param_root],
            optimizer=optimizer_root,
            sub_alg1=alg_1,
            sub_alg2=alg_2,
            name="root")

        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root)

        all_optimizers = alg_root.optimizers()
        # a number of training steps
        step_num = 0
        ckpt_mngr.save(step_num)
        step_num = 1
        set_learning_rate(all_optimizers, 0.01)

        alg_root.apply(weights_init_ones)
        ckpt_mngr.save(step_num)

        # load checkpoints
        ckpt_mngr.load(0)

        # check the recovered optimizers
        expected = torch.Tensor([0.1, 0.1])
        np.testing.assert_array_almost_equal(
            get_learning_rate(all_optimizers), expected)


class TestWithIgnoredSubAlgorithm(unittest.TestCase):
    def test_with_ignored_sub_algorithm(self):

        ckpt_dir = "/tmp/ckpt_data/ignored_sub_alg/"

        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        # construct algorithms
        param_1 = nn.Parameter(torch.Tensor([1]))
        alg_1 = SimpleAlg(params=[param_1], name="alg_1")

        param_2 = nn.Parameter(torch.Tensor([2]))
        optimizer_2 = torch.optim.Adam(lr=0.2)
        alg_2 = SimpleAlg(
            params=[param_2], optimizer=optimizer_2, name="alg_2")

        optimizer_root = torch.optim.Adam(lr=0.1)
        param_root = nn.Parameter(torch.Tensor([0]))
        alg_root = ComposedAlgWithIgnore(
            params=[param_root],
            optimizer=optimizer_root,
            sub_alg1=alg_1,
            sub_alg2=alg_2,
            name="root")

        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root)

        # a number of training steps
        step_num = 0
        ckpt_mngr.save(step_num)

        step_num = 1
        all_optimizers = alg_root.optimizers()
        decay_learning_rate(all_optimizers, 1.0 / 10)
        ckpt_mngr.save(step_num)

        ckpt_mngr.load(1)
        expected = torch.Tensor([0.01, 0.02])
        np.testing.assert_array_almost_equal(
            get_learning_rate(all_optimizers), expected)


class TestWithParamSharing(unittest.TestCase):
    def test_with_param_sharing(self):

        ckpt_dir = "/tmp/ckpt_data/sub_alg_param_sharing/"

        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        # construct algorithms
        param_1 = nn.Parameter(torch.Tensor([1]))
        alg_1 = SimpleAlg(params=[param_1], name="alg_1")

        param_2 = nn.Parameter(torch.Tensor([2]))
        optimizer_2 = torch.optim.Adam(lr=0.2)
        alg_2 = SimpleAlg(
            params=[param_2], optimizer=optimizer_2, name="alg_2")
        alg_2.p = param_1

        optimizer_root = torch.optim.Adam(lr=0.1)
        param_root = nn.Parameter(torch.Tensor([0]))
        alg_root = ComposedAlg(
            params=[param_root],
            optimizer=optimizer_root,
            sub_alg1=alg_1,
            sub_alg2=alg_2,
            name="root")

        all_optimizers = alg_root.optimizers()

        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root)

        # a number of training steps
        step_num = 0
        ckpt_mngr.save(step_num)

        ckpt_mngr.load(0)
        expected = torch.Tensor([0.1, 0.2])
        np.testing.assert_array_almost_equal(
            get_learning_rate(all_optimizers), expected)

        sd = alg_root.state_dict()
        recovered_values = [list(sd.values())[i] for i in [0, 1, 2, 4]]
        self.assertTrue((
            recovered_values == [
                torch.tensor([1.0]),
                torch.tensor([1.0]),  # the shared-parameter
                torch.tensor([2.0]),
                torch.tensor([0.0])
            ]))


class TestWithCycle(unittest.TestCase):
    def test_with_cycle(self):

        ckpt_dir = "/tmp/ckpt_data/sub_alg_cycle/"

        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        # construct algorithms
        param_1 = nn.Parameter(torch.Tensor([1]))
        alg_1 = SimpleAlg(params=[param_1], name="alg_1")

        param_2 = nn.Parameter(torch.Tensor([2]))
        optimizer_2 = torch.optim.Adam(lr=0.2)
        alg_2 = SimpleAlg(
            params=[param_2], optimizer=optimizer_2, name="alg_2")

        optimizer_root = torch.optim.Adam(lr=0.1)
        param_root = nn.Parameter(torch.Tensor([0]))

        # case 1: test cycle detection
        alg_root = ComposedAlg(
            params=[param_root],
            optimizer=optimizer_root,
            sub_alg1=alg_1,
            sub_alg2=alg_2,
            name="root")

        alg_2.root = alg_root
        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root)
        self.assertRaises(AssertionError, ckpt_mngr.save, 0)

        # case 2: test cycle detection when some sub-algorithms are 'ignored'
        alg_root2 = ComposedAlgWithIgnore(
            params=[param_root],
            optimizer=optimizer_root,
            sub_alg1=alg_1,
            sub_alg2=alg_2,
            name="root")

        alg_2.root = alg_root2
        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root2)
        self.assertRaises(AssertionError, ckpt_mngr.save, 0)


class TestModelMismatch(unittest.TestCase):
    def test_model_mismatch(self):
        # test model mis-match

        ckpt_dir = "/tmp/ckpt_data/model_mis_match/"

        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        # construct algorithms
        param_1 = nn.Parameter(torch.Tensor([1]))
        alg_1 = SimpleAlg(params=[param_1], name="alg_1")

        param_2 = nn.Parameter(torch.Tensor([2]))
        optimizer_2 = torch.optim.Adam(lr=0.2)
        alg_2 = SimpleAlg(
            params=[param_2], optimizer=optimizer_2, name="alg_2")

        optimizer_root = torch.optim.Adam(lr=0.1)
        param_root = nn.Parameter(torch.Tensor([0]))
        alg_root12 = ComposedAlg(
            params=[param_root],
            optimizer=optimizer_root,
            sub_alg1=alg_1,
            sub_alg2=alg_2,
            name="root")

        alg_root1 = ComposedAlg(
            params=[param_root],
            optimizer=optimizer_root,
            sub_alg1=alg_1,
            name="root")

        # case 1: save using alg_root12 and load using alg_root1
        step_num = 0
        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root12)
        ckpt_mngr.save(step_num)
        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root1)
        self.assertRaises(RuntimeError, ckpt_mngr.load, step_num)

        # case 2: save using alg_root1 and load using alg_root12
        step_num = 0
        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root1)
        ckpt_mngr.save(step_num)
        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root12)
        self.assertRaises(RuntimeError, ckpt_mngr.load, step_num)


class TestOptMismatch(unittest.TestCase):
    def test_opt_mismatch(self):
        # test optimizer mis-match

        ckpt_dir = "/tmp/ckpt_data/opt_mis_match/"

        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        param_1 = nn.Parameter(torch.Tensor([1]))
        optimizer_1 = torch.optim.Adam(lr=0.2)
        alg_1_no_op = SimpleAlg(params=[param_1], name="alg_1_no_op")
        alg_1 = SimpleAlg(
            params=[param_1], optimizer=optimizer_1, name="alg_1")

        param_2 = nn.Parameter(torch.Tensor([2]))
        optimizer_2 = torch.optim.Adam(lr=0.2)
        alg_2 = SimpleAlg(
            params=[param_2], optimizer=optimizer_2, name="alg_2")

        optimizer_root = torch.optim.Adam(lr=0.1)
        param_root = nn.Parameter(torch.Tensor([0]))
        alg_root_1_no_op = ComposedAlg(
            params=[param_root],
            optimizer=optimizer_root,
            sub_alg1=alg_1_no_op,
            sub_alg2=alg_2,
            name="root")

        alg_root_1 = ComposedAlg(
            params=[param_root],
            optimizer=optimizer_root,
            sub_alg1=alg_1,
            sub_alg2=alg_2,
            name="root")

        # case 1: save using alg_root_1_no_op and load using alg_root_1
        step_num = 0
        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root_1_no_op)
        ckpt_mngr.save(step_num)
        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root_1)
        self.assertRaises(RuntimeError, ckpt_mngr.load, step_num)

        # case 2: save using alg_root_1 load using alg_root_1_no_op
        step_num = 0
        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root_1)
        ckpt_mngr.save(step_num)
        ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root_1_no_op)
        self.assertRaises(RuntimeError, ckpt_mngr.load, step_num)


if __name__ == '__main__':
    unittest.main()
