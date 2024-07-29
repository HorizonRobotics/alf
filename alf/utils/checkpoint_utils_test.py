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

from absl.testing import parameterized
from collections import OrderedDict
import numpy as np
import functools
import json
import os
import shutil
import tempfile
import warnings

import torch
import torch.nn as nn

import alf
from alf.data_structures import LossInfo
from alf.algorithms.algorithm import Algorithm
import alf.utils.checkpoint_utils as ckpt_utils

from alf.networks.encoding_networks import EncodingNetwork
from alf.networks.encoding_networks import LSTMEncodingNetwork
from alf.networks.encoding_networks import ParallelEncodingNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.tensor_specs import TensorSpec
from alf.utils import common
import unittest


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

    def _trainable_attributes_to_ignore(self):
        return ['_sub_alg2']


class TestNetAndOptimizer(alf.test.TestCase):
    def test_net_and_optimizer(self):
        net = Net()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as ckpt_dir:
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
            self.assertRaises(FileNotFoundError, ckpt_mngr.load, 2)
            self.assertTrue(get_learning_rate(optimizer)[0] == 0.01)
            for para in list(net.parameters()):
                self.assertTrue((para == 1).all())


class TestMultiAlgSingleOpt(alf.test.TestCase):
    def test_multi_algo_single_opt(self):

        with tempfile.TemporaryDirectory() as ckpt_dir:
            # construct algorithms
            param_1 = nn.Parameter(torch.Tensor([1.0]))
            alg_1 = SimpleAlg(params=[param_1], name="alg_1")

            param_2_1 = nn.Parameter(torch.Tensor([2.1]))
            alg_2_1 = SimpleAlg(params=[param_2_1], name="alg_2_1")

            param_2 = nn.Parameter(torch.Tensor([2]))
            alg_2 = SimpleAlg(
                params=[param_2], sub_algs=[alg_2_1], name="alg_2")

            optimizer_root = alf.optimizers.Adam(lr=0.1)
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

            # check the recovered parameter values for all modules
            sd = alg_root.state_dict()
            self.assertTrue((list(sd.values())[0:4] == [
                torch.tensor([1]),
                torch.tensor([2.1]),
                torch.tensor([2.0]),
                torch.tensor([0.0])
            ]))


class TestMultiAlgMultiOpt(alf.test.TestCase):
    def test_multi_alg_multi_opt(self):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            # construct algorithms
            param_1 = nn.Parameter(torch.Tensor([1]))
            alg_1 = SimpleAlg(params=[param_1], name="alg_1")

            param_2 = nn.Parameter(torch.Tensor([2]))
            optimizer_2 = alf.optimizers.Adam(lr=0.2)
            alg_2 = SimpleAlg(
                params=[param_2], optimizer=optimizer_2, name="alg_2")

            optimizer_root = alf.optimizers.Adam(lr=0.1)
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
            expected = [0.1, 0.2]
            np.testing.assert_array_almost_equal(
                get_learning_rate(all_optimizers), expected)


class TestWithParamSharing(alf.test.TestCase):
    def test_with_param_sharing(self):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            # construct algorithms
            param_1 = nn.Parameter(torch.Tensor([1]))
            alg_1 = SimpleAlg(params=[param_1], name="alg_1")

            param_2 = nn.Parameter(torch.Tensor([2]))
            optimizer_2 = alf.optimizers.Adam(lr=0.2)
            alg_2 = SimpleAlg(
                params=[param_2], optimizer=optimizer_2, name="alg_2")
            alg_2.ignored_param = param_1

            optimizer_root = alf.optimizers.Adam(lr=0.1)
            param_root = nn.Parameter(torch.Tensor([0]))
            alg_root = ComposedAlg(
                params=[param_root],
                optimizer=optimizer_root,
                sub_alg1=alg_1,
                sub_alg2=alg_2,
                name="root")

            ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root)

            # only one copy of the shared param is returned from state_dict
            self.assertTrue(
                '_sub_alg2.ignored_param' not in alg_root.state_dict())

            # a number of training steps
            step_num = 0
            ckpt_mngr.save(step_num)

            # modify the shared param after saving
            with torch.no_grad():
                alg_root._sub_alg2.state_dict()['ignored_param'].copy_(
                    torch.Tensor([-1]))

            self.assertTrue((alg_root._sub_alg2.state_dict()['ignored_param']
                             == torch.Tensor([-1])))
            self.assertTrue((alg_root.state_dict()['_sub_alg1._param_list.0']
                             == torch.Tensor([-1])))

            # the value of the shared parameter is recovered back to saved value
            ckpt_mngr.load(0)
            self.assertTrue((alg_root._sub_alg2.state_dict()['ignored_param']
                             == torch.Tensor([1])))
            self.assertTrue((alg_root.state_dict()['_sub_alg1._param_list.0']
                             == torch.Tensor([1])))


class TestWithCycle(alf.test.TestCase):
    def test_with_cycle(self):
        # checkpointer should work regardless of cycles
        with tempfile.TemporaryDirectory() as ckpt_dir:
            # construct algorithms
            param_1 = nn.Parameter(torch.Tensor([1]))
            alg_1 = SimpleAlg(params=[param_1], name="alg_1")

            param_2 = nn.Parameter(torch.Tensor([2]))
            optimizer_2 = alf.optimizers.Adam(lr=0.2)
            alg_2 = SimpleAlg(
                params=[param_2], optimizer=optimizer_2, name="alg_2")

            optimizer_root = alf.optimizers.Adam(lr=0.1)
            param_root = nn.Parameter(torch.Tensor([0]))

            # case 1: cycle without ignore
            alg_root = ComposedAlg(
                params=[param_root],
                optimizer=optimizer_root,
                sub_alg1=alg_1,
                sub_alg2=alg_2,
                name="root")

            alg_2.root = alg_root

            expected_state_dict = OrderedDict([
                ('_sub_alg1._param_list.0', torch.tensor([1.])),
                ('_sub_alg2._param_list.0', torch.tensor([2.])),
                (
                    '_sub_alg2._optimizers.0',
                    {
                        'state': {},
                        'param_groups': [
                            {
                                'lr': 0.2,
                                'betas': (0.9, 0.999),
                                'capturable': False,
                                'differentiable': False,
                                'eps': 1e-08,
                                'foreach': None,
                                'fused': None,
                                'weight_decay': 0,
                                'amsgrad': False,
                                'maximize': False,
                                'params': []
                            },
                            {
                                'lr': 0.2,
                                'betas': (0.9, 0.999),
                                'capturable': False,
                                'differentiable': False,
                                'eps': 1e-08,
                                'foreach': None,
                                'fused': None,
                                'weight_decay': 0,
                                'amsgrad': False,
                                'maximize': False,
                                'params': [0]  # order index instead of id
                            }
                        ]
                    }),
                ('_param_list.0', torch.tensor([0.])),
                (
                    '_optimizers.0',
                    {
                        'state': {},
                        'param_groups': [
                            {
                                'lr': 0.1,
                                'betas': (0.9, 0.999),
                                'capturable': False,
                                'differentiable': False,
                                'eps': 1e-08,
                                'foreach': None,
                                'fused': None,
                                'weight_decay': 0,
                                'amsgrad': False,
                                'maximize': False,
                                'params': []
                            },
                            {
                                'lr': 0.1,
                                'betas': (0.9, 0.999),
                                'capturable': False,
                                'differentiable': False,
                                'eps': 1e-08,
                                'foreach': None,
                                'fused': None,
                                'weight_decay': 0,
                                'amsgrad': False,
                                'maximize': False,
                                'params': [0, 1]  # order indices instead of id
                            }
                        ]
                    })
            ])

            # cycles are not allowed with explicit ignoring
            self.assertRaises(AssertionError, alg_root.state_dict)

            # case 2: cycle with ignore, which also resembles the case where a
            # self-training module (alg_2) is involved
            alg_root2 = ComposedAlgWithIgnore(
                params=[param_root],
                optimizer=optimizer_root,
                sub_alg1=alg_1,
                sub_alg2=alg_2,
                name="root")

            alg_2.root = alg_root2

            ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_root2)
            ckpt_mngr.save(0)

            # modify some parameter values after saving
            with torch.no_grad():
                alg_root._sub_alg1._param_list[0].copy_(torch.Tensor([-1]))

            self.assertTrue((alg_root2.state_dict() != expected_state_dict))

            # recover the expected values after loading
            ckpt_mngr.load(0)
            self.assertTrue((alg_root2.state_dict() == expected_state_dict))


class TestModelMismatch(alf.test.TestCase):
    def test_model_mismatch(self):
        # test model mismatch
        with tempfile.TemporaryDirectory() as ckpt_dir:
            # construct algorithms
            param_1 = nn.Parameter(torch.Tensor([1]))
            alg_1 = SimpleAlg(params=[param_1], name="alg_1")

            param_2 = nn.Parameter(torch.Tensor([2]))
            optimizer_2 = alf.optimizers.Adam(lr=0.2)
            alg_2 = SimpleAlg(
                params=[param_2], optimizer=optimizer_2, name="alg_2")

            optimizer_root = alf.optimizers.Adam(lr=0.1)
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


class TestOptMismatch(alf.test.TestCase):
    def test_opt_mismatch(self):
        # test optimizer mismatch
        with tempfile.TemporaryDirectory() as ckpt_dir:
            param_1 = nn.Parameter(torch.Tensor([1]))
            optimizer_1 = alf.optimizers.Adam(lr=0.2)
            alg_1_no_op = SimpleAlg(params=[param_1], name="alg_1_no_op")
            alg_1 = SimpleAlg(
                params=[param_1], optimizer=optimizer_1, name="alg_1")

            param_2 = nn.Parameter(torch.Tensor([2]))
            optimizer_2 = alf.optimizers.Adam(lr=0.2)
            alg_2 = SimpleAlg(
                params=[param_2], optimizer=optimizer_2, name="alg_2")

            optimizer_root = alf.optimizers.Adam(lr=0.1)
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


class TestLoadStateDictForParallelNetwork(parameterized.TestCase,
                                          alf.test.TestCase):
    @parameterized.parameters((False, ), (True, ))
    def test_parallel_network_state_dict_and_params(self, lstm):
        input_spec = TensorSpec((10, ))

        input_preprocessors = EmbeddingPreprocessor(
            input_spec, embedding_dim=10)

        if lstm:
            network_ctor = functools.partial(
                LSTMEncodingNetwork,
                hidden_size=(1, ),
                post_fc_layer_params=(2, 2))
        else:
            network_ctor = functools.partial(
                EncodingNetwork, fc_layer_params=(10, 10))

        network_wo_preprocessor = network_ctor(input_tensor_spec=input_spec)
        network_w_preprocessor = network_ctor(
            input_tensor_spec=input_spec,
            input_preprocessors=input_preprocessors)

        # 1) test parameter copy for the corresponding parallel net
        def _check_parallel_param(p_net_source):
            p_net_target = p_net_source.copy()
            p_net_target.load_state_dict(p_net_source.state_dict())
            for ws, wt in zip(p_net_source.parameters(),
                              p_net_target.parameters()):
                self.assertTensorEqual(ws, wt)

        replicas = 2
        for network in [network_wo_preprocessor, network_w_preprocessor]:
            p_net = network.make_parallel(replicas)
            _check_parallel_param(p_net)
            n_net = alf.networks.network.NaiveParallelNetwork(
                network, replicas)
            _check_parallel_param(n_net)

        # 2) test parameter number for networks with preprocessor
        p_net_w_preprocessor = network_w_preprocessor.make_parallel(replicas)

        self.assertEqual(
            len(p_net_w_preprocessor.state_dict()),
            len(list(p_net_w_preprocessor.parameters())))

        network_w_shared_preprocessor = network_ctor(
            input_tensor_spec=input_spec,
            input_preprocessors=EmbeddingPreprocessor(
                input_spec, embedding_dim=10).singleton())

        p_net_w_shared_preprocessor = network_w_shared_preprocessor.make_parallel(
            replicas)

        # the number of parameters of a parallel network with a shared
        # input_preprocessor should be equal to that of the parallel network
        # with non-shared input processor - (replicas-1) * the number of parameters
        # of input processors
        self.assertEqual(
            len(p_net_w_shared_preprocessor.state_dict()),
            len(p_net_w_preprocessor.state_dict()) -
            (replicas - 1) * len(input_preprocessors.state_dict()))

        self.assertEqual(
            len(p_net_w_shared_preprocessor.state_dict()),
            len(list(p_net_w_shared_preprocessor.parameters())))


class TestCheckpointStructure(alf.test.TestCase):
    def test_checkpoint_structure(self):
        net = Net()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as ckpt_dir:
            ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, net=net)

            step_num = 0
            net.apply(weights_init_zeros)
            ckpt_mngr.save(step_num)

            ckpt_mngr.load(global_step=0)

            model_structure_file = os.path.join(ckpt_dir,
                                                'ckpt-structure.json')

            with open(model_structure_file, 'r') as f:
                model_structure = json.load(f)

            expected_model_structure = {
                'global_step': -1,
                'net': {
                    'conv1.bias': -1,
                    'conv1.weight': -1,
                    'conv2.bias': -1,
                    'conv2.weight': -1,
                    'fc1.bias': -1,
                    'fc1.weight': -1,
                    'fc2.bias': -1,
                    'fc2.weight': -1
                }
            }

            self.assertEqual(model_structure, expected_model_structure)


class TestCheckpointMapLocation(alf.test.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "gpu is unavailable")
    def test_map_location(self):
        # test that that loading checkpoint file using cpu as the map_location
        # works in the subsequent ``load_state_dict`` both for cpu and gpu models
        model_on_cpu = Net()
        model_on_gpu = Net().cuda()

        with tempfile.TemporaryDirectory() as ckpt_dir:
            ckpt_path = ckpt_dir + '/test.ckpt'
            torch.save(model_on_cpu.state_dict(), ckpt_path)

            # load checkpoint file on cpu
            device = torch.device('cpu')
            state_dict = torch.load(ckpt_path, map_location=device)

            # test load state-dict for cpu model
            model_on_cpu.load_state_dict(state_dict)

            # test load state-dict for cpu model
            model_on_gpu.load_state_dict(state_dict)

            for p1, p2 in zip(model_on_cpu.parameters(),
                              model_on_gpu.parameters()):
                self.assertTrue('cpu' == str(p1.device))
                self.assertTrue(p2.is_cuda)

                self.assertTensorClose(p1, p2.cpu())


if __name__ == '__main__':
    alf.test.main()
