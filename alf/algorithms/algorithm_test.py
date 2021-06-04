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

from absl import logging
import json
import pprint
import torch
import torch.nn as nn

import alf
from alf.data_structures import LossInfo
from alf.algorithms.algorithm import Algorithm, _get_optimizer_params


class MyAlg(Algorithm):
    def __init__(self, optimizer=None, sub_algs=[], params=[], name="MyAlg"):
        super().__init__(optimizer=optimizer, name=name)
        self._module_list = nn.ModuleList(sub_algs)
        self._param_list = nn.ParameterList(params)

    def calc_loss(self):
        loss = torch.tensor(0.)
        for p in self.parameters():
            loss = loss + torch.sum(p)
        return LossInfo(loss=loss)

    def _trainable_attributes_to_ignore(self):
        return ['ignored_param']


class AlgorithmTest(alf.test.TestCase):
    def test_flatten_module(self):
        a = nn.Module()
        b = nn.Module()
        c = nn.Module()
        d = nn.Module()
        pa = nn.Parameter()
        pb = nn.Parameter()
        pc = nn.Parameter()
        pd = nn.Parameter()
        nest = nn.ModuleDict({
            'a': a,
            'b': b,
            'list': nn.ModuleList([c, d]),
            'plist': nn.ParameterList([pa, pb]),
            'pdict': nn.ParameterDict({
                'pc': pc,
                'pd': pd
            })
        })
        flattend = alf.algorithms.algorithm._flatten_module(nest)
        self.assertEqual(
            set(map(id, flattend)), set(map(id, [a, b, c, d, pa, pb, pc, pd])))

    def test_get_optimizer_info(self):
        param_1 = nn.Parameter(torch.Tensor([1]))
        alg_1 = MyAlg(params=[param_1], name="alg_1")
        param_2 = nn.Parameter(torch.Tensor([2]))
        alg_2 = MyAlg(params=[param_2], name="alg_2")

        alg_root = MyAlg(
            optimizer=alf.optimizers.Adam(lr=0.25),
            sub_algs=[alg_1],
            name="root")
        info = json.loads(alg_root.get_optimizer_info())
        self.assertEqual(len(info), 1)
        self.assertEqual(info[0]['parameters'],
                         [alg_root.get_param_name(param_1)])

        alg_1 = MyAlg(params=[param_1, param_1], name="alg_1")
        alg_root = MyAlg(
            optimizer=alf.optimizers.Adam(lr=0.25),
            sub_algs=[alg_1, alg_1],
            name="root")
        info = json.loads(alg_root.get_optimizer_info())
        self.assertEqual(len(info), 1)
        self.assertEqual(info[0]['parameters'],
                         [alg_root.get_param_name(param_1)])

        alg_root = MyAlg(
            optimizer=alf.optimizers.Adam(lr=0.25),
            sub_algs=[alg_1, alg_2],
            name="root")
        alg_root.add_optimizer(alf.optimizers.Adam(lr=0.5), [alg_2])
        info = json.loads(alg_root.get_optimizer_info())
        logging.info(pprint.pformat(info))
        self.assertEqual(len(info), 2)
        self.assertTrue(info[0]['hypers']['lr'] == 0.25
                        or info[1]['hypers']['lr'] == 0.25)
        self.assertTrue(info[0]['hypers']['lr'] == 0.5
                        or info[1]['hypers']['lr'] == 0.5)

        if info[0]['hypers']['lr'] == 0.25:
            opt_default = info[0]
            opt_2 = info[1]
        else:
            opt_default = info[1]
            opt_2 = info[0]

        self.assertEqual(opt_default['parameters'],
                         [alg_root.get_param_name(param_1)])
        self.assertEqual(opt_2['parameters'],
                         [alg_root.get_param_name(param_2)])

        alg_root = MyAlg(sub_algs=[alg_1, alg_2], name="root")
        alg_root.add_optimizer(alf.optimizers.Adam(lr=0.5), [alg_2])
        info = json.loads(alg_root.get_optimizer_info())
        self.assertEqual(len(info), 2)
        self.assertEqual(info[0]['optimizer'], 'None')
        self.assertEqual(info[0]['parameters'],
                         [alg_root.get_param_name(param_1)])
        self.assertEqual(info[1]['parameters'],
                         [alg_root.get_param_name(param_2)])

        # Test cycle detection
        alg_2.root = alg_root
        self.assertRaises(AssertionError, alg_root.get_optimizer_info)

        # Test duplicated handling detection
        alg_root.add_optimizer(alf.optimizers.Adam(lr=0.25), [alg_1])
        alg_2.root = None
        alg_2.p = param_1
        self.assertRaises(AssertionError, alg_root.get_optimizer_info)

        # Test _trainable_attributes_to_ignore
        alg_2.p = None
        alg_2.ignored_param = param_1
        info = json.loads(alg_root.get_optimizer_info())
        self.assertEqual(len(info), 2)
        self.assertEqual(info[0]['parameters'],
                         [alg_root.get_param_name(param_2)])
        self.assertEqual(info[1]['parameters'],
                         [alg_root.get_param_name(param_1)])

        # test __repr__
        logging.info("\n" + repr(alg_root))

    def test_get_optimizer_info2(self):
        # test shared module in used by sub-algorithms
        layer = alf.layers.FC(2, 3)
        param_1 = nn.Parameter(torch.Tensor([1]))
        alg_1 = MyAlg(params=[param_1], name="alg_1")
        alg_1.layer = layer
        param_2 = nn.Parameter(torch.Tensor([2]))
        alg_2 = MyAlg(params=[param_2], name="alg_2")
        alg_2.layer = layer
        alg_root = MyAlg(
            sub_algs=[alg_1, alg_2],
            optimizer=alf.optimizers.Adam(lr=0.25),
            name="root")
        info = json.loads(alg_root.get_optimizer_info())
        self.assertEqual(
            set(info[0]['parameters']),
            set(
                alg_root.get_param_name(p)
                for p in [param_1, param_2] + list(layer.parameters())))

    def test_optimizer_params(self):
        # test that the order of parameters is deterministic
        opt1 = alf.optimizers.Adam(lr=0.25)
        alg_1 = MyAlg(optimizer=opt1)
        alg_1.a = nn.Parameter(torch.rand(1, 4))
        alg_1.b = nn.Parameter(torch.rand(2, 4))
        alg_1.c = nn.Parameter(torch.rand(3, 4))
        alg_1.d = nn.Parameter(torch.rand(4, 4))
        alg_1.e = nn.Parameter(torch.rand(5, 4))
        alg_1.f = nn.Parameter(torch.rand(6, 4))
        alg_1.get_optimizer_info()
        params1 = _get_optimizer_params(opt1)
        shapes1 = [p.shape for p in params1]

        opt2 = alf.optimizers.Adam(lr=0.25)
        alg_2 = MyAlg(optimizer=opt2)
        alg_2.a = nn.Parameter(torch.rand(1, 4))
        alg_2.b = nn.Parameter(torch.rand(2, 4))
        alg_2.c = nn.Parameter(torch.rand(3, 4))
        alg_2.d = nn.Parameter(torch.rand(4, 4))
        alg_2.e = nn.Parameter(torch.rand(5, 4))
        alg_2.f = nn.Parameter(torch.rand(6, 4))
        alg_2.get_optimizer_info()
        params2 = _get_optimizer_params(opt2)
        shapes2 = [p.shape for p in params2]
        self.assertEqual(shapes1, shapes2)

    def test_update_with_gradient(self):
        param_1 = nn.Parameter(torch.Tensor([1]))
        alg_1 = MyAlg(params=[param_1], name="alg_1")
        param_2 = nn.Parameter(torch.Tensor([2]))
        alg_2 = MyAlg(params=[param_2], name="alg_2")

        alg_root = MyAlg(sub_algs=[alg_1, alg_2], name="root")
        alg_root.add_optimizer(alf.optimizers.Adam(lr=0.5), [alg_2])
        loss = alg_root.calc_loss()
        self.assertRaises(AssertionError, alg_root.update_with_gradient, loss)

        alg_root = MyAlg(
            optimizer=alf.optimizers.Adam(lr=0.25),
            sub_algs=[alg_1, alg_2],
            name="root")
        alg_root.add_optimizer(alf.optimizers.Adam(lr=0.5), [alg_2])
        loss_info, params = alg_root.update_with_gradient(alg_root.calc_loss())
        self.assertEqual(set(params), set(alg_root.named_parameters()))
        for param in alg_root.parameters():
            self.assertTrue(torch.all(param.grad == 1.0))
        self.assertEqual(loss_info.loss, 3.)


if __name__ == '__main__':
    logging.use_absl_handler()
    logging.set_verbosity(logging.INFO)
    alf.test.main()
