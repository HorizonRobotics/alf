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
import pprint
import torch
import torch.nn as nn
import unittest

import alf
from alf.data_structures import LossInfo, TrainingInfo
from alf.algorithms.algorithm import Algorithm


class MyAlg(Algorithm):
    def __init__(self, optimizer=None, sub_algs=[], params=[], name="MyAlg"):
        super().__init__(optimizer=optimizer, name=name)
        self._module_list = nn.ModuleList(sub_algs)
        self._param_list = nn.ParameterList(params)

    def calc_loss(self, training_info):
        loss = torch.tensor(0.)
        for p in self.parameters():
            loss = loss + torch.sum(p)
        return LossInfo(loss=loss)


class AlgorithmTest(unittest.TestCase):
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
            optimizer=torch.optim.Adam(lr=0.25), sub_algs=[alg_1], name="root")
        info = json.loads(alg_root.get_optimizer_info())
        self.assertEqual(len(info), 1)
        self.assertEqual(info[0]['parameters'], [id(param_1)])

        alg_1 = MyAlg(params=[param_1, param_1])
        alg_root = MyAlg(
            optimizer=torch.optim.Adam(lr=0.25),
            sub_algs=[alg_1, alg_1],
            name="root")
        info = json.loads(alg_root.get_optimizer_info())
        self.assertEqual(len(info), 1)
        self.assertEqual(info[0]['parameters'], [id(param_1)])

        alg_root = MyAlg(
            optimizer=torch.optim.Adam(lr=0.25),
            sub_algs=[alg_1, alg_2],
            name="root")
        alg_root.add_optimizer(torch.optim.Adam(lr=0.5), [alg_2])
        info = json.loads(alg_root.get_optimizer_info())
        pprint.pprint(info)
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

        self.assertEqual(opt_default['parameters'], [id(param_1)])
        self.assertEqual(opt_2['parameters'], [id(param_2)])

        alg_root = MyAlg(sub_algs=[alg_1, alg_2], name="root")
        alg_root.add_optimizer(torch.optim.Adam(lr=0.5), [alg_2])
        info = json.loads(alg_root.get_optimizer_info())
        self.assertEqual(len(info), 2)
        self.assertEqual(info[0]['optimizer'], 'None')
        self.assertEqual(info[0]['parameters'], [id(param_1)])
        self.assertEqual(info[1]['parameters'], [id(param_2)])

    def test_train_complete(self):
        param_1 = nn.Parameter(torch.Tensor([1]))
        alg_1 = MyAlg(params=[param_1], name="alg_1")
        param_2 = nn.Parameter(torch.Tensor([2]))
        alg_2 = MyAlg(params=[param_2], name="alg_2")

        alg_root = MyAlg(sub_algs=[alg_1, alg_2], name="root")
        alg_root.add_optimizer(torch.optim.Adam(lr=0.5), [alg_2])
        self.assertRaises(AssertionError, alg_root.train_complete,
                          TrainingInfo())

        alg_root = MyAlg(
            optimizer=torch.optim.Adam(lr=0.25),
            sub_algs=[alg_1, alg_2],
            name="root")
        alg_root.add_optimizer(torch.optim.Adam(lr=0.5), [alg_2])
        loss_info, params = alg_root.train_complete(TrainingInfo())
        self.assertEqual(set(params), set(alg_root.parameters()))
        for param in alg_root.parameters():
            self.assertTrue(torch.all(param.grad == 1.0))
        self.assertEqual(loss_info.loss, 3.)


if __name__ == '__main__':
    unittest.main()
