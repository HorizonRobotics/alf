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
import copy
import json
import os
import pprint
import tempfile
import torch
import torch.nn as nn

import alf
from alf.data_structures import LossInfo
from alf.algorithms.algorithm import Algorithm, _get_optimizer_params
import alf.utils.checkpoint_utils as ckpt_utils


class MyAlg(Algorithm):
    def __init__(self,
                 optimizer=None,
                 sub_algs=[],
                 params=[],
                 checkpoint=None,
                 name="MyAlg"):
        super().__init__(optimizer=optimizer, checkpoint=checkpoint, name=name)
        self._module_list = nn.ModuleList(sub_algs)
        self._param_list = nn.ParameterList(params)

    def calc_loss(self):
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
                 name="ComposedAlg"):

        super().__init__(
            optimizer=optimizer,
            name=name,
        )
        self._sub_alg1 = sub_alg1
        self._sub_alg2 = sub_alg2
        self._param_list = nn.ParameterList(params)


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
        flattened = alf.algorithms.algorithm._flatten_module(nest)
        self.assertEqual(
            set(map(id, flattened)), set(
                map(id, [a, b, c, d, pa, pb, pc, pd])))

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

    def test_optimizer_name(self):
        optimizer1 = alf.optimizers.Adam(lr=0.25)
        sub_algorithm = MyAlg(optimizer=optimizer1)
        my_algorithm = MyAlg(sub_algs=[sub_algorithm])
        self.assertTrue('_optimizers.0' in sub_algorithm.state_dict())
        self.assertTrue(
            '_module_list.0._optimizers.0' in my_algorithm.state_dict())

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

    def test_full_checkpoint_preloading(self):
        # test the case where the checkpoint directly matches with the algorithm
        with tempfile.TemporaryDirectory() as ckpt_dir:
            # 1) construct the first algorithm instance and save a checkpoint
            param = nn.Parameter(torch.Tensor([1]))
            alg_1 = MyAlg(params=[param], name="alg")

            ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir, alg=alg_1)
            ckpt_mngr.save(0)

            ckpt_path = ckpt_dir + '/ckpt-0'

            # 2) construct a seoncd algorithm instance with different parameter
            # values, without in-algorithm checkpoint pre-loading
            new_alg = MyAlg(
                params=[nn.Parameter(torch.Tensor([-10]))], name="new_alg")

            # 3) new_alg's state_dict should be different from alg_1's state dict
            self.assertTrue(new_alg.state_dict() != alg_1.state_dict())

            # 4) construct a second algorithm instance with different parameter
            # values and use in-algorithm pre-loading of a previously saved
            # checkpoint from alg_1
            new_alg = MyAlg(
                params=[nn.Parameter(torch.Tensor([-10]))],
                checkpoint=ckpt_path,  # an example where the prefix is omitted
                name="new_alg")

            # 5) new_alg's state_dict should match with alg_1's state dict
            self.assertTrue(new_alg.state_dict() == alg_1.state_dict())

    def test_subcheckpoint_preloading(self):
        # test can load from a sub-set of a full checkpoint which corresponds
        # to the full state dict of an algorithm
        with tempfile.TemporaryDirectory() as ckpt_dir:
            # 1) construct a composed algorithm
            param_1 = nn.Parameter(torch.Tensor([1]))
            alg_1 = MyAlg(params=[param_1], name="alg_1")

            old_alg_1_state_dict = alg_1.state_dict()

            param_2 = nn.Parameter(torch.Tensor([2]))
            optimizer_2 = alf.optimizers.Adam(lr=0.2)
            alg_2 = MyAlg(
                params=[param_2], optimizer=optimizer_2, name="alg_2")

            optimizer_root = alf.optimizers.Adam(lr=0.1)
            param_root = nn.Parameter(torch.Tensor([0]))

            alg_composed = ComposedAlg(
                params=[param_root],
                optimizer=optimizer_root,
                sub_alg1=alg_1,
                sub_alg2=alg_2,
                name="root")

            # 2）save a checkpoint for the composed algorithm
            ckpt_dir_composed = ckpt_dir + '/alg_composed/'
            os.mkdir(ckpt_dir_composed)

            ckpt_mngr_composed = ckpt_utils.Checkpointer(
                ckpt_dir_composed, alg=alg_composed)
            ckpt_mngr_composed.save(0)

            ckpt_path = ckpt_dir_composed + '/ckpt-0'

            # 3）test checkpoint loading with prefix
            # construct another MyAlg instance, which is a sub-alg
            # of alg_composed
            new_alg_1 = MyAlg(
                params=[nn.Parameter(torch.Tensor([-200]))],
                checkpoint="alg._sub_alg1@" + ckpt_path)

            # 4) test new_alg_1 loaded successfully from the composed checkpoint
            self.assertTrue(new_alg_1.state_dict() == old_alg_1_state_dict)

    def test_partial_preloading_and_then_checkpoint_loading(self):
        # test the scenario that the sub-algorithm of a composed algorithm is
        # pre-loaded first, and then load the full checkpoint using the standard
        # checkpoint manager (as done in policy trainer).
        # This scenario simulates the case when we want to use the parameters
        # from a particular checkpoint for a sub-algorithm, instead of the
        # parameters in the full checkpoint of the composed algorithm.

        with tempfile.TemporaryDirectory() as ckpt_dir:
            # 1) construct sub-algorithm alg_1 and save a checkpoint
            param_1 = nn.Parameter(torch.Tensor([1]))
            alg_1 = MyAlg(params=[param_1], name="alg_1")
            ckpt_dir_1 = ckpt_dir + '/alg_1/'
            os.mkdir(ckpt_dir_1)
            ckpt_mngr_1 = ckpt_utils.Checkpointer(ckpt_dir_1, alg=alg_1)
            ckpt_mngr_1.save(0)
            ckpt_path = ckpt_dir_1 + '/ckpt-0'

            # 2) construct a composed algorithm, where alg_1's parameter value
            # is updated; then save the composed checkpoint
            alg_1._param_list[0] = nn.Parameter(torch.Tensor([1000]))

            param_2 = nn.Parameter(torch.Tensor([2]))
            optimizer_2 = alf.optimizers.Adam(lr=0.2)
            alg_2 = MyAlg(
                params=[param_2], optimizer=optimizer_2, name="alg_2")
            optimizer_root = alf.optimizers.Adam(lr=0.1)
            param_root = nn.Parameter(torch.Tensor([0]))

            alg_composed = ComposedAlg(
                params=[param_root],
                optimizer=optimizer_root,
                sub_alg1=alg_1,
                sub_alg2=alg_2,
                name="root")

            ckpt_dir_composed = ckpt_dir + '/alg_composed/'
            os.mkdir(ckpt_dir_composed)

            ckpt_mngr_composed = ckpt_utils.Checkpointer(
                ckpt_dir_composed, alg=alg_composed)
            ckpt_mngr_composed.save(0)

            # 3) construct a new sub-algorithm instance new_alg_1, with initial
            # parameter value different from alg_1, with pre-loading from alg_1's
            # checkpoint
            new_alg_1 = MyAlg(
                params=[nn.Parameter(torch.Tensor([-200]))],
                checkpoint=ckpt_path)  # an example where the prefix is omitted

            # 4) construct a new composed algorithm alg_composed_new using new_alg_1
            alg_composed_new = ComposedAlg(
                params=[param_root],
                optimizer=alf.optimizers.Adam(lr=0.1),
                sub_alg1=new_alg_1,
                sub_alg2=alg_2,
                name="root")

            # 5) load the composed checkpoint for alg_composed_new using the
            # checkpoint manager (simulating the case in policy trainer)
            ckpt_mngr_composed = ckpt_utils.Checkpointer(
                ckpt_dir_composed, alg=alg_composed_new)
            ckpt_mngr_composed.load()

            # 6) test checkpoint manager's loading won't overwrite in-algorithm
            # loading
            self.assertTrue(alg_composed_new.state_dict()
                            ['_sub_alg1._param_list.0'] == torch.Tensor([1]))

    def test_mismatch_checkpoint(self):
        # test can detect the case when there is mismatch between
        # keys of the checkpoint and model
        with tempfile.TemporaryDirectory() as ckpt_dir:
            # 1) construct a composed algorithm
            param_1 = nn.Parameter(torch.Tensor([1]))
            alg_1 = MyAlg(params=[param_1], name="alg_1")

            param_2 = nn.Parameter(torch.Tensor([2]))
            optimizer_2 = alf.optimizers.Adam(lr=0.2)
            alg_2 = MyAlg(
                params=[param_2], optimizer=optimizer_2, name="alg_2")

            optimizer_root = alf.optimizers.Adam(lr=0.1)
            param_root = nn.Parameter(torch.Tensor([0]))

            alg_composed = ComposedAlg(
                params=[param_root],
                optimizer=optimizer_root,
                sub_alg1=alg_1,
                sub_alg2=alg_2,
                name="root")

            # 2）save a checkpoint for the composed algorithm
            ckpt_dir_composed = ckpt_dir + '/alg_composed/'
            os.mkdir(ckpt_dir_composed)

            ckpt_mngr_composed = ckpt_utils.Checkpointer(
                ckpt_dir_composed, alg=alg_composed)
            ckpt_mngr_composed.save(0)

            ckpt_path = ckpt_dir_composed + '/ckpt-0'

            # 3）Checkpoint loading with a missing key case by specifying
            # a wrong prefix (``alg._sub_alg3``). Test Exception will be raised
            # and missing key message will be generated.
            with self.assertRaises(AssertionError) as context:
                new_alg_1 = MyAlg(
                    params=[nn.Parameter(torch.Tensor([-200]))],
                    checkpoint="alg._sub_alg3@" + ckpt_path)

            ckpt_check_msg = "(keys in model but not in checkpoint): ['_param_list.0']"
            self.assertTrue(ckpt_check_msg in str(context.exception))

            # 4) test can detect the case when there is dimension mismatch between
            # the parameter of checkpoint and that of the model
            with self.assertRaises(RuntimeError) as context:
                new_alg_2 = MyAlg(
                    params=[nn.Parameter(torch.Tensor([1, 1]))],
                    checkpoint="alg._sub_alg1@" + ckpt_path)
            ckpt_check_msg = (
                "size mismatch for _param_list.0: copying a param "
                "with shape torch.Size([1]) from checkpoint, the shape in current "
                "model is torch.Size([2])")
            self.assertTrue(ckpt_check_msg in str(context.exception))


if __name__ == '__main__':
    logging.use_absl_handler()
    logging.set_verbosity(logging.INFO)
    alf.test.main()
