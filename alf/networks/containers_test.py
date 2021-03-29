# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import alf

from alf.utils.spec_utils import is_same_spec


def _randn_from_spec(specs, batch_size):
    return alf.nest.map_structure(
        lambda spec: torch.randn(batch_size, *spec.shape), specs)


class ContainersTest(alf.test.TestCase):
    def _verify_parameter_copy(self, src, copy):
        """net.copy() only copy the structure, not the values of parameters."""
        for s, c in zip(src.parameters(), copy.parameters()):
            if (s == 0).all():
                self.assertTrue((c == 0).all())
            else:
                self.assertFalse((c == 0).all())
                self.assertFalse((c == s).all())

    def test_sequential1(self):
        net = alf.nn.Sequential(
            alf.layers.FC(4, 6), alf.nn.GRUCell(6, 8), alf.nn.GRUCell(8, 12))

        self.assertEqual(net.input_tensor_spec, alf.TensorSpec((4, )))
        self.assertTrue(
            alf.utils.spec_utils.is_same_spec(
                net.state_spec,
                [(), alf.TensorSpec(
                    (8, )), alf.TensorSpec((12, ))]))

        batch_size = 24
        x = _randn_from_spec(net.input_tensor_spec, batch_size)
        state = _randn_from_spec(net.state_spec, batch_size)
        y, new_state = net(x, state)

        x1 = net[0](x)
        x2, s1 = net[1](x1, state[1])
        x3, s2 = net[2](x2, state[2])
        self.assertEqual(x3, y)
        self.assertEqual((), new_state[0])
        self.assertEqual(s1, new_state[1])
        self.assertEqual(s2, new_state[2])

        net_copy = net.copy()
        self._verify_parameter_copy(net, net_copy)

    def test_sequential_complex1(self):
        net = alf.nn.Sequential(
            alf.layers.FC(4, 6),
            a=alf.nn.GRUCell(6, 8),
            b=alf.nn.GRUCell(8, 12),
            c=('a', alf.nn.GRUCell(8, 16)),
            output=('b', 'c'))
        self.assertEqual(net.input_tensor_spec, alf.TensorSpec((4, )))
        self.assertTrue(
            alf.utils.spec_utils.is_same_spec(net.state_spec,
                                              [(),
                                               alf.TensorSpec((8, )),
                                               alf.TensorSpec((12, )),
                                               alf.TensorSpec((16, ))]))

        batch_size = 24
        x = _randn_from_spec(net.input_tensor_spec, batch_size)
        state = _randn_from_spec(net.state_spec, batch_size)
        y, new_state = net(x, state)

        x1 = net[0](x)
        a, s1 = net[1](x1, state[1])
        b, s2 = net[2](a, state[2])
        c, s3 = net[3](a, state[3])
        self.assertEqual(len(y), 2)
        self.assertEqual(type(y), tuple)
        self.assertEqual(b, y[0])
        self.assertEqual(c, y[1])
        self.assertEqual((), new_state[0])
        self.assertEqual(s1, new_state[1])
        self.assertEqual(s2, new_state[2])
        self.assertEqual(s3, new_state[3])

    def test_sequential2(self):
        net = alf.nn.Sequential(
            alf.layers.FC(4, 6), alf.layers.FC(6, 8), alf.layers.FC(8, 12))

        self.assertEqual(net.input_tensor_spec, alf.TensorSpec((4, )))
        self.assertEqual(net.state_spec, ())

        batch_size = 24
        x = _randn_from_spec(net.input_tensor_spec, batch_size)
        y, new_state = net(x)
        self.assertEqual(new_state, ())

        x1 = net[0](x)
        x2 = net[1](x1)
        x3 = net[2](x2)
        self.assertEqual(x3, y)

        net_copy = net.copy()
        self._verify_parameter_copy(net, net_copy)

    def test_sequential_complex2(self):
        net = alf.nn.Sequential(
            alf.layers.FC(4, 6),
            a=alf.layers.FC(6, 8),
            b=alf.layers.FC(8, 12),
            c=(('a', 'b'), alf.layers.NestConcat()))

        self.assertEqual(net.input_tensor_spec, alf.TensorSpec((4, )))
        self.assertEqual(net.state_spec, ())

        batch_size = 24
        x = _randn_from_spec(net.input_tensor_spec, batch_size)
        y, new_state = net(x)
        self.assertEqual(new_state, ())

        x1 = net[0](x)
        x2 = net[1](x1)
        x3 = net[2](x2)
        x4 = net[3]((x2, x3))
        self.assertEqual(x4, y)

        net_copy = net.copy()
        self._verify_parameter_copy(net, net_copy)

    def test_sequential_complex3(self):
        net = alf.nn.Sequential(
            alf.layers.FC(4, 6),
            a=alf.layers.FC(6, 8),
            b=alf.layers.FC(8, 8),
            c=(('a', 'b'), lambda x: x[0] + x[1]))

        self.assertEqual(net.input_tensor_spec, alf.TensorSpec((4, )))
        self.assertEqual(net.state_spec, ())

        batch_size = 24
        x = _randn_from_spec(net.input_tensor_spec, batch_size)
        y, new_state = net(x)
        self.assertEqual(new_state, ())

        x1 = net[0](x)
        x2 = net[1](x1)
        x3 = net[2](x2)
        x4 = x2 + x3
        self.assertEqual(x4, y)

        net_copy = net.copy()
        self._verify_parameter_copy(net, net_copy)

    def test_parallel1(self):
        net = alf.nn.Parallel((alf.layers.FC(4, 6), alf.nn.GRUCell(6, 8),
                               alf.nn.GRUCell(8, 12)))

        self.assertTrue(
            is_same_spec(net.input_tensor_spec, (alf.TensorSpec(
                (4, )), alf.TensorSpec((6, )), alf.TensorSpec((8, )))))
        self.assertTrue(
            is_same_spec(net.state_spec, ((), alf.TensorSpec(
                (8, )), alf.TensorSpec((12, )))))

        batch_size = 24
        x = _randn_from_spec(net.input_tensor_spec, batch_size)
        state = _randn_from_spec(net.state_spec, batch_size)
        y, new_state = net(x, state)

        y0, state0 = net.networks[0](x[0])
        y1, state1 = net.networks[1](x[1], state[1])
        y2, state2 = net.networks[2](x[2], state[2])
        self.assertEqual(y[0], y0)
        self.assertEqual(y[1], y1)
        self.assertEqual(y[2], y2)
        self.assertEqual(new_state[0], state0)
        self.assertEqual(new_state[1], state1)
        self.assertEqual(new_state[2], state2)

        net_copy = net.copy()
        self._verify_parameter_copy(net, net_copy)

    def test_parallel2(self):
        net = alf.nn.Parallel((alf.layers.FC(4, 6), alf.layers.FC(6, 8),
                               alf.layers.FC(8, 12)))

        self.assertTrue(
            is_same_spec(net.input_tensor_spec, (alf.TensorSpec(
                (4, )), alf.TensorSpec((6, )), alf.TensorSpec((8, )))))
        self.assertEqual(net.state_spec, ())

        batch_size = 24
        x = _randn_from_spec(net.input_tensor_spec, batch_size)
        y, new_state = net(x)
        self.assertEqual(new_state, ())

        y0 = net.networks[0](x[0])[0]
        y1 = net.networks[1](x[1])[0]
        y2 = net.networks[2](x[2])[0]
        self.assertEqual(y[0], y0)
        self.assertEqual(y[1], y1)
        self.assertEqual(y[2], y2)

        net_copy = net.copy()
        self._verify_parameter_copy(net, net_copy)

    def test_branch1(self):
        net = alf.nn.Branch((alf.layers.FC(4, 6), alf.nn.GRUCell(4, 8),
                             alf.nn.GRUCell(4, 12)))

        self.assertEqual(net.input_tensor_spec, alf.TensorSpec((4, )))
        self.assertTrue(
            is_same_spec(net.state_spec, ((), alf.TensorSpec(
                (8, )), alf.TensorSpec((12, )))))

        batch_size = 24
        x = _randn_from_spec(net.input_tensor_spec, batch_size)
        state = _randn_from_spec(net.state_spec, batch_size)
        y, new_state = net(x, state)

        y0, state0 = net.networks[0](x)
        y1, state1 = net.networks[1](x, state[1])
        y2, state2 = net.networks[2](x, state[2])
        self.assertEqual(y[0], y0)
        self.assertEqual(y[1], y1)
        self.assertEqual(y[2], y2)
        self.assertEqual(new_state[0], state0)
        self.assertEqual(new_state[1], state1)
        self.assertEqual(new_state[2], state2)

        net_copy = net.copy()
        self._verify_parameter_copy(net, net_copy)

    def test_branch2(self):
        net = alf.nn.Branch((alf.layers.FC(4, 6), alf.layers.FC(4, 8),
                             alf.layers.FC(4, 12)))

        self.assertEqual(net.input_tensor_spec, alf.TensorSpec((4, )))
        self.assertEqual(net.state_spec, ())

        batch_size = 24
        x = _randn_from_spec(net.input_tensor_spec, batch_size)
        y, new_state = net(x)
        self.assertEqual(new_state, ())

        y0 = net.networks[0](x)[0]
        y1 = net.networks[1](x)[0]
        y2 = net.networks[2](x)[0]
        self.assertEqual(y[0], y0)
        self.assertEqual(y[1], y1)
        self.assertEqual(y[2], y2)

        net_copy = net.copy()
        self._verify_parameter_copy(net, net_copy)


if __name__ == '__main__':
    alf.test.main()
