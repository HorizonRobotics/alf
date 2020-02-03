# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

import tensorflow as tf

import alf.utils.common as common
from alf.data_structures import namedtuple


class ImageScaleTransformerTest(tf.test.TestCase):
    def test_transform_image(self):
        shape = [10]
        observation = tf.zeros(shape, dtype=tf.uint8)
        common.image_scale_transformer(observation)

        T1 = namedtuple('T1', ['x', 'y'])
        T2 = namedtuple('T2', ['a', 'b', 'c'])
        T3 = namedtuple('T3', ['l', 'm'])
        observation = T1(
            x=T2(
                a=tf.ones(shape, dtype=tf.uint8) * 255,
                b=T3(l=tf.zeros(shape, dtype=tf.uint8))))
        transformed_observation = common.image_scale_transformer(
            observation, fields=["x.a", "x.b.l"])

        tf.debugging.assert_equal(transformed_observation.x.a,
                                  tf.ones(shape, dtype=tf.float32))
        tf.debugging.assert_equal(transformed_observation.x.b.l,
                                  tf.ones(shape, dtype=tf.float32) * -1)

        with self.assertRaises(Exception) as _:
            common.image_scale_transformer(
                observation, fields=["x.b.m"])  # empty ()

        observation = dict(x=dict(a=observation.x.a))
        common.image_scale_transformer(observation, fields=["x.a"])


class F(tf.Module):
    def __init__(self, name):
        super().__init__(name=name)
        self._created = False

    @common.function
    def train(self, a, b):
        with self.name_scope as scope:
            if not self._created:
                self._v = tf.Variable(initial_value=1.0, shape=())
                self._created = True
            return a + b, scope

    @common.function(experimental_relax_shapes=True)
    def train2(self, a, b):
        with self.name_scope as scope:
            if not self._created:
                self._v = tf.Variable(initial_value=1.0, shape=())
                self._created = True
            return a + b, scope


@common.function
def func(a, b):
    with tf.name_scope("func") as scope:
        return a + b, scope


class FunctionTest(tf.test.TestCase):
    @tf.function
    def f(self):
        with tf.name_scope("f") as scope:
            return scope

    def g(self):
        with tf.name_scope("g") as scope:
            return scope

    @common.function
    def h(self):
        with tf.name_scope("h") as scope:
            return scope

    def test_function(self):
        with tf.name_scope("main"):
            f1 = F("f1")
            f2 = F("f2")
            self.assertEqual(self.f(), "f/")
            self.assertEqual(self.g(), "main/g/")
            self.assertEqual(self.h(), "main/h/")
            self.assertEqual(f1.train(1, 3)[0], 4)
            self.assertEqual(f1.train(1, 3)[1], "main/f1/")
            self.assertEqual(f2.train(2, 3)[0], 5)
            self.assertEqual(f2.train(2, 3)[1], "main/f2/")
            self.assertEqual(f2.train2(2, 3)[0], 5)
            self.assertEqual(f2.train2(2, 3)[1], "main/f2/")
            self.assertEqual(func(2, 4)[0], 6)
            self.assertEqual(func(2, 4)[1], "main/func/")

        self.assertEqual(self.f(), "f/")
        self.assertEqual(self.g(), "g/")
        self.assertEqual(self.h(), "h/")
        self.assertEqual(f1.train(5, 3)[0], 8)
        self.assertEqual(f1.train(5, 3)[1], "main/f1/")
        self.assertEqual(f2.train(6, 3)[0], 9)
        self.assertEqual(f2.train(6, 3)[1], "main/f2/")
        self.assertEqual(func(3, 5)[0], 8)
        self.assertEqual(func(3, 5)[1], "func/")


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
