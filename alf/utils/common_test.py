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
            f = self.f()
            self.assertEqual(f, "f/")
            g = self.g()
            self.assertEqual(g, "main/g/")
            h = self.h()
            self.assertEqual(h, "main/h/")

        f = self.f()
        self.assertEqual(f, "f/")
        g = self.g()
        self.assertEqual(g, "g/")
        h = self.h()
        self.assertEqual(h, "h/")


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
