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

import math
import unittest

from absl import logging
from absl.testing import parameterized
import tensorflow as tf

from tf_agents.networks.network import Network
from alf.algorithms.generator import Generator
from alf.algorithms.mi_estimator import MIEstimator


class Net(Network):
    def __init__(self, dim):
        super().__init__(
            input_tensor_spec=tf.TensorSpec(shape=(dim, )),
            state_spec=(),
            name="Net")
        self._w = tf.Variable(
            initial_value=[[1, 2], [-1, 1], [1, 1]],
            shape=(3, dim),
            dtype=tf.float32)

    def call(self, input):
        return tf.matmul(input, self._w), ()


class Net2(Network):
    def __init__(self, dim):
        super().__init__(
            input_tensor_spec=[
                tf.TensorSpec(shape=(dim, )),
                tf.TensorSpec(shape=(dim, ))
            ],
            state_spec=(),
            name="Net")
        self._w = tf.Variable(
            initial_value=[[1, 2], [1, 1]], shape=(dim, dim), dtype=tf.float32)
        self._u = tf.Variable(
            initial_value=tf.zeros((dim, dim)),
            shape=(dim, dim),
            dtype=tf.float32)

    def call(self, input):
        return tf.matmul(input[0], self._w) + tf.matmul(input[1], self._u), ()


class GeneratorTest(parameterized.TestCase, unittest.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(tf.reduce_max(abs(x - y))), eps)

    @parameterized.parameters(
        dict(mode='STEIN'),
        dict(mode='ML'),
        dict(mode='ML', mi_weight=1),
    )
    def test_generator_unconditional(self, mode='ML', mi_weight=None):
        """
        The generator is trained to match(STEIN)/maximize(ML) the likelihood
        of a Gaussian distribution with zero mean and diagonal variance (1, 4).
        After training, w^T w is the variance of the distribution implied by the
        generator. So it should be diag(1,4) for STEIN and 0 for 'ML'.
        """
        logging.info("mode: %s mi_weight: %s" % (mode, mi_weight))
        dim = 2
        batch_size = 512
        net = Net(dim)
        generator = Generator(
            dim,
            noise_dim=3,
            mode=mode,
            net=net,
            mi_weight=mi_weight,
            optimizer=tf.optimizers.Adam(learning_rate=1e-3))

        var = tf.constant([1, 4], dtype=tf.float32)
        precision = 1. / var

        def _neglogprob(x):
            return tf.squeeze(
                0.5 * tf.matmul(x * x, tf.reshape(precision, (dim, 1))),
                axis=-1)

        @tf.function
        def _train():
            with tf.GradientTape() as tape:
                alg_step = generator.train_step(
                    inputs=None, loss_func=_neglogprob, batch_size=batch_size)
            generator.train_complete(tape, alg_step.info)

        for i in range(5000):
            _train()
            # older version of tf complains about directly multiplying two
            # variables.
            learned_var = tf.matmul((1. * net._w), (1. * net._w),
                                    transpose_a=True)
            if i % 500 == 0:
                tf.print(i, "learned var=", learned_var)

        if mode == 'STEIN':
            self.assertArrayEqual(tf.linalg.diag(var), learned_var, 0.1)
        elif mode == 'ML':
            if mi_weight is None:
                self.assertArrayEqual(tf.zeros((dim, dim)), learned_var, 0.1)
            else:
                self.assertGreater(
                    float(tf.reduce_sum(tf.abs(learned_var))), 0.5)

    @parameterized.parameters(
        dict(mode='STEIN'),
        dict(mode='ML'),
        dict(mode='ML', mi_weight=1),
    )
    def test_generator_conditional(self, mode='ML', mi_weight=None):
        """
        The target conditional distribution is N(yu; diag(1, 4)). After training
        net._u should be u for both STEIN and ML. And w^T*w should be diag(1, 4)
        for STEIN and 0 for ML.
        """
        logging.info("mode: %s mi_weight: %s" % (mode, mi_weight))
        dim = 2
        batch_size = 512
        net = Net2(dim)
        generator = Generator(
            dim,
            noise_dim=dim,
            mode=mode,
            net=net,
            mi_weight=mi_weight,
            input_tensor_spec=tf.TensorSpec((dim, )),
            optimizer=tf.optimizers.Adam(learning_rate=1e-3))

        var = tf.constant([1, 4], dtype=tf.float32)
        precision = 1. / var
        u = tf.constant([[-0.3, 1], [1, 2]], dtype=tf.float32)

        def _neglogprob(xy):
            x, y = xy
            d = x - tf.matmul(y, u)
            return tf.squeeze(
                0.5 * tf.matmul(d * d, tf.reshape(precision, (dim, 1))),
                axis=-1)

        @tf.function
        def _train():
            y = tf.random.normal(shape=(batch_size, dim))
            with tf.GradientTape() as tape:
                alg_step = generator.train_step(
                    inputs=y, loss_func=_neglogprob)
            generator.train_complete(tape, alg_step.info)

        for i in range(5000):
            _train()
            # older version of tf complains about directly multiplying two
            # variables.
            learned_var = tf.matmul((1. * net._w), (1. * net._w),
                                    transpose_a=True)
            if i % 500 == 0:
                tf.print(i, "learned var=", learned_var)
                tf.print("u=", net._u)

        if mi_weight is not None:
            self.assertGreater(float(tf.reduce_sum(tf.abs(learned_var))), 0.5)
        elif mode == 'STEIN':
            self.assertArrayEqual(net._u, u, 0.1)
            self.assertArrayEqual(tf.linalg.diag(var), learned_var, 0.1)
        elif mode == 'ML':
            self.assertArrayEqual(net._u, u, 0.1)
            self.assertArrayEqual(tf.zeros((dim, dim)), learned_var, 0.1)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    unittest.main()
