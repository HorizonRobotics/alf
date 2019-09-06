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

from alf.algorithms.mi_estimator import MINEstimator, ScalarAdaptiveAverager


class MINEstimatorTest(parameterized.TestCase, unittest.TestCase):
    @parameterized.parameters(
        dict(estimator='MINE', rho=0.0, eps=0.02),
        dict(estimator='MINE', rho=0.5, eps=0.3),
        dict(estimator='MINE', rho=0.9, eps=7.0),
        dict(estimator='MINE-f', rho=0.0, eps=0.02),
        dict(estimator='MINE-f', rho=0.5, eps=0.3),
        dict(estimator='MINE-f', rho=0.9, eps=7.0),
    )
    def test_mi_estimator(self,
                          estimator='MINE',
                          rho=0.9,
                          eps=1.0,
                          buffer_size=65536,
                          dim=20):
        mi_estimator = MINEstimator(
            x_spec=tf.TensorSpec(shape=(dim, ), dtype=tf.float32),
            y_spec=tf.TensorSpec(shape=(dim, ), dtype=tf.float32),
            fc_layers=(512, ),
            buffer_size=1048576,
            estimator_type=estimator,
            averager=ScalarAdaptiveAverager(),
            optimizer=tf.optimizers.Adam(learning_rate=1e-4))

        a = 0.5 * (math.sqrt(1 + rho) + math.sqrt(1 - rho))
        b = 0.5 * (math.sqrt(1 + rho) - math.sqrt(1 - rho))
        # This matrix transforms standard Gaussian to a Gaussian with variance
        # [[1, rho], [rho, 1]]
        w = tf.constant([[a, b], [b, a]], dtype=tf.float32)
        var = tf.matmul(w, w)
        entropy = 0.5 * tf.math.log(tf.linalg.det(2 * math.pi * math.e * var))
        entropy_x = 0.5 * tf.math.log(2 * math.pi * math.e * var[0, 0])
        entropy_y = 0.5 * tf.math.log(2 * math.pi * math.e * var[1, 1])
        mi = float(dim * (entropy_x + entropy_y - entropy))

        def _get_batch(batch_size):
            xy = tf.random.normal(shape=(batch_size * dim, 2))
            xy = tf.matmul(xy, w)
            x = xy[:, 0]
            y = xy[:, 1]
            x = tf.reshape(x, (-1, dim))
            y = tf.reshape(y, (-1, dim))
            return x, y

        def _calc_estimated_mi(i, alg_step):
            estimated_mi, var = tf.nn.moments(alg_step.outputs, axes=(0, ))
            estimated_mi = float(estimated_mi)
            # For MINE estimator, the following std is an approximated std.
            logging.info(
                "%s estimated mi=%s std=%s" %
                (i, estimated_mi, math.sqrt(var / alg_step.outputs.shape[0])))
            return estimated_mi

        batch_size = 512
        logging.info("mi=%s" % float(mi))
        for i in range(5000):
            x, y = _get_batch(batch_size)
            with tf.GradientTape() as tape:
                alg_step = mi_estimator.train_step((x, y))
            mi_estimator.train_complete(tape, alg_step.info)
            if i % 1000 == 0:
                _calc_estimated_mi(i, alg_step)
        x, y = _get_batch(16384)
        alg_step = mi_estimator.predict((x, y))
        estimated_mi = _calc_estimated_mi(estimator, alg_step)
        self.assertLess(estimated_mi, mi)
        self.assertGreater(estimated_mi, mi - eps)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    unittest.main()
