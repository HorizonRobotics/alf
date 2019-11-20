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

from absl import logging
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_probability as tfp

from alf.algorithms.mi_estimator import MIEstimator, ScalarAdaptiveAverager


class MIEstimatorTest(parameterized.TestCase, tf.test.TestCase):
    # @parameterized.parameters(
    #     dict(estimator='DV', rho=0.0, eps=0.02),
    #     dict(estimator='KLD', rho=0.0, eps=0.02),
    #     dict(estimator='JSD', rho=0.0, eps=0.02),
    #     dict(estimator='DV', rho=0.5, eps=0.4),
    #     dict(estimator='DV', rho=0.5, eps=0.4, sampler='double_buffer'),
    #     dict(estimator='DV', rho=0.5, eps=0.4, buffer_size=1048576),
    #     dict(estimator='DV', rho=0.5, eps=0.6, sampler='shuffle'),
    #     dict(estimator='DV', rho=0.5, eps=0.4, sampler='shift'),
    #     dict(estimator='KLD', rho=0.5, eps=0.4),
    #     dict(estimator='JSD', rho=0.5, eps=0.4),
    #     dict(estimator='DV', rho=0.9, eps=7.0),
    #     dict(estimator='KLD', rho=0.9, eps=7.0),
    #     dict(estimator='JSD', rho=0.9, eps=12.0),
    # )
    # def test_mi_estimator(self,
    #                       estimator='DV',
    #                       sampler='buffer',
    #                       rho=0.9,
    #                       eps=1000.0,
    #                       buffer_size=65536,
    #                       dim=20):
    #     mi_estimator = MIEstimator(
    #         x_spec=[
    #             tf.TensorSpec(shape=(dim // 3, ), dtype=tf.float32),
    #             tf.TensorSpec(shape=(dim - dim // 3, ), dtype=tf.float32)
    #         ],
    #         y_spec=[
    #             tf.TensorSpec(shape=(dim // 2, ), dtype=tf.float32),
    #             tf.TensorSpec(shape=(dim // 2, ), dtype=tf.float32)
    #         ],
    #         fc_layers=(512, ),
    #         buffer_size=buffer_size,
    #         estimator_type=estimator,
    #         sampler=sampler,
    #         averager=ScalarAdaptiveAverager(),
    #         optimizer=tf.optimizers.Adam(learning_rate=1e-4))

    #     a = 0.5 * (math.sqrt(1 + rho) + math.sqrt(1 - rho))
    #     b = 0.5 * (math.sqrt(1 + rho) - math.sqrt(1 - rho))
    #     # This matrix transforms standard Gaussian to a Gaussian with variance
    #     # [[1, rho], [rho, 1]]
    #     w = tf.constant([[a, b], [b, a]], dtype=tf.float32)
    #     var = tf.matmul(w, w)
    #     entropy = 0.5 * tf.math.log(tf.linalg.det(2 * math.pi * math.e * var))
    #     entropy_x = 0.5 * tf.math.log(2 * math.pi * math.e * var[0, 0])
    #     entropy_y = 0.5 * tf.math.log(2 * math.pi * math.e * var[1, 1])
    #     mi = float(dim * (entropy_x + entropy_y - entropy))

    #     def _get_batch(batch_size):
    #         xy = tf.random.normal(shape=(batch_size * dim, 2))
    #         xy = tf.matmul(xy, w)
    #         x = xy[:, 0]
    #         y = xy[:, 1]
    #         x = tf.reshape(x, (-1, dim))
    #         y = tf.reshape(y, (-1, dim))
    #         x = [x[..., :dim // 3], x[..., dim // 3:]]
    #         y = [y[..., :dim // 2], y[..., dim // 2:]]
    #         return x, y

    #     def _calc_estimated_mi(i, mi_samples):
    #         estimated_mi, var = tf.nn.moments(mi_samples, axes=(0, ))
    #         estimated_mi = float(estimated_mi)
    #         # For DV estimator, the following std is an approximated std.
    #         logging.info(
    #             "%s estimated mi=%s std=%s" %
    #             (i, estimated_mi, math.sqrt(var / mi_samples.shape[0])))
    #         return estimated_mi

    #     batch_size = 512
    #     info = "mi=%s estimator=%s buffer_size=%s sampler=%s dim=%s" % (
    #         float(mi), estimator, buffer_size, sampler, dim)

    #     @tf.function
    #     def _train():
    #         x, y = _get_batch(batch_size)
    #         with tf.GradientTape() as tape:
    #             alg_step = mi_estimator.train_step((x, y))
    #         mi_estimator.train_complete(tape, alg_step.info)
    #         return alg_step

    #     for i in range(5000):
    #         alg_step = _train()
    #         if i % 1000 == 0:
    #             _calc_estimated_mi(i, alg_step.outputs)
    #     x, y = _get_batch(16384)
    #     log_ratio = mi_estimator.calc_pmi(x, y)
    #     estimated_mi = _calc_estimated_mi(info, log_ratio)
    #     if estimator == 'JSD':
    #         self.assertAlmostEqual(estimated_mi, mi, delta=eps)
    #     else:
    #         self.assertLess(estimated_mi, mi)
    #         self.assertGreater(estimated_mi, mi - eps)
    #     return mi, estimated_mi

    def test_conditional_mi_estimator(self, estimator='JSD', eps=0.3, dim=1):
        """ estimate the conditional mutual information MI(X;Y|Z)

        X, Y and Z are generated by the following procedure:
            Z ~ N(0, 1)
            X|z ~ N(z, 1)
            if z >= 0:
                Y|x,z ~ N(z + xz, 1)
            else:
                Y|x,z ~ N(0, 1)
        When z>0,
            [X, Y] ~ N([z, z+z^2], [[1, z], [z, 1+z^2]])
            MI(X;Y|z) = 0.5 * log(1+z^2)
        """
        mi_estimator = MIEstimator(
            x_spec=[
                tf.TensorSpec(shape=(dim, ), dtype=tf.float32),  # z
                tf.TensorSpec(shape=(dim, ), dtype=tf.float32)  # x
            ],
            y_spec=tf.TensorSpec(shape=(dim, ), dtype=tf.float32),  # y
            fc_layers=(1024, 1024),
            estimator_type=estimator,
            optimizer=tf.optimizers.Adam(learning_rate=2e-6))

        z = tf.random.normal((10000, ))
        mi = 0.25 * tf.reduce_mean(tf.math.log(1 + z * z))

        def _get_batch(batch_size, z=None):
            if z is None:
                z = tf.random.normal(shape=(batch_size, dim))
            x_dist = tfp.distributions.Normal(loc=z, scale=tf.ones_like(z))
            y_dist = tfp.distributions.Normal(
                loc=z + z * z, scale=tf.sqrt(1 + z * z))
            x = x_dist.sample()
            y = (z + x * z) * tf.cast(z > 0, tf.float32)
            y += tf.random.normal(shape=(batch_size, dim))
            return x, y, z, x_dist, y_dist

        def _calc_estimated_mi(i, mi_samples):
            estimated_mi, var = tf.nn.moments(mi_samples, axes=(0, ))
            estimated_mi = float(estimated_mi)
            logging.info(
                "%s estimated mi=%s std=%s" %
                (i, estimated_mi, math.sqrt(var / mi_samples.shape[0])))
            return estimated_mi

        batch_size = 1024

        info = "mi=%s estimator=%s dim=%s" % (float(mi), estimator, dim)

        @tf.function
        def _train():
            x, y, z, x_dist, y_dist = _get_batch(batch_size)
            with tf.GradientTape() as tape:
                alg_step = mi_estimator.train_step(([z, x], y),
                                                   y_distribution=y_dist)
            mi_estimator.train_complete(tape, alg_step.info)
            return alg_step

        for i in range(5000):
            _train()
            if i % 1000 == 0:
                x, y, z, x_dist, y_dist = _get_batch(batch_size)
                pmi = mi_estimator.calc_pmi([z, x], y)
                _calc_estimated_mi(i, pmi)

        batch_size = 16384
        x, y, z, x_dist, y_dist = _get_batch(batch_size)
        log_ratio = mi_estimator.calc_pmi([z, x], y)
        estimated_mi = _calc_estimated_mi(info, log_ratio)
        if estimator == 'JSD':
            self.assertAlmostEqual(estimated_mi, mi, delta=eps)
        else:
            self.assertLess(estimated_mi, mi)
            self.assertGreater(estimated_mi, mi - eps)
        for z1 in tf.range(-2., 2.001, 0.125):
            x, y, z, x_dist, y_dist = _get_batch(
                batch_size, z1 * tf.ones((batch_size, dim)))
            pmi = 0.5 * (tf.square(y - z - z * z) / (1 + z * z) -
                         tf.square(y - z - x * z) + tf.math.log(1 + z * z))
            pmi = pmi * tf.cast(z1 > 0, tf.float32)
            estimated_pmi = mi_estimator.calc_pmi([z, x], y)
            pmi_rmse = tf.sqrt(tf.reduce_mean(tf.square(pmi - estimated_pmi)))
            info = "z={z} mi={mi} pmi_rmse={pmi_rmse}".format(
                z=float(z1),
                mi=float(
                    0.5 * tf.math.log(1 + z * z) * tf.cast(z > 0, tf.float32)),
                pmi_rmse=float(pmi_rmse))
            _calc_estimated_mi(info, estimated_pmi)

        return mi, estimated_mi


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    tf.test.main()
