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

import unittest

from absl import logging
from absl.testing import parameterized

import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp

import alf.utils.dist_utils as dist_utils


class EstimatedEntropyTest(parameterized.TestCase, unittest.TestCase):
    def assertArrayAlmostEqual(self, x, y, eps):
        self.assertLess(tf.reduce_max(tf.abs(x - y)), eps)

    @parameterized.parameters(False, True)
    def test_estimated_entropy(self, assume_reparametrization):
        logging.info("assume_reparametrization=%s" % assume_reparametrization)
        num_samples = 1000000
        seed_stream = tfp.distributions.SeedStream(
            seed=1, salt='test_estimated_entropy')
        batch_shape = (2, )
        loc = tf.random.normal(shape=batch_shape, seed=seed_stream())
        scale = tf.abs(tf.random.normal(shape=batch_shape, seed=seed_stream()))

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(scale)
            dist = tfp.distributions.Normal(loc=loc, scale=scale)
            analytic_entropy = dist.entropy()
            est_entropy, est_entropy_for_gradient = dist_utils.estimated_entropy(
                dist=dist,
                seed=seed_stream(),
                assume_reparametrization=assume_reparametrization,
                num_samples=num_samples)

        analytic_grad = tape.gradient(analytic_entropy, scale)
        est_grad = tape.gradient(est_entropy_for_gradient, scale)
        logging.info("scale=%s" % scale)
        logging.info("analytic_entropy=%s" % analytic_entropy)
        logging.info("estimated_entropy=%s" % est_entropy)
        self.assertArrayAlmostEqual(analytic_entropy, est_entropy, 2e-2)

        logging.info("analytic_entropy_grad=%s" % analytic_grad)
        logging.info("estimated_entropy_grad=%s" % est_grad)
        self.assertArrayAlmostEqual(analytic_grad, est_grad, 1e-2)
        if not assume_reparametrization:
            est_grad_wrong = tape.gradient(est_entropy, scale)
            logging.info("estimated_entropy_grad_wrong=%s", est_grad_wrong)
            self.assertLess(tf.reduce_max(tf.abs(est_grad_wrong)), 2e-3)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    unittest.main()
