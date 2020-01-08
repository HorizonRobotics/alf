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

from tf_agents.utils.tensor_normalizer import EMATensorNormalizer
from tf_agents.utils.common import create_variable
from tf_agents.specs.tensor_spec import TensorSpec
from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal


class AdaptiveNormalizer(EMATensorNormalizer):
    def __init__(self,
                 tensor_spec,
                 speed=2.0,
                 auto_update=True,
                 variance_epsilon=1e-3):
        """Create reward normalizer

        This normalizer gives higher weight to more recent samples for
        calculating mean and variance. Roughly speaking, the weight for each
        sample at time t is proportional to (t/T)^(speed-1), where T is the
        current time step. See docs/streaming_averaging_amd_sampling.py for
        detail.

        Args:
            tensor_spec (TensorSpec): spec of the mean of tensors to be
              normlized.
            speed (float): speed of updating mean and variance.
            auto_update (bool): If True, automatically update mean and variance
              for each call to `normalize()`. Otherwise, the user needs to call
              `update()`
        """
        self._total_env_steps = tf.Variable(
            int(speed), dtype=tf.int64, trainable=False)
        self._update_ema_rate = tf.Variable(
            1.0, dtype=tf.float32, trainable=False)
        self._speed = speed
        self._auto_update = auto_update
        self._variance_epsilon = variance_epsilon
        super(AdaptiveNormalizer, self).__init__(
            tensor_spec, norm_update_rate=self._update_ema_rate)

    def _create_variables(self):
        """Creates the variables needed for EMATensorNormalizer.
           Override this function by using self._var_running for
           a more accurate moving average variance estimation.
           See _update() for more details.
        """
        self._mean_moving_avg = tf.nest.map_structure(
            lambda spec: create_variable('mean', 0, spec.shape, tf.float32),
            self._flat_tensor_spec)
        self._var_running = tf.nest.map_structure(
            lambda spec: create_variable('var_run', 0, spec.shape, tf.float32),
            self._flat_tensor_spec)

    @property
    def _var_moving_avg(self):
        return tf.nest.map_structure(
            lambda t: t / (tf.cast(self._total_env_steps, tf.float32) - 1),
            self._var_running)

    @property
    def variables(self):
        """Returns a tuple of tf variables owned by this EMATensorNormalizer."""
        return (tf.nest.pack_sequence_as(self._tensor_spec,
                                         self._mean_moving_avg),
                tf.nest.pack_sequence_as(self._tensor_spec, self._var_running))

    def _update_ops(self, tensor, outer_dims):
        """Returns a list of update obs and EMATensorNormalizer mean and var.
           Override this function for a more accurate computation of moving
           average variance. The algorithm implemented is from Donald Knuth's
           book. See detailed formulas:
           https://www.johndcook.com/blog/standard_deviation/
        """

        def _tensor_update_ops(single_tensor, mean_var, var_var):
            """Make update ops for a single non-nested tensor.
               Maybe some equations here?
            """
            mean = tf.reduce_mean(input_tensor=single_tensor, axis=outer_dims)
            var = tf.reduce_mean(
                input_tensor=tf.square(single_tensor - mean_var),
                axis=outer_dims)
            with tf.control_dependencies([mean, var]):
                update_ops = [
                    mean_var.assign_add(
                        self._norm_update_rate * (mean - mean_var)),
                    var_var.assign_add((1 - self._norm_update_rate) * var)
                ]
            return update_ops

        # Aggregate update ops for all parts of potentially nested tensor.
        tensor = tf.nest.flatten(tensor)
        updates = tf.nest.map_structure(_tensor_update_ops, tensor,
                                        self._mean_moving_avg,
                                        self._var_running)
        all_update_ops = tf.nest.flatten(updates)

        return all_update_ops

    def normalize(self, tensors, clip_value=-1.0):
        """Normalized the reward

        Args:
            tensors (nested Tensor): tensors to be normalized
            clip_value (float): the normalized values will be clipped to +/-
                this value. If it's negative, then ignore
        Returns:
            normalized reward. with mean equal to 0 and variance equal to 1
              over the time.
        """
        if self._auto_update:
            self.update(tensors)
        n_tensors = super(AdaptiveNormalizer, self).normalize(
            tensors,
            clip_value=clip_value,
            center_mean=True,
            variance_epsilon=self._variance_epsilon)
        return n_tensors

    def denormalize(self, tensor):
        """Denormalize the input tensor. """
        tf.nest.assert_same_structure(tensor, self._tensor_spec)
        tensor = tf.nest.flatten(tensor)
        tensor = tf.nest.map_structure(lambda t: tf.cast(t, tf.float32),
                                       tensor)

        with tf.name_scope(self._scope + '/denormalize'):
            mean_estimate, var_estimate = self._get_mean_var_estimates()

        def _denormalize_single_tensor(single_tensor, single_mean, single_var):
            return single_tensor * single_var + single_mean

        denormalized_tensor = nest.map_structure_up_to(
            self._flat_tensor_spec,
            _denormalize_single_tensor,
            tensor,
            mean_estimate,
            var_estimate,
            check_types=False)
        denormalized_tensor = tf.nest.pack_sequence_as(self._tensor_spec,
                                                       denormalized_tensor)
        return denormalized_tensor

    def update(self, tensors):
        self._update_ema_rate.assign(
            self._speed / tf.cast(self._total_env_steps, tf.float32))
        super(AdaptiveNormalizer, self).update(tensors)
        self._total_env_steps.assign_add(1)


class ScalarAdaptiveNormalizer(AdaptiveNormalizer):
    def __init__(self, speed=2.0, auto_update=True, variance_epsilon=1e-10):
        super(ScalarAdaptiveNormalizer, self).__init__(
            tensor_spec=TensorSpec((), dtype=tf.float32),
            speed=speed,
            auto_update=auto_update,
            variance_epsilon=variance_epsilon)
