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

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks.utils import BatchSquash
from tf_agents.utils.nest_utils import get_outer_rank

from alf.algorithms.algorithm import Algorithm, AlgorithmStep
from alf.utils.averager import ScalarAdaptiveAverager
from alf.utils.data_buffer import DataBuffer
from alf.utils.encoding_network import EncodingNetwork


class MINEstimator(Algorithm):
    """Mutual Infomation Neural Estimator.

    Implements MINE and MINE-f estimator in paper
    Belghazi et al "Mutual Information Neural Estimation"
    http://proceedings.mlr.press/v80/belghazi18a/belghazi18a.pdf
    """

    def __init__(self,
                 x_spec: tf.TensorSpec,
                 y_spec: tf.TensorSpec,
                 model=None,
                 fc_layers=(256, ),
                 buffer_size=65536,
                 optimizer: tf.optimizers.Optimizer = None,
                 estimator_type='MINE',
                 averager=ScalarAdaptiveAverager(),
                 name="MIEstimator"):
        """Create a MIEstimator.

        Args:
            x_spec (TensorSpec): spec of x
            y_spec (TensorSpec): spec of y
            model (Network): can be called as model([x, y]) and return a Tensor
                with shape=[batch_size, 1]. If None, a default MLP with
                fc_layers will be created.
            fc_layers (tuple[int]): size of hidden layers. Only used if model is
                None.
            buffer_size (int): capacity of buffer for storing y
            optimzer (tf.optimizers.Optimzer): optimizer
            estimator_type (str): one of 'MINE' and 'MINE-f'
            averager (EMAverager): averager used to maintain a moving average
                of exp(T). Only used for 'MINE' estimator
            name (str): name of this estimator
        """
        assert estimator_type in ['MINE', 'MINE-f']
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)
        self._x_spec = x_spec
        self._y_spec = y_spec
        if model is None:
            model = EncodingNetwork(
                name="MIEstimator",
                input_tensor_spec=[x_spec, y_spec],
                fc_layer_params=fc_layers,
                last_layer_size=1)
        self._model = model
        self._type = estimator_type
        self._y_buffer = DataBuffer(y_spec, capacity=buffer_size)
        self._z_averager = averager

    def train_step(self, inputs, state=None):
        """Perform training on one batch of inputs.

        Args:
            inputs (tuple(Tensor, Tensor)): tuple of x and y
            state: not used
        Returns:
            AlgorithmStep
                outputs (Tensor): shape=[batch_size], its mean is the estimated
                    MI
                state: not used
                info (LossInfo): info.loss is the loss
        """
        (x, y) = inputs
        num_outer_dims = get_outer_rank(x, self._x_spec)
        batch_squash = BatchSquash(num_outer_dims)
        x = batch_squash.flatten(x)
        y = batch_squash.flatten(y)
        batch_size = y.shape[0]
        self._y_buffer.add_batch(y)
        y1 = self._y_buffer.get_batch(batch_size)
        log_ratio = self._model([x, y])[0]
        ratio = tf.math.exp(self._model([x, y1])[0])

        if self._type == 'MINE':
            z = tf.stop_gradient(tf.reduce_mean(ratio))
            if self._z_averager:
                self._z_averager.update(z)
                unbiased_z = tf.stop_gradient(self._z_averager.get())
            else:
                unbiased_z = z
            # estimated MI = reduce_mean(mi)
            # ratio/z-1 does not contribute to the final estimated MI, since
            # mean(ratio/z-1) = 0. We add it so that we can have an estimation
            # of the variance of the MI estimator
            mi = log_ratio - (tf.math.log(z) + ratio / z - 1)
            loss = ratio / unbiased_z - log_ratio
        else:
            mi = log_ratio - ratio + 1
            loss = -mi

        return AlgorithmStep(
            outputs=mi, state=(), info=LossInfo(loss, extra=()))

    def calc_log_ratio(self, x, y):
        """Return estimated log P(x|y)/P(x) = log P(y|x)/P(y)
        """
        return self._model([x, y])[0]
