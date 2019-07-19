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
"""Algorithm base class."""

from abc import abstractmethod
from collections import namedtuple

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.trajectories.time_step import StepType

import tensorflow as tf

AlgorithmStep = namedtuple("AlgorithmStep", ["outputs", "state", "info"])


class Algorithm(tf.Module):
    """Algorithm base class.

    Algorithm is a generic interface for supervised training algorithms.

    User needs to implement train_step() and calc_loss()/train_complete().

    train_step() is called to generate actions for every environment step.
    It also needs to generate necessary information for training.

    train_complete() is called every train_interval steps (specified in
    TrainingPolicy). All the training information collected at each previous
    train_step() are batched and provided as arguments for train_complete().

    The following is the pseudo code to illustrate how Algoirhtm is for
    training.

    ```python
    while training not ends:
        training_info = []
        with GradientTape() as tape:
        for i in range(train_interval):
            get inputs
            outputs, state, info = train_step(inputs, state)
            add info to training_info

        train_complete(tape, batched_training_info)
    ```
    """

    def __init__(self,
                 train_state_spec=None,
                 predict_state_spec=None,
                 optimizer=None,
                 debug_summaries=False,
                 name="Algorithm"):
        """Create an Algorithm.

        Args:
            train_state_spec (nested TensorSpec): for the network state of
                `train_step()`
            predict_state_spec (nested TensorSpec): for the network state of
                `predict()`. If None, it's assume to be same as train_state_spec
            optimizer (tf.optimizers.Optimizer): The optimizer for training.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this algorithm.
        """
        super(Algorithm, self).__init__(name=name)

        self._train_state_spec = train_state_spec
        if predict_state_spec is None:
            predict_state_spec = train_state_spec
        self._predict_state_spec = predict_state_spec
        self._optimizer = optimizer
        self._debug_summaries = debug_summaries

    @property
    def optimizer(self):
        """Return the optimizer for this algorithm."""
        return self._optimizer

    @property
    def predict_state_spec(self):
        """Returns the RNN state spec for predict()."""
        return self._predict_state_spec

    @property
    def train_state_spec(self):
        """Returns the RNN state spec for train_step()."""
        return self._train_state_spec

    #------------- User need to implement the following functions -------

    # Subclass may override predict() to allow more efficient implementation
    def predict(self, inputs, state=None):
        """Predict for one step of inputs.

        Args:
            inputs (nested Tensor): inputs for prediction
            state (nested Tensor): network state (for RNN)

        Returns:
            AlgorithmStep
                outputs (nested Tensor): prediction result
                state (nested Tensor): should match `predict_state_spec`
        """
        algorithm_step = self.train_step(inputs, state)
        return algorithm_step._replace(info=None)

    @abstractmethod
    def train_step(self, inputs, state=None):
        """Perform one step of predicting and training computation.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            inputs (nested Tensor): inputs for train
            state (nested Tensor): consistent with train_state_spec

        Returns:
            AlgorithmStep
                outputs (nested Tensor): predict outputs
                state (nested Tensor): should match `predict_state_spec`
                info (nested Tensor): information for training. If this is
                    LossInfo, calc_loss() in Algorithm can be used. Otherwise,
                    the user needs to override calc_loss() to calculate loss or
                    override train_complete() to do customized training.
        """
        pass

    # Subclass may override train_complete() to allow customized training
    def train_complete(self,
                       tape: tf.GradientTape,
                       training_info,
                       valid_masks=None):
        """Complte one iteration of training.

        `train_complete` should calcuate gradients and update parameters using
        those gradients.

        Args:
            tape (tf.GradientTape): the tape which are used for calculating
                gradient. All the previous `train_interval` `train_step()` for
                are called under the context of this tape.
            training_info (nested Tensor): information collected for training.
                It is batched from each `info` returned bt `train_step()`
            valid_masks (tf.Tensor): masks indicating which samples are valid.
                shape=(T, B), dtype=tf.float32
        Returns:
            loss_info (LossInfo): loss information
            grads_and_vars (list[tuple]): list of gradient and variable tuples
        """

        with tape:
            loss_info = self.calc_loss(training_info)
            if valid_masks is not None:
                loss_info = tf.nest.map_structure(
                    lambda l: tf.reduce_mean(l * valid_masks), loss_info)
            else:
                loss_info = tf.nest.map_structure(lambda l: tf.reduce_mean(l),
                                                  loss_info)

        vars = self.variables
        grads = tape.gradient(loss_info.loss, vars)
        grads_and_vars = tuple(zip(grads, vars))
        self._optimizer.apply_gradients(grads_and_vars)
        return loss_info, grads_and_vars

    # Subclass may override calc_loss() to allow more sophiscated loss
    def calc_loss(self, training_info):
        """Calculate the loss at each step for each sample.

        Args:
            training_info (nested Tensor): information collected for training.
                It is batched from each `info` returned bt `train_step()`
        Returns:
            loss_info (LossInfo): loss at each time step for each sample in the
                batch. The shapes of the tensors in loss_info should be (T, B)
        """
        assert isinstance(training_info, LossInfo), (
            "training_info returned by"
            " train_step() should be LossInfo. Otherwise you need override"
            " calc_loss() to generate LossInfo from training_info")
        return training_info
