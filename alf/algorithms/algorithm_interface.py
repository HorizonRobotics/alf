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

import torch.nn as nn

from alf.nest.utils import get_nested_field
from alf.data_structures import AlgStep, LossInfo
from alf.networks import Network

# Experience = namedtuple(
#     'Experience', ['root_inputs', 'rollout_state', 'rollout_info'])

# BatchInfo = namedtuple(
#     "BatchInfo", ["env_ids", "positions", "importance_weights", "replay_buffer"])


class AlgorithmInterface(nn.Module):
    @property
    def path(self):
        """Path from the root algorithm to this algorithm.

        The input state to rollout_step() can be retrieved by

        .. code-block:: python

            nest.get_field(experience.rollout_state, self.path)

        The info from  rollout_step() can be retrieved by:

        .. code-block:: python

            nest.get_field(experience.rollout_info, self.path)

        Returns:
            str: path from the root algorithm to this algorithm
        """

    def is_on_policy(self):
        """Whether is on-policy training.

        For on-policy training, ``train_step()`` will not be called. And ``info``
        passed to ``calc_loss()`` is info collected from ``rollout_step()``.

        For off-policy training, ``train_step()`` will be called with the experience
        from replay buffer. And ``info`` passed to ``calc_loss()`` is info collected
        from ``train_step``.

        Returns:
            bool | None: True if on-policy training, False if off-policy training,
                None if not set.
        """

    def predict_step(self, inputs, state):
        """Predict for one step of inputs.

        Args:
            inputs (nested Tensor): inputs for prediction.
            state (nested Tensor): network state (for RNN).
        Returns:
            AlgStep:
            - output (nested Tensor): prediction result.
            - state (nested Tensor): should match ``predict_state_spec``.
        """

    def rollout_step(self, inputs, state):
        """Rollout for one step of inputs.

        Args:
            inputs (nested Tensor): inputs for prediction.
            state (nested Tensor): network state (for RNN).
        Returns:
            AlgStep:
            - output (nested Tensor): prediction result.
            - state (nested Tensor): should match ``rollout_state_spec``.
            - info (nested Tensor): For on-policy training it will be temporally
              batched and passed as ``info`` for calc_loss(). For off-policy
              training, it will be retrieved from replay buffer and passed as
              ``rollout_info`` for train_step().
        """

    def train_step(self, inputs, state, rollout_info):
        """Perform one step of training computation.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            inputs (nested Tensor): inputs for train.
            state (nested Tensor): consistent with ``train_state_spec``.
            rollout_info (nested Tensor): info from rollout_step()
            batch_info (BatchInfo): information about this batch of data
        Returns:
            AlgStep:
            - output (nested Tensor): prediction result.
            - state (nested Tensor): should match ``train_state_spec``.
            - info (nested Tensor): information for training. It will temporally
              batched and passed as ``info`` for calc_loss(). If this is
              ``LossInfo``, ``calc_loss()`` in ``Algorithm`` can be used.
              Otherwise, the user needs to override ``calc_loss()`` to
              calculate loss or override ``update_with_gradient()`` to do
              customized training.
        """

    def calc_loss(self, info):
        """Calculate the loss at each step for each sample.

        Args:
            info (nest): information collected for training. It is batched
                from each ``AlgStep.info`` returned by ``rollout_step()``
                (on-policy training) or ``train_step()`` (off-policy training).
        Returns:
            LossInfo: loss at each time step for each sample in the
                batch. The shapes of the tensors in loss info should be
                :math:`(T, B)`.
        """

    def preprocess_experience(self, root_inputs, rollout_info, batch_info):
        """This function is called on the experiences obtained from a replay
        buffer. An example usage of this function is to calculate advantages and
        returns in ``PPOAlgorithm``.

        The shapes of tensors in experience are assumed to be :math:`(B, T, ...)`.

        Args:
            root_inputs (nest): input for rollout_step() of the root algorithm.
                This is from replay buffer. Note that in general, this is not
                same as the input of rollout_step() of this algorithm.
            rollout_info (nested Tensor): ``AlgStep.info`` from rollout_step()
                for this algorithm.
            batch_info (BatchInfo): information about this batch of data
        Returns:
            tuple:
            - processed root_inputs
            - processed rollout_info
        """
        return root_inputs, rollout_info

    def after_update(self, root_inputs, info):
        """Do things after completing one gradient update (i.e. ``update_with_gradient()``).
        This function can be used for post-processings following one minibatch
        update, such as copy a training model to a target model in SAC, DQN, etc.

        Args:
            experience (nest): experiences collected for the most recent
                ``update_with_gradient()``.
            info (nest): information collected for training.
                It is batched from each ``AlgStep.info`` returned by ``rollout_step()``
                or ``train_step()``.
        """

    def after_train_iter(self, root_inputs, rollout_info):
        """Do things after completing one training iteration (i.e. ``train_iter()``
        that consists of one or multiple gradient updates). This function can
        be used for training additional modules that have their own training logic
        (e.g., on/off-policy, replay buffers, etc). These modules should be added
        to ``_trainable_attributes_to_ignore`` in the parent algorithm.

        Other things might also be possible as long as they should be done once
        every training iteration.

        This function will serve the same purpose with ``after_update`` if there
        is always only one gradient update in each training iteration. Otherwise
        it's less frequently called than ``after_update``.

        Args:
            root_inputs (nest): inputs for the rollout_step() of the root algorithm
                collected during ``unroll()``.
            rollout_info (nest): information collected from rollout_step() for
                this algorithm during ``unroll()``.
        """

    def train_iter(self):
        """Perform one iteration of training.

        Users may choose to implement their own ``train_iter()``.

        Returns:
            int:
            - number of samples being trained on (including duplicates).
        """
