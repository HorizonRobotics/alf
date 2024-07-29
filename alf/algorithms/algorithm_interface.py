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

from alf.data_structures import AlgStep, LossInfo


class AlgorithmInterface(nn.Module):
    """The interface for algorithm.

    It is a generic interface for reinforcement learning (RL) and non-RL
    algorithms. The key interface functions are:

    1. ``predict_step()``: one step of computation of action for evaluation.
    2. ``rollout_step()``: one step of computation for rollout. It is used for
       collecting experiences during training. Different from ``predict_step``,
       ``rollout_step`` may include additional computations for training. An
       algorithm could immediately use the collected experiences to update
       parameters after one rollout (multiple rollout steps) is performed; or it
       can put these collected experiences into a replay buffer.
    3. ``train_step()``: only used by algorithms that put experiences into
       replay buffers. The training data are sampled from the replay buffer
       filled by ``rollout_step()``.
    4. ``train_from_unroll()``: perform a training iteration from the unrolled
       result.
    5. ``train_from_replay_buffer()``: perform a training iteration from a
       replay buffer.
    6. ``update_with_gradient()``: do one gradient update based on the loss. It
       is used by the default ``train_from_unroll()`` and
       ``train_from_replay_buffer()`` implementations. You can override to
       implement your own ``update_with_gradient()``.
    7. ``calc_loss()``: calculate loss based on the ``info`` collected from
       ``rollout_step()`` or ``train_step()``. It
       is used by the default implementations of ``train_from_unroll()`` and
       ``train_from_replay_buffer()``. If you want to use these two functions,
       you need to implement ``calc_loss()``.
    8. ``after_update()``: called by ``train_iter()`` after every call to
       ``update_with_gradient()``, mainly for some postprocessing steps such as
       copying a training model to a target model in SAC or DQN.
    9. ``after_train_iter()``: called by ``train_iter()`` after every call to
       ``train_from_unroll()`` (on-policy training iter) or
       ``train_from_replay_buffer`` (off-policy training iter). It's mainly for
       training additional modules that have their own training logic (e.g.,
       on/off-policy, replay buffers, etc). Other things might also be possible
       as long as they should be done once every training iteration.

    For algorithms that have additional offline training flows, they can be
    implemented by using the following additional interface functions:
    10. ``train_step_offline()``: only used by algorithms that has offline
        training flows. The training data are sampled from a replay buffer
        that is loaded from an offline replay buffer checkpoint.
    11. ``calc_loss_offline()``: It calculates the loss based on the ``info``
        collected from ``train_step_offline()``.

    The offline training flows can be invoked by specifying a valid path to a
    replay buffer for ``TrainerConfig.offline_buffer_dir``.

    .. note::
        A non-RL algorithm will not directly interact with an
        environment. The iteration loop will always be driven by an
        ``RLAlgorithm`` that outputs actions and gets rewards. So a
        non-RL algorithm is always attached to an ``RLAlgorithm`` and cannot
        change the timing of (when to launch) a training iteration. However, it
        can have its own logic of a training iteration (e.g.,
        ``train_from_unroll()`` and ``train_from_replay_buffer()``) which can be
        triggered by a parent ``RLAlgorithm`` inside its ``after_train_iter()``.
    """

    @property
    def path(self):
        """Path from the root algorithm to this algorithm.

        Currently, path is useful when an algorithm needs to directly access
        the data about itself in replay buffer. There are two types of
        data about an algorithm are stored in replay buffer: one is ``rollout_info``,
        which is ``AlgStep.info`` returned by rollout_step(), the other is ``state``,
        which is the ``state`` argument used to call ``rollout_step()``. The data
        in replay buffer is organized as ``Experience`` which includes ``rollout_info``
        and ``state``.

        Given an experience structure, the input state to ``rollout_step()`` can be
        retrieved by:

        .. code-block:: python

            nest.get_field(experience.state, self.path)

        The info from ``rollout_step()`` can be retrieved by:

        .. code-block:: python

            nest.get_field(experience.rollout_info, self.path)

        Returns:
            str: path from the root algorithm to this algorithm
        """
        raise NotImplementedError()

    def set_path(self, path):
        """Set the path from the root algorithm to this algorithm.

        See ``AlgorithmInterface.path`` for description about path.
        This function is called by the trainer before training starts.
        It needs to be implemented if the algorithm contains some other
        sub-algorithms.

        If an algorithm does not have any sub-algorithm or its sub-algorithm does
        not need to access the root replay buffer directly, it does not implement
        this function.
        """
        raise NotImplementedError()

    @property
    def on_policy(self):
        """Whether is on-policy training.

        For on-policy training, ``train_step()`` will not be called. And ``info``
        passed to ``calc_loss()`` is info collected from ``rollout_step()``.

        For off-policy training, ``train_step()`` will be called with the experience
        from replay buffer. And ``info`` passed to ``calc_loss()`` is info collected
        from ``train_step``.

        An algorithm can override this to indicate whether it is an on-policy or
        off-policy algorithm. If an algorithm does not override this, it needs to
        support both on-policy and off-policy training, which means that ``rollout_step()``
        and ``train_step()`` need to have the correct behavior for on-policy and
        off-policy training. It can check whether it is on-policy training by
        calling this function.

        Returns:
            bool | None: True if on-policy training, False if off-policy training,
                None if not set.
        """
        raise NotImplementedError()

    def set_on_policy(self, is_on_policy):
        """Set whether this algorithm is on-policy or not.

        Args:
            is_on_policy (bool):
        """
        raise NotImplementedError()

    def predict_step(self, inputs, state):
        """Predict for one step of inputs.

        Args:
            inputs (nested Tensor): inputs for prediction.
            state (nested Tensor): network state (for RNN).
        Returns:
            AlgStep:
            - output (nested Tensor): prediction result.
            - state (nested Tensor): should match ``predict_state_spec``.
            - info (nest): information for analyzing the agent. In particular,
                if an element of the info is ``alf.summary.render.Image``, it
                will be rendered during play. See alf/summary/render.py for
                detail.
        """
        raise NotImplementedError()

    def rollout_step(self, inputs, state):
        """Rollout for one step of inputs.

        It is called to calculate output for every environment step. For on-policy
        training, it also needs to generate necessary information for ``calc_loss()``.
        For off-policy training, it needs to generate necessary information for
        ``train_step()``.

        Args:
            inputs (nested Tensor): inputs for prediction.
            state (nested Tensor): network state (for RNN).
        Returns:
            AlgStep:
            - output (nested Tensor): prediction result.
            - state (nested Tensor): should match ``rollout_state_spec``.
            - info (nested Tensor): For on-policy training it will be temporally
              batched and passed as ``info`` for calc_loss(). For off-policy
              training, it will be stored into retrieved from replay buffer and
              and retrieved for ``train_step()`` as ``rollout_info``.
        """
        raise NotImplementedError()

    def train_step(self, inputs, state, rollout_info):
        """Perform one step of training computation.

        It is called to calculate output for every time step for a batch of
        experience from replay buffer. It also needs to generate necessary
        information for ``calc_loss()``.

        Args:
            inputs (nested Tensor): inputs for train.
            state (nested Tensor): consistent with ``train_state_spec``.
            rollout_info (nested Tensor): info from ``rollout_step()``. It is
                retrieved from replay buffer.
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
        raise NotImplementedError()

    def calc_loss(self, info):
        """Calculate the loss for one mini-batch.

        Args:
            info (nest): information collected for training. It is batched
                from each ``AlgStep.info`` returned by ``rollout_step()``
                (on-policy training) or ``train_step()`` (off-policy training).
                The shape of the tensors in info is ``(T, B, ...)``, where T
                is the mini-batch length and B is the mini-batch size.
        Returns:
            LossInfo: loss at each time step for each sample in the
                batch. The shapes of the tensors in loss info should be
                :math:`(T, B)`.
        """
        raise NotImplementedError()

    def preprocess_experience(self, root_inputs, rollout_info, batch_info):
        """This function is called on the experiences obtained from a replay
        buffer. An example usage of this function is to calculate advantages and
        returns in ``PPOAlgorithm``.

        The shapes of tensors in experience are assumed to be :math:`(B, T, ...)`.

        Args:
            root_inputs (nest): input for rollout_step() of the root algorithm.
                This is from replay buffer. Note this is not same as the input
                of rollout_step() of self unless self is the root algorithm.
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
            root_inputs (nest): temporally batched inputs for the ``rollout_step()``
                of the root algorithm collected during ``unroll()``.
            info (nest): information collected for training.
                It is batched from each ``AlgStep.info`` returned by ``rollout_step()``
                for on-policy training or ``train_step()`` for off-policy training.
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
            root_inputs (nest|None): temporally batched inputs for the
                ``rollout_step()`` of the root algorithm collected during
                ``unroll()``. In the case where no data is available from
                the ``rollout_step()`` (e.g. in a offline pre-training phase
                where the online interaction is not started yet) ``root_inputs``
                will be None.
            rollout_info (nest|None): information collected from
                ``rollout_step()`` for this algorithm during ``unroll()``.
                In the case where no data is available from the
                ``rollout_step()`` (e.g. in a offline pre-training phase
                where the online interaction is not started yet) ``rollout_info``
                will be None.
        """

    def train_iter(self):
        """Perform one iteration of training.

        Users may choose to implement their own ``train_iter()``.

        Returns:
            int:
            - number of samples being trained on (including duplicates).
        """

    def train_from_unroll(self, experience, train_info):
        """Train given the info collected from ``unroll()``. This function can
        be called by any child algorithm that doesn't have the unroll logic but
        has a different training logic with its parent.

        Args:
            experience (Experience): collected during ``unroll()``.
            train_info (nest): ``AlgStep.info`` returned by ``rollout_step()``.

        Returns:
            int: number of steps that have been trained
        """
        raise NotImplementedError()

    def train_from_replay_buffer(self, update_global_counter=False):
        """This function can be called by any algorithm that has its own
        replay buffer configured.

        Args:
            update_global_counter (bool): controls whether this function changes
                the global counter for summary. If there are multiple
                algorithms, then only the parent algorithm should change this
                quantity and child algorithms should disable the flag. When it's
                ``True``, it will affect the counter only if
                ``config.update_counter_every_mini_batch=True``.
        """
        raise NotImplementedError()

    def train_step_offline(self, inputs, state, rollout_info, pre_train=False):
        """Perform one step of offline training computation.

        It is called to calculate output for every time step for a batch of
        experience from offline replay buffer. It also needs to generate
        necessary information for ``calc_loss_offline()``.

        Args:
            inputs (nested Tensor): inputs for train.
            state (nested Tensor): consistent with ``train_state_spec``.
            rollout_info (nested Tensor): info from ``rollout_step()``. It is
                retrieved from replay buffer.
            pre_train (bool): whether in pre_training phase. This flag
                can be used for algorithms that need to implement different
                training procedures at different phases.
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
        raise NotImplementedError()

    def calc_loss_offline(self, info, pre_train=False):
        """Calculate the loss for one mini-batch.

        Args:
            info (nest): information collected for training. It is batched
                from each ``AlgStep.info`` returned by ``rollout_step()``
                (on-policy training) or ``train_step()`` (off-policy training).
                The shape of the tensors in info is ``(T, B, ...)``, where T
                is the mini-batch length and B is the mini-batch size.
            pre_train (bool): whether in pre_training phase. This flag
                can be used for algorithms that need to implement different
                training procedures at different phases.
        Returns:
            LossInfo: loss at each time step for each sample in the
                batch. The shapes of the tensors in loss info should be
                :math:`(T, B)`.
        """
        raise NotImplementedError()
