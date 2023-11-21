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
"""LagrangianRewardWeightAlgorithm."""

from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import namedtuple, AlgStep, LossInfo
from alf.tensor_specs import TensorSpec
from alf.utils import tensor_utils
from alf.utils.averager import EMAverager

LagInfo = namedtuple("LagInfo", ["rollout_reward"], default_value=())


def _inv_softplus(tensor):
    return torch.where(tensor > 20., tensor, tensor.expm1().log())


@alf.configurable(blacklist=["reward_spec"])
class LagrangianRewardWeightAlgorithm(Algorithm):
    """An algorithm that adjusts reward weights according to untransformed
    rollout rewards. The adjustment is expected to be performed after every
    training iteration.

    Generally speaking, for each reward dimension, the algorithm compares an
    individual reward per step to an average expected threshold, and if the
    reward is greater than the threshold (requirement satisfied) then it decreases
    the reward weight; otherwise it increases the weight.

    Note: The parent algorithm should not call ``train_step()`` and ``calc_loss()``
    of this algorithm. Instead, it should call ``after_train_iter()`` to perform
    one gradient step of updating the reward weights.

    .. note::

        This algorithm doesn't put a constraint on per-step basis since it only
        learns a single, state-independent weight for each reward dim. Also, a
        reward is always assumed to be the higher the better.
    """

    def __init__(self,
                 reward_spec,
                 reward_thresholds,
                 optimizer,
                 init_weights=1.,
                 max_weight=None,
                 reward_weight_normalization=True,
                 lambda_transform=F.softplus,
                 debug_summaries=False,
                 name="LagrangianRewardWeightAlgorithm"):
        """
        Args:
            reward_spec (TensorSpec): a rank-1 tensor spec representing multi-dim
                rewards.
            reward_thresholds (list[float]|None]): a list of floating numbers,
                each representing a desired minimum reward threshold in expectation.
                If any entry is None, then the corresponding reward weight won't be
                tuned; either its init value or its normalized init value
                (if ``reward_weight_normalization=True``) will be used.
            optimizer (optimizer): optimizer for learning the reward weights.
            init_weights (float|list[float]): the initial reward weights.
            max_weight (float): the reward weights will be clipped up to this value
            reward_weight_normalization (bool): whether project the weights to
                a simplex (sum-to-one normalization)
            lambda_transform (Callable): the transform function to make sure all
                lambdas (reward weights) are positive. Currently only support
                ``F.softplus`` and ``torch.exp``.
            debug_summaries (bool):
            name (str):
        """
        super(LagrangianRewardWeightAlgorithm, self).__init__(
            debug_summaries=debug_summaries, name=name)

        self._reward_spec = reward_spec

        assert reward_spec.numel > 1, (
            "Only multi-dim reward needs this algorithm!")
        assert (isinstance(reward_thresholds, (list, tuple))
                and len(reward_thresholds) == reward_spec.numel), (
                    "Mismatch between len(reward_weights)=%s and reward_dim=%s"
                    % (len(reward_thresholds), reward_spec.numel))

        self._reward_training_mask = torch.tensor(
            [t is not None for t in reward_thresholds], dtype=torch.float32)
        self._reward_thresholds = torch.tensor(
            [0. if t is None else t for t in reward_thresholds])

        self._reward_weight_normalization = reward_weight_normalization

        lambda_init = torch.tensor(init_weights)
        if lambda_init.ndim == 0:
            lambda_init = tensor_utils.tensor_extend_new_dim(
                lambda_init, 0, reward_spec.numel)
        assert torch.all(
            lambda_init >= 0.), "Initial weights must be non-negative!"

        inv_mapping = dict()
        inv_mapping[F.softplus] = _inv_softplus
        inv_mapping[torch.exp] = torch.log

        # convert to softplus space
        self._lambda_transform = lambda_transform
        self._inv_lambda_transform = inv_mapping[lambda_transform]

        self._lambdas = nn.Parameter(self._inv_lambda_transform(lambda_init))
        if max_weight is not None:
            self._max_lambda = self._inv_lambda_transform(
                torch.tensor(max_weight))
        else:
            self._max_lambda = None
        self._optimizer = optimizer
        self._optimizer.add_param_group({'params': self._lambdas})

    @property
    def reward_weights(self):
        """Return the detached reward weights. These weights are expected not to
        be changed by external code."""
        weights = self._lambda_transform(self._lambdas).detach().clone()
        if self._reward_weight_normalization:
            weights = weights / weights.sum()
        return weights

    def _trainable_attributes_to_ignore(self):
        return ["_lambdas"]

    def predict_step(self, inputs, state=None):
        return AlgStep()

    def rollout_step(self, inputs, state=None):
        return AlgStep(
            info=LagInfo(rollout_reward=inputs.untransformed.reward))

    def _calc_loss(self, train_info: LagInfo):
        """Retrieve *untransformed* rollout rewards from ``train_info``
        and compute the loss for training lambdas.
        """
        # [T, B, reward_dim]
        reward_weights = self._lambda_transform(self._lambdas)
        loss = ((train_info.rollout_reward - self._reward_thresholds).detach()
                * (reward_weights * self._reward_training_mask))
        loss = loss.sum(dim=-1).mean()
        return LossInfo(scalar_loss=loss, extra=reward_weights)

    def after_train_iter(self, root_inputs, train_info: LagInfo):
        """Perform one gradient step of updating lambdas."""
        loss = self._calc_loss(train_info)
        loss, reward_weights = loss.scalar_loss, loss.extra

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # capped at the upper limit
        if self._max_lambda is not None:
            self._lambdas.data.copy_(
                torch.minimum(self._lambdas, self._max_lambda))

        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("cost", loss)
                for i in range(len(self._reward_thresholds)):
                    alf.summary.scalar("reward_threshold/%d" % i,
                                       self._reward_thresholds[i])
                    alf.summary.scalar("lambda/%d" % i, reward_weights[i])


@alf.configurable(blacklist=["reward_spec"])
class LagrangianPredRewardWeightAlgorithm(LagrangianRewardWeightAlgorithm):
    """Similar to ``LagrangianRewardWeightAlgorithm``, except that the rewards
    used to compare with the thresholds are collected by prediction steps instead
    of by rollout steps. For harsh target constraints, it is important to remove
    the rollout stochasticity otherwise the agent's constraint satisfaction ability
    will usually be under-estimated.

    Because prediction output is not directly passed to training, in order to use the
    rewards from prediction to train the weights, here we use an ``Averager`` to
    maintain the reward statistics. Inside every ``after_train_iter`` we perform
    a gradient step by querying the current averager value.

    .. note::

        This algorithm asserts ``TrainerConfig.evaluate=True``.
    """

    def __init__(self,
                 reward_spec,
                 reward_thresholds,
                 optimizer,
                 init_weights=1.,
                 max_weight=None,
                 reward_weight_normalization=True,
                 pred_rewards_averager_ctor=partial(
                     EMAverager, update_rate=1e-4),
                 debug_summaries=False,
                 name="LagrangianPredRewardWeightAlgorithm"):
        """
        Args:
            reward_spec (TensorSpec): a rank-1 tensor spec representing multi-dim
                rewards.
            reward_thresholds (list[float]|None]): a list of floating numbers,
                each representing a desired minimum reward threshold in expectation.
                If any entry is None, then the corresponding reward weight won't be
                tuned; either its init value or its normalized init value
                (if ``reward_weight_normalization=True``) will be used.
            optimizer (optimizer): optimizer for learning the reward weights.
            init_weights (float|list[float]): the initial reward weights.
            max_weight (float): the reward weights will be clipped up to this value
            reward_weight_normalization (bool): whether project the weights to
                a simplex (sum-to-one normalization)
            pred_rewards_averager_ctor (Callable): callable for creating an
                averager to maintain a moving average of prediction rewards.
                If None, ``EMAverager`` with an update rate of ``1e-4`` will be
                used.
            debug_summaries (bool):
            name (str):
        """
        assert alf.get_config_value('TrainerConfig.evaluate'), (
            "This algorithm must have the evaluation mode turned on!")

        super(LagrangianPredRewardWeightAlgorithm, self).__init__(
            reward_spec=reward_spec,
            reward_thresholds=reward_thresholds,
            optimizer=optimizer,
            init_weights=init_weights,
            max_weight=max_weight,
            reward_weight_normalization=reward_weight_normalization,
            debug_summaries=debug_summaries,
            name=name)

        assert not alf.get_config_value("TrainerConfig.async_eval"), (
            "This algorithm doesn't support async evaluation!")
        self._pred_rewards_averager = pred_rewards_averager_ctor(reward_spec)

    def predict_step(self, inputs, state=None):
        self._pred_rewards_averager.update(inputs.untransformed.reward)
        return AlgStep()

    def _calc_loss(self, train_info: LagInfo):
        """Retrieve *untransformed* prediction rewards from the averager
        and train lambdas.
        """
        # [T, B, reward_dim]
        reward_weights = self._lambda_transform(self._lambdas)
        pred_rewards = self._pred_rewards_averager.get()
        loss = ((pred_rewards - self._reward_thresholds).detach() *
                (reward_weights * self._reward_training_mask))
        loss = loss.sum()

        if self._debug_summaries:
            with alf.summary.scope(self._name):
                for i in range(len(self._reward_thresholds)):
                    alf.summary.scalar("average_pred_reward/%d" % i,
                                       pred_rewards[i])
        return LossInfo(scalar_loss=loss, extra=reward_weights)
