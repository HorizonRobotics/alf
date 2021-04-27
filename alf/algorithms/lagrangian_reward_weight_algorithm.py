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

import numpy as np
import gin

import torch
import torch.nn as nn
from torch.nn import functional as F

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import namedtuple, AlgStep
from alf.nest import nest
from alf.tensor_specs import TensorSpec
from alf.utils import tensor_utils

LagInfo = namedtuple("LagInfo", ["rollout_reward"], default_value=())


@alf.configurable(blacklist=["reward_spec"])
class LagrangianRewardWeightAlgorithm(Algorithm):
    """An algorithm that adjusts reward weights according to untransformed
    rollout rewards. The adjustment is expected to be performed after every
    training iteration.

    Generally speaking, for each reward dimension, the algorithm compares an
    individual reward per step to an average expected threshold, and if the
    reward is greater than the threshold (requirement satisfied) then it decreases
    the reward weight; otherwise it increases the weight.

    Note that this algorithm doesn't put a constraint on per-step basis since it
    only learns a single weight for each reward dim. A reward is always the higher
    the better.
    """

    def __init__(self,
                 reward_spec,
                 reward_thresholds,
                 optimizer,
                 init_weights=1.,
                 debug_summaries=False,
                 name="LagrangianRewardWeightAlgorithm"):
        """
        Args:
            reward_spec (TensorSpec): a rank-1 tensor spec representing multi-dim
                rewards.
            reward_thresholds (list[float|None]): a list of floating numbers,
                each representing a desired minimum reward threshold in expectation.
                If any entry is None, then that reward weight won't be tuned and
                its init value is always used.
            optimizer (optimizer): optimizer for learning the reward weights.
            init_weights (float|list[float]): the initial reward weights.
            debug_summaries (bool):
            name (str):
        """
        super(LagrangianRewardWeightAlgorithm, self).__init__(
            debug_summaries=debug_summaries, name=name)

        self._reward_spec = reward_spec
        self._reward_thresholds = reward_thresholds

        assert reward_spec.numel > 1, (
            "Only multi-dim reward needs this algorithm!")
        assert len(reward_thresholds) == reward_spec.numel, (
            "Mismatch between len(reward_weights)=%s and reward_dim=%s" %
            (len(reward_thresholds), reward_spec.numel))

        self._reward_training_mask = torch.tensor(
            [t is not None for t in reward_thresholds], dtype=torch.float32)
        self._reward_thresholds = torch.tensor(
            [0. if t is None else t for t in reward_thresholds])

        lambda_init = torch.tensor(init_weights)
        if lambda_init.ndim == 0:
            lambda_init = tensor_utils.tensor_extend_new_dim(
                lambda_init, 0, reward_spec.numel)
        assert torch.all(
            lambda_init >= 0.), "Initial weights must be non-negative!"

        # convert to softplus space
        self._lambdas = nn.Parameter(self._inv_softplus(lambda_init))
        self._optimizer = optimizer
        self._optimizer.add_param_group({'params': self._lambdas})

    def _inv_softplus(self, tensor):
        return torch.where(tensor > 20., tensor, tensor.expm1().log())

    @property
    def reward_weights(self):
        """Return the detached reward weights. These weights are expected not to
        be changed by external code."""
        return F.softplus(self._lambdas).detach().clone()

    def _trainable_attributes_to_ignore(self):
        return ["_lambdas"]

    def rollout_step(self, time_step, state):
        return AlgStep(
            info=LagInfo(rollout_reward=time_step.untransformed.reward))

    def after_train_iter(self, experience, train_info: LagInfo):
        """Retrieve *untransformed* rollout rewards from ``train_info``
        and train lambdas.
        """
        # [T, B, reward_dim]
        reward_weights = F.softplus(self._lambdas)
        loss = ((train_info.rollout_reward - self._reward_thresholds).detach()
                * (reward_weights * self._reward_training_mask))
        loss = loss.sum(dim=-1).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("cost", loss)
                for i in range(len(self._reward_thresholds)):
                    alf.summary.scalar("reward_threshold/%d" % i,
                                       self._reward_thresholds[i])
                    alf.summary.scalar("lambda/%d" % i, reward_weights[i])
