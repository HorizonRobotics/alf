# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from functools import partial
import torch

import alf

from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep
from alf.utils.averager import EMAverager


@alf.configurable(blacklist=["reward_spec"])
class ProportionalRewardWeightAlgorithm(Algorithm):
    """Adjust reward weights according to untransformed rollout rewards.

    The adjustment is expected to be performed after every training iteration.
    The reward weight is calculated as "(ratios @ average_rward) / average_reward"

    Args:
        reward_spec (TensorSpec): the spec for the reward to be averaged
        ratios (List[List[float]]): a 2D list of ratios to be multiplied with the
            average reward. The shape of the list should be (reward_spec.numel,
            reward_spec.numel)
        reward_averager_ctor (Callable): a callable that constructs an averager
            for the reward. The averager should have an update method and a get
            method.
        epsilon (float): a small value to avoid division by zero
        max_weight (float): the maximum value for the reward weights
        debug_summaries (bool): True if debug summaries should be created
        name (str): the name of this algorithm
    """

    def __init__(self,
                 reward_spec,
                 ratios,
                 reward_averager_ctor=partial(EMAverager, update_rate=1e-4),
                 epsilon=1e-6,
                 max_weight=None,
                 debug_summaries=False,
                 name="ProportionalRewardWeightAlgorithm"):
        super().__init__(debug_summaries=debug_summaries, name=name)
        assert isinstance(reward_spec, alf.TensorSpec), (
            "reward_spec must be a TensorSpec! Got: %s" % reward_spec)
        assert reward_spec.numel > 1 and reward_spec.ndim == 1, (
            "Only multi-dim reward needs this algorithm!")

        self._reward_spec = reward_spec
        self._reward_averager = reward_averager_ctor(reward_spec)
        self._max_weight = max_weight
        self._ratios = torch.tensor(ratios, dtype=torch.float32)
        assert self._ratios.shape == (reward_spec.numel, reward_spec.numel)
        self._epsilon = epsilon

    @property
    def reward_weights(self):
        reward = self._reward_averager.get()
        reward = reward + reward.sign() * self._epsilon
        reward_weights = (self._ratios @ reward) / reward
        if self._max_weight is not None:
            reward_weights = torch.clamp(
                reward_weights, min=-self._max_weight, max=self._max_weight)
        return reward_weights

    def predict_step(self, inputs, state):
        return AlgStep()

    def predict_step(self, inputs, state):
        return AlgStep()

    def rollout_step(self, inputs, state):
        return AlgStep(info=inputs.untransformed.reward)

    def after_train_iter(self, root_inputs, info):
        reward = info
        self._reward_averager.update(reward)

        if self._debug_summaries:
            with alf.summary.scope(self._name):
                reward_weights = self.reward_weights
                for i in range(reward_weights.shape[0]):
                    alf.summary.scalar("reward_weight/%d" % i,
                                       reward_weights[i])
