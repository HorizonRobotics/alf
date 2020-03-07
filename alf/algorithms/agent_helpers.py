# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
"""Some helper functions for constructing an Agent instance."""

import os

import torch

import alf
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import LossInfo
from alf.utils.math_ops import add_ignore_empty


class AgentStateSpecs(object):
    def __init__(self):
        self.train_state_spec = alf.algorithms.agent.AgentState()
        self.rollout_state_spec = alf.algorithms.agent.AgentState()
        self.predict_state_spec = alf.algorithms.agent.AgentState()

    def collect_algorithm_state_specs(self, alg, alg_field):
        """Collect state specs from algorithms. For code conciseness, we collect
        all three state specs even though some of them will not be used during
        `unroll` or `train`.

        Args:
            alg (Algorithm):
            alg_field (str): the corresponding algorithm field in an
                `AgentState`.
        """
        self.train_state_spec = self.train_state_spec._replace(
            **{alg_field: alg.train_state_spec})
        self.rollout_state_spec = self.rollout_state_spec._replace(
            **{alg_field: alg.rollout_state_spec})
        self.predict_state_spec = self.predict_state_spec._replace(
            **{alg_field: alg.predict_state_spec})


def accumulate_algortihm_rewards(rewards, weights, names, summary_prefix,
                                 summarize_fn):
    """Sum a list of rewards by their weights. Also summarize the rewards
    statistics given their names.

    Args:
        rewards (list[Tensor]): a list of rewards tensors
        weights (list[float]): a list of floating numbers
        names (list[str]): a list of reward names
        summary_prefix (str): a string prefix for summary
        summarize_fn (Callable): a summarize function that accepts a name and
            a reward.

    Returns:
        A single reward after accumulation.
    """
    assert len(rewards) > 0
    assert len(rewards) == len(weights)
    assert len(rewards) == len(names)

    reward = torch.zeros_like(rewards[0])
    for r, w, name in zip(rewards, weights, names):
        if r is not None:
            summarize_fn(os.path.join(summary_prefix, name), r)
            reward += w * r

    summarize_fn(os.path.join(summary_prefix, "overall"), reward)
    return reward


def accumulate_loss_info(training_info, algorithms, names):
    """Given an overall Agent training info that contains various training infos
    for different algorithms, compute the accumulated loss info for updating
    parameters.

    Args:
        training_info (nested Tensor): information collected for training
            algorithms. It is batched from each `info` returned by `train_step()`.
        algorithms (list[Algorithm]): the list of algorithms whose loss infos
            are to be accumulated. Note that we always assume the first algorithm
            is the main RL algorithm which might have an additional field
            `rollout_info` that needs special handling.
        names (list[str]): the algorithm names that should appear as fields in
            `training_info`.
    """

    def _make_rl_training_info(training_info, name):
        if training_info.rollout_info == ():
            rollout_info = ()
        else:
            rollout_info = getattr(training_info.rollout_info, name)
        info = getattr(training_info.info, name)
        return training_info._replace(info=info, rollout_info=rollout_info)

    def _update_loss(loss_info, training_info, name, algorithm):
        if algorithm is None:
            return loss_info
        new_loss_info = algorithm.calc_loss(getattr(training_info.info, name))
        loss_info.extra[name] = new_loss_info.extra
        return LossInfo(
            loss=add_ignore_empty(loss_info.loss, new_loss_info.loss),
            scalar_loss=add_ignore_empty(loss_info.scalar_loss,
                                         new_loss_info.scalar_loss),
            extra=loss_info.extra)

    assert len(algorithms) > 0
    assert len(algorithms) == len(names)
    assert isinstance(algorithms[0], RLAlgorithm)

    rl_loss_info = algorithms[0].calc_loss(
        _make_rl_training_info(training_info, names[0]))
    loss_info = rl_loss_info._replace(extra=dict(rl=rl_loss_info.extra))
    for alg, name in zip(algorithms[1:], names[1:]):
        loss_info = _update_loss(loss_info, training_info, name, alg)
    return loss_info
