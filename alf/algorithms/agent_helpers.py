# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.data_structures import LossInfo
from alf.utils.math_ops import add_ignore_empty


def _make_alg_experience(experience, name):
    """Given an experience, extracts the ``rollout_info`` field for an
    algorithm.
    """
    if experience.rollout_info == ():
        rollout_info = ()
    else:
        rollout_info = getattr(experience.rollout_info, name)
    return experience._replace(
        rollout_info=rollout_info,
        rollout_info_field=experience.rollout_info_field + '.' + name)


class AgentHelper(object):
    def __init__(self, state_ctor):
        """Create three state specs given the state creator."""
        self._train_state_spec = state_ctor()
        self._rollout_state_spec = state_ctor()
        self._predict_state_spec = state_ctor()
        self._alg_to_field_mapping = dict()

    def register_algorithm(self, alg, alg_field):
        """Collect state specs from algorithms. For code conciseness, we collect
        all three state specs even though some of them will not be used during
        ``unroll`` or ``train``.

        This function also registers ``alg`` with ``alg_field``.

        Args:
            alg (Algorithm): a child algorithm in the agent.
            alg_field (str): the corresponding algorithm field in an
                ``AgentState`` or ``AgentInfo``.
        """
        self._alg_to_field_mapping[alg] = alg_field
        if alg_field in self._train_state_spec._fields:
            self._train_state_spec = self._train_state_spec._replace(
                **{alg_field: alg.train_state_spec})
            self._rollout_state_spec = self._rollout_state_spec._replace(
                **{alg_field: alg.rollout_state_spec})
            self._predict_state_spec = self._predict_state_spec._replace(
                **{alg_field: alg.predict_state_spec})

    def _get_algorithm_field(self, alg):
        assert alg in self._alg_to_field_mapping, \
                "Should first register this algorithm %s!" % alg.name
        return self._alg_to_field_mapping[alg]

    def state_specs(self):
        """Return the state specs collected from child algorithms."""
        return dict(
            train_state_spec=self._train_state_spec,
            rollout_state_spec=self._rollout_state_spec,
            predict_state_spec=self._predict_state_spec)

    @staticmethod
    def accumulate_algorithm_rewards(rewards, weights, names, summary_prefix,
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
            Tensor: A single reward after accumulation.
        """
        assert len(rewards) > 0

        reward = torch.zeros_like(rewards[0])
        for r, w, name in zip(rewards, weights, names):
            summarize_fn(os.path.join(summary_prefix, name), r)
            reward += w * r

        if len(rewards) > 1:
            summarize_fn(os.path.join(summary_prefix, "overall"), reward)
        return reward

    def accumulate_loss_info(self,
                             algorithms,
                             train_info,
                             offline=False,
                             pre_train=False):
        """Given an overall Agent training info that contains various training infos
        for different algorithms, compute the accumulated loss info for updating
        parameters.

        Args:
            algorithms (list[Algorithm]): the list of algorithms whose loss infos
                are to be accumulated.
            experience (Experience): experience used for gradient update.
            train_info (nested Tensor): information collected for training
                algorithms. It is batched from each ``AlgStep.info`` returned by
                ``train_step()`` or ``rollout_step()``.
            offline (bool): whether the accumulation is done for offline RL part
                or the online RL part.
            pre_train (bool): whether in pre_training phase. This flag
                can be used for algorithms that need to implement different
                training procedures at different phases.
        Returns:
            LossInfo: the accumulated loss info.
        """

        def _update_loss(loss_info, algorithm, name):
            info = getattr(train_info, name)
            if not offline:
                new_loss_info = algorithm.calc_loss(info)
            else:
                new_loss_info = algorithm.calc_loss_offline(info, pre_train)
            if loss_info is None:
                return new_loss_info._replace(
                    extra={name: new_loss_info.extra})
            else:
                loss_info.extra[name] = new_loss_info.extra
                return LossInfo(
                    loss=add_ignore_empty(loss_info.loss, new_loss_info.loss),
                    scalar_loss=add_ignore_empty(loss_info.scalar_loss,
                                                 new_loss_info.scalar_loss),
                    extra=loss_info.extra,
                    priority=add_ignore_empty(loss_info.priority,
                                              new_loss_info.priority))

        loss_info = None
        for alg in algorithms:
            field = self._get_algorithm_field(alg)
            loss_info = _update_loss(loss_info, alg, field)
        assert loss_info is not None, "No loss info is calculated!"
        return loss_info

    def after_update(self, algorithms, root_inputs, train_info):
        """For each provided algorithm, call its ``after_update()`` to do things after
        the agent completes one gradient update (i.e. ``update_with_gradient()``).

        Args:
            algorithms (list[Algorithm]): the list of algorithms whose
                ``after_update`` is to be called.
            root_inputs (TimeStep): experience used for the gradient update.
            train_info (AgentInfo): information collected for training
                algorithms. It is batched from each ``AlgStep.info`` returned by
                ``train_step()`` or ``rollout_step()``.
        """
        for alg in algorithms:
            field = self._get_algorithm_field(alg)
            info = getattr(train_info, field)
            alg.after_update(root_inputs, info)

    def after_train_iter(self, algorithms, root_inputs, rollout_info=None):
        """For each provided algorithm, call its ``after_train_iter()`` to do
        things after the agent finishes one training iteration (i.e.,
        ``train_iter()``).

        Args:
            algorithms (list[Algorithm]): the list of algorithms whose
                ``after_train_iter`` is to be called.
            root_inputs (TimeStep): experience collected from ``rollout_step()``.
            rollout_info (AgentInfo): information collected for training
                algorithms. It is batched from each ``AlgStep.info`` returned by
                ``rollout_step()``.
        """
        for alg in algorithms:
            field = self._get_algorithm_field(alg)
            info = (None if rollout_info is None else getattr(
                rollout_info, field))
            alg.after_train_iter(root_inputs, info)

    def set_path(self, path):
        """Set the path for the sub-algorithms."""
        prefix = path
        if path:
            prefix = prefix + '.'
        for alg, name in self._alg_to_field_mapping.items():
            alg.set_path(path + name)
