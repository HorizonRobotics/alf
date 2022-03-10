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
"""An algorithm for adjusting entropy regularization strength."""
from absl import logging
import copy
import numpy as np
import torch
from typing import Callable, Union

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import namedtuple, AlgStep, LossInfo, StepType
from alf.summary import should_record_summaries
from alf.utils.averager import ScalarWindowAverager
from alf.utils.dist_utils import calc_default_target_entropy, entropy_with_fallback
from alf.utils.schedulers import ConstantScheduler

EntropyTargetLossInfo = namedtuple("EntropyTargetLossInfo", ["neg_entropy"])
EntropyTargetInfo = namedtuple("EntropyTargetInfo", ["step_type", "loss"])


@alf.configurable
class EntropyTargetAlgorithm(Algorithm):
    """Algorithm for adjusting entropy regularization.

    It tries to adjust the entropy regularization (i.e. alpha) so that the
    the entropy is not smaller than ``target_entropy``.

    The algorithm has three stages:

    0. init stage. This is an optional stage. If the initial entropy is already
       below ``max_entropy``, then this stage is skipped. Otherwise, the alpha will
       be slowly decreased so that the entropy will land at ``max_entropy`` to
       trigger the next ``free_stage``. Basically, this stage let the user to choose
       an arbitrary large init alpha without considering every specific case.
    1. free stage. During this stage, the alpha is not changed. It transitions
       to adjust_stage once entropy drops below ``target_entropy``.
    2. adjust stage. During this stage, ``log_alpha`` is adjusted using this formula:

       .. code-block:: python

            ((below + 0.5 * above) * decreasing - (above + 0.5 * below) * increasing) * update_rate

       Note that ``log_alpha`` will always be decreased if entropy is increasing
       even when the entropy is below the target entropy. This is to prevent
       overshooting ``log_alpha`` to a too big value. Same reason for always
       increasing ``log_alpha`` even when the entropy is above the target entropy.
       ``update_rate`` is initialized to ``fast_update_rate`` and is reduced by a
       factor of 0.9 whenever the entropy crosses ``target_entropy``. ``udpate_rate``
       is reset to ``fast_update_rate`` if entropy drops too much below
       ``target_entropy`` (i.e., ``fast_stage_thresh`` in the code, which is the half
       of ``target_entropy`` if it is positive, and twice of ``target_entropy`` if
       it is negative.

    ``EntropyTargetAlgorithm`` can be used to approximately reproduce the learning
    of temperature in `Soft Actor-Critic Algorithms and Applications <https://arxiv.org/abs/1812.05905>`_.
    To do so, you need to use the same ``target_entropy``, set ``skip_free_stage``
    to True, and  set ``slow_update_rate`` and ``fast_update_rate`` to the 4
    times of the learning rate for temperature.
    """

    def __init__(self,
                 action_spec,
                 initial_alpha=0.1,
                 skip_free_stage=False,
                 max_entropy=None,
                 target_entropy=None,
                 very_slow_update_rate=0.001,
                 slow_update_rate=0.01,
                 fast_update_rate=np.log(2),
                 min_alpha=1e-4,
                 average_window=2,
                 debug_summaries=False,
                 name="EntropyTargetAlgorithm"):
        """
        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            initial_alpha (float): initial value for alpha; make sure that it's
                large enough for initial meaningful exploration
            skip_free_stage (bool): If True, directly goes to the adjust stage.
            max_entropy (float|None): the upper bound of the total entropy. If it is None,
                ``min(initial_entropy * 0.8, initial_entropy / 0.8)`` is used.
                initial_entropy is estimated from the first ``average_window``
                steps. 0.8 is to ensure that we can get a policy a less random
                as the initial policy before starting the free stage.
            target_entropy (float|None): the lower bound of the total entropy.
                If it is None, a default value proportional to the action dimension
                is used. This value should be less or equal than ``max_entropy``.
            very_slow_update_rate (float): a tiny update rate for ``log_alpha``;
                used in stage 0.
            slow_update_rate (float): minimal update rate for ``log_alpha``; used
                in stage 2.
            fast_update_rate (float): maximum update rate for ``log_alpha``; used
                in state 2.
            min_alpha (float): the minimal value of alpha. If <=0, :math:`e^{-100}`
                is used.
            average_window (int): window size for averaging past entropies.
            debug_summaries (bool): True if debug summaries should be created.
        """
        super().__init__(debug_summaries=debug_summaries, name=name)

        self.register_buffer(
            '_log_alpha',
            torch.tensor(np.log(initial_alpha), dtype=torch.float32))
        self.register_buffer('_stage', torch.tensor(-2, dtype=torch.int32))
        self._avg_entropy = ScalarWindowAverager(average_window)
        self.register_buffer(
            "_update_rate", torch.tensor(
                fast_update_rate, dtype=torch.float32))
        self._action_spec = action_spec
        self._min_log_alpha = -100.
        if min_alpha >= 0.:
            self._min_log_alpha = np.log(min_alpha)
        self._min_log_alpha = torch.tensor(self._min_log_alpha)

        flat_action_spec = alf.nest.flatten(self._action_spec)
        if target_entropy is None:
            target_entropy = np.sum(
                list(map(calc_default_target_entropy, flat_action_spec)))
            logging.info("target_entropy=%s" % target_entropy)

        if not isinstance(target_entropy, Callable):
            target_entropy = ConstantScheduler(target_entropy)

        if max_entropy is None:
            # max_entropy will be estimated in the first `average_window` steps.
            max_entropy = 0.
            self._stage.fill_(-2 - average_window)
        else:
            assert target_entropy() <= max_entropy, (
                "Target entropy %s should be less or equal than max entropy %s!"
                % (target_entropy(), max_entropy))
        self.register_buffer("_max_entropy",
                             torch.tensor(max_entropy, dtype=torch.float32))

        if skip_free_stage:
            self._stage.fill_(1)

        self._target_entropy = target_entropy
        self._very_slow_update_rate = very_slow_update_rate

        # need to explicitly specify dtype to be the same as `self._update_rate`
        # as required by the `torch.where` function later. This was not needed
        # in lower version of pytorch (e.g. 1.4) as it will cast a np.float64
        # to torch.float32.
        self._slow_update_rate = torch.tensor(
            slow_update_rate, dtype=torch.float32)
        self._fast_update_rate = torch.tensor(
            fast_update_rate, dtype=torch.float32)

    def predict_step(self, distribution_and_step_type, state):
        return AlgStep()

    def rollout_step(self, distribution_and_step_type, state=None):
        """Rollout step.

        Args:
            distribution (nested Distribution): action distribution from the
                policy.
            step_type (StepType): the step type for the distributions.
            on_policy_training (bool): If False, this step does nothing.

        Returns:
            AlgStep: ``info`` field is ``LossInfo``, other fields are empty. All
            fields are empty If ``on_policy_training=False``.
        """
        if self.on_policy:
            return self.train_step(distribution_and_step_type)
        else:
            return AlgStep()

    def train_step(self,
                   distribution_and_step_type,
                   state=None,
                   rollout_info=None):
        """Train step.

        Args:
            distribution (nested Distribution): action distribution from the
                policy.
            step_type (StepType): the step type for the distributions.
        Returns:
            AlgStep: ``info`` field is ``LossInfo``, other fields are empty.
        """
        distribution, step_type = distribution_and_step_type
        entropy, entropy_for_gradient = entropy_with_fallback(distribution)
        return AlgStep(
            output=(),
            state=(),
            info=EntropyTargetInfo(
                step_type=step_type,
                loss=LossInfo(
                    loss=-entropy_for_gradient,
                    extra=EntropyTargetLossInfo(neg_entropy=-entropy))))

    def calc_loss(self, info: EntropyTargetInfo, valid_mask=None):
        """Calculate loss.

        Args:
            info (EntropyTargetInfo): for computing loss.
            valid_mask (tensor): valid mask to be applied on time steps.

        Returns:
            LossInfo:
        """
        loss_info = info.loss
        mask = (info.step_type != StepType.LAST).type(torch.float32)
        if valid_mask:
            mask = mask * (valid_mask).type(torch.float32)
        entropy = -loss_info.extra.neg_entropy * mask
        num = torch.sum(mask)
        not_empty = num > 0
        num = max(num, 1)
        entropy2 = torch.sum(entropy**2) / num
        entropy = torch.sum(entropy) / num
        entropy_std = torch.sqrt(
            torch.max(torch.tensor(0.0), entropy2 - entropy * entropy))

        if not_empty:
            self.adjust_alpha(entropy)
            if self._debug_summaries and should_record_summaries():
                with alf.summary.scope(self.name):
                    alf.summary.scalar("entropy_std", entropy_std)

        alpha = torch.exp(self._log_alpha)
        return loss_info._replace(loss=loss_info.loss * alpha)

    def adjust_alpha(self, entropy):
        """Adjust alpha according to the current entropy.

        Args:
            entropy (scalar Tensor): the current entropy.
        Returns:
            adjusted entropy regularization
        """
        prev_avg_entropy = self._avg_entropy.get()
        avg_entropy = self._avg_entropy.average(entropy)

        target_entropy = self._target_entropy()

        if target_entropy > 0:
            fast_stage_thresh = 0.5 * target_entropy
        else:
            fast_stage_thresh = 2.0 * target_entropy

        def _init_entropy():
            self._max_entropy.fill_(
                torch.min(0.8 * avg_entropy, avg_entropy / 0.8))
            self._stage.add_(1)

        def _init():
            below = avg_entropy < self._max_entropy
            decreasing = (avg_entropy < prev_avg_entropy).type(torch.float32)
            # -1 * (1 - decreasing) + 0.5 * decreasing
            update_rate = (-1 + 1.5 * decreasing) * self._very_slow_update_rate
            self._stage.add_(below.type(torch.int32))
            self._log_alpha.fill_(
                torch.max(self._log_alpha + update_rate, self._min_log_alpha))

        def _free():
            crossing = avg_entropy < target_entropy
            self._stage.add_(crossing.type(torch.int32))

        def _adjust():
            previous_above = self._stage.type(torch.bool)
            above = avg_entropy > target_entropy
            self._stage.fill_(above.type(torch.int32))
            crossing = above != previous_above
            update_rate = self._update_rate
            update_rate = torch.where(crossing, 0.9 * update_rate, update_rate)
            update_rate = torch.max(update_rate, self._slow_update_rate)
            update_rate = torch.where(entropy < fast_stage_thresh,
                                      self._fast_update_rate, update_rate)
            self._update_rate.fill_(update_rate)
            above = above.type(torch.float32)
            below = 1 - above
            decreasing = (avg_entropy < prev_avg_entropy).type(torch.float32)
            increasing = 1 - decreasing
            log_alpha = self._log_alpha + (
                (below + 0.5 * above) * decreasing -
                (above + 0.5 * below) * increasing) * update_rate
            log_alpha = torch.max(log_alpha, self._min_log_alpha)
            self._log_alpha.fill_(log_alpha)

        if self._stage < -2:
            _init_entropy()
        if self._stage == -2:
            _init()
        if self._stage == -1:
            _free()
        if self._stage >= 0:
            _adjust()
        alpha = torch.exp(self._log_alpha)

        if self._debug_summaries and should_record_summaries():
            with alf.summary.scope(self.name):
                alf.summary.scalar("alpha", alpha)
                alf.summary.scalar("avg_entropy", avg_entropy)
                alf.summary.scalar("stage", self._stage)
                alf.summary.scalar("update_rate", self._update_rate)
                if type(self._target_entropy) != ConstantScheduler:
                    alf.summary.scalar("target_entropy", target_entropy)

        return alpha


@alf.configurable
class NestedEntropyTargetAlgorithm(Algorithm):
    """Algorithm for adjusting entropy regularization.

    Similar to ``EntropyTargetAlgorithm``, ``NestedEntropyTargetAlgorithm``
    adjusts the entropy regularization for each action in a nested action so that
    the entropy for each action in the nest is not smaller than the corresponding
    ``target_entropy``. It uses ``EntropyTargetAlgorithm`` to do the actual work.
    See ``EntropyTargetAlgorithm`` for how it works.
    """

    def __init__(self,
                 action_spec,
                 initial_alpha=0.1,
                 skip_free_stage=False,
                 max_entropy=None,
                 target_entropy=None,
                 very_slow_update_rate=0.001,
                 slow_update_rate=0.01,
                 fast_update_rate=np.log(2),
                 min_alpha=1e-4,
                 average_window=2,
                 debug_summaries=False,
                 name="EntropyTargetAlgorithm"):
        """
        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            initial_alpha (float): initial value for alpha; make sure that it's
                large enough for initial meaningful exploration
            skip_free_stage (bool): If True, directly goes to the adjust stage.
            max_entropy (Nested[float|None]): the upper bound of the entropy for
                each corresponding action in ``action_spec``. If it is None,
                ``min(initial_entropy * 0.8, initial_entropy / 0.8)`` is used.
                initial_entropy is estimated from the first ``average_window``
                steps. 0.8 is to ensure that we can get a policy a less random
                as the initial policy before starting the free stage.
                If ``target_entropy`` is nested and:

                - If ``max_entropy`` is None: the max entropy of each of the distribution
                  in ``action_spec`` is calculated as using the estimated initial
                  entropy for that distribution.
                - If ``max_entropy`` is nested: it should have the same structure
                  as ``action_spec`` and each element indicates the max entropy
                  for the corresponding distribution in ``action_spec``.
                - If ``max_entropy`` is a float: it is the max entropy for each of
                  the distributions in ``action_spec``

            target_entropy (Nested[float|None]): the lower bound of the
                the entropy for each corresponding action in ``action_spec``.
                If it is None, a default value proportional to the action dimension
                is used. This value should be less or equal than ``max_entropy``.
                If ``action_spec`` is nested, ``target_entropy`` can also be a nest
                with the same structure and each element indicates the target entropy
                for the corresponding distribution in ``action_spec``.
            very_slow_update_rate (float): a tiny update rate for ``log_alpha``;
                used in stage 0.
            slow_update_rate (float): minimal update rate for ``log_alpha``; used
                in stage 2.
            fast_update_rate (float): maximum update rate for ``log_alpha``; used
                in state 2.
            min_alpha (float): the minimal value of alpha. If <=0, :math:`e^{-100}`
                is used.
            average_window (int): window size for averaging past entropies.
            debug_summaries (bool): True if debug summaries should be created.
        """

        kwargs = copy.copy(locals())
        del kwargs['self']
        del kwargs['__class__']
        super().__init__(debug_summaries=debug_summaries, name=name)

        def _create_et(path, action_spec, target_entropy, max_entropy):
            kwargs.update(
                action_spec=action_spec,
                target_entropy=target_entropy,
                max_entropy=max_entropy,
                name=name + "/" + path)
            return EntropyTargetAlgorithm(**kwargs)

        alf.nest.assert_same_structure(target_entropy, action_spec)
        if alf.nest.is_nested(max_entropy):
            alf.nest.assert_same_structure(max_entropy, action_spec)
        else:
            max_entropy = alf.nest.map_structure(lambda x: max_entropy,
                                                 action_spec)
        algs = alf.nest.py_map_structure_with_path(_create_et, action_spec,
                                                   target_entropy, max_entropy)
        self._algs = algs
        self._algs_flattened = alf.nest.flatten(algs)
        if alf.nest.is_nested(algs):
            self._nested_algs = alf.nest.utils.make_nested_module(algs)

    def predict_step(self, distribution_and_step_type, state=None):
        return AlgStep()

    def rollout_step(self, distribution_and_step_type, state=None):
        if self.on_policy:
            return self.train_step(distribution_and_step_type)
        else:
            return AlgStep()

    def train_step(self,
                   distribution_and_step_type,
                   state=None,
                   rollout_info=None):
        distribution, step_type = distribution_and_step_type
        infos = alf.nest.map_structure(
            lambda alg, dist: alg.train_step((dist, step_type)).info._replace(
                step_type=()), self._algs, distribution)
        return AlgStep(output=(), state=(), info=(step_type, infos))

    def calc_loss(self, info: EntropyTargetInfo, valid_mask=None):
        step_type, info = info
        info_flattened = alf.nest.flatten_up_to(self._algs, info)
        loss_infos = list(
            map(
                lambda alg, inf: alg.calc_loss(
                    inf._replace(step_type=step_type), valid_mask),
                self._algs_flattened, info_flattened))
        loss = sum(loss_info.loss for loss_info in loss_infos)
        extra = alf.nest.pack_sequence_as(
            self._algs, [loss_info.extra for loss_info in loss_infos])
        return LossInfo(loss=loss, extra=extra)


@alf.configurable
class SGDEntropyTargetAlgorithm(Algorithm):
    """Adjusting the entropy weight using SGD according to a target, similar to
    the way of SAC.
    """

    def __init__(self,
                 action_spec: alf.tensor_specs.TensorSpec,
                 initial_alpha: float = 0.1,
                 target_entropy: Union[Callable[[], float], float] = None,
                 window_size: int = 1,
                 optimizer: torch.optim.Optimizer = None,
                 debug_summaries: bool = False,
                 name: str = "SGDEntropyTargetAlgorithm"):
        """
        Args:
            action_spec: nested tensor spec for the action
            initial_alpha: initial value for alpha; make sure that it's
                large enough for initial meaningful exploration
            target_entropy: the target of the total entropy. If it is None,
                a default value proportional to the action dimension is used.
            window_size: window size for averaging past entropies.
            optimizer: the optimizer for adjusting the weight
            debug_summaries: whether to turn on debugging info
            name: name of the class
        """

        super().__init__(
            optimizer=optimizer, debug_summaries=debug_summaries, name=name)

        self._log_alpha = torch.nn.Parameter(
            torch.tensor(np.log(initial_alpha), dtype=torch.float32))

        self._action_spec = action_spec
        flat_action_spec = alf.nest.flatten(self._action_spec)
        if target_entropy is None:
            target_entropy = np.sum(
                list(map(calc_default_target_entropy, flat_action_spec)))
            logging.info("target_entropy=%s" % target_entropy)

        if not isinstance(target_entropy, Callable):
            target_entropy = ConstantScheduler(target_entropy)

        self._target_entropy = target_entropy
        self._entropy_averager = ScalarWindowAverager(window_size)

    def predict_step(self, distribution_and_step_type):
        return AlgStep()

    def rollout_step(self, distribution_and_step_type):
        """
        Args:
            distribution_and_step_type (nested Distribution): action distribution
                from the policy, and the step type for the distributions.

        Returns:
            AlgStep: ``info`` is ``EntropyTargetInfo`` and ``info.loss`` is
                ``LossInfo``, other fields are empty. All fields are empty for
                off-policy training.
        """
        if self.on_policy:
            return self.train_step(distribution_and_step_type)
        else:
            return AlgStep()

    def train_step(self, distribution_and_step_type):
        """
        Args:
            distribution_and_step_type (nested Distribution): action distribution
                from the policy, and the step type for the distributions.

        Returns:
            AlgStep: ``info`` is ``EntropyTargetInfo`` and ``info.loss`` is
                ``LossInfo``, other fields are empty.
        """
        distribution, _ = distribution_and_step_type
        entropy, entropy_for_gradient = entropy_with_fallback(distribution)
        return AlgStep(
            output=(),
            state=(),
            info=EntropyTargetInfo(
                step_type=(),
                loss=LossInfo(
                    loss=-entropy_for_gradient,
                    extra=EntropyTargetLossInfo(neg_entropy=-entropy))))

    def calc_loss(self, info: EntropyTargetInfo):
        """Calculate the losses for training. It will compute two losses, one for
        training the entropy weight, and the other for maximizing the entropy of
        the action distribution.
        """
        loss_info = info.loss
        avg_entropy = self._entropy_averager.average(
            -loss_info.extra.neg_entropy)
        alpha_loss = (
            (avg_entropy - self._target_entropy()).detach() * self._log_alpha)
        alpha = torch.exp(self._log_alpha).detach()
        entropy_loss = loss_info.loss * alpha

        if self._debug_summaries:
            with alf.summary.scope(self.name):
                alf.summary.scalar("alpha", alpha)
                alf.summary.scalar("target_entropy", self._target_entropy())

        return LossInfo(
            loss=alpha_loss + entropy_loss,
            extra=dict(
                neg_entropy=loss_info.extra.neg_entropy,
                alpha_loss=alpha_loss,
                entropy_loss=entropy_loss))
