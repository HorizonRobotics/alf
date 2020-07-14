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
"""Prior action policies for KL regularized RL."""

import gin
import numpy as np
import torch
import torch.distributions as td
from torch.distributions import Categorical, Independent, Uniform

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, Experience, TimeStep, StepType
from alf.networks import Network
from alf.tensor_specs import TensorSpec, BoundedTensorSpec


def normcdf(a, b):
    c = 0.7071067811865476  # = math.sqrt(0.5)
    return 0.5 * (torch.erf(c * b) - torch.erf(c * a))


class TruncatedNormal(td.Distribution):
    def __init__(self, loc, scale, low, high, validate_args=None):
        """Normal distribution truncated to the range between ``low`` and ``high``.

        Currently, only ``log_prob()`` is implemented.

        Args:
            loc (Tensor): mean of the untruncated Normal
            scale (Tensor): standard deviation of the untruncated Normal
            low (Tensor): lower range of the truncation range
            high (Tensor): upper range of the truncation range
        """
        self._loc = loc
        self._scale = scale
        self._low = low
        self._high = high
        super().__init__(batch_shape=loc.shape, validate_args=validate_args)

    def log_prob(self, value):
        """Log-probability of ``value``.

        Args:
            value (Tensor): the samples whose log_prob is to calculated
        Returns:
            log probability of ``value``
        """
        scale = self._scale
        loc = self._loc
        var = scale**2
        log_scale = scale.log()
        low = (self._low - loc) / scale
        high = (self._high - loc) / scale
        # 0.9189385332046727 = math.log(math.sqrt(2 * math.pi))
        log_prob = -(
            (value - loc)**2) / (2 * var) - log_scale - 0.9189385332046727
        return log_prob - normcdf(low, high).log()

    def sample(self):
        raise NotImplementedError()

    def rsample(self):
        raise NotImplementedError()


class MixtureSameFamily(td.Distribution):
    """
    ``MixtureSameFamily`` was introduced in pytorch 1.5. Since we are currently
    using pytorch 1.4, here we only implement a subset of its functions. See
    `<https://pytorch.org/docs/stable/distributions.html#mixturesamefamily>`_
    for full description of it.

    TODO: remove this after upgrading to pytorch 1.5.
    """

    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution
        event_shape = component_distribution.event_shape
        self._event_ndims = len(event_shape)
        super().__init__(
            batch_shape=component_distribution.batch_shape[:-1],
            event_shape=event_shape,
            validate_args=validate_args)

    def log_prob(self, x):
        """Log-probability of sample ``x``."""
        x = x.unsqueeze(-1 - self._event_ndims)
        log_prob_x = self._component_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = self._mixture_distribution.logits  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]


@gin.configurable
class SameActionPriorActor(Algorithm):
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 same_action_noise=0.1,
                 same_action_prob=0.9,
                 debug_summaries=False,
                 name="PriorActor"):
        """
        ``SameActionPriorActor`` can be used as a prior for KLD regularized RL-algorithms.
        It encodes the prior intuition that the next action should be same as the
        previous action most of time. More specifically, the distribution for each
        action dimension is a mixture of two components:
        1. a flat ``TruncatedNormal`` with ``loc`` equal to the median of the
            action range ``scale`` equal to the action range.
        2. a sharp ``TruncatedNormal`` with ``loc`` equal to the previous action
            and scale equal to the action range multiplied by ``same_action_noise``.

        The mixture weight depends on step_type:
        1. If the step_type is FIRST, the mixture weight is [1.0, 0]
        2. Otherwise the mixture weight is [1-same_actin_prob, same_actin_prob]

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            same_action_noise (float): the noise added to the previous action if
                the new action is the same as the previous action.
            same_action_prob (float): the probability that the next action is same
                as the previous action.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        super().__init__(
            train_state_spec=(), debug_summaries=debug_summaries, name=name)

        def _prepare_spec(action_spec):
            spec = {}
            assert action_spec.is_continuous, "Discrete action is not supported"
            spec['minimum'] = torch.as_tensor(
                np.broadcast_to(action_spec.minimum,
                                action_spec.shape)).reshape(
                                    1, *action_spec.shape, 1)
            spec['maximum'] = torch.as_tensor(
                np.broadcast_to(action_spec.maximum,
                                action_spec.shape)).reshape(
                                    1, *action_spec.shape, 1)
            spec['background_loc'] = 0.5 * (
                spec['minimum'] + spec['maximum']).squeeze(-1)
            spec['scale'] = torch.cat([
                spec['maximum'] - spec['minimum'],
                (spec['maximum'] - spec['minimum']) * same_action_noise
            ],
                                      dim=-1)
            mix_prob = torch.tensor([1. - same_action_prob, same_action_prob])
            spec['mix_logits'] = mix_prob.log().reshape(
                1, *([1] * len(action_spec.shape)), 2)
            spec['pure_logits'] = torch.tensor([0., -100.])
            spec['shape'] = action_spec.shape
            spec['continuous'] = True
            return spec

        self._action_spec = action_spec
        flat_action_spec = alf.nest.flatten(action_spec)
        self._prepared_specs = [
            _prepare_spec(spec) for spec in flat_action_spec
        ]

    def _make_dist(self, step_type, prev_action, spec):
        logits = spec['mix_logits'].expand(*prev_action.shape, -1).clone()
        logits[step_type == StepType.FIRST] = spec['pure_logits']
        mix = Categorical(logits=logits)
        loc = torch.stack(
            [spec['background_loc'].expand_as(prev_action), prev_action],
            dim=-1)
        components = TruncatedNormal(loc, spec['scale'], spec['minimum'],
                                     spec['maximum'])
        return Independent(
            base_distribution=MixtureSameFamily(mix, components),
            reinterpreted_batch_ndims=prev_action.ndim - 1)

    def predict_step(self, time_step: TimeStep, state):
        """Calculate the disribution of the next action.

        Args:
            time_step (TimeStep): time step structure
        Returns:
            AlgStep:
            - output (Distribution): the distribution of the action
            - state: ()
            - info: ()
        """
        flat_prev_action = alf.nest.flatten(time_step.prev_action)
        dists = [
            self._make_dist(time_step.step_type, prev_action,
                            spec) for prev_action, spec in zip(
                                flat_prev_action, self._prepared_specs)
        ]
        return AlgStep(
            output=alf.nest.pack_sequence_as(self._action_spec, dists),
            state=(),
            info=())

    def rollout_step(self, time_step: TimeStep, state):
        return self.predict_step(time_step, state)

    def train_step(self, exp: Experience, state):
        return self.predict_step(exp, state)


@gin.configurable
class UniformPriorActor(Algorithm):
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 debug_summaries=False,
                 name="UniformPriorActor"):
        """
        UniformPriorActor can be used as a prior for KLD regularized RL-algorithms. It
        generate a prior distribution for the next action using limited information,
        which can be used as the prior distribution in KLD.

        The action distribution is always an uniform distribution defined by the
        valid range of the action specified in ``action_spec``

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        super().__init__(
            train_state_spec=(), debug_summaries=debug_summaries, name=name)

        def _prepare_spec(action_spec):
            spec = {}
            spec['minimum'] = torch.as_tensor(
                np.broadcast_to(action_spec.minimum,
                                action_spec.shape)).reshape(
                                    1, *action_spec.shape)
            spec['maximum'] = torch.as_tensor(
                np.broadcast_to(action_spec.maximum,
                                action_spec.shape)).reshape(
                                    1, *action_spec.shape)
            spec['shape'] = action_spec.shape
            return spec

        self._action_spec = action_spec
        flat_action_spec = alf.nest.flatten(action_spec)
        self._prepared_specs = [
            _prepare_spec(spec) for spec in flat_action_spec
        ]

    def _make_dist(self, step_type, prev_action, spec):
        low = spec['minimum'].expand_as(prev_action)
        high = spec['maximum'].expand_as(prev_action)
        return Independent(
            base_distribution=Uniform(low, high),
            reinterpreted_batch_ndims=prev_action.ndim - 1)

    def predict_step(self, time_step: TimeStep, state):
        flat_prev_action = alf.nest.flatten(time_step.prev_action)
        dists = [
            self._make_dist(time_step.step_type, prev_action,
                            spec) for prev_action, spec in zip(
                                flat_prev_action, self._prepared_specs)
        ]
        return AlgStep(
            output=alf.nest.pack_sequence_as(self._action_spec, dists),
            state=(),
            info=())

    def rollout_step(self, time_step: TimeStep, state):
        return self.predict_step(time_step, state)

    def train_step(self, exp: Experience, state):
        return self.predict_step(exp, state)
