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
"""Trusted Region Actor critic algorithm."""

from typing import Callable

import gin
import tensorflow as tf

from tf_agents.networks.network import Network, DistributionNetwork
from tf_agents.specs.distribution_spec import nested_distributions_from_specs

from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm, ActorCriticState
from alf.algorithms.rl_algorithm import ActionTimeStep, StepType
from alf.utils import common
from alf.utils.common import namedtuple, run_if
from alf.utils.data_buffer import DataBuffer
from alf.utils import math_ops

nest_map = tf.nest.map_structure

TracExperience = namedtuple(
    "TracExperience", ["observation", "step_type", "state", "action_param"])


class TrustedUpdater(object):
    """Adjust variables based on the change calculated by `change_f()`

    The motivation is that if some quatity changes too much after an SGD update,
    the SGD step might be too big. We want to shink that step so that the
    concerned quatity does not change too much. We can also monitor multiple
    quantities to make sure none of them has sudden big jump.

    It adjusts variables provided at `__init__` if the change calculated by
    `change_f` is too big:
    ```
    change = change_f()
    if change > max_change:
        var <= old_var + 0.9 * (max_change/change) * (var - old_var)
    ```
    The above procedure is repeated until `change` is not bigger than
    `max_change`. Note that `change` and `max_change` can be nests of
    scalars. In this case, the inequality is understood as if any one of the
    changes is greater than its corresponding max_change.
    """

    def __init__(self, variables):
        """Create a TrustedUpdater instance.

        Args:
            varialbes (list[Variables]): variables to be monitored.
        """
        self._variables = variables
        assert len(self._variables) > 0
        self._prev_variables = [
            tf.Variable(initial_value=v, dtype=v.dtype) for v in variables
        ]

    @tf.function
    def adjust_step(self, change_f: Callable, max_change):
        """Adjust `variables` based change calculated by change_f

        This function will copy the new values of the variables to
        a backup to be used for the next call of adjust_step.
        Args:
            change_f (Callable): a function calculate a (nested) change based on
                current variable.
            max_change (float): (nested) max change allowed.
        Returns:
            the actual change after variables are adjusted
            the number of step to adjust variables
        """

        def _adjust_step(ratio):
            # 0.9 is to prevent infinite loop when `actual_change` is close to
            # `max_change`
            r = 0.9 / ratio
            for var, prev_var in zip(self._variables, self._prev_variables):
                var.assign(prev_var + r * (var - prev_var))

        counter = tf.zeros((), tf.int32)
        ratio = tf.constant(2.)
        change = max_change  # TF require `change` to be defined before the loop
        while ratio > 1.:
            change = change_f()
            ratio = nest_map(lambda c, m: tf.abs(c) / m, change, max_change)
            ratio = math_ops.max_n(tf.nest.flatten(ratio))
            if ratio > 1.:
                _adjust_step(ratio)
            counter += 1

        for var, prev_var in zip(self._variables, self._prev_variables):
            prev_var.assign(var)

        return change, counter


@gin.configurable
class TracAlgorithm(ActorCriticAlgorithm):
    """Trust-region actor-critic.

    It compares the action distributions after the SGD with the action
    distributions from the previous model. If the KL-divergence is too big,
    the new parameters are shrinked as:
        w_new' = old_w + max_kl / kl * (w_new - w_old)
    """

    def __init__(self,
                 action_spec,
                 observation_spec,
                 batch_size,
                 buffer_size,
                 actor_network: DistributionNetwork,
                 value_network: Network,
                 kl_clip=0.1,
                 loss=None,
                 loss_class=ActorCriticLoss,
                 optimizer=None,
                 debug_summaries=False,
                 name="TracAlgorithm"):
        """Create an instance TracAlgorithm.

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            observation_spec (nested TensorSpec): spec of the observation.
            batch_size (int): batch size of the parallel evironment batch.
            buffer_size (int): use the past so many steps for estimating KL
                divergence.
            kl_clip (float): adjust step if KL divergence exceeds this value.
            actor_network (DistributionNetwork): A network that returns nested
                tensor of action distribution for each observation given observation
                and network state.
            value_network (Network): A function that returns value tensor from neural
                net predictions for each observation given observation and nwtwork
                state.
            loss (None|ActorCriticLoss): an object for calculating loss. If
                None, a default loss of class loss_class will be used.
            loss_class (type): the class of the loss. The signature of its
                constructor: loss_class(action_spec, debug_summaries)
            optimizer (tf.optimizers.Optimizer): The optimizer for training
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """
        super().__init__(
            action_spec=action_spec,
            actor_network=actor_network,
            value_network=value_network,
            loss=loss,
            loss_class=loss_class,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)
        action_param_spec = nest_map(lambda spec: spec.input_params_spec,
                                     actor_network.output_spec)
        spec = TracExperience(
            observation=observation_spec,
            step_type=tf.TensorSpec((), tf.int32),
            state=self.train_state_spec,
            action_param=action_param_spec)
        self._buffer = DataBuffer(spec, capacity=buffer_size * batch_size)
        self._trusted_updater = None
        self._kl_clips = nest_map(lambda _: kl_clip, self.action_spec)
        self._batch_size = batch_size

    def rollout(self, time_step: ActionTimeStep, state: ActorCriticState):
        policy_step = super().rollout(time_step, state)
        action_param = common.get_distribution_params(policy_step.action)
        exp = TracExperience(
            observation=time_step.observation,
            step_type=time_step.step_type,
            state=state,
            action_param=action_param)

        # run_if() is unnecessary. But TF has error if run_if is removed.
        # TODO: find the minimum condition to reproduce this bug.
        run_if(
            tf.shape(time_step.step_type)[0] >
            0, lambda: self._buffer.add_batch(exp))

        if self._trusted_updater is None:
            self._trusted_updater = TrustedUpdater(
                self._actor_network.trainable_variables)
        return policy_step

    def after_train(self):
        """Adjust actor parameter according to KL-divergence."""
        kls, steps = self._trusted_updater.adjust_step(self._calc_kl,
                                                       self._kl_clips)

        def _summarize():
            with self.name_scope:
                for i, kl in enumerate(tf.nest.flatten(kls)):
                    tf.summary.scalar("action_kl/%s" % i, kl)
                tf.summary.scalar("adjust_steps", steps)

        common.run_if(common.should_record_summaries(), _summarize)
        super().after_train()

    @tf.function
    def _calc_kl(self):
        """Calculate the symmetrical KL-devengence between old/new action dist.

        Equation: KL(d1||d2) + KL(d2||d1)
        """
        # Here we use stored states to calculate action. It might be better
        # to use the states calculated using the new weights.
        size = self._buffer.current_size
        total_kls = nest_map(lambda _: tf.zeros(()), self.action_spec)
        for t in tf.range(0, size, self._batch_size):
            exp = self._buffer.get_batch_by_indices(
                tf.range(t, t + self._batch_size))
            new_action, _ = self._actor_network(
                exp.observation,
                step_type=exp.step_type,
                network_state=exp.state)
            old_action = nested_distributions_from_specs(
                self.action_distribution_spec, exp.action_param)
            kls = nest_map(
                lambda d1, d2: d1.kl_divergence(d2) + d2.kl_divergence(d1),
                old_action, new_action)
            valid_masks = tf.cast(
                tf.not_equal(exp.step_type, StepType.LAST), tf.float32)
            kls = nest_map(lambda kl: tf.reduce_sum(kl * valid_masks), kls)
            total_kls = nest_map(lambda x, y: x + y, total_kls, kls)

        size = tf.cast(size, tf.float32)
        total_kls = nest_map(lambda total_kl: total_kl / size, total_kls)
        return total_kls
