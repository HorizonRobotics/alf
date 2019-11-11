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
import numbers

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks.network import Network, DistributionNetwork
from tf_agents.specs.distribution_spec import nested_distributions_from_specs
from tf_agents.specs import tensor_spec

from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm, ActorCriticState
from alf.algorithms.rl_algorithm import ActionTimeStep, StepType
from alf.optimizers.trusted_updater import TrustedUpdater
from alf.utils import common
from alf.utils.common import namedtuple, run_if
from alf.utils.data_buffer import DataBuffer

nest_map = tf.nest.map_structure

TracExperience = namedtuple(
    "TracExperience", ["observation", "step_type", "state", "action_param"])


@gin.configurable
class TracAlgorithm(ActorCriticAlgorithm):
    """Trust-region actor-critic.

    It compares the action distributions after the SGD with the action
    distributions from the previous model. If the average distance is too big,
    the new parameters are shrinked as:
        w_new' = old_w + max_kl / kl * (w_new - w_old)

    If the distribution is Categorical, the distance is ||logits_1 - logits_2||^2,
    otherwise it's KL(d1||d2) + KL(d2||d1).

    The reason of using ||logits_1 - logits_2||^2 for Categorical distribution
    is that KL can be small even if there are large differences in logits when
    the entropy is small. This means that KL cannot fully capture how much the
    change is.
    """

    def __init__(self,
                 action_spec,
                 observation_spec,
                 batch_size,
                 buffer_size,
                 actor_network: DistributionNetwork,
                 value_network: Network,
                 action_dist_clip_per_dim=0.01,
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
            action_dist_clip_per_dim (float): adjust step if the average
                distance between the old and new action distributions exceeds
                this value multiplied with dimension.
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
            state=self.train_state_spec.actor,
            action_param=action_param_spec)
        self._buffer = DataBuffer(
            spec, capacity=(buffer_size + 1) * batch_size)
        self._trusted_updater = None

        def _get_clip(spec):
            dims = np.product(spec.shape.as_list())
            if tensor_spec.is_discrete(spec):
                dims *= spec.maximum - spec.minimum + 1
            return float(action_dist_clip_per_dim * dims)

        self._action_dist_clips = nest_map(_get_clip, self.action_spec)
        self._batch_size = batch_size

    def rollout(self, time_step: ActionTimeStep, state: ActorCriticState):
        """Rollout for one step."""
        policy_step = super().rollout(time_step, state)
        action_param = common.get_distribution_params(policy_step.action)
        exp = TracExperience(
            observation=time_step.observation,
            step_type=time_step.step_type,
            state=state.actor,
            action_param=action_param)

        # run_if() is unnecessary. But TF has error if run_if is removed.
        # See TF issue: https://github.com/tensorflow/tensorflow/issues/34090
        # The bug seems to be fixed by the latest tf-nightly build.
        run_if(
            tf.shape(time_step.step_type)[0] >
            0, lambda: self._buffer.add_batch(exp))

        if self._trusted_updater is None:
            self._trusted_updater = TrustedUpdater(
                self._actor_network.trainable_variables)
        return policy_step

    def after_train(self):
        """Adjust actor parameter according to KL-divergence."""
        dists, steps = self._trusted_updater.adjust_step(
            self._calc_change, self._action_dist_clips)

        def _summarize():
            with self.name_scope:
                for i, d in enumerate(tf.nest.flatten(dists)):
                    tf.summary.scalar("unadjusted_action_dist/%s" % i, d)
                tf.summary.scalar("adjust_steps", steps)

        common.run_if(common.should_record_summaries(), _summarize)
        super().after_train()

    @tf.function
    def _calc_change(self):
        """Calculate the distance between old/new action distributions.

        The distance is:
        ||logits_1 - logits_2||^2 for Categorical distribution
        KL(d1||d2) + KL(d2||d1) for others
        """
        # Remove the samples from the last step from the buffer because it may
        # repeated if the final_step_mode of OnPolicyDriver is FINAL_STEP_REDO.
        # See comment of OnPolicyDriver for detail.
        self._buffer.pop(self._batch_size)
        size = self._buffer.current_size
        total_dists = nest_map(lambda _: tf.zeros(()), self.action_spec)

        def _dist(d1, d2):
            if isinstance(d1, tfp.distributions.Categorical):
                return tf.reduce_sum(tf.square(d1.logits - d2.logits), axis=-1)
            else:
                return d1.kl_divergence(d2) + d2.kl_divergence(d1)

        def _update_total_dists(new_action, exp, total_dists):
            old_action = nested_distributions_from_specs(
                self.action_distribution_spec, exp.action_param)
            dists = nest_map(_dist, old_action, new_action)
            valid_masks = tf.cast(
                tf.not_equal(exp.step_type, StepType.LAST), tf.float32)
            dists = nest_map(lambda kl: tf.reduce_sum(kl * valid_masks), dists)
            return nest_map(lambda x, y: x + y, total_dists, dists)

        if len(tf.nest.flatten(self.train_state_spec)) == 0:
            # Not RNN
            for t in tf.range(0, size, self._batch_size):
                exp = self._buffer.get_batch_by_indices(
                    tf.range(t, t + self._batch_size))
                new_action, state = self._actor_network(
                    exp.observation,
                    step_type=exp.step_type,
                    network_state=exp.state)
                total_dists = _update_total_dists(new_action, exp, total_dists)
        else:
            # RNN
            exp = self._buffer.get_batch_by_indices(tf.range(self._batch_size))
            initial_state = common.zero_tensor_from_nested_spec(
                self.train_state_spec.actor, self._batch_size)
            state = exp.state
            for t in tf.range(0, size, self._batch_size):
                exp = self._buffer.get_batch_by_indices(
                    tf.range(t, t + self._batch_size))
                state = common.reset_state_if_necessary(
                    state, initial_state, exp.step_type == StepType.FIRST)
                new_action, state = self._actor_network(
                    exp.observation,
                    step_type=exp.step_type,
                    network_state=state)
                total_dists = _update_total_dists(new_action, exp, total_dists)

        size = tf.cast(size, tf.float32)
        total_dists = nest_map(lambda d: d / size, total_dists)
        return total_dists
