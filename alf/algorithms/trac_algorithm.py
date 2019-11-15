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

from tf_agents.networks.network import DistributionNetwork
from tf_agents.specs.distribution_spec import nested_distributions_from_specs
from tf_agents.specs import tensor_spec
from tf_agents.distributions.utils import SquashToSpecNormal

from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.off_policy_algorithm import Experience
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.rl_algorithm import ActionTimeStep, StepType
from alf.optimizers.trusted_updater import TrustedUpdater
from alf.utils import common
from alf.utils.common import namedtuple

nest_map = tf.nest.map_structure

TracExperience = namedtuple(
    "TracExperience", ["observation", "step_type", "state", "action_param"])

TracInfo = namedtuple("TracInfo", ["observation", "state", "ac"])


@gin.configurable
class TracAlgorithm(OnPolicyAlgorithm):
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
                 ac_algorithm_cls=ActorCriticAlgorithm,
                 action_dist_clip_per_dim=0.01,
                 debug_summaries=False,
                 name="TracAlgorithm"):
        """Create an instance TracAlgorithm.

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            ac_algorithm_cls (type): Actor Critic Algorithm cls (AC, PPO, SAC...)
            action_dist_clip_per_dim (float): action dist clip per dimension
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """
        ac_algorithm = ac_algorithm_cls(
            action_spec=action_spec, debug_summaries=debug_summaries)

        assert isinstance(ac_algorithm._actor_network, DistributionNetwork)

        super().__init__(
            action_spec=action_spec,
            train_state_spec=ac_algorithm.train_state_spec,
            predict_state_spec=ac_algorithm.predict_state_spec,
            action_distribution_spec=ac_algorithm.action_distribution_spec,
            debug_summaries=debug_summaries,
            name=name)

        self._ac_algorithm = ac_algorithm
        self._trusted_updater = None

        def _get_clip(spec):
            dims = np.product(spec.shape.as_list())
            if tensor_spec.is_discrete(spec):
                dims *= spec.maximum - spec.minimum + 1
            return float(action_dist_clip_per_dim * dims)

        self._action_dist_clips = nest_map(_get_clip, self.action_spec)

    def rollout(self, time_step: ActionTimeStep, state):
        """Rollout for one step."""
        policy_step = self._ac_algorithm.rollout(time_step, state)
        if self._trusted_updater is None:
            self._trusted_updater = TrustedUpdater(
                self._ac_algorithm._actor_network.trainable_variables)
        return policy_step._replace(
            info=TracInfo(
                observation=time_step.observation,
                state=state,
                ac=policy_step.info))

    def train_step(self, exp: Experience, state):
        policy_step = self._ac_algorithm.train_step(exp, state)
        return policy_step._replace(
            info=TracInfo(
                observation=exp.observation, state=state, ac=policy_step.info))

    def calc_loss(self, training_info):
        return self._ac_algorithm.calc_loss(
            training_info._replace(info=training_info.info.ac))

    def after_train(self, training_info):
        """Adjust actor parameter according to KL-divergence."""
        exp_array = TracExperience(
            observation=training_info.info.observation,
            step_type=training_info.step_type,
            action_param=common.get_distribution_params(
                training_info.action_distribution),
            state=training_info.info.state)
        exp_array = common.create_and_unstack_tensor_array(
            exp_array, clear_after_read=False)
        dists, steps = self._trusted_updater.adjust_step(
            lambda: self._calc_change(exp_array), self._action_dist_clips)

        def _summarize():
            with self.name_scope:
                for i, d in enumerate(tf.nest.flatten(dists)):
                    tf.summary.scalar("unadjusted_action_dist/%s" % i, d)
                tf.summary.scalar("adjust_steps", steps)

        common.run_if(common.should_record_summaries(), _summarize)
        self._ac_algorithm.after_train(
            training_info._replace(info=training_info.info.ac))

    @tf.function
    def _calc_change(self, exp_array):
        """Calculate the distance between old/new action distributions.

        The distance is:
        ||logits_1 - logits_2||^2 for Categorical distribution
        KL(d1||d2) + KL(d2||d1) for others
        """

        def _dist(d1, d2):
            if isinstance(d1, tfp.distributions.Categorical):
                return tf.reduce_sum(tf.square(d1.logits - d2.logits), axis=-1)
            elif isinstance(d1, SquashToSpecNormal):
                # TODO spec means and magnitudes equal check fails in graph mode
                return tf.reduce_sum(
                    d1.input_distribution.kl_divergence(d2.input_distribution)
                    + d2.input_distribution.kl_divergence(
                        d1.input_distribution),
                    axis=-1)
            else:
                return tf.reduce_sum(
                    d1.kl_divergence(d2) + d2.kl_divergence(d1), )

        def _update_total_dists(new_action, exp, total_dists):
            old_action = nested_distributions_from_specs(
                self.action_distribution_spec, exp.action_param)
            dists = nest_map(_dist, old_action, new_action)
            valid_masks = tf.cast(
                tf.not_equal(exp.step_type, StepType.LAST), tf.float32)
            dists = nest_map(lambda kl: tf.reduce_sum(kl * valid_masks), dists)
            return nest_map(lambda x, y: x + y, total_dists, dists)

        num_steps = exp_array.step_type.size()
        # element_shape for `TensorArray` can be (None, ...)
        batch_size = tf.shape(exp_array.step_type.read(0))[0]
        state = tf.nest.map_structure(lambda x: x.read(0), exp_array.state)
        # exp_array.state is no longer needed
        exp_array = exp_array._replace(state=())
        initial_state = common.zero_tensor_from_nested_spec(
            self.train_state_spec, batch_size)
        total_dists = nest_map(lambda _: tf.zeros(()), self.action_spec)
        for t in tf.range(num_steps):
            exp = tf.nest.map_structure(lambda x: x.read(t), exp_array)
            state = common.reset_state_if_necessary(
                state, initial_state, exp.step_type == StepType.FIRST)
            time_step = ActionTimeStep()
            time_step = time_step._replace(
                observation=exp.observation,
                step_type=exp.step_type,
            )
            policy_step = self._ac_algorithm.predict(
                time_step=time_step, state=state)
            new_action, state = policy_step.action, policy_step.state
            total_dists = _update_total_dists(new_action, exp, total_dists)

        size = tf.cast(num_steps * batch_size, tf.float32)
        total_dists = nest_map(lambda d: d / size, total_dists)
        return total_dists

    def preprocess_experience(self, exp: Experience):
        return self._ac_algorithm.preprocess_experience(
            exp._replace(info=exp.info.ac))
