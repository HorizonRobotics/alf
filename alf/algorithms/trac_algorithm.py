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
import gin
import numpy as np
import torch
import torch.distributions as td

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.data_structures import Experience, namedtuple, StepType, TimeStep
from alf.optimizers.trusted_updater import TrustedUpdater
from alf.utils import common, dist_utils

nest_map = alf.nest.map_structure

TracExperience = namedtuple(
    "TracExperience",
    ["observation", "step_type", "state", "action_param", "prev_action"])

TracInfo = namedtuple(
    "TracInfo",
    ["action_distribution", "observation", "state", "ac", "prev_action"])


@gin.configurable
class TracAlgorithm(OnPolicyAlgorithm):
    """Trust-region actor-critic.

    It compares the action distributions after the SGD with the action
    distributions from the previous model. If the average distance is too big,
    the new parameters are shrinked as:
        w_new' = old_w + 0.9 * distance_clip / distance * (w_new - w_old)

    If the distribution is Categorical, the squared distance is
    ||logits_1 - logits_2||^2, and if the distribution is Deterministic, it is
    ||loc_1 - loc_2||^2,  otherwise it's KL(d1||d2) + KL(d2||d1).

    The reason of using ||logits_1 - logits_2||^2 for Categorical distribution
    is that KL can be small even if there are large differences in logits when
    the entropy is small. This means that KL cannot fully capture how much the
    change is.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 env=None,
                 config=None,
                 ac_algorithm_cls=ActorCriticAlgorithm,
                 action_dist_clip_per_dim=0.01,
                 debug_summaries=False,
                 name="TracAlgorithm"):
        """Create an instance TracAlgorithm.

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            ac_algorithm_cls (type): Actor Critic Algorithm cls.
            action_dist_clip_per_dim (float): action dist clip per dimension
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """
        ac_algorithm = ac_algorithm_cls(
            observation_spec=observation_spec,
            action_spec=action_spec,
            debug_summaries=debug_summaries)

        assert hasattr(ac_algorithm, '_actor_network')

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            env=env,
            config=config,
            train_state_spec=ac_algorithm.train_state_spec,
            predict_state_spec=ac_algorithm.predict_state_spec,
            debug_summaries=debug_summaries,
            name=name)

        self._ac_algorithm = ac_algorithm
        self._trusted_updater = None
        self._action_distribution_spec = None

        def _get_clip(spec):
            dims = np.product(spec.shape)
            if spec.is_discrete:
                dims *= spec.maximum - spec.minimum + 1
            return np.sqrt(action_dist_clip_per_dim * dims)

        self._action_dist_clips = nest_map(_get_clip, self.action_spec)

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        return self._ac_algorithm.predict_step(time_step, state,
                                               epsilon_greedy)

    def _make_policy_step(self, time_step, state, policy_step):
        assert (
            alf.nest.is_namedtuple(policy_step.info)
            and "action_distribution" in policy_step.info._fields), (
                "PolicyStep.info from ac_algorithm.rollout_step() or "
                "ac_algorithm.train_step() should be a namedtuple containing "
                "`action_distribution` in order to use TracAlgorithm.")
        action_distribution = policy_step.info.action_distribution
        if self._action_distribution_spec is None:
            self._action_distribution_spec = dist_utils.extract_spec(
                action_distribution)
        ac_info = policy_step.info._replace(action_distribution=())
        # EntropyTargetAlgorithm need info.action_distribution
        return policy_step._replace(
            info=TracInfo(
                action_distribution=action_distribution,
                observation=time_step.observation,
                prev_action=time_step.prev_action,
                state=self._ac_algorithm.convert_train_state_to_predict_state(
                    state),
                ac=ac_info))

    def rollout_step(self, time_step: TimeStep, state):
        """Rollout for one step."""
        policy_step = self._ac_algorithm.rollout_step(time_step, state)
        return self._make_policy_step(time_step, state, policy_step)

    def train_step(self, exp: Experience, state):
        ac_info = exp.rollout_info.ac._replace(
            action_distribution=exp.rollout_info.action_distribution)
        exp = exp._replace(rollout_info=ac_info)
        policy_step = self._ac_algorithm.train_step(exp, state)
        return self._make_policy_step(exp, state, policy_step)

    def calc_loss(self, training_info):
        if self._trusted_updater is None:
            self._trusted_updater = TrustedUpdater(
                list(self._ac_algorithm._actor_network.parameters()))
        rollout_ac_info = ()
        if training_info.rollout_info != ():
            rollout_ac_info = training_info.rollout_info.ac._replace(
                action_distribution=training_info.rollout_info.
                action_distribution)
        ac_info = training_info.info.ac._replace(
            action_distribution=training_info.info.action_distribution)
        return self._ac_algorithm.calc_loss(
            training_info._replace(rollout_info=rollout_ac_info, info=ac_info))

    def after_update(self, training_info):
        """Adjust actor parameter according to KL-divergence."""
        action_param = dist_utils.distributions_to_params(
            training_info.info.action_distribution)
        exp_array = TracExperience(
            observation=training_info.info.observation,
            step_type=training_info.step_type,
            action_param=action_param,
            prev_action=training_info.info.prev_action,
            state=training_info.info.state)
        dists, steps = self._trusted_updater.adjust_step(
            lambda: self._calc_change(exp_array), self._action_dist_clips)

        if alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                for i, d in enumerate(alf.nest.flatten(dists)):
                    alf.summary.scalar("unadjusted_action_dist/%s" % i, d)
                alf.summary.scalar("adjust_steps", steps)

        ac_info = training_info.info.ac._replace(
            action_distribution=training_info.info.action_distribution)
        self._ac_algorithm.after_update(training_info._replace(info=ac_info))

    def _calc_change(self, exp_array):
        """Calculate the distance between old/new action distributions.

        The squared distance is:
        ||logits_1 - logits_2||^2 for Categorical distribution
        ||loc_1 - loc_2||^2 for Deterministic distribution
        KL(d1||d2) + KL(d2||d1) for others
        """

        def _get_base_dist(dist: td.Distribution):
            """Get the base distribution of `dist`."""
            if isinstance(dist, (td.Independent, td.TransformedDistribution)):
                return _get_base_dist(dist.base_dist)
            return dist

        def _dist(d1, d2):
            d1_base = _get_base_dist(d1)
            d2_base = _get_base_dist(d2)
            if isinstance(d1_base, td.Categorical):
                dist = (d1_base.logits - d2_base.logits)**2
            elif isinstance(d1, torch.Tensor):
                dist = (d1 - d2)**2
            else:
                dist = td.kl.kl_divergence(d1, d2) + td.kl.kl_divergence(
                    d2, d1)
            return dist.sum(list(range(1, dist.ndim)))

        def _update_total_dists(new_action, exp, total_dists):
            old_action = dist_utils.params_to_distributions(
                exp.action_param, self._action_distribution_spec)
            valid_masks = (exp.step_type != StepType.LAST).to(torch.float32)
            return nest_map(
                lambda d1, d2, total_dist: (_dist(d1, d2) * valid_masks).sum()
                + total_dist, old_action, new_action, total_dists)

        num_steps, batch_size = exp_array.step_type.shape
        state = nest_map(lambda x: x[0], exp_array.state)
        # exp_array.state is no longer needed
        exp_array = exp_array._replace(state=())
        initial_state = common.zero_tensor_from_nested_spec(
            self.predict_state_spec, batch_size)
        total_dists = nest_map(lambda _: torch.tensor(0.), self.action_spec)
        for t in range(num_steps):
            exp = nest_map(lambda x: x[t], exp_array)
            state = common.reset_state_if_necessary(
                state, initial_state, exp.step_type == StepType.FIRST)
            time_step = TimeStep(
                observation=exp.observation,
                step_type=exp.step_type,
                prev_action=exp.prev_action)
            policy_step = self._ac_algorithm.predict_step(
                time_step=time_step, state=state, epsilon_greedy=1.0)
            assert (
                alf.nest.is_namedtuple(policy_step.info)
                and "action_distribution" in policy_step.info._fields
            ), ("AlgStep.info from ac_algorithm.predict_step() should be "
                "a namedtuple containing `action_distribution` in order to "
                "use TracAlgorithm.")
            new_action = policy_step.info.action_distribution
            state = policy_step.state
            total_dists = _update_total_dists(new_action, exp, total_dists)

        size = num_steps * batch_size
        total_dists = nest_map(lambda d: torch.sqrt(d / size), total_dists)
        return total_dists

    def preprocess_experience(self, exp: Experience):
        ac_info = exp.rollout_info.ac._replace(
            action_distribution=exp.rollout_info.action_distribution)
        new_exp = self._ac_algorithm.preprocess_experience(
            exp._replace(rollout_info=ac_info))
        return new_exp._replace(
            rollout_info=exp.rollout_info._replace(ac=new_exp.rollout_info))
