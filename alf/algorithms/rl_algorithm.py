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
"""Base class for RL algorithms."""

from abc import abstractmethod
from absl import logging
from collections import namedtuple
import os
import time
import torch
from typing import Callable, Optional
from absl import logging

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.async_unroller import AsyncUnroller
from alf.experience_replayers.replay_buffer import ReplayBuffer
from alf.data_structures import (AlgStep, Experience, make_experience,
                                 StepType, TimeStep, BasicRolloutInfo,
                                 BasicRLInfo)
from alf.utils import common, dist_utils, summary_utils
from alf.utils.summary_utils import record_time
from alf.utils.distributed import data_distributed_when
from alf.tensor_specs import TensorSpec
from .config import TrainerConfig


def adjust_replay_buffer_length(config: TrainerConfig,
                                num_earliest_frames_ignored: int = 0) -> int:
    """Adjust the replay buffer length for whole replay buffer training.

    Normally we just respect the replay buffer length set in the
    config. However, for a specific case where the user asks to do
    "whole replay buffer training", we need to adjust the user
    provided length to achieve desired behavior.

    Args:

        config: The trainer config of the training session
        num_earliest_frames_ignored: ignore the earliest so many
           frames from the buffer when sampling or gathering. This is
           typically required when FrameStacker is used. See
           ``ReplayBuffer`` for details.

    Returns:

        An integer representing the adjusted replay buffer length.

    """
    if not config.whole_replay_buffer_training:
        return config.replay_buffer_length

    adjusted = config.replay_buffer_length

    if config.clear_replay_buffer:
        # Here the clear replay buffer (after each training iteration)
        # is achieved by setting the replay buffer size to the unroll
        # length, while disregarding config.replay_buffer_length.
        #
        # Remember that the replay buffer is under the hood a ring
        # buffer. The next iteration will push ``unroll_length``
        # batches of experiences into the replay buffer. It
        # effectively "clears" the experiences collected from the last
        # iteration when the replay buffer length is set so.
        #
        # The actual replay buffer length should have an extra 1 added
        # to it. This is to prevent the last batch of experiences in
        # each iteration from never getting properly trained.
        if config.clear_replay_buffer_but_keep_one_step:
            adjusted = config.unroll_length + 1
        else:
            adjusted = config.unroll_length

    # The replay buffer length is extended by num_earliest_frames_ignored so
    # that after FrameStacker transformation the number of experiences matches
    # ``unroll_length``.
    adjusted += num_earliest_frames_ignored

    common.info(f'Actual replay buffer length is adjusted to {adjusted}.')

    return adjusted


@alf.configurable
class RLAlgorithm(Algorithm):
    """Abstract base class for RL Algorithms.

    ``RLAlgorithm`` provide basic functions and generic interface for rl algorithms.

    The key interface functions are:

    1. ``predict_step()``: one step of computation of action for evaluation.
    2. ``rollout_step()``: one step of computation for rollout. It is
       used for collecting experiences during training. Different from
       ``predict_step``, ``rollout_step`` may include additional computations for
       training. For on-policy algorithms (e.g., AC, PPO, etc), the collected
       experiences will be immediately used to update parameters after one
       rollout (multiple rollout steps) is performed; for off-policy algorithms
       (e.g., SAC, DDPG, etc), these collected experiences will be put into a
       replay buffer.
    3. ``train_step()``: only used for off-policy training. The training data are
       sampled from the replay buffer filled by ``rollout_step()``.
    4. ``train_iter()``: perform one iteration of training (rollout [and train]).
       ``train_iter()`` is called ``num_iterations`` times by ``Trainer``.
       We provide a default implementation. Users can choose to implement
       their own ``train_iter()``.
    5. ``update_with_gradient()``: Do one gradient update based on the loss. It is
       used by the default ``train_iter()`` implementation. You can override to
       implement your own ``update_with_gradient()``.
    6. ``calc_loss()``: calculate loss based the ``experience`` and the ``train_info``
       collected from ``rollout_step()`` or ``train_step()``. It is used by the
       default implementation of ``train_iter()``. If you want to use the default
       ``train_iter()``, you need to implement ``calc_loss()``.
    7. ``after_update()``: called by ``train_iter()`` after every call to
       ``update_with_gradient()``, mainly for some postprocessing steps such as
       copying a training model to a target model in SAC or DQN.
    8. ``after_train_iter()``: called by ``train_iter()`` after every call to
       ``train_from_unroll()`` (on-policy training iter) or
       ``train_from_replay_buffer`` (off-policy training iter). It's mainly for
       training additional modules that have their own training logic
       (e.g., on/off-policy, replay buffers, etc). Other things might also be
       possible as long as they should be done once every training iteration.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 train_state_spec,
                 reward_spec=TensorSpec(()),
                 predict_state_spec=None,
                 rollout_state_spec=None,
                 is_on_policy=None,
                 reward_weights=None,
                 env=None,
                 config: TrainerConfig = None,
                 optimizer=None,
                 checkpoint=None,
                 is_eval: bool = False,
                 overwrite_policy_output=False,
                 debug_summaries=False,
                 name="RLAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            train_state_spec (nested TensorSpec): for the network state of
                ``train_step()``.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            rollout_state_spec (nested TensorSpec): for the network state of
                ``predict_step()``. If None, it's assumed to be the same as
                ``train_state_spec``.
            predict_state_spec (nested TensorSpec): for the network state of
                ``predict_step()``. If None, it's assumed to be the same as
                ``rollout_state_spec``.
            is_on_policy (None|bool): whether the algorithm is on-policy or not.
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. If not None, the weighted sum of rewards is
                the reward for training. Otherwise, the sum of rewards is used.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple
                simulations simultaneously. Running multiple environments in
                parallel is crucial to on-policy algorithms as it increases the
                diversity of data and decreases temporal correlation. ``env`` only
                needs to be provided to the root ``Algorithm``.
            config (TrainerConfig): config for training. ``config`` only needs to
                be provided to the algorithm which performs a training iteration
                by itself.
            optimizer (torch.optim.Optimizer): The default optimizer for training.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            is_eval (bool): True if this algorithm is used for evaluation only,
                during deployment.  In this case, the algorithm do not need to
                create certain components such as value_network for ActorCriticAlgorithm,
                critic_networks for SacAlgorithm.
            overwrite_policy_output (bool): if True, overwrite the policy output
                with next_step.prev_action. This option can be used in some
                cases such as data collection.
            debug_summaries (bool): If True, debug summaries will be created.
            name (str): Name of this algorithm.
        """
        super(RLAlgorithm, self).__init__(
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec,
            predict_state_spec=predict_state_spec,
            is_on_policy=is_on_policy,
            optimizer=optimizer,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)
        self._is_eval = is_eval

        self._env = env
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        assert reward_spec.ndim <= 1, "reward_spec must be rank-0 or rank-1!"
        self._reward_spec = reward_spec

        if reward_spec.numel > 1:
            if reward_weights:
                assert reward_spec.numel == len(reward_weights), (
                    "Mismatch between len(reward_weights)=%s and reward_dim=%s"
                    % (len(reward_weights), reward_spec.numel))
                # Note that if training or playing from a checkpoint while specifying
                # a reward weight vector different from the original one, this new
                # specified vector will be overwritten by the checkpoint.
                self.register_buffer(
                    "_reward_weights",
                    torch.tensor(reward_weights, dtype=torch.float32))
            else:
                self.register_buffer(
                    "_reward_weights",
                    torch.ones(reward_spec.shape, dtype=torch.float32))
        else:
            self._reward_weights = None
            assert reward_weights is None, (
                "reward_weights cannot be used for one dimensional reward")

        self._rollout_info_spec = None

        self._current_time_step = None
        self._current_policy_state = None
        self._current_transform_state = None

        if self._env is not None and not self.on_policy:
            replay_buffer_length = adjust_replay_buffer_length(
                config, self._num_earliest_frames_ignored)

            if config.whole_replay_buffer_training and config.clear_replay_buffer:
                # For whole replay buffer training, we would like to be sure
                # that the replay buffer have enough samples in it to perform
                # the training, which will most likely happen in the 2nd
                # iteration. The minimum_initial_collect_steps guarantees that.
                minimum_initial_collect_steps = replay_buffer_length * self._env.batch_size
                if config.initial_collect_steps < minimum_initial_collect_steps:
                    common.info(
                        'Set the initial_collect_steps to minimum required '
                        f'value {minimum_initial_collect_steps} because '
                        'whole_replay_buffer_training is on.')
                    config.initial_collect_steps = minimum_initial_collect_steps

            self.set_replay_buffer(self._env.batch_size, replay_buffer_length,
                                   config.priority_replay)

        if config:
            self._offline_buffer_dir = config.offline_buffer_dir  # default None

            if self._offline_buffer_dir:
                # TODO: add support to on-policy algorithm
                assert not self.on_policy, (
                    "currently only support "
                    "hybrid training for off-policy algorithms")
                self._has_offline = True
            else:
                self._has_offline = False

        env = self._env
        if env is not None:
            metric_buf_size = max(self._config.metric_min_buffer_size,
                                  self._env.batch_size)
            example_time_step = env.reset()
            self._metrics = [
                alf.metrics.NumberOfEpisodes(),
                alf.metrics.EnvironmentSteps(),
                alf.metrics.AverageReturnMetric(
                    buffer_size=metric_buf_size,
                    example_time_step=example_time_step),
                alf.metrics.AverageEpisodeLengthMetric(
                    example_time_step=example_time_step,
                    buffer_size=metric_buf_size),
                alf.metrics.AverageEnvInfoMetric(
                    example_time_step=example_time_step,
                    buffer_size=metric_buf_size),
                alf.metrics.AverageDiscountedReturnMetric(
                    buffer_size=metric_buf_size,
                    example_time_step=example_time_step),
                alf.metrics.AverageRewardMetric(
                    buffer_size=metric_buf_size,
                    example_time_step=example_time_step),
                alf.metrics.EpisodicStartAverageDiscountedReturnMetric(
                    buffer_size=metric_buf_size,
                    example_time_step=example_time_step)
            ]
        self._async_unroller = None
        self._original_rollout_step = self.rollout_step
        self.rollout_step = self._rollout_step
        self._overwrite_policy_output = overwrite_policy_output
        self._remaining_unroll_length_fraction = 0
        self._ensure_rollout_summary = alf.summary.EnsureSummary()
        self._offline_replay_buffer = None

    def is_rl(self):
        """Always return True for RLAlgorithm."""
        return True

    @property
    def observation_spec(self):
        """Return the observation spec."""
        return self._observation_spec

    @property
    def rollout_info_spec(self):
        """The spec for the ``AlgStep.info`` returned from ``rollout_step()``."""
        assert self._rollout_info_spec is not None, (
            "rollout_step() has not "
            " been used. rollout_info_spec is not available.")
        return self._rollout_info_spec

    @property
    def action_spec(self):
        """Return the action spec."""
        return self._action_spec

    @torch.no_grad()
    def set_reward_weights(self, reward_weights):
        """Update reward weights; this function can be called at any step during
        training. Once called, the updated reward weights are expected to be used
        by the algorithm in the next.

        Args:
            reward_weights (Tensor): a tensor that is compatible with
                ``self._reward_spec``.
        """
        assert self.has_multidim_reward(), (
            "Can't update weights for a scalar reward!")
        self._reward_weights.copy_(reward_weights)

    def has_multidim_reward(self):
        """Check if the algorithm uses multi-dim reward or not.

        Returns:
            bool: True if the reward has multiple dims.
        """
        return self._reward_spec.numel > 1

    @property
    def reward_weights(self):
        """Return the current reward weights."""
        return self._reward_weights

    def get_step_metrics(self):
        """Get step metrics that used for generating summaries against

        Returns:
            list[StepMetric]: step metrics ``EnvironmentSteps`` and
            ``NumberOfEpisodes``.
        """
        return self._metrics[:2]

    def get_metrics(self):
        """Returns the metrics monitored by this driver.

        Returns:
            list[StepMetric]:
        """
        return self._metrics

    def summarize_reward(self, name, rewards):
        if self._debug_summaries:
            assert 2 <= rewards.ndim <= 3, (
                "The shape of rewards should be [T, B] or [T, B, k]")
            if rewards.ndim == 2:
                alf.summary.histogram(name + "/value", rewards)
                alf.summary.scalar(
                    name + "/mean",
                    torch.mean(rewards),
                    average_over_summary_interval=True)
            else:
                for i in range(rewards.shape[2]):
                    r = rewards[..., i]
                    alf.summary.histogram('%s/%s/value' % (name, i), r)
                    alf.summary.scalar(
                        '%s/%s/mean' % (name, i),
                        torch.mean(r),
                        average_over_summary_interval=True)

    @alf.configurable(whitelist=["custom_summary"])
    def summarize_rollout(
            self,
            experience: Experience,
            custom_summary: Optional[Callable[[Experience], None]] = None):
        """Generate summaries for rollout.

        Args:
            experience: experience collected from ``rollout_step()``.
            custom_summary: when specified it is a function that will be called every
               time when this ``summarize_rollout`` hook is called. This provides
               a convenient way for the user to extend ``summarize_rollout`` from
               ALF configs.
        """
        if self._debug_summaries:
            summary_utils.summarize_action(experience.action,
                                           self._action_spec, "rollout_action")
            self.summarize_reward("rollout_reward/extrinsic",
                                  experience.reward)

        if self._config.summarize_action_distributions:
            field = alf.nest.find_field(experience.rollout_info,
                                        'action_distribution')
            if len(field) == 1:
                summary_utils.summarize_distribution("rollout_action_dist",
                                                     field[0])

        if custom_summary is not None:
            custom_summary(experience)

    def summarize_train(self, experience, train_info, loss_info, params):
        """Generate summaries for training & loss info after each gradient update.

        For on-policy algorithms, ``experience.rollout_info`` is empty, while for
        off-policy algorithms, it is available. However, the statistics in both
        ``train_info`` and ``experience.rollout_info` are for the data sampled
        from the replay buffer. They store the update-to-date model outputs and
        the historical model outputs (on the past rollout data), respectively.
        They do not represent the model outputs on the current on-going rollout.

        Args:
            experience (Experience): experiences collected from the most recent
                ``unroll()`` or from a replay buffer. It also has been used for
                the most recent ``update_with_gradient()``.
            train_info (nested Tensor): ``AlgStep.info`` returned by either
                ``rollout_step()`` (on-policy training) or ``train_step()``
                (off-policy training).
            loss_info (LossInfo): loss
            params (list[Parameter]): list of parameters with gradients
        """
        super(RLAlgorithm, self).summarize_train(experience, train_info,
                                                 loss_info, params)

        if self._debug_summaries:
            summary_utils.summarize_action(experience.action,
                                           self._action_spec)
            self.summarize_reward("training_reward", experience.reward)

        if self._config.summarize_action_distributions:
            field = alf.nest.find_field(train_info, 'action_distribution')
            if len(field) == 1:
                summary_utils.summarize_distribution("action_dist", field[0])

    @alf.configurable(whitelist=["custom_summary"])
    def summarize_play(
            self,
            experience: Experience,
            custom_summary: Optional[Callable[[Experience], None]] = None):
        """Generate summaries for play or evaluate.

        Args:
            experience: experience of one step rollout in the environment during
            play or evaluation, this is in contrast to the experience input to
            ``summarize_rollout`` where a sequence of multiple rollout steps
            are collected.
            custom_summary: when specified it is a function that will be called every
               time when this ``summarize_play`` hook is called. This provides
               a convenient way for the user to customize ``summarize_play`` from
               ALF configs.
        """
        if custom_summary is not None:
            custom_summary(experience)

    def summarize_metrics(self):
        """Generate summaries for metrics ``AverageEpisodeLength``,
        ``AverageReturn``, etc.
        """
        if not alf.summary.should_record_summaries():
            return

        if self._metrics:
            for metric in self._metrics:
                metric.gen_summaries(
                    train_step=alf.summary.get_global_counter(),
                    step_metrics=self._metrics[:2])

    # Subclass may override predict_step() to allow more efficient implementation
    def predict_step(self, inputs: TimeStep, state):
        r"""Predict for one step of observation.

        This only used for evaluation. So it only need to perform computations
        for generating action distribution.

        Args:
            time_step (TimeStep): Current observation and other inputs for computing
                action.
            state (nested Tensor): should be consistent with predict_state_spec
        Returns:
            AlgStep:
            - output (nested Tensor): should be consistent with ``action_spec``.
            - state (nested Tensor): should be consistent with ``predict_state_spec``.
        """
        policy_step = self.rollout_step(inputs, state)
        return policy_step._replace(info=())

    def _rollout_step(self, time_step: TimeStep, state):
        """A wrapper around user-defined ``rollout_step``. For every rl algorithm,
        this wrapper ensures that the rollout info spec will be computed.
        """
        policy_step = self._original_rollout_step(time_step, state)
        if self._rollout_info_spec is None:
            self._rollout_info_spec = dist_utils.extract_spec(policy_step.info)
        return policy_step

    @common.mark_rollout
    def unroll(self, unroll_length: int):
        if self._config.async_unroll:
            return self._async_unroll(unroll_length)
        else:
            return self._sync_unroll(unroll_length)

    def _async_unroll(self, unroll_length: int):
        if self._async_unroller is None:
            # env is no longer needed for rollout in the main process.
            # Close it to release resources.
            # self._env.close()
            self._async_unroller = AsyncUnroller(self, self._config)
        elif alf.summary.get_global_counter(
        ) % self._config.unroll_parameter_update_period == 0:
            self._async_unroller.update_parameter(self)

        assert not self._overwrite_policy_output, (
            "async_unroll does not "
            "support overwrite_policy_output")
        assert not self.on_policy, ("async_unroll does not support on-policy "
                                    "training")

        if self._current_transform_state is None:
            self._current_transform_state = self.get_initial_transform_state(
                self._env.batch_size)
        trans_state = self._current_transform_state

        experience_list = []
        original_reward_list = []
        env_step_time = 0.
        store_exp_time = 0.
        step_time = 0.
        max_step_time = 0.
        qsize = self._async_unroller.get_queue_size()
        unroll_results = self._async_unroller.gather_unroll_results(
            unroll_length, self._config.max_unroll_length)
        if self._rollout_info_spec is None and len(unroll_results) > 0:
            self._rollout_info_spec = dist_utils.extract_spec(
                unroll_results[0].policy_step.info)

        for unroll_result in unroll_results:
            time_step = unroll_result.time_step
            policy_step = unroll_result.policy_step
            policy_state = unroll_result.policy_state
            # Some data transformers contain parameters which are updated during
            # rollout (e.g. ObservationNormalizer and RewardNormalizer with
            # update_mode="rollout"). So we need to redo the transform_timestep
            # so that those parameters are correctly updated.
            transformed_time_step, trans_state = self.transform_timestep(
                time_step, trans_state)

            env_step_time += unroll_result.env_step_time
            step_time += unroll_result.step_time
            max_step_time = max(max_step_time, unroll_result.step_time)

            store_exp_time += self._process_unroll_step(
                policy_step, policy_step.output, time_step,
                transformed_time_step, policy_state, experience_list,
                original_reward_list)

        alf.summary.scalar(
            "time/unroll_env_step",
            env_step_time,
            average_over_summary_interval=True)
        alf.summary.scalar("time/unroll_store_exp", store_exp_time)
        if unroll_length == 0:
            alf.summary.scalar("async_unroll/unroll_length",
                               float(len(unroll_results)))
        alf.summary.scalar("async_unroll/queue_size", qsize)
        if not unroll_results:
            return None

        alf.summary.scalar("time/avg_unroll_step_time",
                           step_time / len(unroll_results))
        alf.summary.scalar("time/max_unroll_step_time", max_step_time)
        original_reward = alf.nest.utils.stack_nests(original_reward_list)
        self.summarize_reward("rollout_reward/original_reward",
                              original_reward)

        experience = alf.nest.utils.stack_nests(experience_list)
        experience = experience._replace(
            rollout_info=dist_utils.params_to_distributions(
                experience.rollout_info, self._rollout_info_spec))

        self._current_transform_state = common.detach(trans_state)

        return experience

    def _process_unroll_step(self, policy_step, action, time_step,
                             transformed_time_step, policy_state,
                             experience_list, original_reward_list):
        self.observe_for_metrics(time_step.cpu())
        exp = make_experience(time_step.cpu(),
                              alf.layers.to_float32(policy_step),
                              alf.layers.to_float32(policy_state))

        store_exp_time = 0
        if not self.on_policy:
            t0 = time.time()
            self.observe_for_replay(exp)
            store_exp_time = time.time() - t0

        exp_for_training = Experience(
            time_step=transformed_time_step,
            action=action,
            rollout_info=dist_utils.distributions_to_params(policy_step.info),
            state=policy_state)

        experience_list.append(exp_for_training)
        original_reward_list.append(time_step.reward)
        return store_exp_time

    def _sync_unroll(self, unroll_length: int):
        r"""Unroll ``unroll_length`` steps using the current policy.

        Because the ``self._env`` is a batched environment. The total number of
        environment steps is ``self._env.batch_size * unroll_length``.

        Args:
            unroll_length (int): number of steps to unroll.
        Returns:
            Experience: The stacked experience with shape :math:`[T, B, \ldots]`
            for each of its members.
        """
        if self._current_time_step is None:
            self._current_time_step = common.get_initial_time_step(self._env)
        if self._current_policy_state is None:
            self._current_policy_state = self.get_initial_rollout_state(
                self._env.batch_size)
        if self._current_transform_state is None:
            self._current_transform_state = self.get_initial_transform_state(
                self._env.batch_size)

        time_step = self._current_time_step
        policy_state = self._current_policy_state
        trans_state = self._current_transform_state

        experience_list = []
        original_reward_list = []
        initial_state = self.get_initial_rollout_state(self._env.batch_size)

        env_step_time = 0.
        store_exp_time = 0.
        for _ in range(unroll_length):
            policy_state = common.reset_state_if_necessary(
                policy_state, initial_state, time_step.is_first())
            transformed_time_step, trans_state = self.transform_timestep(
                time_step, trans_state)
            policy_step = self.rollout_step(transformed_time_step,
                                            policy_state)

            action = common.detach(policy_step.output)

            t0 = time.time()
            next_time_step = self._env.step(action)
            env_step_time += time.time() - t0

            # For typical cases, there is no impact since the action at the
            # current time step is the same as the prev_action of the next
            # time step. In some cases, for example, for data collection,
            # this step is useful for updating the action to be saved into
            # replay buffer with the actual action that is used (e.g. from
            # an expert), which can be recordered in next_time_step.prev_action.
            if self._overwrite_policy_output:
                policy_step = policy_step._replace(
                    output=next_time_step.prev_action)
            store_exp_time += self._process_unroll_step(
                policy_step, action, time_step, transformed_time_step,
                policy_state, experience_list, original_reward_list)

            time_step = next_time_step
            policy_state = policy_step.state

        alf.summary.scalar(
            "time/unroll_env_step",
            env_step_time,
            average_over_summary_interval=True)
        alf.summary.scalar("time/unroll_store_exp", store_exp_time)
        original_reward = alf.nest.utils.stack_nests(original_reward_list)
        self.summarize_reward("rollout_reward/original_reward",
                              original_reward)

        experience = alf.nest.utils.stack_nests(experience_list)
        experience = experience._replace(
            rollout_info=dist_utils.params_to_distributions(
                experience.rollout_info, self._rollout_info_spec))

        self._current_time_step = time_step
        # Need to detach so that the graph from this unroll is disconnected from
        # the next unroll. Otherwise backward() will report error for on-policy
        # training after the next unroll.
        self._current_policy_state = common.detach(policy_state)
        self._current_transform_state = common.detach(trans_state)

        return experience

    def train_iter(self):
        """Perform one iteration of training.

        Users may choose to implement their own ``train_iter()``.

        Returns:
            int: the number of samples being trained on (including duplicates).
        """
        assert self.on_policy is not None

        if self._config.empty_cache:
            torch.cuda.empty_cache()
        if self.on_policy:
            return self._train_iter_on_policy()
        else:
            return self._train_iter_off_policy()

    @data_distributed_when(lambda algorithm: algorithm.on_policy)
    def _compute_train_info_and_loss_info_on_policy(self, unroll_length):
        with record_time("time/unroll"):
            with torch.cuda.amp.autocast(self._config.enable_amp):
                experience = self.unroll(self._config.unroll_length)
            self.summarize_metrics()

        train_info = experience.rollout_info
        experience = experience._replace(rollout_info=())
        loss_info = self.calc_loss(train_info)
        self.summarize_rollout(experience)
        return train_info, loss_info, experience

    def _train_iter_on_policy(self):
        """User may override this for their own training procedure."""
        alf.summary.increment_global_counter()

        train_info, loss_info, experience = self._compute_train_info_and_loss_info_on_policy(
            self._config.unroll_length)

        with record_time("time/train"):
            valid_masks = (experience.step_type != StepType.LAST).to(
                torch.float32)
            loss_info, params = self.update_with_gradient(
                loss_info, valid_masks)
            self.after_update(experience.time_step, train_info)
            self.summarize_train(experience, train_info, loss_info, params)
            shape = alf.nest.get_nest_shape(experience)
            steps = shape[0] * shape[1]

        with record_time("time/after_train_iter"):
            root_inputs = experience.time_step if self._config.use_root_inputs_for_after_train_iter else None
            self.after_train_iter(root_inputs, train_info)

        return steps

    def _unroll_iter_off_policy(self):
        """Perform an unroll before the next off-policy training iteration.

        Returns:
            tuple:
            - unrolled: if the ``unroll()`` has been called. Sometimes for fractional
                unroll length, it may not have been called.
            - root_inputs: root-level time step returned by the unroll
            - rollout_info: rollout info returned by the unroll
        """
        config: TrainerConfig = self._config

        if not config.update_counter_every_mini_batch:
            alf.summary.increment_global_counter()

        unroll_length = self._remaining_unroll_length_fraction + config.unroll_length
        self._remaining_unroll_length_fraction = unroll_length - int(
            unroll_length)
        unroll_length = int(unroll_length)

        self._ensure_rollout_summary.tick()

        unrolled = False
        root_inputs = None
        rollout_info = None
        if (alf.summary.get_global_counter() >=
                self._rl_train_after_update_steps
                and (unroll_length > 0 or config.unroll_length == 0) and
            (config.num_env_steps == 0
             or self.get_step_metrics()[1].result() < config.num_env_steps)):
            unrolled = True
            with torch.set_grad_enabled(
                    config.unroll_with_grad), torch.cuda.amp.autocast(
                        config.enable_amp):
                with record_time("time/unroll"):
                    self.eval()
                    # The period of performing unroll may not be an integer
                    # divider of config.summary_interval if config.unroll_length is not an
                    # integer. In order to make sure the summary for unroll is
                    # still written out about every summary_interval steps, we
                    # need to remember whether summary has been written between
                    # two unrolls.
                    with self._ensure_rollout_summary:
                        experience = self.unroll(unroll_length)
                        if experience:
                            self.summarize_rollout(experience)
                            self.summarize_metrics()
                            rollout_info = experience.rollout_info
                            if config.use_root_inputs_for_after_train_iter:
                                root_inputs = experience.time_step
                            del experience
        return unrolled, root_inputs, rollout_info

    def _train_iter_off_policy(self):
        """User may override this for their own training procedure."""
        unrolled, root_inputs, rollout_info = self._unroll_iter_off_policy()

        # replay buffer may not have been created for two different reasons:
        # 1. in online RL training (``has_offline`` is False), unroll is not
        # performed yet. In this case, we simply return from here.
        # 2. in offline RL training case (``has_offline`` is True), there is no
        # online replay buffer. In this case, we move on and continue with the
        # offline training.
        if self._replay_buffer is None and not self.has_offline:
            return 0

        self.train()
        steps = self.train_from_replay_buffer(update_global_counter=True)

        if unrolled:
            with record_time("time/after_train_iter"):
                self.after_train_iter(root_inputs, rollout_info)

        # For now, we only return the steps of the primary algorithm's training
        return steps

    def load_offline_replay_buffer(self, untransformed_observation_spec):
        """Load replay buffer from a replay buffer checkpoint.
        It will construct a replay buffer (``self._offline_replay_buffer``)
        holding the data loaded from the checkpoint, which can be used for
        model training, e.g. in the hybrid training pipeline or in other ways.

        Args:
            untransformed_observation_spec (nested TensorSpec): spec that
                describes the structure of the utransformed observations.
        """

        if self._offline_buffer_dir is None or self._offline_buffer_dir == "":
            # no offline buffer is provided
            return
        else:
            logging.info('------offline replay buffer loading started------')

            offline_buffer_dir_list = common.as_list(self._offline_buffer_dir)

            def _get_full_key(dict, partial_key):
                full_key = next((key for key in dict if partial_key in key),
                                None)
                assert full_key is not None, (
                    "key containing {} "
                    "is not found.".format(partial_key))
                return full_key

            # pre-calculate the individual and total buffer length
            if self._config.offline_buffer_length is None:
                buffer_lens = []
                for buffer_dir in offline_buffer_dir_list:
                    map_location = None
                    if not torch.cuda.is_available():
                        map_location = torch.device('cpu')
                    replay_buffer_checkpoint = torch.load(
                        buffer_dir, map_location=map_location)

                    buffer_dict = replay_buffer_checkpoint['algorithm']
                    reward_key = _get_full_key(buffer_dict, "time_step|reward")
                    replay_buffer_length = buffer_dict[reward_key].shape[1]
                    buffer_lens.append(replay_buffer_length)
            else:
                buffer_lens = ([self._config.offline_buffer_length] *
                               len(offline_buffer_dir_list))

            total_replay_buffer_length = sum(buffer_lens)

            for i, buffer_dir in enumerate(offline_buffer_dir_list):
                map_location = None
                if not torch.cuda.is_available():
                    map_location = torch.device('cpu')
                replay_buffer_checkpoint = torch.load(
                    buffer_dir, map_location=map_location)

                buffer_dict = replay_buffer_checkpoint['algorithm']

                # prepare specs for buffer resonctruction
                reward_key = _get_full_key(buffer_dict, "time_step|reward")
                step_type_key = _get_full_key(buffer_dict,
                                              "time_step|step_type")
                discount_key = _get_full_key(buffer_dict, "time_step|discount")
                env_id_key = _get_full_key(buffer_dict, "time_step|env_id")

                env_batch_size = buffer_dict[reward_key].shape[0]
                replay_buffer_length = buffer_dict[reward_key].shape[1]

                step_type_spec = dist_utils.extract_spec(
                    buffer_dict[step_type_key], from_dim=2)
                reward_spec = dist_utils.extract_spec(
                    buffer_dict[reward_key], from_dim=2)
                discount_spec = dist_utils.extract_spec(
                    buffer_dict[discount_key], from_dim=2)

                env_id_spec = dist_utils.extract_spec(
                    buffer_dict[env_id_key], from_dim=2)

                time_step_spec = TimeStep(
                    step_type=step_type_spec,
                    reward=reward_spec,
                    discount=discount_spec,
                    observation=untransformed_observation_spec,
                    prev_action=self._action_spec,
                    env_id=env_id_spec)

                exp_spec_wo_info = Experience(
                    time_step=time_step_spec, action=self._action_spec)

                # assumes a typical Agent structure
                exp_spec = Experience(
                    time_step=time_step_spec,
                    action=self._action_spec,
                    rollout_info=BasicRolloutInfo(
                        rl=BasicRLInfo(action=self._action_spec),
                        rewards={},
                        repr={},
                    ))
                self._offline_experience_spec = exp_spec

                self._populate_offline_replay_buffer(
                    exp_spec, exp_spec_wo_info, buffer_lens[i],
                    total_replay_buffer_length, env_batch_size,
                    replay_buffer_checkpoint)

        logging.info('------loading completed; total_size '
                     '{}------'.format(
                         self._offline_replay_buffer.total_size.item()))

    def _populate_offline_replay_buffer(
            self, exp_spec, exp_spec_wo_info, number_of_samples,
            total_replay_buffer_length, env_batch_size,
            replay_buffer_checkpoint):
        """Initialize the experience replay buffer from a offline replay buffer
        checkpoint. It will construct ``_offline_replay_buffer`` if it is not
        constructed yet. Then the first ``number_of_samples`` data samples from
        ``replay_buffer_checkpoint`` will be added to the
        ``_offline_replay_buffer``.
        TODO: a non-sequential version.

        Args:
            exp_spec (nested spec): spec for the ``Experience`` structure.
            exp_spec_wo_info (nested spec): spec for the ``Experience`` structure
                without the rollout_info field.
            number_of_samples (int): max number of samples to be added to the
                ``_offline_replay_buffer`` from the ``replay_buffer_checkpoint``.
            total_replay_buffer_length (int): the full length of the
                ``_offline_replay_buffer``. Used for constructing the buffer.
            env_batch_size (int): environment batch size
            replay_buffer_checkpoint (dict): the buffer dictionary loaded from
                the saved checkpoint file.
        """

        if self._offline_replay_buffer is None:
            self._offline_replay_buffer = ReplayBuffer(
                data_spec=exp_spec,
                num_environments=env_batch_size,
                max_length=total_replay_buffer_length,
                prioritized_sampling=self._prioritized_sampling,
                num_earliest_frames_ignored=self._num_earliest_frames_ignored,
                name=f'{self._name}_offline_replay_buffer')

        # prepare data for re-loading
        # 1) filter out irrelevant items (this is algorithm dependent)
        replay_buffer_from_ckpt = replay_buffer_checkpoint['algorithm']
        buffer_dict = {}
        for name, buf in replay_buffer_from_ckpt.items():
            # the actual action not the rollout.action
            # and also not prev_action
            if 'action' in name and (not 'rollout_info' in name
                                     and not 'prev_action' in name):
                buffer_dict[name] = buf
            elif ('time_step|prev_action' in name
                  or 'time_step|env_id' in name):
                buffer_dict[name] = buf
            elif ('time_step|step_type' in name or 'time_step|reward' in name
                  or 'time_step|discount' in name
                  or 'time_step|observation' in name
                  or 'time_step|prev_action' in name
                  or 'time_step|env_id' in name):
                buffer_dict[name] = buf

        # 2) pack nest
        flat_buffer = list(buffer_dict.values())
        buffer_dict = alf.nest.pack_sequence_as(exp_spec_wo_info, flat_buffer)

        # 3) wrap as experience
        time_step_dict = buffer_dict.time_step
        time_step = TimeStep(
            step_type=time_step_dict.step_type,
            reward=time_step_dict.reward,
            discount=time_step_dict.discount,
            observation=time_step_dict.observation,
            prev_action=time_step_dict.prev_action,
            env_id=time_step_dict.env_id)

        exp = Experience(
            time_step=time_step,
            action=buffer_dict.action,
            rollout_info=BasicRolloutInfo(
                rl=BasicRLInfo(action=buffer_dict.action),
                rewards={},
                repr={},
            ))

        # load data
        def _load_data(exp):
            """
            For the sync driver, `exp` has the shape (`env_batch_size`, ...)
            with `num_envs`==1 and `unroll_length`==1.
            """
            outer_rank = alf.nest.utils.get_outer_rank(exp, exp_spec)

            if outer_rank == 2:
                # The shape is [env_batch_size, mini_batch_length, ...], where
                # mini_batch_length denotes the length of the mini_batch
                for t in range(min(number_of_samples, exp.step_type.shape[1])):
                    bat = alf.nest.map_structure(lambda x: x[:, t, ...], exp)
                    self._offline_replay_buffer.add_batch(bat, bat.env_id)
            else:
                raise ValueError(
                    "Unsupported outer rank %s of `exp`" % outer_rank)

        _load_data(exp)

    def finish_train(self):
        """Finish training and release resources if necessary."""
        if self._async_unroller is not None:
            self._async_unroller.close()
            self._async_unroller = None
