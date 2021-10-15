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
import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Callable
from absl import logging

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, Experience, make_experience, TimeStep
from alf.utils import common, dist_utils, summary_utils
from alf.utils.summary_utils import record_time
from alf.tensor_specs import TensorSpec
from .config import TrainerConfig


class _UnrollPerformer(torch.nn.Module):
    """Wraps RLAlgorithm.unroll() as a forward()

    In fact, UnrollPerformer.forward() is just a delegation to
    RLAlgorithm.unroll(). The reason that we need this wrapper is to trick DDP
    to add necessary callbacks to the unroll process. DDP wrapper only
    recognizes and hijacks the method named "forward()".

    """

    def __init__(self, algorithm: 'RLAlgorithm'):
        """Construct the unroll() wrapper

        Args:
            algorithm (RLAlgorithm): an RLAlgorithm instance whose unroll() is
                wrapped by UnrollPerformer's forward().
        """
        super().__init__()
        self.inner_algorithm = algorithm

        # DDP will panic if the wrapped module (i.e. UnrollPerformer here) has
        # member in its state_dict() that is not a Tensor. Here such state_dict
        # members are picked and thrown into _ddp_params_and_buffers_to_ignore.
        # By contract this implicitly instruct DDP wrapper to not include them
        # in its parameter/buffer synchronization.
        self._ddp_params_and_buffers_to_ignore = []
        for name, value in self.state_dict().items():
            if type(value) is not torch.Tensor:
                self._ddp_params_and_buffers_to_ignore.append(name)

    def forward(self, unroll_length: int):
        """This is simply a delegation to the underlying algorithm's unroll()

        Args:
            unroll_length (int): the number of steps to unroll
        """
        return self.inner_algorithm._unroll(unroll_length)


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
        adjusted = config.unroll_length + 1

    # The replay buffer length is exteneded by num_earliest_frames_ignored so
    # that after FrameStacker transformation the number of experiences matches
    # ``unroll_length``.
    adjusted += num_earliest_frames_ignored

    logging.info(f'Actual replay buffer length is adjusted to {adjusted}.')

    return adjusted


@alf.configurable
class RLAlgorithm(Algorithm):
    """Abstract base class for RL Algorithms.

    ``RLAlgorithm`` provide basic functions and generic interface for rl algorithms.

    The key interface functions are:

    1. ``predict_step()``: one step of computation of action for evaluation.
    2. ``rollout_step()``: one step of computation for rollout. It is
       used for collecting experiences during training. Different from
       ``predict_step``, ``rollout_step`` may include addtional computations for
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
            debug_summaries=debug_summaries,
            name=name)

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

        # When in DDP (distributed data parallel) mode, self._unroll_performer
        # will be set to a DDP wrapped nn.odule that does the unroll(), so that
        # gradients derived in the next backward() will be aggregated and
        # syncrhonized across all DDP processes. The caller decides whether to
        # activate the DDP (as a result the unroll performer too) by calling
        # RLAlgorithm.activate_ddp().
        self._unroll_performer = None
        self._rollout_info_spec = None

        self._current_time_step = None
        self._current_policy_state = None
        self._current_transform_state = None

        if self._env is not None and not self.on_policy:
            replay_buffer_length = adjust_replay_buffer_length(
                config, self._num_earliest_frames_ignored)

            if config.whole_replay_buffer_training:
                # For whole replay buffer training, we would like to be sure
                # that the replay buffer have enough samples in it to perform
                # the training, which will most likely happen in the 2nd
                # iteration. The minimum_initial_collect_steps guarantees that.
                minimum_initial_collect_steps = replay_buffer_length * self._env.batch_size
                if config.initial_collect_steps < minimum_initial_collect_steps:
                    logging.warn(
                        'Set the initial_collect_steps to minimum required '
                        f'value {minimum_initial_collect_steps} because '
                        'whole_replay_buffer_training is on.')
                    config.initial_collect_steps = minimum_initial_collect_steps

            self.set_replay_buffer(self._env.batch_size, replay_buffer_length,
                                   config.priority_replay)

        env = self._env
        if env is not None:
            metric_buf_size = max(self._config.metric_min_buffer_size,
                                  self._env.batch_size)
            self._metrics = [
                alf.metrics.NumberOfEpisodes(),
                alf.metrics.EnvironmentSteps(),
                alf.metrics.AverageReturnMetric(
                    batch_size=env.batch_size,
                    buffer_size=metric_buf_size,
                    reward_shape=reward_spec.shape),
                alf.metrics.AverageEpisodeLengthMetric(
                    batch_size=env.batch_size, buffer_size=metric_buf_size),
                alf.metrics.AverageEnvInfoMetric(
                    example_env_info=env.reset().env_info,
                    batch_size=env.batch_size,
                    buffer_size=metric_buf_size),
                alf.metrics.AverageDiscountedReturnMetric(
                    batch_size=env.batch_size,
                    buffer_size=metric_buf_size,
                    reward_shape=reward_spec.shape)
            ]

        self._original_rollout_step = self.rollout_step
        self.rollout_step = self._rollout_step

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

    def activate_ddp(self, rank: int):
        """Prepare the RLAlgorithm with DistributedDataParallel wrapper

        Note that RLAlgorithm does not need to remember the rank of the device.

        Args:
            rank (int): DDP wrapper needs to know on which GPU device this
                module's parameters and buffers are supposed to be.
        """
        # The DDP wrapped module is still a module. To prevent it from being
        # taken into the state_dict of this RLAlgorithm, it is put into a tuple
        # to avoid triggering automatic state_dict inclusion by __setattr__.
        self._unroll_performer = (DDP(
            _UnrollPerformer(self), device_ids=[rank]), )

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
                alf.summary.scalar(name + "/mean", torch.mean(rewards))
            else:
                for i in range(rewards.shape[2]):
                    r = rewards[..., i]
                    alf.summary.histogram('%s/%s/value' % (name, i), r)
                    alf.summary.scalar('%s/%s/mean' % (name, i), torch.mean(r))

    def summarize_rollout(self, experience):
        """Generate summaries for rollout.

        Args:
            experience (Experience): experience collected from ``rollout_step()``.
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
                summary_utils.summarize_action_dist(
                    action_distributions=field[0], name="rollout_action_dist")

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
            if not self.on_policy:
                self.summarize_reward("training_reward", experience.reward)

        if self._config.summarize_action_distributions:
            field = alf.nest.find_field(train_info, 'action_distribution')
            if len(field) == 1:
                summary_utils.summarize_action_dist(field[0])

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

    def unroll(self, unroll_length: int):
        """Unroll ``unroll_length`` steps using the current policy.

        This is a delegation over _unroll() which decides whether to call it
        with DDP wrapper on or off.

        """
        if self._unroll_performer is not None:
            # If DDP is on, self._unroll_performer will be activated. In this
            # case, call its foward().
            return self._unroll_performer[0](unroll_length)
        else:
            return self._unroll(unroll_length)

    @common.mark_rollout
    def _unroll(self, unroll_length: int):
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
            # save the untransformed time step in case that sub-algorithms need
            # to store it in replay buffers
            transformed_time_step = transformed_time_step._replace(
                untransformed=time_step)
            policy_step = self.rollout_step(transformed_time_step,
                                            policy_state)
            # release the reference to ``time_step``
            transformed_time_step = transformed_time_step._replace(
                untransformed=())

            action = common.detach(policy_step.output)

            t0 = time.time()
            next_time_step = self._env.step(action)
            env_step_time += time.time() - t0

            self.observe_for_metrics(time_step.cpu())

            exp = make_experience(time_step.cpu(), policy_step, policy_state)

            t0 = time.time()
            self.observe_for_replay(exp)
            store_exp_time += time.time() - t0

            exp_for_training = Experience(
                action=action,
                reward=transformed_time_step.reward,
                discount=transformed_time_step.discount,
                step_type=transformed_time_step.step_type,
                state=policy_state,
                prev_action=transformed_time_step.prev_action,
                observation=transformed_time_step.observation,
                rollout_info=dist_utils.distributions_to_params(
                    policy_step.info),
                env_id=transformed_time_step.env_id)

            experience_list.append(exp_for_training)
            original_reward_list.append(time_step.reward)
            time_step = next_time_step
            policy_state = policy_step.state

        alf.summary.scalar("time/unroll_env_step", env_step_time)
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

        if self.on_policy:
            return self._train_iter_on_policy()
        else:
            return self._train_iter_off_policy()

    def _train_iter_on_policy(self):
        """User may override this for their own training procedure."""
        alf.summary.increment_global_counter()

        with record_time("time/unroll"):
            experience = self.unroll(self._config.unroll_length)
            self.summarize_metrics()

        with record_time("time/train"):
            train_info = experience.rollout_info
            experience = experience._replace(rollout_info=())
            steps = self.train_from_unroll(experience, train_info)

        with record_time("time/after_train_iter"):
            self.after_train_iter(experience, train_info)

        return steps

    def _train_iter_off_policy(self):
        """User may override this for their own training procedure."""
        config: TrainerConfig = self._config

        if not config.update_counter_every_mini_batch:
            alf.summary.increment_global_counter()

        with torch.set_grad_enabled(config.unroll_with_grad):
            with record_time("time/unroll"):
                self.eval()
                experience = self.unroll(config.unroll_length)
                self.summarize_rollout(experience)
                self.summarize_metrics()

        self.train()
        steps = self.train_from_replay_buffer(update_global_counter=True)

        with record_time("time/after_train_iter"):
            train_info = experience.rollout_info
            experience = experience._replace(rollout_info=())
            self.after_train_iter(experience, train_info)

        # For now, we only return the steps of the primary algorithm's training
        return steps
