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
import psutil
import time
import torch
from typing import Callable

import gin

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, Experience, make_experience, TimeStep
from alf.utils import common, dist_utils, summary_utils, math_ops
from .config import TrainerConfig


@gin.configurable
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
                 predict_state_spec=None,
                 rollout_state_spec=None,
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
            rollout_state_spec (nested TensorSpec): for the network state of
                ``predict_step()``. If None, it's assumed to be the same as
                ``train_state_spec``.
            predict_state_spec (nested TensorSpec): for the network state of
                ``predict_step()``. If None, it's assumed to be the same as
                ``rollout_state_spec``.
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
            optimizer=optimizer,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        self._env = env
        self._observation_spec = observation_spec
        self._action_spec = action_spec

        self._proc = psutil.Process(os.getpid())

        self._rollout_info_spec = None

        self._current_time_step = None
        self._current_policy_state = None
        self._current_transform_state = None

        if self._env is not None and not self.is_on_policy():
            if config.whole_replay_buffer_training and config.clear_replay_buffer:
                replayer = "one_time"
            else:
                replayer = "uniform"
            self.set_exp_replayer(replayer, self._env.batch_size,
                                  config.replay_buffer_length,
                                  config.priority_replay)

        env = self._env
        if env is not None:
            metric_buf_size = max(10, self._env.batch_size)
            self._metrics = [
                alf.metrics.NumberOfEpisodes(),
                alf.metrics.EnvironmentSteps(),
                alf.metrics.AverageReturnMetric(
                    batch_size=env.batch_size, buffer_size=metric_buf_size),
                alf.metrics.AverageEpisodeLengthMetric(
                    batch_size=env.batch_size, buffer_size=metric_buf_size),
                alf.metrics.AverageEnvInfoMetric(
                    example_env_info=env.reset().env_info,
                    batch_size=env.batch_size,
                    buffer_size=metric_buf_size)
            ]

        self._original_rollout_step = self.rollout_step
        self.rollout_step = self._rollout_step

    def is_rl(self):
        """Always return True for RLAlgorithm."""
        return True

    @abstractmethod
    def is_on_policy(self):
        """Whehter this algorithm is an on-policy algorithm.

        If it's on-policy algorihtm, ``train_iter()`` will use
        ``_train_iter_on_policy()`` to train. Otherwise, it will use
        ``_train_iter_off_policy()``.
        """
        pass

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

        mem = self._proc.memory_info().rss // 1e6
        alf.summary.scalar(name='memory/cpu', data=mem)
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() // 1e6
            alf.summary.scalar(name='memory/gpu_allocated', data=mem)
            mem = torch.cuda.memory_reserved() // 1e6
            alf.summary.scalar(name='memory/gpu_reserved', data=mem)
            mem = torch.cuda.max_memory_allocated() // 1e6
            alf.summary.scalar(name='memory/max_gpu_allocated', data=mem)
            mem = torch.cuda.max_memory_reserved() // 1e6
            alf.summary.scalar(name='memory/max_gpu_reserved', data=mem)
            torch.cuda.reset_max_memory_allocated()
            # TODO: consider using torch.cuda.empty_cache() to save memory.

    # Subclass may override predict_step() to allow more efficient implementation
    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        r"""Predict for one step of observation.

        This only used for evaluation. So it only need to perform computations
        for generating action distribution.

        Args:
            time_step (TimeStep): Current observation and other inputs for computing
                action.
            state (nested Tensor): should be consistent with predict_state_spec
            epsilon_greedy (float): a floating value in :math:`[0,1]`, representing
                the chance of action sampling instead of taking argmax.
                This can help prevent a dead loop in some deterministic environment
                like `Breakout`.
        Returns:
            AlgStep:
            - output (nested Tensor): should be consistent with ``action_spec``.
            - state (nested Tensor): should be consistent with ``predict_state_spec``.
        """
        policy_step = self.rollout_step(time_step, state)
        return policy_step._replace(info=())

    def _rollout_step(self, time_step: TimeStep, state):
        """A wrapper around user-defined ``rollout_step``. For every rl algorithm,
        this wrapper ensures that the rollout info spec will be computed.
        """
        policy_step = self._original_rollout_step(time_step, state)
        if self._rollout_info_spec is None:
            self._rollout_info_spec = dist_utils.extract_spec(policy_step.info)
        return policy_step

    @abstractmethod
    def rollout_step(self, time_step: TimeStep, state):
        """Perform one step of rollout.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            time_step (TimeStep):
            state (nested Tensor): should be consistent with ``train_state_spec``.

        Returns:
            AlgStep:
            - output (nested Tensor): should be consistent with ``action_spec``.
            - state (nested Tensor): should be consistent with ``train_state_spec``.
            - info (nested Tensor): everything necessary for training. Note that
              ("action", "reward", "discount", "is_last") are automatically
              collected in ``unroll()``. So the user only need to put other
              stuff (e.g. value estimation) into ``policy_step.info``.
        """
        pass

    @abstractmethod
    def train_step(self, experience: Experience, state):
        """Perform one step of training computation.

        Args:
            experience (Experience):
            state (nested Tensor): should be consistent with ``train_state_spec``.

        Returns:
            AlgStep:
            - output (nested Tensor): should be consistent with ``action_spec``.
            - state (nested Tensor): should be consistent with ``train_state_spec``.
            - info (nested Tensor): everything necessary for training.
        """
        pass

    @abstractmethod
    def calc_loss(self, experience, train_info):
        """Calculate the loss for each step.

        ``calc_loss()`` does not need to mask out the loss at invalid steps as
        ``train_iter()`` will apply the mask automatically.

        Args:
            experience (Experience): experiences collected from the most recent
                ``unroll()`` or from a replay buffer. It's used for the most
                recent ``update_with_gradient()``.
            train_info (nest): information collected for training.
                It is batched from each ``AlgStep.info`` returned by
                ``rollout_step()`` (on-policy training) or ``train_step()``
                (off-policy training).

        Returns:
            LossInfo: loss at each time step for each sample in the batch. The
            shapes of the tensors in it should be :math:`[T, B]`.
        """
        pass

    @common.mark_rollout
    def unroll(self, unroll_length):
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

            if self._exp_replayer_type == "one_time":
                exp = make_experience(time_step, policy_step, policy_state)
            else:
                exp = make_experience(time_step.cpu(), policy_step,
                                      policy_state)

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
            time_step = next_time_step
            policy_state = policy_step.state

        alf.summary.scalar("time/unroll_env_step", env_step_time)
        alf.summary.scalar("time/unroll_store_exp", store_exp_time)
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
            int:
            - number of samples being trained on (including duplicates).
        """
        if self.is_on_policy():
            return self._train_iter_on_policy()
        else:
            return self._train_iter_off_policy()

    def _train_iter_on_policy(self):
        """Implemented in ``OnPolicyAlgorithm``."""
        raise NotImplementedError()

    def _train_iter_off_policy(self):
        """Implemented in ``OffPolicyAlgorithm``."""
        raise NotImplementedError()
