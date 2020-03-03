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
from collections import Iterable
import os
import psutil
import time
import torch
from typing import Callable

import gin

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, Experience, make_experience, TimeStep, TrainingInfo
from alf.utils import common, dist_utils, summary_utils
from alf.experience_replayers.experience_replay import (
    OnetimeExperienceReplayer, SyncUniformExperienceReplayer)
from .config import TrainerConfig


@gin.configurable
class RLAlgorithm(Algorithm):
    """Abstract base class for RL Algorithms.

    RLAlgorithm provide basic functions and generic interface for rl algorithms.

    The key interface functions are:
    1. predict_step(): one step of computation of action for evaluation.
    2. rollout_step(): one step of computation for rollout. `rollout_step()` is
       used for collecting experiences during training. Different from
       `predict_step`, `rollout_step` may include addtional computations for
       training. For on-policy algorithms (e.g., AC, PPO, etc), the collected
       experiences will be immediately used to update parameters after one
       rollout (multiple rollout steps) is performed; for off-policy algorithms
       (e.g., SAC, DDPG, etc), these collected experiences will be put into a
       replay buffer.
    3. train_step(): only used for off-policy training. The training data are
       sampled from the replay buffer filled by `rollout_step()`.
    4. train_iter(): perform one iteration of training (rollout [and train]).
       `train_iter()` are called `num_iterations` time by `Trainer`.
       We provide a default implementation. Users can choose to implement
       their own `train_iter()`.
    5. update_with_gradient(): Do one gradient update based on the loss. It is
       used by the default `train_iter()` implementation. You can override to
       implement your own `update_with_gradient()`.
    6. calc_loss(): calculate loss based the training_info collected from
       `rollout_step()` or `train_step()`. It is used by the default
       implementation of `train_iter()`. If you want to use the default
       `train_iter()`, you need to implement `calc_loss()`.
    7. after_update(): called by `train_iter()` after every call to
       `update_with_gradient()`, mainly for some postprocessing steps such as
        copying a training model to a target model in SAC or DQN.
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
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 reward_shaping_fn: Callable = None,
                 observation_transformer=common.cast_transformer,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 summarize_action_distributions=False,
                 name="RLAlgorithm"):
        """Create a RLAlgorithm.

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            train_state_spec (nested TensorSpec): for the network state of
                `train_step()`.
            rollout_state_spec (nested TensorSpec): for the network state of
                `predict_step()`. If None, it's assumed to be the same as
                `train_state_spec`.
            predict_state_spec (nested TensorSpec): for the network state of
                `predict_step()`. If None, it's assumed to be the same as
                `rollout_state_spec`.
            env (Environment): The environment to interact with. `env` is a
                batched environment, which means that it runs multiple
                simulations simultaneously. Running multiple environments in
                parallel is crucial to on-policy algorithms as it increases the
                diversity of data and decreases temporal correlation. `env` only
                needs to be provided to the root `Algorithm`.
            config (TrainerConfig): config for training. `config` only needs to
                be provided to the algorithm which performs `train_iter()` by
                itself.
            optimizer (torch.optim.Optimizer): The default optimizer for training.
            reward_shaping_fn (Callable): a function that transforms extrinsic
                immediate rewards.
            observation_transformer (Callable | list[Callable]): transformation(s)
                applied to `time_step.observation`.
            debug_summaries (bool): If True, debug summaries will be created.
            name (str): Name of this algorithm.
            summarize_grads_and_vars (bool): If True, gradient and network
                variable summaries will be written during training.
            summarize_action_distributions (bool): If True, generate summaries
                for the action distributions.
        """
        super(RLAlgorithm, self).__init__(
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec,
            predict_state_spec=predict_state_spec,
            optimizer=optimizer,
            gradient_clipping=gradient_clipping,
            clip_by_global_norm=clip_by_global_norm,
            debug_summaries=debug_summaries,
            name=name)

        self._env = env
        self._config = config
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._reward_shaping_fn = reward_shaping_fn
        if isinstance(observation_transformer, Iterable):
            observation_transformers = list(observation_transformer)
        else:
            observation_transformers = [observation_transformer]
        self._observation_transformers = observation_transformers
        self._proc = psutil.Process(os.getpid())
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._summarize_action_distributions = summarize_action_distributions
        self._use_rollout_state = False

        self._rollout_info_spec = None
        self._train_info_spec = None
        self._processed_experience_spec = None

        self._current_time_step = None
        self._current_policy_state = None

        self._observers = []

        self._exp_replayer = None
        self._exp_replayer_type = None
        if self._env is not None and not self.is_on_policy():
            self.set_exp_replayer("uniform", self._env.batch_size,
                                  config.replay_buffer_length)

        self._metrics = []
        env = self._env
        if env is not None:
            metric_buf_size = max(10, self._env.batch_size)
            standard_metrics = [
                alf.metrics.NumberOfEpisodes(),
                alf.metrics.EnvironmentSteps(),
                alf.metrics.AverageReturnMetric(
                    batch_size=env.batch_size, buffer_size=metric_buf_size),
                alf.metrics.AverageEpisodeLengthMetric(
                    batch_size=env.batch_size, buffer_size=metric_buf_size),
            ]
            self._metrics = standard_metrics
            self._observers.extend(self._metrics)

        if config:
            self.use_rollout_state = config.use_rollout_state

    def _set_children_property(self, property_name, value):
        """Set the property named `property_name` in child RLAlgorithm to `value`."""
        for child in self._get_children():
            if isinstance(child, RLAlgorithm):
                child.__setattr__(property_name, value)

    @abstractmethod
    def is_on_policy(self):
        """Whehter this algorithm is an on-policy algorithm.

        If it's on-policy algorihtm, train_iter() will use
         _train_iter_on_policy() to train. Otherwise, it will use
        _train_iter_off_policy()
        """
        pass

    @property
    def use_rollout_state(self):
        return self._use_rollout_state

    @use_rollout_state.setter
    def use_rollout_state(self, flag):
        self._use_rollout_state = flag
        self._set_children_property('use_rollout_state', flag)

    def need_full_rollout_state(self):
        """Whether AlgStep.state from rollout_step should be full.

        If True, it means that rollout_step() should return the complete state
        for train_step().
        """
        return self._is_rnn and self._use_rollout_state

    @property
    def observation_spec(self):
        """Return the observation spec."""
        return self._observation_spec

    @property
    def rollout_info_spec(self):
        """The spec for the AlgStep.info returned from rollout_step()."""
        assert self._rollout_info_spec is not None, (
            "rollout_step() has not "
            " been used. rollout_info_spec is not available.")
        return self._rollout_info_spec

    @property
    def experience_spec(self):
        """Spec for experience."""
        policy_step_spec = AlgStep(
            output=self.action_spec,
            state=self.train_state_spec,
            info=self.rollout_info_spec)
        exp_spec = make_experience(self.time_step_spec, policy_step_spec,
                                   policy_step_spec.state)
        if not self._use_rollout_state:
            exp_spec = exp_spec._replace(state=())
        return exp_spec

    @property
    def time_step_spec(self):
        """Return spec for ActionTimeStep."""
        return TimeStep(
            step_type=alf.TensorSpec((), 'int32'),
            reward=alf.TensorSpec((), 'float32'),
            discount=alf.TensorSpec((), 'float32'),
            observation=self.observation_spec,
            prev_action=self.action_spec,
            env_id=alf.TensorSpec((), 'int32'))

    @property
    def action_spec(self):
        """Return the action spec."""
        return self._action_spec

    @property
    def exp_observers(self):
        """Return experience observers."""
        return self._exp_observers

    def get_step_metrics(self):
        """Get step metrics that used for generating summaries against

        Returns:
             list[StepMetric]: step metrics `EnvironmentSteps` and `NumberOfEpisodes`
        """
        return self._metrics[:2]

    def get_metrics(self):
        """Returns the metrics monitored by this driver.

        Returns:
            list[StepMetric]
        """
        return self._metrics

    def set_summary_settings(self,
                             summarize_grads_and_vars=False,
                             summarize_action_distributions=False):
        """Set summary flags."""
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._summarize_action_distributions = summarize_action_distributions

    def summarize_reward(self, name, rewards):
        if self._debug_summaries:
            alf.summary.histogram(name + "/value", rewards)
            alf.summary.scalar(name + "/mean", torch.mean(rewards))

    def add_experience_observer(self, observer: Callable):
        """Add an observer to receive experience.

        Args:
            observer (Callable): callable which accept Experience as argument.
        """

    def set_exp_replayer(self, exp_replayer: str, num_envs, max_length: int):
        """Set experience replayer.

        Args:
            exp_replayer (str): type of experience replayer. One of ("one_time",
                "uniform")
            num_envs (int): the total number of environments from all batched
                environments.
            max_length (int): the maximum number of steps the replay
                buffer store for each environment.
        """
        assert exp_replayer in ("one_time", "uniform"), (
            "Unsupported exp_replayer: %s" % exp_replayer)
        self._exp_replayer_type = exp_replayer
        self._exp_replayer_num_envs = num_envs
        self._exp_replayer_length = max_length

    def _set_exp_replayer(self, exp_replayer: str, num_envs):
        if exp_replayer == "one_time":
            self._exp_replayer = OnetimeExperienceReplayer()
        elif exp_replayer == "uniform":
            exp_spec = dist_utils.to_distribution_param_spec(
                self.experience_spec)
            self._exp_replayer = SyncUniformExperienceReplayer(
                exp_spec, self._exp_replayer_num_envs,
                self._exp_replayer_length)
        else:
            raise ValueError("invalid experience replayer name")
        self._observers.append(self._exp_replayer.observe)

    def observe(self, exp: Experience):
        """An algorithm can override to manipulate experience.

        Args:
            exp (Experience): The shapes can be either [Q, T, B, ...] or
                [B, ...], where Q is `learn_queue_cap` in `AsyncOffPolicyDriver`,
                T is the sequence length, and B is the batch size of the batched
                environment.
        """
        if not self._use_rollout_state:
            exp = exp._replace(state=())
        exp = dist_utils.distributions_to_params(exp)

        if self._exp_replayer is None and self._exp_replayer_type:
            self._set_exp_replayer(self._exp_replayer_type,
                                   self._config.num_envs)

        for observer in self._observers:
            observer(exp)

    def summarize_rollout(self, training_info):
        """Generate summaries for rollout.

        Note that training_info.info is empty here. Should use
        training_info.rollout_info to generate the summaries.

        Args:
            training_info (TrainingInfo): TrainingInfo structure collected from
                `rollout_step()`.
        Returns:
            None
        """
        if self._debug_summaries:
            summary_utils.summarize_action(training_info.action,
                                           self._action_spec, "rollout_action")
            self.summarize_reward("rollout_reward/extrinsic",
                                  training_info.reward)

        if self._config.summarize_action_distributions:
            field = alf.nest.find_field(training_info.rollout_info,
                                        'action_distribution')
            if len(field) == 1:
                summary_utils.summarize_action_dist(
                    action_distributions=field[0],
                    action_specs=self._action_spec,
                    name="rollout_action_dist")

    def summarize_train(self, training_info, loss_info, params):
        """Generate summaries for training & loss info.

        For on-policy algorithms, training_info.info is available.
        For off-policy alogirthms, both training_info.info and training_info.rollout_info
        are available. However, the statistics in these two structures are for
        the data sampled from the replay buffer. They store the update-to-date
        model outputs and the historical model outputs (on the past rollout data),
        respectively. They do not represent the model outputs on the current
        on-going rollout.

        Args:
            training_info (TrainingInfo): TrainingInfo structure collected from
                `rollout_step` (on-policy training) or `train_step` (off-policy
                training).
            loss_info (LossInfo): loss
            params (list[Parameter]): list of parameters with gradients
        Returns:
            None
        """
        if self._config.summarize_grads_and_vars:
            summary_utils.summarize_variables(params)
            summary_utils.summarize_gradients(params)
        if self._debug_summaries:
            summary_utils.summarize_action(training_info.action,
                                           self._action_spec)
            summary_utils.summarize_loss(loss_info)

        if self._config.summarize_action_distributions:
            field = alf.nest.find_field(training_info.info,
                                        'action_distribution')
            if len(field) == 1:
                summary_utils.summarize_action_dist(field[0],
                                                    self._action_spec)

    def summarize_metrics(self):
        """Generate summaries for metrics `AverageEpisodeLength`, `AverageReturn`..."""
        if self._metrics:
            for metric in self._metrics:
                metric.gen_summaries(
                    train_step=alf.summary.get_global_counter(),
                    step_metrics=self._metrics[:2])

        mem = self._proc.memory_info().rss // 1e6
        alf.summary.scalar(name='memory_usage', data=mem)

    # Subclass may override predict_step() to allow more efficient implementation
    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        """Predict for one step of observation.

        This only used for evaluation. So it only need to perform computations
        for generating action distribution.

        Args:
            time_step (ActionTimeStep): Current observation and other inputs
                for computing action.
            state (nested Tensor): should be consistent with predict_state_spec
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout.
        Returns:
            policy_step (AlgStep):
              output (nested Tensor): should be consistent with
                `action_spec`
              state (nested Tensor): should be consistent with
                `predict_state_spec`
        """
        policy_step = self.rollout_step(time_step, state)
        return policy_step._replace(info=())

    @abstractmethod
    def rollout_step(self, time_step: TimeStep, state):
        """Perform one step of rollout.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec

        Returns:
            policy_step (AlgStep):
              output (nested Tensor): should be consistent with
                `action_spec`
              state (nested Tensor): should be consistent with `train_state_spec`
              info (nested Tensor): everything necessary for training. Note that
                ("action", "reward", "discount", "is_last") are automatically
                collected in `unroll()`. So the user only need to put other
                stuff (e.g. value estimation) into `policy_step.info`
        """
        pass

    def transform_timestep(self, time_step):
        """Transform time_step.

        `transform_timestep` is called for all raw time_step got from
        the environment before passing to `predict_step` and 'rollout_step`. For
        off-policy algorithms, the replay buffer stores raw time_step. So when
        experiences are retrieved from the replay buffer, they are tranformed by
        `transform_timestep` in OffPolicyAlgorithm before passing to `_update()`.

        It includes tranforming observation and reward and should be stateless.

        Args:
            time_step (TimeStep | Experience): time step
        Returns:
            TimeStep | Experience: transformed time step
        """
        if self._reward_shaping_fn is not None:
            time_step = time_step._replace(
                reward=self._reward_shaping_fn(time_step.reward))
        if self._observation_transformers is not None:
            for observation_transformer in self._observation_transformers:
                time_step = time_step._replace(
                    observation=observation_transformer(time_step.observation))
        return time_step

    @abstractmethod
    def train_step(self, experience: Experience, state):
        """Perform one step of training computation.

        Args:
            experience (Experience):
            state (nested Tensor): should be consistent with train_state_spec

        Returns (AlgStep):
            output (nested Tensor): should be consistent with `action_spec`
            state (nested Tensor): should be consistent with `train_state_spec`
            info (nested Tensor): everything necessary for training.
        """
        pass

    @abstractmethod
    def calc_loss(self, training_info: TrainingInfo):
        """Calculate the loss for each step.

        `calc_loss()` does not need to mask out the loss at invalid steps as
        `train_iter()` will apply the mask automatically.

        Args:
            training_info (TrainingInfo): information collected for training.
                training_info.info are batched from each policy_step.info
                returned by train_step(). Note that training_info.next_discount
                is 0 if the next step is the last step in an episode.

        Returns (LossInfo):
            loss at each time step for each sample in the batch. The shapes of
            the tensors in loss_info should be (T, B)
        """
        pass

    def unroll(self, unroll_length):
        """Unroll `unroll_length` steps using the current policy.

        Because the self._env is a batched environment. The total number of
        environment steps is `self._env.batch_size * unroll_length`.

        Args:
            unroll_length (int): number of steps to unroll.
        Returns:
            training_info (TrainingInfo): The stacked information with shape
                (T, B, ...) for each of its members.
        """
        if self._current_time_step is None:
            self._current_time_step = common.get_initial_time_step(self._env)
        if self._current_policy_state is None:
            self._current_policy_state = self.get_initial_rollout_state(
                self._env.batch_size)
        time_step = self._current_time_step
        policy_state = self._current_policy_state

        training_info_list = []
        initial_state = self.get_initial_rollout_state(self._env.batch_size)

        env_step_time = 0.
        for _ in range(unroll_length):
            policy_state = common.reset_state_if_necessary(
                policy_state, initial_state, time_step.is_first())
            transformed_time_step = self.transform_timestep(time_step)
            policy_step = self.rollout_step(transformed_time_step,
                                            policy_state)
            if self._rollout_info_spec is None:
                self._rollout_info_spec = dist_utils.extract_spec(
                    policy_step.info)

            t0 = time.time()
            next_time_step = self._env.step(policy_step.output)
            env_step_time += time.time() - t0

            exp = make_experience(time_step, policy_step, policy_state)
            self.observe(exp)

            action = alf.nest.map_structure(lambda t: t.detach(),
                                            policy_step.output)

            training_info = TrainingInfo(
                action=action,
                reward=transformed_time_step.reward,
                discount=transformed_time_step.discount,
                step_type=transformed_time_step.step_type,
                rollout_info=dist_utils.distributions_to_params(
                    policy_step.info),
                env_id=transformed_time_step.env_id)

            training_info_list.append(training_info)
            time_step = next_time_step
            policy_state = policy_step.state

        alf.summary.scalar("time/env_step", env_step_time)
        training_info = alf.nest.utils.stack_nests(training_info_list)
        training_info = training_info._replace(
            rollout_info=dist_utils.params_to_distributions(
                training_info.rollout_info, self._rollout_info_spec))

        self._current_time_step = time_step
        self._current_policy_state = policy_state

        return training_info

    def train_iter(self):
        """Perform one iteration of training.

        Users may choose to implement their own train_iter()
        Returns:
            number of samples being trained on (including duplicates)
        """
        if self.is_on_policy():
            return self._train_iter_on_policy()
        else:
            return self._train_iter_off_policy()

    def _train_iter_on_policy(self):
        """Implemented in OnPolicyAlgorithm."""
        raise NotImplementedError()

    def _train_iter_off_policy(self):
        """Implemented in OffPolicyAlgorithm."""
        raise NotImplementedError()
