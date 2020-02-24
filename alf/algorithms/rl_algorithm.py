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
import torch
from typing import Callable

import gin

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, Experience, make_experience, StepType, TimeStep, TrainingInfo
from alf.utils import common, dist_utils, summary_utils


@gin.configurable
class RLAlgorithm(Algorithm):
    """Abstract base class for  RL Algorithms.

    RLAlgorithm provide basic functions and generic interface for rl algorithms.

    The key interface functions are:
    1. predict(): one step of computation of action for evaluation.
    2. rollout(): one step of computation for rollout. rollout() is used for
        collecting experiences during training. Different from `predict`,
        `rollout` may include addtional computations for training.
    3. train_step(): only used for off-policy training.
    4. train_complete(): Complete one training iteration based on the
       information collected from rollout() and/or train_step()

    See OnPolicyAlgorithm and OffPolicyAlgorithm for detail.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 train_state_spec,
                 predict_state_spec=None,
                 optimizer=None,
                 trainable_module_sets=None,
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
                `rollout()`
            action_distribution_spec (nested DistributionSpec): for the action
                distributions.
            predict_state_spec (nested TensorSpec): for the network state of
                `predict()`. If None, it's assumed to be the same as train_state_spec
            optimizer (tf.optimizers.Optimizer | list[Optimizer]): The
                optimizer(s) for training.
            reward_shaping_fn (Callable): a function that transforms extrinsic
                immediate rewards
            observation_transformer (Callable | list[Callable]): transformation(s)
                applied to `time_step.observation`
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
            summarize_grads_and_vars (bool): If True, gradient and network
                variable summaries will be written during training.
            summarize_action_distributions (bool): If True, generate summaris
                for the action distributions.
        """
        super(RLAlgorithm, self).__init__(
            train_state_spec=train_state_spec,
            predict_state_spec=predict_state_spec,
            optimizer=optimizer,
            trainable_module_sets=trainable_module_sets,
            gradient_clipping=gradient_clipping,
            clip_by_global_norm=clip_by_global_norm,
            debug_summaries=debug_summaries,
            name=name)

        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._reward_shaping_fn = reward_shaping_fn
        if isinstance(observation_transformer, Iterable):
            observation_transformers = list(observation_transformer)
        else:
            observation_transformers = [observation_transformer]
        self._observation_transformers = observation_transformers
        self._exp_observers = []
        self._metrics = None
        self._proc = psutil.Process(os.getpid())
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._summarize_action_distributions = summarize_action_distributions
        self._use_rollout_state = False

        self._rollout_info_spec = None
        self._train_step_info_spec = None
        self._processed_experience_spec = None

        self._current_time_step = None
        self._current_policy_state = None

    def _set_children_property(self, property_name, value):
        """Set the property named `property_name` in child RLAlgorithm to `value`."""
        for child in self._get_children():
            if isinstance(child, RLAlgorithm):
                child.__setattr__(property_name, value)

    @property
    def use_rollout_state(self):
        return self._use_rollout_state

    @use_rollout_state.setter
    def use_rollout_state(self, flag):
        self._use_rollout_state = flag
        self._set_children_property('use_rollout_state', flag)

    def need_full_rollout_state(self):
        """Whether PolicyStep.state from rollout should be full.

        If True, it means that rollout() should return the complete state
        for train_step().
        """
        return self._is_rnn and self._use_rollout_state

    @property
    def observation_spec(self):
        """Return the observation spec."""
        return self._observation_spec

    @property
    def train_step_info_spec(self):
        """The spec for the PolicyInfo.info returned from train_step()."""
        if self._train_step_info_spec is not None:
            return self._train_step_info_spec
        batch_size = 4
        processed_exp = common.zeros_from_spec(self.processed_experience_spec,
                                               batch_size)
        state = common.zeros_from_spec(self.train_state_spec, batch_size)
        policy_step = self.train_step(processed_exp, state)
        self._train_step_info_spec = common.extract_spec(policy_step.info)
        return self._train_step_info_spec

    @property
    def rollout_info_spec(self):
        """The spec for the PolicyInfo.info returned from rollout()."""
        if self._rollout_info_spec is not None:
            return self._rollout_info_spec
        batch_size = 4
        time_step = common.zeros_from_spec(self.time_step_spec, batch_size)
        state = common.zeros_from_spec(self.train_state_spec, batch_size)
        policy_step = self.rollout(
            self.transform_timestep(time_step), state,
            RLAlgorithm.PREPARE_SPEC)
        self._rollout_info_spec = common.extract_spec(policy_step.info)
        return self._rollout_info_spec

    @property
    def experience_spec(self):
        """Spec for experience."""
        policy_step_spec = PolicyStep(
            action=self.action_spec,
            state=self.train_state_spec,
            info=self.rollout_info_spec)
        exp_spec = make_experience(self.time_step_spec, policy_step_spec,
                                   policy_step_spec.state)
        if not self._use_rollout_state:
            exp_spec = exp_spec._replace(state=())
        return exp_spec

    @property
    def processed_experience_spec(self):
        """Spec for processed experience.

        Returns:
            Spec for the experience returned by preprocess_experience().
        """
        if self._processed_experience_spec is not None:
            return self._processed_experience_spec
        batch_size = 4
        exp = common.zeros_from_spec(self.experience_spec, batch_size)
        transformed_exp = self.transform_timestep(exp)
        processed_exp = self.preprocess_experience(transformed_exp)
        self._processed_experience_spec = self.experience_spec._replace(
            observation=common.extract_spec(processed_exp.observation),
            rollout_info=common.extract_spec(processed_exp.rollout_info))
        if not self._use_rollout_state:
            self._procesed_experience_spec = \
                self._processed_experience_spec._replace(state=())
        return self._processed_experience_spec

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
    def action_distribution_spec(self):
        """Return the action distribution spec for the action distributions."""
        return self._action_distribution_spec

    @property
    def exp_observers(self):
        """Return experience observers."""
        return self._exp_observers

    def set_metrics(self, metrics=[]):
        """Set metrics.

        metrics (list[TFStepMetric]): An optional list of metrics
            len(metrics) >= 2 as required by calling "self._metrics[:2]" in training_summary()
        """
        self._metrics = metrics

    def set_summary_settings(self,
                             summarize_grads_and_vars=False,
                             summarize_action_distributions=False):
        """Set summary flags."""
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._summarize_action_distributions = summarize_action_distributions

    def add_reward_summary(self, name, rewards):
        if self._debug_summaries:
            tf.summary.histogram(name + "/value", rewards)
            tf.summary.scalar(name + "/mean", tf.reduce_mean(rewards))

    def add_experience_observer(self, observer: Callable):
        """Add an observer to receive experience.

        Args:
            observer (Callable): callable which accept Experience as argument.
        """
        self._exp_observers.append(observer)

    def set_exp_replayer(self, exp_replayer: str, num_envs):
        """Set experience replayer.

        Args:
            exp_replayer (str): type of experience replayer. One of ("one_time",
                "uniform")
            num_envs (int): the total number of environments from all batched
                environments.
        """
        if exp_replayer == "one_time":
            self._exp_replayer = OnetimeExperienceReplayer()
        elif exp_replayer == "uniform":
            exp_spec = alf.nest.utils.to_distribution_param_spec(
                self.experience_spec)
            self._exp_replayer = SyncUniformExperienceReplayer(
                exp_spec, num_envs)
        else:
            raise ValueError("invalid experience replayer name")
        self.add_experience_observer(self._exp_replayer.observe)

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
        exp = alf.nest.utils.distributions_to_params(exp)
        for observer in self._exp_observers:
            observer(exp)

    def summarize_rollout(self, training_info):
        """Generate summaries for rollout.

        Note that training_info.info is empty here. Should use
        training_info.rollout_info to generate the summaries.

        Args:
            training_info (TrainingInfo): TrainingInfo structure collected from
                rollout.
        Returns:
            None
        """
        if self._debug_summaries:
            summary_utils.add_action_summaries(
                training_info.action, self._action_spec, "rollout_action")
            self.add_reward_summary("rollout_reward/extrinsic",
                                    training_info.reward)

        if self._summarize_action_distributions:
            field = alf.nest.find_field(training_info.rollout_info,
                                        'action_distribution')
            if len(field) == 1:
                summary_utils.summarize_action_dist(
                    action_distributions=field[0],
                    action_specs=self._action_spec,
                    name="rollout_action_dist")

    def summarize_train(self, training_info, loss_info, grads_and_vars):
        """Generate summaries for training & loss info.

        For on-policy algorithms, training_info.info is available.
        For off-policy alogirthms, both training_info.info and training_info.rollout_info
        are available. However, the statistics for these two structure are for
        the data batch sampled from the replay buffer. They do not represent
        the statistics of current on-going rollout.

        Args:
            training_info (TrainingInfo): TrainingInfo structure collected from
                rollout (on-policy training) or train_step (off-policy training).
            loss_info (LossInfo): loss
            grads_and_vars (tuple of (grad, var) pairs): list of gradients and
                their corresponding variables
        Returns:
            None
        """
        if self._summarize_grads_and_vars:
            summary_utils.add_variables_summaries(grads_and_vars)
            summary_utils.add_gradients_summaries(grads_and_vars)
        if self._debug_summaries:
            summary_utils.add_action_summaries(training_info.action,
                                               self._action_spec)
            summary_utils.add_loss_summaries(loss_info)

        if self._summarize_action_distributions:
            field = alf.nest.find_field(training_info.info,
                                        'action_distribution')
            if len(field) == 1:
                summary_utils.summarize_action_dist(field[0],
                                                    self._action_spec)

    def summarize_metrics(self):
        """Generate summaries for metrics `AverageEpisodeLength`, `AverageReturn`..."""
        if self._metrics:
            for metric in self._metrics:
                metric.tf_summaries(
                    train_step=common.get_global_counter(),
                    step_metrics=self._metrics[:2])

        mem = self._proc.memory_info().rss // 1e6
        alf.summary.scalar(name='memory_usage', data=mem)

    # Subclass may override predict() to allow more efficient implementation
    def predict(self, time_step: TimeStep, state, epsilon_greedy):
        """Predict for one step of observation.

        This only used for evaluation. So it only need to perform compuations
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
        policy_step = self.rollout(time_step, state, mode=self.ROLLOUT)
        return policy_step._replace(info=())

    ON_POLICY_TRAINING = 0
    OFF_POLICY_TRAINING = 1
    ROLLOUT = 2
    PREPARE_SPEC = 3

    @abstractmethod
    def rollout(self, time_step: TimeStep, state, mode):
        """Perform one step of rollout.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec
            mode (int): one of (ON_POLICY_TRAINING, OFF_POLICY_TRAINING, ROLLOUT).
                ON_POLICY_TRAINING: called during on-policy training
                OFF_POLICY_TRAINING: called during the training phase off-policy
                    training
                ROLLOUT: called during the rollout phase of off-policy training
                PREPARE_SPEC: called using fake data for preparing various specs.
                    rollout() should not make any side effect during this, such
                    as making changes to Variable using the provided time_step.
        Returns:
            policy_step (AlgStep):
              output (nested Tensor): should be consistent with
                `action_spec`
              state (nested Tensor): should be consistent with `train_state_spec`
              info (nested Tensor): everything necessary for training. Note that
                ("action", "reward", "discount", "is_last") are automatically
                collected by OnPolicyDriver. So the user only need to put other
                stuff (e.g. value estimation) into `policy_step.info`
        """
        pass

    def transform_timestep(self, time_step):
        """Transform time_step.

        `transform_timestep` is called by driver for all raw time_step got from
        the environment before passing to `predict`, 'rollout`. For off-policy
        algorithms, the replay buffer stores the raw time_step. So when
        experiences are retrieved from the replay buffer, they are tranformed by
        `transform_timestep` in OffPolicyDriver before passing to `train_step`.

        It includes tranforming observation and reward and should be stateless.

        Args:
            time_step (ActionTimeStep | Experience): time step
        Returns:
            ActionTimeStep | Experience: transformed time step
        """
        if self._reward_shaping_fn is not None:
            time_step = time_step._replace(
                reward=self._reward_shaping_fn(time_step.reward))
        if self._observation_transformers is not None:
            for observation_transformer in self._observation_transformers:
                time_step = time_step._replace(
                    observation=observation_transformer(time_step.observation))
        return time_step

    def preprocess_experience(self, experience: Experience):
        """Preprocess experience.

        preprocess_experience is called for the experiences got from replay
        buffer. An example is to calculate advantages and returns in PPOAlgorithm.

        The shapes of tensors in experience are assumed to be (B, T, ...)

        Args:
            experience (Experience): original experience
        Returns:
            processed experience
        """
        return experience

    @abstractmethod
    def train_step(self, experience: Experience, state):
        """Perform one step of training computation.

        Args:
            experience (Experience):
            state (nested Tensor): should be consistent with train_state_spec

        Returns (AlgStep):
            output (nested tf.distribution): should be consistent with
                `distribution_spec`
            state (nested Tensor): should be consistent with `train_state_spec`
            info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OffPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        pass

    # Subclass may override train_complete() to allow customized training
    def train_complete(self, training_info: TrainingInfo, weight=1.0):
        """Complete one iteration of training.

        `train_complete` should calculate gradients and update parameters using
        those gradients.

        Args:
            training_info (TrainingInfo): information collected for training.
                training_info.info are the batched from each policy_step.info
                returned by train_step()
            weight (float): weight for this batch. Loss will be multiplied with
                this weight before calculating gradient
        Returns:
            a tuple of the following:
            loss_info (LossInfo): loss information
            grads_and_vars (list[tuple]): list of gradient and variable tuples
        """
        valid_masks = (training_info.step_type != StepType.LAST).to(
            torch.float32)

        return super().train_complete(training_info, valid_masks, weight)

    @abstractmethod
    def calc_loss(self, training_info: TrainingInfo):
        """Calculate the loss for each step.

        `calc_loss()` does not need to mask out the loss at invalid steps as
        train_complete() will apply the mask automatically.

        Args:
            training_info (TrainingInfo): information collected for training.
                training_info.info are the batched from each policy_step.info
                returned by train_step(). Note that training_info.next_discount
                is 0 if the next step is the last step in an episode.

        Returns (LossInfo):
            loss at each time step for each sample in the batch. The shapes of
            the tensors in loss_info should be (T, B)
        """
        pass

    def unroll(self, time_step, policy_state, unroll_length):
        training_info_list = []

        for t in range(unroll_length):
            policy_state = common.reset_state_if_necessary(
                policy_state, self._initial_state, time_step.is_first())
            transformed_time_step = self.transform_timestep(time_step)
            policy_step = self.rollout(
                transformed_time_step, policy_state, mode=RLAlgorithm.ROLLOUT)
            next_time_step = self._env.step(policy_step.output)

            exp = make_experience(time_step, policy_step, policy_state)
            self.observe(exp)

            action = alf.nest.map_structure(lambda t: t.detach(),
                                            policy_step.output)

            if t == 0:
                rollout_info_spec = dist_utils.extract_spec(policy_step.info)

            training_info = TrainingInfo(
                action=action,
                reward=transformed_time_step.reward,
                discount=transformed_time_step.discount,
                step_type=transformed_time_step.step_type,
                rollout_info=nest_utils.distributions_to_params(
                    policy_step.info),
                env_id=transformed_time_step.env_id)

            training_info_list.append(training_info)
            time_step = next_time_step
            policy_state = policy_step.state

        def _stack(*tensors):
            return torch.cat([t.unsqueeze(0) for t in tensors])

        training_info = alf.nest.map_structure(_stack, *training_info)
        training_info = training_info._replace(
            rollout_info=dist_utils.params_to_distributions(
                training_info.rollout_info, rollout_info_spec))
        return time_step, policy_state, training_info

    def train_iter(self):
        """Perform one iteration of training.

        Returns:
            #(samples precessed) * #(repeats)
        """

        if self._on_policy:
            return self._train_iter_on_policy()
        else:
            return self._train_iter_off_policy()

    def _train_iter_on_policy(self):
        if self._current_time_step is None:
            self._current_time_step = self.get_initial_time_step()
        if self._current_policy_state is None:
            self._current_policy_state = self.get_initial_policy_state()

        self._current_time_step, self._current_policy_state, training_info = \
            self.unroll(self._current_time_step,
                    self._current_policy_state,
                    self._config.unroll_length)

        loss_info, params = self.train_complete(training_info)
        self.summarize_train(training_info, loss_info, params)
        self.summarize_metrics()
        alf.summary.get_global_counter().add_(1)
        return training_info.step_type.shape().prod()
