# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from functools import partial
import numpy as np
from typing import Callable, NamedTuple
import torch.distributions as td
from torch import nn
import torch.nn.functional as F
import torch
from alf.utils import common

import alf
from alf.algorithms.actor_critic_loss import normalize
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.ppo_algorithm import PPOLoss
from alf.data_structures import LossInfo, AlgStep, StepType, TimeStep
from alf.nest.utils import convert_device
from alf.tensor_specs import TensorSpec
from alf.utils import dist_utils, tensor_utils, value_ops, summary_utils
from alf.utils.model_averager import create_averaged_model
from alf.utils.normalizers import ScalarAdaptiveNormalizer
from alf.utils.losses import element_wise_squared_loss
from alf.utils.summary_utils import safe_mean_hist_summary


class ARModel(nn.Module):
    @property
    def sequence_length(self):
        """The length of the output of the model."""
        raise NotImplementedError()

    @property
    def event_shape(self):
        """The shape of one step of the  output of the model."""
        raise NotImplementedError()

    def log_prob(self, input, sample, prefix_length):
        raise NotImplementedError()

    def sample_with_prefix(self,
                           input,
                           prefix,
                           prefix_length,
                           return_log_prob=False):
        return self._sample(input, prefix, prefix_length,
                            dist_utils.sample_action_distribution,
                            return_log_prob)

    def epsilon_greedy_sample_with_prefix(self, input, prefix, prefix_length,
                                          epsilon):
        return self._sample(
            input, prefix, prefix_length,
            partial(dist_utils.epsilon_greedy_sample, eps=epsilon))


@alf.configurable(
    blacklist=["input_tensor_spec", "output_spec", "sequence_length"])
class RNNARModel(ARModel):
    def __init__(self,
                 input_tensor_spec,
                 output_spec,
                 sequence_length,
                 cell_ctor,
                 hidden_sizes,
                 continuous_projection_net_ctor: Callable = alf.nn.
                 NormalProjectionNetwork,
                 discrete_projection_net_ctor: Callable = alf.nn.
                 CategoricalProjectionNetwork):
        super().__init__()
        input_dim = input_tensor_spec.numel
        output_dim = output_spec.numel
        if output_spec.is_discrete:
            self._output_to_rnn_input = alf.layers.Sequential(
                torch.nn.Embedding(
                    np.max(output_spec.maximum) + 1, hidden_sizes[0]),
                torch.nn.Flatten())
        else:
            self._output_to_rnn_input = alf.layers.FC(output_dim,
                                                      hidden_sizes[0])
        self._input_to_rnn_input = alf.layers.FC(input_dim, hidden_sizes[0])
        if output_spec.is_discrete:
            projection_net_ctor = discrete_projection_net_ctor
        else:
            projection_net_ctor = continuous_projection_net_ctor
        self._proj_net = projection_net_ctor(
            input_size=hidden_sizes[-1], action_spec=output_spec)
        self._sequence_length = sequence_length
        self._rnn = alf.nn.Sequential(*[
            cell_ctor(hidden_sizes[i - 1], hidden_sizes[i])
            for i in range(1, len(hidden_sizes))
        ])
        self._event_shape = torch.Size((sequence_length, output_dim))

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def event_shape(self):
        return self._event_shape

    def log_prob(self, input, sample):
        assert sample.size(1) == self._sequence_length
        state = common.zero_tensor_from_nested_spec(self._rnn.state_spec,
                                                    input.size(0))
        rnn_input = self._input_to_rnn_input(input)
        log_probs = []
        for i in range(self._sequence_length):
            rnn_output, state = self._rnn(rnn_input, state)
            output_dist = self._proj_net(rnn_output)[0]
            lp = output_dist.log_prob(sample[:, i])
            log_probs.append(lp)
            rnn_input = self._output_to_rnn_input(sample[:, i])
        return torch.stack(log_probs, dim=1)

    def _sample(self,
                input,
                prefix,
                prefix_length,
                sample_func,
                return_log_prob=False):
        samples = []
        state = common.zero_tensor_from_nested_spec(self._rnn.state_spec,
                                                    input.size(0))
        rnn_input = self._input_to_rnn_input(input)
        if return_log_prob:
            log_probs = []
        for i in range(self._sequence_length):
            rnn_output, state = self._rnn(rnn_input, state)
            output_dist = self._proj_net(rnn_output)[0]
            sample = sample_func(output_dist)
            if prefix is not None:
                use_pre = i < prefix_length
                samples.append(
                    torch.where(use_pre[:, None], prefix[:, i], sample))
            else:
                samples.append(sample)
            rnn_input = self._output_to_rnn_input(samples[-1])
            if return_log_prob:
                logp = output_dist.log_prob(sample)
                if prefix is not None:
                    # The reason of not using samples[-1] for log_prob is to utilize
                    # the caching mechanism of td.Transform so that log_prob is more
                    # accurate for samples near the boundary of the action space.
                    ologp = output_dist.log_prob(prefix[:, i])
                    logp = torch.where(use_pre, ologp, logp)
                log_probs.append(logp)

        samples = torch.stack(samples, dim=1)
        if return_log_prob:
            log_probs = torch.stack(log_probs, dim=1)
            return samples, log_probs
        else:
            return samples


class MixtureARDistribution(td.Distribution):
    has_rsample = True
    """Auto-regressive distribution.

    :param input: [batch_size, input_dim]
    :param prefix: [batch_size, prefix_length, per_step_dim]
    :param new_sample_logit: [batch_size], its sigmoid is the probability of
        generating a new sample
    :param model: ARModel
    """

    def __init__(self, input, prefix, new_sample_logit, model: ARModel):
        super().__init__(
            batch_shape=input.shape[:-1],
            event_shape=(model.event_shape.numel(), ))
        assert input.shape[:-1] == prefix.shape[:-2] == new_sample_logit.shape
        self._input = input
        self._model = model
        self._new_sample_logit = new_sample_logit
        self._prefix_length = prefix.size(-2)
        padding = torch.zeros(
            prefix.shape[:-2] + (model.sequence_length - self._prefix_length,
                                 prefix.size(-1)),
            dtype=prefix.dtype,
            device=prefix.device)
        prefix = torch.cat([prefix, padding], dim=-2)
        self._prefix = prefix

    @property
    def arg_constraints(self):
        return {}

    def _batch_squash_call(self, f, *args, **kwargs):
        outer_rank = len(self.batch_shape)
        bs = tensor_utils.BatchSquash(outer_rank)

        def _flatten(x):
            if isinstance(x, torch.Tensor) and x.ndim >= outer_rank:
                return bs.flatten(x)
            else:
                return x

        args, kwargs = alf.nest.map_structure(_flatten, (args, kwargs))
        result = f(*args, **kwargs)
        return alf.nest.map_structure(bs.unflatten, result)

    def log_prob(self, sample):
        """
        :param sample: [batch_size, per_step_dim * sequence_length]
        :return: [batch_size]
        """
        sample_shape = sample.shape[:-1 - len(self.batch_shape)]
        assert len(sample_shape) == 0
        assert sample.shape[-1 - len(self.batch_shape):-1] == self.batch_shape
        sample = sample.reshape(*sample.shape[:-1], self._model.event_shape[1],
                                self._model.event_shape[0]).transpose(-2, -1)
        log_probs = self._batch_squash_call(self._model.log_prob, self._input,
                                            sample)
        return self._calc_log_prob(sample, log_probs)

    def _calc_log_prob(self, sample, log_probs):
        same_as_prefix = (sample == self._prefix)[..., :self._prefix_length, :]
        same_as_prefix = same_as_prefix.all(-1).all(-1)  # [B]
        logp_new = -F.softplus(-self._new_sample_logit) + log_probs.sum(-1)
        logp_old = (-F.softplus(self._new_sample_logit) +
                    log_probs[..., self._prefix_length:].sum(-1))
        if sample.dtype.is_floating_point:
            # P(a_prefix, a|prefix, s) =
            #       (1-P_switch) * delta(a_prefix-prefix) * Q(a|prefix, s)
            #       + P_switch * Q(a_prefix, a|s)
            # where delta is the Dirac delta function. If a_prefix is same as prefix,
            # the first term dominates and we can ignore the second term.
            logp = torch.where(same_as_prefix, logp_old, logp_new)
        else:
            logp = torch.logsumexp(
                torch.stack([logp_new, logp_old], dim=-1), dim=-1)
        return torch.where(same_as_prefix, logp, logp_new)

    def sample(self, sample_shape=torch.Size(), return_log_prob=False):
        if return_log_prob:
            sample, log_prob = self.rsample(sample_shape, return_log_prob)
            return sample.detach(), log_prob
        else:
            with torch.no_grad():
                return self.rsample(sample_shape, return_log_prob)

    def rsample(self, sample_shape=torch.Size(), return_log_prob=False):
        """
        :return: [batch_size, per_step_dim * sequence_length]
        """
        sample_shape = torch.Size(sample_shape)
        assert sample_shape.numel() == 1
        is_old_sample = torch.rand(
            self._input.shape[:-1]) > self._new_sample_logit.sigmoid()
        prefix_length = self._prefix_length * is_old_sample
        ret = self._batch_squash_call(self._model.sample_with_prefix,
                                      self._input, self._prefix, prefix_length,
                                      return_log_prob)

        def _as_sample_shape(x):
            return x.expand(sample_shape + x.shape)

        if return_log_prob:
            sample, log_probs = ret
            log_prob = self._calc_log_prob(sample, log_probs)
            return (_as_sample_shape(sample.transpose(-2, -1).flatten(-2, -1)),
                    _as_sample_shape(log_prob))
        else:
            return _as_sample_shape(ret.transpose(-2, -1).flatten(-2, -1))

    def epsilon_greedy_sample(self, epsilon: float):
        """
        :param epsilon:
        :return: [batch_size, per_step_dim * sequence_length]
        """
        B = self._input.shape[:-1]
        is_old_sample = torch.rand(B) > self._new_sample_logit.sigmoid()
        # is_old_sample = is_old_sample.where(
        #     torch.rand(B) < epsilon, self._new_sample_logit < 0.0)
        prefix_length = self._prefix_length * is_old_sample
        sample = self._batch_squash_call(
            self._model.epsilon_greedy_sample_with_prefix, self._input,
            self._prefix, prefix_length, epsilon)
        return sample.transpose(-2, -1).flatten(-2, -1)

    @property
    def rmode(self):
        return self.epsilon_greedy_sample(0.0)

    @property
    def mode(self):
        return self.rmode.detach()

    def get_builder(self):
        return partial(
            MixtureARDistribution, model=self._model), {
                "input": self._input,
                "prefix": self._prefix[..., :self._prefix_length, :],
                "new_sample_logit": self._new_sample_logit,
            }


class TrajPPOInfo(NamedTuple):
    step_type: torch.Tensor = ()
    reward: torch.Tensor = ()
    discount: torch.Tensor = ()
    action_distribution: torch.Tensor = ()
    action: torch.Tensor = ()
    log_prob: torch.Tensor = ()
    rollout_log_prob: torch.Tensor = ()
    value: torch.Tensor = ()
    critic: torch.Tensor = ()
    rollout_log_prob: torch.Tensor = ()
    value_diff: torch.Tensor = ()
    switched: torch.Tensor = ()
    returns: torch.Tensor = ()
    advantages: torch.Tensor = ()
    normalized_advantages: torch.Tensor = ()
    reward_weights: torch.Tensor = ()
    rollout_action_distribution: torch.Tensor = ()


class TrajPPOState(NamedTuple):
    steps_since_last_switch: torch.Tensor = ()
    action_shifter: torch.Tensor = ()


class DefaultActionShifter(object):
    @property
    def state_spec(self):
        return ()

    def __call__(self, prev_action, inputs, state):
        return prev_action[:, 1:], state


@alf.configurable(blacklist=[
    "observation_spec",
    "action_spec",
    "reward_spec",
    "env",
    "config",
    "debug_summaries",
])
class TrajectoryPPOAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 trajectory_length,
                 reward_spec=TensorSpec(()),
                 reward_weights=None,
                 actor_network_ctor: Callable = RNNARModel,
                 value_network_ctor: Callable = alf.nn.ValueNetwork,
                 critic_network_ctor: Callable = alf.nn.CriticNetwork,
                 action_shifter=DefaultActionShifter(),
                 target_switch_steps=10,
                 switch_threshold_learning_rate=1e-4,
                 initial_switch_threshold=0.0,
                 distribution_adapter_ctor=None,
                 epsilon_greedy=None,
                 env=None,
                 config: TrainerConfig = None,
                 loss_class: Callable = PPOLoss,
                 predict_average_type: str = "none",
                 optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 name="TrajectoryPPOAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
                ``action_spec.ndim`` must be 1. And the action can be reshaped as
                ``[batch_size, per_step_action_dim, trajectory_length]``.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the v values
                is used for training the actor if reward_weights is not None.
                Otherwise, the sum of the v values is used.
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            actor_network_ctor (Callable): Function to construct the actor network.
                ``actor_network_ctor`` needs to accept ``input_tensor_spec`` and
                ``action_spec`` as its arguments and return an actor network.
                The constructed network will be called with ``forward(observation, state)``.
            value_network_ctor (None | Callable): Function to construct the value network.
                ``value_network_ctor`` needs to accept ``input_tensor_spec`` as its
                arguments and return a value network. The constructed network will be
                called with ``forward(observation, state)`` and returns value tensor for
                each observation given observation and network state. Note that if the
                algorithm is constructed for evaluation or deployment only, the
                value_network_ctor can be set to None and the value network will not be
                constructed at all.
            loss_ctor: the class of the loss. The signature of its
                constructor: ``loss_class(debug_summaries)``
            optimizer (torch.optim.Optimizer): The optimizer for training
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.

        """
        assert action_spec.ndim == 1
        assert action_spec.numel % trajectory_length == 0, (
            f"action_spec.numel={action_spec.numel}, "
            f"trajectory_length={trajectory_length}")
        assert distribution_adapter_ctor is None, "Not supported yet"

        per_step_action_dim = action_spec.numel // trajectory_length
        self._per_step_action_dim = per_step_action_dim

        # We use the minimum and maximum of the last step of the trajectory as the
        # minimum and maximum of the per_step_action_spec
        per_step_action_spec = alf.BoundedTensorSpec(
            (per_step_action_dim, ),
            minimum=action_spec.minimum[trajectory_length -
                                        1::trajectory_length],
            maximum=action_spec.maximum[trajectory_length -
                                        1::trajectory_length],
            dtype=action_spec.dtype)

        prefix_action_spec = alf.TensorSpec(
            ((trajectory_length - 1) * per_step_action_dim, ),
            dtype=action_spec.dtype)
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy
        actor_model = actor_network_ctor(
            input_tensor_spec=observation_spec,
            output_spec=per_step_action_spec,
            sequence_length=trajectory_length)
        value_network = value_network_ctor(
            input_tensor_spec=observation_spec, output_tensor_spec=reward_spec)
        critic_network = critic_network_ctor(
            input_tensor_spec=(observation_spec, prefix_action_spec),
            output_tensor_spec=reward_spec)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            reward_weights=reward_weights,
            predict_state_spec=TrajPPOState(
                action_shifter=action_shifter.state_spec),
            train_state_spec=TrajPPOState(
                steps_since_last_switch=alf.TensorSpec(()),
                action_shifter=action_shifter.state_spec),
            env=env,
            config=config,
            optimizer=optimizer,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_model = actor_model
        self._value_network = value_network
        self._critic_network = critic_network
        loss = loss_class(
            reward_dim=reward_spec.numel, debug_summaries=debug_summaries)
        self._loss = loss

        self._predict_model = create_averaged_model(self._actor_model,
                                                    predict_average_type)
        self._action_shifter = action_shifter
        self.register_buffer(
            "_switch_threshold",
            torch.tensor(initial_switch_threshold, dtype=torch.float32))
        self._switch_threshold_learning_rate = switch_threshold_learning_rate
        self.register_buffer("_avg_switch_steps", torch.tensor(1.0))
        self._target_switch_steps = target_switch_steps
        self.register_buffer("_total_num_switches",
                             torch.tensor(10, dtype=torch.int64))
        self._value_diff_normalizer = ScalarAdaptiveNormalizer(
            auto_update=False)
        self._critic_loss_func = element_wise_squared_loss

    def _unflatten_action(self, action):
        return action.reshape(*action.shape[:-1], self._per_step_action_dim,
                              -1).transpose(-2, -1)

    def _extract_prev_action(self, inputs, state: TrajPPOState):
        # return [batch_size, trajectory_length - 1, per_step_action_dim]
        prev_action = self._unflatten_action(inputs.prev_action)
        return self._action_shifter(prev_action, inputs, state.action_shifter)

    def _calc_action_distribution(self, inputs, state, model):
        value, _ = self._value_network(inputs.observation)
        prev_action, action_shifter_state = self._extract_prev_action(
            inputs, state)
        prev_q, _ = self._critic_network((inputs.observation,
                                          prev_action.flatten(-2, -1)))
        switch_logit = self._calc_switch_logit(prev_q, value, inputs.step_type)
        action_dist = MixtureARDistribution(inputs.observation, prev_action,
                                            switch_logit, model)
        return action_dist, action_shifter_state, value, prev_q, prev_action

    def predict_step(self, inputs: TimeStep, state: TrajPPOState):
        action_dist, action_shifter_state, value, prev_q, prev_action = self._calc_action_distribution(
            inputs, state, self._predict_model)
        action = action_dist.epsilon_greedy_sample(self._epsilon_greedy)
        return AlgStep(
            output=action,
            state=TrajPPOState(action_shifter=action_shifter_state),
            info=TrajPPOInfo(action_distribution=action_dist, action=action))

    def rollout_step(self, inputs: TimeStep, state: TrajPPOState):
        action_dist, action_shifter_state, value, prev_q, prev_action = self._calc_action_distribution(
            inputs, state, self._actor_model)
        action, log_prob = action_dist.sample(return_log_prob=True)

        # update state
        action_prefix = self._unflatten_action(action)[:, :-1, :]
        switched = ~(prev_action == action_prefix).all(-1).all(-1)
        switched = switched & (inputs.step_type != StepType.FIRST)
        steps_since_last_switch = state.steps_since_last_switch + 1
        state = TrajPPOState(
            action_shifter=action_shifter_state,
            steps_since_last_switch=torch.where(switched, 0,
                                                steps_since_last_switch))

        # update self._avg_switch_steps
        num_switches = switched.sum()
        rate = (100 * num_switches / self._total_num_switches).clip(max=1)
        self._total_num_switches += num_switches
        avg_length = (steps_since_last_switch * switched).sum() / (
            num_switches + 1e-10)
        self._avg_switch_steps.lerp_(avg_length, rate)

        return AlgStep(
            output=action,
            state=state,
            info=TrajPPOInfo(
                step_type=inputs.step_type,
                reward=inputs.reward,
                discount=inputs.discount,
                log_prob=log_prob,
                value_diff=value - prev_q,
                action_distribution=action_dist,
                action=action,
                value=value))

    def train_step(self, inputs: TimeStep, state: TrajPPOState,
                   rollout_info: TrajPPOInfo):
        action_dist, action_shifter_state, value, prev_q, prev_action = self._calc_action_distribution(
            inputs, state, self._actor_model)

        action_prefix = self._unflatten_action(rollout_info.action)[:, :-1, :]
        switched = ~(prev_action == action_prefix).all(-1).all(-1)
        switched = switched & (inputs.step_type != StepType.FIRST)
        steps_since_last_switch = state.steps_since_last_switch + 1
        state = TrajPPOState(
            action_shifter=action_shifter_state,
            steps_since_last_switch=torch.where(switched, 0,
                                                steps_since_last_switch))

        critic, _ = self._critic_network((inputs.observation,
                                          action_prefix.flatten(-2, -1)))

        # update self._switch_threshold
        diff = self._avg_switch_steps - self._target_switch_steps
        self._switch_threshold -= self._switch_threshold_learning_rate * diff

        return AlgStep(
            output=rollout_info.action,
            state=state,
            info=TrajPPOInfo(
                step_type=inputs.step_type,
                reward=inputs.reward,
                discount=inputs.discount,
                action_distribution=action_dist,
                action=rollout_info.action,
                value=value,
                critic=critic,
                value_diff=rollout_info.value_diff,
                switched=switched,
                rollout_log_prob=rollout_info.rollout_log_prob,
                returns=rollout_info.returns,
                advantages=rollout_info.advantages,
                normalized_advantages=rollout_info.normalized_advantages))

    def _calc_switch_logit(self, prev_q, value, step_type):
        if self.has_multidim_reward():
            prev_q = prev_q @ self.reward_weights
            value = value @ self.reward_weights
        diff = value - prev_q
        if common.is_rollout():
            self._value_diff_normalizer.update(diff)
        diff = self._value_diff_normalizer.normalize(diff)
        logit = (diff - self._switch_threshold).detach()
        logit[step_type == StepType.FIRST] = 100.0
        logit.clamp_(-100, 100)
        return logit

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        """Compute advantages and put it into exp.rollout_info."""

        # The device of rollout_info can be different from the default device
        # when ReplayBuffer.gather_all.convert_to_default_device is configured
        # to False to save gpu memory.
        step_type = convert_device(rollout_info.step_type)
        discount = convert_device(rollout_info.discount)
        reward = convert_device(rollout_info.reward)
        value = convert_device(rollout_info.value)

        if rollout_info.reward.ndim == 3:
            # [B, T, D] or [B, T, 1]
            discounts = discount.unsqueeze(-1) * self._loss.gamma
        else:
            # [B, T]
            discounts = discount * self._loss.gamma

        advantages = value_ops.generalized_advantage_estimation(
            rewards=reward,
            values=value,
            step_types=step_type,
            discounts=discounts,
            td_lambda=self._loss._lambda,
            time_major=False)

        if self.has_multidim_reward():
            assert self._loss.normalizing_scalar_advantages
            scalar_advantages = advantages @ self.reward_weights
        else:
            scalar_advantages = advantages
        normalized_advantages = normalize(self._loss._adv_norm,
                                          scalar_advantages.reshape(-1, 1))
        normalized_advantages = normalized_advantages.reshape_as(
            scalar_advantages)
        normalized_advantages = tensor_utils.tensor_extend_zero(
            normalized_advantages, dim=1)

        advantages = tensor_utils.tensor_extend_zero(advantages, dim=1)
        returns = value + advantages
        return root_inputs, TrajPPOInfo(
            value_diff=rollout_info.value_diff,
            rollout_log_prob=rollout_info.log_prob,
            returns=returns,
            action=rollout_info.action,
            advantages=advantages,
            normalized_advantages=normalized_advantages,
        )

    def after_update(self, root_inputs: TimeStep, info: TrajPPOInfo):
        if self._predict_model != self._actor_model:
            self._predict_model.update_parameters(self._actor_model)

    def calc_loss(self, info: TrajPPOInfo):
        """Calculate loss."""
        loss_info = self._loss(info)
        critic_loss = self._critic_loss_func(info.critic, info.returns)
        mask = info.step_type != StepType.LAST

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope("TrajPPOAlgorithm"):
                alf.summary.scalar("switched", info.switched.float().mean())
                alf.summary.scalar("avg_switch_steps", self._avg_switch_steps)
                alf.summary.scalar("switch_threshold", self._switch_threshold)
                alf.summary.scalar("total_num_switches",
                                   self._total_num_switches)
                alf.summary.scalar("value_diff_mean",
                                   self._value_diff_normalizer.mean)
                alf.summary.scalar("value_diff_std",
                                   self._value_diff_normalizer.variance**0.5)

                value_diff = info.value_diff
                if self.has_multidim_reward():
                    value_diff = value_diff @ self.reward_weights
                diff = self._value_diff_normalizer.normalize(value_diff)
                prob = torch.sigmoid(diff - self._switch_threshold).detach()
                prob[info.step_type == StepType.FIRST] = 1.0
                not_first = info.step_type != StepType.FIRST
                summary_utils.safe_mean_hist_summary("value_diff", value_diff,
                                                     not_first)
                summary_utils.safe_mean_hist_summary("switch_prob", prob,
                                                     not_first)

                def _summarize(v, r, td, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_critics" + suffix,
                        tensor_utils.explained_variance(v, r, mask))

                    safe_mean_hist_summary('critics' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("critic_td_error" + suffix, td,
                                           mask)

                returns = info.returns
                critics = info.critic
                td_error = returns - critics
                if critics.ndim == 2:
                    _summarize(critics, returns, td_error, '')
                else:
                    for i in range(critics.size(2)):
                        suffix = '/' + str(i)
                        _summarize(critics[..., i], returns[..., i],
                                   td_error[..., i], suffix)

        if critic_loss.ndim == 3:
            critic_loss = critic_loss.mean(dim=2)
        critic_loss = critic_loss * mask

        return LossInfo(
            loss=loss_info.loss + critic_loss,
            extra=dict(
                td_loss=loss_info.extra.td_loss,
                pg_loss=loss_info.extra.pg_loss,
                neg_entropy=loss_info.extra.neg_entropy,
                critic=critic_loss))
