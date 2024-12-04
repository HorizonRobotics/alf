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
"""Actor critic algorithm."""

import torch
from typing import Tuple

import alf
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.data_structures import TimeStep, AlgStep, namedtuple
from alf.utils import common, dist_utils, tensor_utils
from alf.tensor_specs import TensorSpec
from .config import TrainerConfig
from alf.utils.model_averager import create_averaged_model
from alf.utils import summary_utils

ActorCriticState = namedtuple(
    "ActorCriticState", ["actor", "value", "adapter"], default_value=())

ActorCriticInfo = namedtuple(
    "ActorCriticInfo", [
        "step_type", "discount", "reward", "action", "log_prob",
        "action_distribution", "value", "reward_weights"
    ],
    default_value=())


@alf.configurable
class ActorCriticAlgorithm(OnPolicyAlgorithm):
    """Actor critic algorithm."""

    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=TensorSpec(()),
                 reward_weights=None,
                 actor_network_ctor=ActorDistributionNetwork,
                 value_network_ctor=ValueNetwork,
                 distribution_adapter_ctor=None,
                 epsilon_greedy=None,
                 top_k_sample: int = 0,
                 top_p_sample: float = 0,
                 env=None,
                 config: TrainerConfig = None,
                 loss=None,
                 loss_class=ActorCriticLoss,
                 predict_average_type: str = "none",
                 optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 name="ActorCriticAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
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
            top_k_sample: If >0, use top-k sampling for action selection in
                ``predict_step()``
            top_p_sample: If >0, use top-p sampling for action selection in
                ``predict_step()``
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
            loss (None|ActorCriticLoss): an object for calculating loss. If
                None, a default loss of class loss_class will be used.
            loss_class (type): the class of the loss. The signature of its
                constructor: ``loss_class(debug_summaries)``
            optimizer (torch.optim.Optimizer): The optimizer for training
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.

        """
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy
        self._top_k_sample = top_k_sample
        self._top_p_sample = top_p_sample
        actor_network = actor_network_ctor(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        value_network = None
        if value_network_ctor is not None:
            value_network = value_network_ctor(
                input_tensor_spec=observation_spec)

            if reward_spec.numel > 1:
                value_network = value_network.make_parallel(
                    reward_spec.numel)  # value->[B,n]

        if distribution_adapter_ctor is not None:
            distribution_adapter = distribution_adapter_ctor(action_spec)
            adapter_state_spec = distribution_adapter.state_spec
        else:
            distribution_adapter = None
            adapter_state_spec = ()
        super(ActorCriticAlgorithm, self).__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            reward_weights=reward_weights,
            predict_state_spec=ActorCriticState(
                actor=actor_network.state_spec, adapter=adapter_state_spec),
            train_state_spec=ActorCriticState(
                actor=actor_network.state_spec,
                value=value_network.state_spec if value_network else (),
                adapter=adapter_state_spec),
            env=env,
            config=config,
            optimizer=optimizer,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network
        self._value_network = value_network
        self._distribution_adapter = distribution_adapter
        if loss is None:
            loss = loss_class(
                reward_dim=reward_spec.numel, debug_summaries=debug_summaries)
        self._loss = loss

        # The following checkpoint loading hook handles the case when value
        # network is not constructed. In this case the value network parameters
        # present in the checkpoint should be ignored.
        def _deployment_hook(state_dict, prefix: str, unused_loacl_metadata,
                             unused_strict, unused_missing_keys,
                             unused_unexpected_keys, unused_error_msgs):
            to_delete = []
            for key in state_dict:
                if not key.startswith(prefix):
                    continue
                if self._value_network is None:
                    if key[len(prefix):].startswith("_value_network"):
                        to_delete.append(key)
            for key in to_delete:
                state_dict.pop(key)

        self._register_load_state_dict_pre_hook(_deployment_hook)

        self._predict_model = create_averaged_model(self._actor_network,
                                                    predict_average_type)

    def after_update(self, root_inputs: TimeStep, info: ActorCriticInfo):
        if self._predict_model != self._actor_network:
            self._predict_model.update_parameters(self._actor_network)

    def _trainable_attributes_to_ignore(self):
        return ['_predict_model']

    def convert_train_state_to_predict_state(self, state):
        return state._replace(value=())

    def predict_step(self, inputs: TimeStep, state: ActorCriticState):
        """Predict for one step."""
        action_dist, actor_state = self._predict_model(
            inputs.observation, state=state.actor)

        if self._distribution_adapter is not None:
            action_dist, adapter_state = self._distribution_adapter(
                (action_dist, inputs.prev_action), state.adapter)
        else:
            adapter_state = ()

        if self._top_k_sample > 0:
            action = action_dist.top_k_sample(self._top_k_sample)
        elif self._top_p_sample > 0:
            action = action_dist.top_p_sample(self._top_p_sample)
        else:
            action = dist_utils.epsilon_greedy_sample(action_dist,
                                                      self._epsilon_greedy)
        return AlgStep(
            output=action,
            state=ActorCriticState(actor=actor_state, adapter=adapter_state),
            info=ActorCriticInfo(action_distribution=action_dist))

    def rollout_step(self, inputs: TimeStep, state: ActorCriticState):
        """Rollout for one step."""
        value, value_state = self._value_network(
            inputs.observation, state=state.value)

        action_distribution, actor_state = self._actor_network(
            inputs.observation, state=state.actor)

        if self._distribution_adapter is not None:
            action_distribution, adapter_state = self._distribution_adapter(
                (action_distribution, inputs.prev_action), state.adapter)
        else:
            adapter_state = ()

        action, log_prob = dist_utils.sample_action_distribution(
            action_distribution, return_log_prob=True)

        if self.has_multidim_reward():
            reward_weights = tensor_utils.tensor_extend_new_dim(
                self.reward_weights, dim=0, n=value.shape[0])
        else:
            reward_weights = ()
        return AlgStep(
            output=action,
            state=ActorCriticState(
                actor=actor_state, value=value_state, adapter=adapter_state),
            info=ActorCriticInfo(
                action=common.detach(action),
                log_prob=common.detach(log_prob),
                value=value,
                step_type=inputs.step_type,
                reward=inputs.reward,
                discount=inputs.discount,
                action_distribution=action_distribution,
                reward_weights=reward_weights))

    def calc_loss(self, info: ActorCriticInfo):
        """Calculate loss."""
        return self._loss(info)


class CorrelatedDistributionAdpater(alf.nn.Network):
    r"""Adapt the action distribution to be correlated with the previous action.

    Sampling using the adapted distribution is equivalent to the following sampling
    process:

    .. math::

        \episilon_t \leftarrow \beta_c \epsilon + \beta N(0, 1)
        x \leftarrow \mu + \sigma (\alpha_c \epsilon + \alpha N(0, 1))

    where :math:`\epsilon` is the state, :math:`\mu` is the mean, :math:`\sigma`,
    :math:`\beta_c = \sqrt{1 - \beta^2}`, :math:`\alpha_c = \sqrt{1 - \alpha^2}`.

    :param dim: the dimension of the normal distribution
    :param alpha: the alpha parameter in the above equation
    :param beta: the beta parameter in the above equation
    """

    def __init__(self, dim: int, alpha: float, beta: float):
        assert alpha == 0, "Only support alpha=0"
        dist = dist_utils.DiagMultivariateNormal(
            loc=torch.zeros((1, dim)), scale=torch.ones((1, dim)))
        dist_spec = dist_utils.extract_spec(dist)
        super().__init__(
            input_tensor_spec=dist_spec,
            state_spec=(alf.TensorSpec((dim, )), alf.TensorSpec((dim, ))))
        self._beta = beta
        self._betac = (1 - beta**2)**0.5

    def forward(self,
                input: Tuple[dist_utils.DiagMultivariateNormal, torch.Tensor],
                state: torch.Tensor):
        dist, prev_action = input
        assert type(dist) == dist_utils.DiagMultivariateNormal
        prev_mean, prev_stddev = state
        is_first = (prev_mean == 0).all(dim=1)[..., None]
        mu_bar = self._betac * (prev_action - prev_mean) / (
            prev_stddev + 1e-30)
        new_mean = dist.mean + mu_bar * dist.stddev
        loc = torch.where(is_first, dist.mean, new_mean)
        scale = torch.where(is_first, dist.stddev, dist.stddev * self._beta)

        if common.is_replay() and alf.summary.should_record_summaries():
            summary_utils.add_mean_hist_summary(
                "/CorrelatedDistributionAdpater/mu_bar", mu_bar)

        return dist_utils.DiagMultivariateNormal(loc, scale), (dist.mean,
                                                               dist.stddev)


def create_distribution_adapter(action_spec, alpha, beta):
    """Create a seed sampler for nested `action_spec`.

    :param action_spec: a nest of `BoundedTensorSpec`
    :param beta: see doc of `SeedSampler`
    :return: a `Network` for sampling from the nested action distribution
    """

    def _get_dist_spec(action_spec):
        assert action_spec.ndim == 1
        dim = action_spec.shape[0]
        dist = dist_utils.DiagMultivariateNormal(
            loc=torch.zeros((1, dim)), scale=torch.ones((1, dim)))
        return dist_utils.extract_spec(dist)

    return alf.nn.Sequential(
        lambda dist_and_prev_action: alf.nest.map_structure(
            lambda d, a: (d, a), *dist_and_prev_action),
        alf.nn.Parallel(
            alf.nest.map_structure(
                lambda spec: CorrelatedDistributionAdpater(
                    spec.shape[0], alpha, beta), action_spec)),
        input_tensor_spec=(alf.nest.map_structure(_get_dist_spec, action_spec),
                           action_spec))
