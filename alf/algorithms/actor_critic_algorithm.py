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

import alf
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.data_structures import TimeStep, AlgStep, namedtuple
from alf.utils import common, dist_utils, tensor_utils
from alf.tensor_specs import TensorSpec
from .config import TrainerConfig

ActorCriticState = namedtuple(
    "ActorCriticState", ["actor", "value"], default_value=())

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
                 epsilon_greedy=None,
                 env=None,
                 config: TrainerConfig = None,
                 loss=None,
                 loss_class=ActorCriticLoss,
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
        actor_network = actor_network_ctor(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        value_network = None
        if value_network_ctor is not None:
            value_network = value_network_ctor(
                input_tensor_spec=observation_spec)

            if reward_spec.numel > 1:
                value_network = value_network.make_parallel(
                    reward_spec.numel)  # value->[B,n]

        super(ActorCriticAlgorithm, self).__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            reward_weights=reward_weights,
            predict_state_spec=ActorCriticState(
                actor=actor_network.state_spec),
            train_state_spec=ActorCriticState(
                actor=actor_network.state_spec,
                value=value_network.state_spec if value_network else ()),
            env=env,
            config=config,
            optimizer=optimizer,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network
        self._value_network = value_network
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

    def convert_train_state_to_predict_state(self, state):
        return state._replace(value=())

    def predict_step(self, inputs: TimeStep, state: ActorCriticState):
        """Predict for one step."""
        action_dist, actor_state = self._actor_network(
            inputs.observation, state=state.actor)

        action = dist_utils.epsilon_greedy_sample(action_dist,
                                                  self._epsilon_greedy)
        return AlgStep(
            output=action,
            state=ActorCriticState(actor=actor_state),
            info=ActorCriticInfo(action_distribution=action_dist))

    def rollout_step(self, inputs: TimeStep, state: ActorCriticState):
        """Rollout for one step."""
        value, value_state = self._value_network(
            inputs.observation, state=state.value)

        action_distribution, actor_state = self._actor_network(
            inputs.observation, state=state.actor)

        action, log_prob = dist_utils.sample_action_distribution(
            action_distribution, return_log_prob=True)

        if self.has_multidim_reward():
            reward_weights = tensor_utils.tensor_extend_new_dim(
                self.reward_weights, dim=0, n=value.shape[0])
        else:
            reward_weights = ()
        return AlgStep(
            output=action,
            state=ActorCriticState(actor=actor_state, value=value_state),
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
