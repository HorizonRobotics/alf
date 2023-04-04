# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Phasic Policy Gradient Algorithm."""

from __future__ import annotations
import torch

from typing import Callable, Optional

import alf
from alf.algorithms.ppg import DisjointPolicyValueNetwork, PPGRolloutInfo, PPGTrainInfo, PPGAuxAlgorithm, PPGAuxOptions, ppg_network_forward
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.ppo_loss import PPOLoss
from alf.networks import Network, EncodingNetwork
from alf.data_structures import TimeStep, AlgStep, LossInfo, make_experience
from alf.tensor_specs import TensorSpec


# TODO(breakds): When needed, implement the support for multi-dimensional reward.
@alf.configurable
class PPGAlgorithm(OffPolicyAlgorithm):
    """PPG Algorithm.

    Implementation of the paper: https://arxiv.org/abs/2009.04416

    PPG can be viewed as a variant of PPO, with two differences:

    1. It uses a special network structure (DisjointPolicyValueNetwork) that has
       an extra auxiliary value head in addition to the policy head and value
       head. In the current implementation, the auxiliary value head also tries
       to estimate the value function, similar to the (actual) value head.

    2. It does PPO update in normal iterations. However, after every specified
       number of iterations, it will perform auxiliary phase updates based on
       auxiliary phase losses (different from PPO loss, see
       algorithms/ppg/ppg_aux_phase_loss.py for details). Auxiliary phase
       updates does not require new rollouts. Instead it is performed on all of
       the experience collected since the last auxiliary phase update.

    """

    def __init__(
            self,
            observation_spec,
            action_spec,
            reward_spec=TensorSpec(()),
            env=None,
            config: Optional[TrainerConfig] = None,
            aux_options: PPGAuxOptions = PPGAuxOptions(),
            encoding_network_ctor: Callable[..., Network] = EncodingNetwork,
            policy_optimizer: Optional[torch.optim.Optimizer] = None,
            aux_optimizer: Optional[torch.optim.Optimizer] = None,
            epsilon_greedy=None,
            checkpoint: Optional[str] = None,
            debug_summaries: bool = False,
            name: str = "PPGAlgorithm"):
        """Args:

            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            env (Environment): The environment to interact with. env is a
                batched environment, which means that it runs multiple
                simulations simultateously. env only needs to be provided to the
                root Algorithm. NOTE: env will default to None if PPGAlgorithm
                is run via Agent.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            aux_options: Options that controls the auxiliary phase training.
            encoding_network_ctor (Callable[[TensorSpec], Network]): Function to
                construct the encoding network from an input tensor spec. The
                constructed network will be called with ``forward(observation,
                state)``.
            policy_optimizer (torch.optim.Optimizer): The optimizer for training
                the policy phase of PPG.
            aux_optimizer (torch.optim.Optimizer): The optimizer for training
                the auxiliary phase of PPG.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
                It is used in ``predict_step()`` during evaluation.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.

        """
        dual_actor_value_network = DisjointPolicyValueNetwork(
            observation_spec=observation_spec,
            action_spec=action_spec,
            encoding_network_ctor=encoding_network_ctor)

        super().__init__(
            config=config,
            env=env,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            predict_state_spec=dual_actor_value_network.state_spec,
            train_state_spec=dual_actor_value_network.state_spec,
            checkpoint=checkpoint,
            optimizer=policy_optimizer)

        # When aux phase update is enabled, a sub algorithm named
        # "PPGAuxAlgorithm" will be created. The sub algorithm shares the same
        # network as the main algorithm, but updates the parameters with a
        # different loss and optimizer. ``_trainable_attributes_to_ignore()`` is
        # defined to prevent the network parameters being managed by two
        # different optimizers.
        if aux_options.enabled:
            self._aux_algorithm = PPGAuxAlgorithm(
                observation_spec=observation_spec,
                action_spec=action_spec,
                reward_spec=reward_spec,
                config=config,
                optimizer=aux_optimizer,
                dual_actor_value_network=dual_actor_value_network,
                aux_options=aux_options,
                debug_summaries=debug_summaries)
        else:
            # A None ``_aux_algorithm`` means not performaning aux
            # phase update at all.
            self._aux_algorithm = None

        self._network = dual_actor_value_network
        self._loss = PPOLoss(debug_summaries=debug_summaries)

        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._predict_step_epsilon_greedy = epsilon_greedy
        self._ensure_summary = alf.summary.EnsureSummary()

    def _trainable_attributes_to_ignore(self):
        return ['_aux_algorithm']

    def rollout_step(self, inputs: TimeStep, state) -> AlgStep:
        """Rollout step for PPG algorithm

        Besides running the network prediction, it does one extra thing to store
        the experience in the auxiliary replay buffer so that it can be consumed
        by the auxiliary phase updates.

        """
        policy_step = ppg_network_forward(self._network, inputs, state)

        if self._aux_algorithm:
            exp = make_experience(inputs.cpu(), policy_step, state)
            self._aux_algorithm.observe_for_aux_replay(exp)

        return policy_step

    def train_step(self, inputs: TimeStep, state,
                   plain_rollout_info: PPGRolloutInfo) -> AlgStep:
        alg_step = ppg_network_forward(
            self._network, inputs, state, require_aux=False)

        train_info = PPGTrainInfo(
            action=plain_rollout_info.action,
            rollout_log_prob=plain_rollout_info.log_prob,
            rollout_value=plain_rollout_info.value,
            rollout_action_distribution=plain_rollout_info.
            action_distribution).absorbed(alg_step.info)

        return alg_step._replace(info=train_info)

    def calc_loss(self, info: PPGTrainInfo) -> LossInfo:
        return self._loss(info)

    def predict_step(self, inputs: TimeStep, state):
        return ppg_network_forward(
            self._network,
            inputs,
            state,
            epsilon_greedy=self._predict_step_epsilon_greedy)

    def after_train_iter(self, experience, info: PPGTrainInfo):
        """Run auxiliary update if conditions are met

        PPG requires running auxiliary update after certain number of
        iterations policy update. This is checked and performed at the
        after_train_iter() hook currently.

        """
        if not self._aux_algorithm:
            return

        self._ensure_summary.tick()

        if alf.summary.get_global_counter(
        ) % self._aux_algorithm.interval == 0:
            with self._ensure_summary:
                with alf.summary.scope(self._aux_algorithm.name):
                    self._aux_algorithm.train_from_replay_buffer(
                        update_global_counter=False)
