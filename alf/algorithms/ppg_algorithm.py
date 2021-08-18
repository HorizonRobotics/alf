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

import torch

from typing import Optional, Tuple
from contextlib import contextmanager

import alf
from alf.algorithms.ppg import DisjointPolicyValueNetwork, PPGAuxPhaseLoss, PPGAuxPhaseLossInfo, PPGRolloutInfo, PPGTrainInfo
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.ppo_loss import PPOLoss
from alf.algorithms.algorithm import Loss
from alf.networks.encoding_networks import EncodingNetwork
from alf.experience_replayers.experience_replay import OnetimeExperienceReplayer
from alf.data_structures import TimeStep, AlgStep, LossInfo
from alf.tensor_specs import TensorSpec

from alf.utils import common, dist_utils, value_ops, tensor_utils
from alf.utils.summary_utils import record_time


class PPGPhaseContext(object):
    """The context for each of the two phases of PPG

    The PPG algorithm has two phases. Each phase will

    1. use a DIFFERENT "loss function"
    2. on a DIFFERENT "replay buffer"
    3. and use a DIFFERENT "optimizer"
    4. on a DIFFRENT copy of the "network"

    Therefore, this context class is created to group such (stateful)
    information for each phase so as to make switching between phases
    as easy as switching the context to use. See the implementation of
    the ``PPGAlgorithm`` for the details.

    Note that in the above list No.4 is not needed in theory because
    the two phases technically operates on the same network. However,
    two separate networks are used to bypass the current constraint
    that different optimizers do not share their managed parameters.

    """

    def __init__(self, main_algorithm: 'PPGAlgorithm',
                 network: DisjointPolicyValueNetwork,
                 optimizer: torch.optim.Optimizer, loss: Loss,
                 exp_replayer: Optional[OnetimeExperienceReplayer]):
        """Establish the context and construct the context instance

        Note that the constructor also register the optimizer and the replay
        buffer (experience replayer)

        Args:

            main_algorithm (PPGAlgorithm): reference to the PPGAlgorithm which
                will be used to register the observer and the optimizer
            network (DisjointPolicyValueNetwork): the network to use during
                training in the corresponding phase
            optimizer (torch.optim.Optimizer): the optimizer to use during
                training in the corresponding phase. Actaully all registered
                optimizers' ``step()`` will be called during training updates no
                matter which phase it is in, but because the loss is only
                computed on the network of the active phase, the other
                optimizers will effectively do nothing
            loss (Loss): the loss function that will be used to compute the loss
                in the corresponding phase
            exp_replayer (Optional[ExperienceReplayer]): the replay buffer to
                draw experience for training during the corresponding phase. If
                set to None, it means that in this phase the main replay buffer
                in the algorithm will be used.

        """
        self._main_algorithm = main_algorithm
        self._network = network
        self._main_algorithm.add_optimizer(optimizer, [self._network])
        self._loss = loss
        self._exp_replayer = exp_replayer
        if self._exp_replayer is not None:
            self._main_algorithm._observers.append(self._exp_replayer.observe)

    @property
    def loss(self):
        return self._loss

    @property
    def exp_replayer(self):
        return self._exp_replayer

    @property
    def network(self):
        return self._network

    def on_context_switch_from(self, previous_context: 'PPGPhaseContext'):
        """Necessary operations when switching from another context

        Currently it is used to handle the special treatment of inheriting the
        state_dict of the (identical) network from the previous context, so that
        in this training phase the parameter updates continue on the most
        update-to-date parameter values.

        Args:

            previous_context (PPGPhaseContext): the context of the phase that it
                is switching from

        """
        self._network.load_state_dict(previous_context._network.state_dict())

    def network_forward(self, inputs: TimeStep, state) -> AlgStep:
        """Evaluates the underlying network for roll out or training

        The signature mimics ``rollout_step()`` of ``Algorithm`` completedly.

        """
        (action_distribution, value, aux), state = self._network(
            inputs.observation, state=state)

        action = dist_utils.sample_action_distribution(action_distribution)

        return AlgStep(
            output=action,
            state=state,
            info=PPGRolloutInfo(
                action_distribution=action_distribution,
                action=common.detach(action),
                value=value,
                aux=aux,
                step_type=inputs.step_type,
                discount=inputs.discount,
                reward=inputs.reward,
                reward_weights=()))


# TODO(breakds): When needed, implement the support for multi-dimensional reward.
@alf.configurable
class PPGAlgorithm(OffPolicyAlgorithm):
    """PPG Algorithm.

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

    def __init__(self,
                 observation_spec: TensorSpec,
                 action_spec: TensorSpec,
                 reward_spec=TensorSpec(()),
                 aux_phase_interval: int = 32,
                 env=None,
                 config: Optional[TrainerConfig] = None,
                 encoding_network_ctor: callable = EncodingNetwork,
                 policy_optimizer: Optional[torch.optim.Optimizer] = None,
                 aux_optimizer: Optional[torch.optim.Optimizer] = None,
                 debug_summaries: bool = False,
                 name: str = "PPGAlgorithm"):
        """Args:

            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            aux_phase_interval (int): perform auxiliary phase update after every
                aux_phase_interval iterations
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            encoding_network_ctor (Callable[..., Network]): Function to
                construct the encoding network. The constructed network will be
                called with ``forward(observation, state)``.
            policy_optimizer (torch.optim.Optimizer): The optimizer for training
                the policy phase of PPG.
            aux_optimizer (torch.optim.Optimizer): The optimizer for training
                the auxiliary phase of PPG.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.

        """

        dual_actor_value_network = DisjointPolicyValueNetwork(
            observation_spec=observation_spec,
            action_spec=action_spec,
            encoding_network_ctor=encoding_network_ctor,
            is_sharing_encoder=False)

        super().__init__(
            config=config,
            env=env,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            predict_state_spec=dual_actor_value_network.state_spec,
            train_state_spec=dual_actor_value_network.state_spec)

        self._aux_phase_interval = aux_phase_interval

        # The policy phase uses the main experience replayer with the PPO Loss
        self._policy_phase = PPGPhaseContext(
            main_algorithm=self,
            network=dual_actor_value_network,
            optimizer=policy_optimizer,
            loss=PPOLoss(debug_summaries=debug_summaries),
            exp_replayer=None)

        # The auxiliary phase uses an extra experience replayer with the loss
        # specific to the auxiliary update phase.
        self._aux_phase = PPGPhaseContext(
            main_algorithm=self,
            network=dual_actor_value_network.copy(),
            optimizer=aux_optimizer,
            loss=PPGAuxPhaseLoss(),
            exp_replayer=OnetimeExperienceReplayer())

        # Register the two networks so that they are picked up as part of the
        # Algorithm by ptyroch.
        self._registered_networks = torch.nn.ModuleList(
            [self._policy_phase.network, self._aux_phase.network])

        self._active_phase = self._policy_phase

    @contextmanager
    def aux_phase_activated(self):
        """A context switch that activates the auxiliary phase

        ... code-block:: python
            with self.aux_phase_activated():
                self.train_from_replay_buffer(...)

        In the above code snippet, the hooks called in the
        ``train_from_replay_buffer()`` function will be operating with the
        network, replay buffer and loss in the auxiliary phase contex.

        """
        backup_exp_replayer = None
        try:
            self._aux_phase.on_context_switch_from(self._policy_phase)
            self._active_phase = self._aux_phase
            # Special handling to shadow the current _exp_replayer
            backup_exp_replayer = self._exp_replayer
            self._exp_replayer = self._aux_phase.exp_replayer
            yield
        finally:
            self._policy_phase.on_context_switch_from(self._aux_phase)
            self._active_phase = self._policy_phase
            # Restore the _exp_replayer
            self._exp_replayer = backup_exp_replayer

    def train_iter(self):
        """PPG operates in a (slightly) different pipleine

        Overrides the original ``train_iter()`` to achieve that

        """
        config: TrainerConfig = self._config

        if not config.update_counter_every_mini_batch:
            alf.summary.increment_global_counter()

        # Run unroll() with environments
        with torch.set_grad_enabled(config.unroll_with_grad):
            with record_time("time/unroll"):
                self.eval()
                experience = self.unroll(config.unroll_length)
                self.summarize_rollout(experience)
                self.summarize_metrics()

        # Run the normal training
        self.train()
        steps = self.train_from_replay_buffer(update_global_counter=True)

        # This is the only difference from the original pipeline, where
        # periodically an extra auxiliary phase update is done.
        if alf.summary.get_global_counter() % self._aux_phase_interval == 0:
            with self.aux_phase_activated():
                print('Aux!')
                steps += self.train_from_replay_buffer(
                    update_global_counter=False)

        with record_time("time/after_train_iter"):
            train_info = experience.rollout_info
            experience = experience._replace(rollout_info=())
            self.after_train_iter(experience, train_info)

        # For now, we only return the steps of the primary algorithm's training
        return steps

    @property
    def on_policy(self) -> bool:
        return False

    def rollout_step(self, inputs: TimeStep, state) -> AlgStep:
        return self._policy_phase.network_forward(inputs, state)

    def preprocess_experience(
            self,
            inputs: TimeStep,  # nest of [B, T, ...]
            rollout_info: PPGRolloutInfo,
            batch_info) -> Tuple[TimeStep, PPGTrainInfo]:
        """Phase context dependent preprocessing

        This hook compute the advantages for policy gradient updates and returns
        for value targets (used in TD-error losses).

        Since both phase has TD-error losses, the returns are needed (which
        implies that the advantages are needed too). Therefore, the computation
        for both phases are identical, except on the context that is currently
        active.

        """

        with torch.no_grad():
            gamma = self._policy_phase.loss.gamma

            if rollout_info.reward.ndim == 3:
                # [B, T, D] or [B, T, 1]
                discounts = rollout_info.discount.unsqueeze(-1) * gamma
            else:
                # [B, T]
                discounts = rollout_info.discount * gamma

            td_lambda = self._policy_phase.loss._lambda

            advantages = value_ops.generalized_advantage_estimation(
                rewards=rollout_info.reward,
                values=rollout_info.value,
                step_types=rollout_info.step_type,
                discounts=discounts,
                td_lambda=td_lambda,
                time_major=False)
            advantages = tensor_utils.tensor_extend_zero(advantages, dim=1)
            returns = rollout_info.value + advantages

        return inputs, PPGTrainInfo(
            action=rollout_info.action,
            rollout_action_distribution=rollout_info.action_distribution,
            returns=returns,
            advantages=advantages).absorbed(rollout_info)

    def train_step(self, inputs: TimeStep, state,
                   prev_train_info: PPGTrainInfo) -> AlgStep:
        """Phase context dependent evaluation on experiences for training
        """
        alg_step = self._active_phase.network_forward(inputs, state)
        return alg_step._replace(info=prev_train_info.absorbed(alg_step.info))

    def calc_loss(self, info: PPGTrainInfo) -> LossInfo:
        """Phase context dependent loss function evaluation
        """
        return self._active_phase.loss(info)
