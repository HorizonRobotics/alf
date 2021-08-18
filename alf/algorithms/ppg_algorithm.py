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
from alf.data_structures import namedtuple
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
from alf.experience_replayers.replay_buffer import ReplayBuffer
from alf.data_structures import TimeStep, AlgStep, LossInfo, make_experience
from alf.tensor_specs import TensorSpec

from alf.utils import common, dist_utils, value_ops, tensor_utils
from alf.utils.summary_utils import record_time

# Data structure to store the options for PPG's auxiliary phase
# training iterations.
PPGAuxOptions = namedtuple(
    'PPGAuxOptions',
    [
        # Set to False to disable aux phase for good so that the PPG algorithm
        # degenerate to PPO
        'enabled',

        # Perform one auxiliary phase update after this number of normal
        # training iterations
        'interval',

        # Counterpart to TrainerConfig.mini_batch_length for aux phase. When set
        # to None, it will automatically overriden by unroll_length.
        'mini_batch_length',

        # Counterpart to TrainerConfig.mini_batch_size for aux phase
        'mini_batch_size',

        # Counterpart to TrainerConfig.num_updates_per_train_iter for aux phase
        'num_updates_per_train_iter',
    ],
    default_values={
        'enabled': True,
        'interval': 32,
        'mini_batch_length': None,
        'mini_batch_size': 8,
        'num_updates_per_train_iter': 6,
    })


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

    def __init__(self, ppg_algorithm: PPGAlgorithm,
                 network: DisjointPolicyValueNetwork,
                 optimizer: torch.optim.Optimizer, loss: Loss,
                 replay_buffer: Optional[ReplayBuffer]):
        """Establish the context and construct the context instance

        Note that the constructor also register the optimizer and the replay
        buffer.

        Args:

            ppg_algorithm (PPGAlgorithm): reference to the PPGAlgorithm which
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
            replay_buffer: the replay buffer to draw experience for
                training during the corresponding phase. If set to
                None, it means that in this phase the main replay
                buffer in the algorithm will be used.

        """
        self._ppg_algorithm = ppg_algorithm
        self._network = network
        self._ppg_algorithm.add_optimizer(optimizer, [self._network])
        self._loss = loss
        self._replay_buffer = replay_buffer

    @property
    def loss(self):
        return self._loss

    @property
    def replay_buffer(self):
        return self._replay_buffer

    def ensure_replay_buffer(self, sample_exp, max_length, name):
        if self._replay_buffer is not None:
            return

        exp_spec = dist_utils.to_distribution_param_spec(
            dist_utils.extract_spec(sample_exp, from_dim=1))
        num_environments = sample_exp.env_id.shape[0]
        self._replay_buffer = ReplayBuffer(
            data_spec=exp_spec,
            num_environments=self._replay_buffer_num_envs,
            max_length=self._replay_buffer_max_length,
            prioritized_sampling=self._prioritized_sampling,
            num_earliest_frames_ignored=self._num_earliest_frames_ignored,
            name=name)

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

    def network_forward(self,
                        inputs: TimeStep,
                        state,
                        epsilon_greedy: Optional[float] = None) -> AlgStep:
        """Evaluates the underlying network for roll out or training

        The signature mimics ``rollout_step()`` of ``Algorithm`` completedly.

        Args:

            inputs (TimeStep): carries the observation that is needed
                as input to the network
            state (nested Tesnor): carries the state for RNN-based network
            epsilon_greedy (Optional[float]): if set to None, the action will be
                sampled strictly based on the action distribution. If set to a
                value in [0, 1], epsilon-greedy sampling will be used to sample
                the action from the action distribution, and the float value
                determines the chance of action sampling instead of taking
                argmax.

        """
        (action_distribution, value, aux), state = self._network(
            inputs.observation, state=state)

        if epsilon_greedy is not None:
            action = dist_utils.epsilon_greedy_sample(action_distribution,
                                                      epsilon_greedy)
        else:
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
                 aux_options: PPGAuxOptions = PPGAuxOptions(),
                 env=None,
                 config: Optional[TrainerConfig] = None,
                 encoding_network_ctor: callable = EncodingNetwork,
                 policy_optimizer: Optional[torch.optim.Optimizer] = None,
                 aux_optimizer: Optional[torch.optim.Optimizer] = None,
                 epsilon_greedy=None,
                 debug_summaries: bool = False,
                 name: str = "PPGAlgorithm"):
        """Args:

            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            aux_phase_interval (int): perform auxiliary phase update after every
                aux_phase_interval iterations
            env (Environment): The environment to interact with. env is a
                batched environment, which means that it runs multiple
                simulations simultateously. env only needs to be provided to the
                root Algorithm. NOTE: env will default to None if PPGAlgorithm
                is run via Agent.
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
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``alf.get_config_value(TrainerConfig.epsilon_greedy)``. It
                is used in ``predict_step()`` during evaluation.
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
            train_state_spec=dual_actor_value_network.state_spec)

        self._aux_options = aux_options

        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value(
                'TrainerConfig.epsilon_greedy')
        self._predict_step_epsilon_greedy = epsilon_greedy

        # The policy phase uses the main replay buffer with the PPO Loss
        self._policy_phase = PPGPhaseContext(
            ppg_algorithm=self,
            network=dual_actor_value_network,
            optimizer=policy_optimizer,
            loss=PPOLoss(debug_summaries=debug_summaries),
            replay_buffer=None)

        # The auxiliary phase uses an extra replay buffer with the loss
        # specific to the auxiliary update phase.
        self._aux_phase = PPGPhaseContext(
            ppg_algorithm=self,
            network=dual_actor_value_network.copy(),
            optimizer=aux_optimizer,
            loss=PPGAuxPhaseLoss(),
            replay_buffer=None)

        # Register the two networks so that they are picked up as part of the
        # Algorithm by pytorch.
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
        previous_replay_buffer = None

        original_mini_batch_length = self._config.mini_batch_length
        original_mini_batch_size = self._config.mini_batch_size
        original_num_updates_per_train_iter = self._config.num_updates_per_train_iter

        try:
            self._aux_phase.on_context_switch_from(self._policy_phase)
            self._active_phase = self._aux_phase
            # Special handling to shadow the current _replay_buffer
            previous_replay_buffer = self._replay_buffer
            self._replay_buffer = self._aux_phase.replay_buffer
            self._config.mini_batch_length = (
                self._aux_options.mini_batch_length
                or self._config.unroll_length)
            self._config.mini_batch_size = self._aux_options.mini_batch_size
            self._config.num_updates_per_train_iter = self._aux_options.num_updates_per_train_iter
            yield
        finally:
            self._policy_phase.on_context_switch_from(self._aux_phase)
            self._active_phase = self._policy_phase
            # Restore the _replay_buffer
            self._replay_buffer = previous_replay_buffer
            self._config.mini_batch_length = original_mini_batch_length
            self._config.mini_batch_size = original_mini_batch_size
            self._config.num_updates_per_train_iter = original_num_updates_per_train_iter

    @property
    def on_policy(self) -> bool:
        return False

    def _observe_for_aux_replay(self, inputs: TimeStep, state,
                                policy_step: AlgStep):
        """Construct the experience and save it in the replay buffer for auxiliary
        phase update.

        Args:

            inputs (nested Tensor): inputs towards network prediction
            state (nested Tensor): state for RNN-based network
            policy_step (AlgStep): a data structure wrapping the information
                fromm the rollout

        """
        if not self._aux_options.enabled:
            return

        # Note that we need to release the ``untransformed`` from the time step
        # (inputs) before constructing the experience.
        lite_time_step = inputs._replace(untransformed=())
        exp = make_experience(lite_time_step, policy_step, state)
        if not self._use_rollout_state:
            exp = exp._replace(state=())
        # Set the experience spec explicitly if it is not set, based on this
        # (sample) experience
        if not self._experience_spec:
            self._experience_spec = dist_utils.extract_spec(exp, from_dim=1)
        exp = dist_utils.distributions_to_params(exp)

        if self._aux_phase._replay_buffer is None:
            exp_spec = dist_utils.to_distribution_param_spec(
                self._experience_spec)
            num_envs = exp.env_id.shape[0]
            max_length = self._config.unroll_length * self._aux_options.interval
            max_length += 1 + self._num_earliest_frames_ignored
            self._aux_phase._replay_buffer = ReplayBuffer(
                data_spec=exp_spec,
                num_environments=exp.env_id.shape[0],
                max_length=max_length,
                prioritized_sampling=False,
                num_earliest_frames_ignored=self._num_earliest_frames_ignored,
                name=f'aux_replay_buffer')

        self._aux_phase.replay_buffer.add_batch(exp, exp.env_id)

    def rollout_step(self, inputs: TimeStep, state) -> AlgStep:
        """Rollout step for PPG algorithm

        Besides running the network prediction, it does one extra thing to store
        the experience in the auxiliary replay buffer so that it can be consumed
        by the auxiliary phase updates.

        """
        policy_step = self._policy_phase.network_forward(inputs, state)
        self._observe_for_aux_replay(inputs.untransformed.cpu(), state,
                                     policy_step)
        return policy_step

    def train_step(self, inputs: TimeStep, state,
                   plain_rollout_info: PPGRolloutInfo) -> AlgStep:
        """Phase context dependent evaluation on experiences for training
        """
        alg_step = self._active_phase.network_forward(inputs, state)

        train_info = PPGTrainInfo(
            action=plain_rollout_info.action,
            rollout_action_distribution=plain_rollout_info.
            action_distribution).absorbed(alg_step.info)

        return alg_step._replace(info=train_info)

    def calc_loss(self, info: PPGTrainInfo) -> LossInfo:
        """Phase context dependent loss function evaluation
        """
        return self._active_phase.loss(info)

    def predict_step(self, inputs: TimeStep, state):
        """Predict for one step."""
        return self._active_phase.network_forward(
            inputs, state, self._predict_step_epsilon_greedy)

    def after_train_iter(self, experience, info: PPGTrainInfo):
        """Run auxiliary update if conditions are met

        PPG requires running auxiliary update after certain number of
        iterations policy update. This is checked and performed at the
        after_train_iter() hook currently.

        """
        if not self._aux_options.enabled:
            return

        if alf.summary.get_global_counter() % self._aux_options.interval == 0:
            with self.aux_phase_activated():
                # TODO(breakds): currently auxiliary update steps are not
                # counted towards the total steps. If needed in the future, we
                # should return this from after_train_iter() and add some logic
                # to handle this in the call site of after_train_iter().
                aux_steps = self.train_from_replay_buffer(
                    update_global_counter=False)
