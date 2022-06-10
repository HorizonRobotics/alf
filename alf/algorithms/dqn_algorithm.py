# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""DQN Algorithm."""

import torch
from typing import Callable, Optional, Union

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.sac_algorithm import SacAlgorithm, ActionType, \
    SacState as DqnState, SacCriticState as DqnCriticState, \
    SacInfo as DqnInfo
from alf.algorithms.td_loss import TDLoss
from alf.data_structures import TimeStep
from alf.environments.alf_environment import AlfEnvironment
from alf.networks import QNetwork
from alf.optimizers import AdamTF
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils.schedulers import as_scheduler, Scheduler


@alf.configurable
class DqnAlgorithm(SacAlgorithm):
    r"""DQN/DDQN algorithm:

    ::

        Mnih et al "Playing Atari with Deep Reinforcement Learning", arXiv:1312.5602
        Hasselt et al "Deep Reinforcement Learning with Double Q-learning", arXiv:1509.06461

    The difference with DQN is that a minimum is taken from the two critics,
    similar to TD3, instead of choosing the maximum action using the Q network
    and evaluating the action value using the target Q network.

    The implementation is based on the SAC algorithm.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec: TensorSpec = TensorSpec(()),
                 q_network_cls: Callable[..., QNetwork] = QNetwork,
                 q_optimizer: Optional[torch.optim.Optimizer] = None,
                 rollout_epsilon_greedy: Union[float, Scheduler] = 0.1,
                 num_critic_replicas: int = 2,
                 env: Optional[AlfEnvironment] = None,
                 config: Optional[TrainerConfig] = None,
                 critic_loss_ctor: Optional[Callable[..., TDLoss]] = None,
                 debug_summaries: bool = False,
                 name: str = "DqnAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                be a mixture of discrete and continuous actions. The number of
                continuous actions can be arbitrary while only one discrete
                action is allowed currently. If it's a mixture, then it must be
                a tuple/list ``(discrete_action_spec, continuous_action_spec)``.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            q_network: is used to construct QNetwork for estimating ``Q(s,a)``
                given that the action is discrete. Its output spec must be consistent with
                the discrete action in ``action_spec``.
            q_optimizer: A custom optimizer for the q network.
                Uses the enclosing algorithm's optimizer if None.
            rollout_epsilon_greedy: epsilon greedy policy for rollout.
                Together with the following two parameters, the SAC algorithm
                can be converted to a DQN or DDQN algorithm when e.g.
                ``rollout_epsilon_greedy=0.3``, ``max_target_action=True``, and
                ``use_entropy_reward=False``.
            num_critic_replicas: number of critics to be used. Default is 2.
            env: The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config: config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            critic_loss_ctor: a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        self._rollout_epsilon_greedy = as_scheduler(rollout_epsilon_greedy)
        # Disable alpha learning:
        alpha_optimizer = AdamTF(lr=0)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=None,
            critic_network_cls=None,
            q_network_cls=q_network_cls,
            # Do not use entropy reward:
            use_entropy_reward=False,
            # Use uniform epsilon greedy action sampling
            uniform_epsilon_sample=True,
            num_critic_replicas=num_critic_replicas,
            env=env,
            config=config,
            critic_loss_ctor=critic_loss_ctor,
            # Allow custom optimizer for q_network:
            critic_optimizer=q_optimizer,
            alpha_optimizer=alpha_optimizer,
            debug_summaries=debug_summaries,
            name=name)
        assert self._act_type == ActionType.Discrete

    def rollout_step(self, inputs: TimeStep, state: DqnState):
        return super().rollout_step(
            inputs, state, eps=self._rollout_epsilon_greedy())

    def _critic_train_step(self, inputs: TimeStep, state: DqnCriticState,
                           rollout_info: DqnInfo, action, action_distribution):
        return super()._critic_train_step(
            inputs,
            state,
            rollout_info,
            action,
            action_distribution,
            # Pick the greedy target action:
            max_target_action=True)
