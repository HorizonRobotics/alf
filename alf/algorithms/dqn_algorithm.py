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
import torch.distributions as td
from typing import Callable, Optional, Union

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.sac_algorithm import SacAlgorithm, ActionType, \
    SacState as DqnState, SacCriticState as DqnCriticState, \
    SacActionState as DqnActionState, \
    SacInfo as DqnInfo, SacCriticInfo as DqnCriticInfo
from alf.algorithms.td_loss import TDLoss
from alf.data_structures import AlgStep, TimeStep
from alf.environments.alf_environment import AlfEnvironment
from alf.networks import QNetwork
from alf.optimizers import AdamTF
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, dist_utils
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

    # Copied and modified from sac_algorithm (discrete actions).
    def _predict_action(self,
                        observation,
                        state: DqnActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        rollout=False):
        """The reason why we want to do action sampling inside this function
        instead of outside is that for the mixed case, once a continuous action
        is sampled here, we should pair it with the discrete action sampled from
        the Q value. If we just return two distributions and sample outside, then
        the actions will not match.
        """
        new_state = DqnActionState()

        critic_network_inputs = (observation, None)
        if self._act_type == ActionType.Mixed:
            critic_network_inputs = (observation, (None, continuous_action))

        q_values, critic_state = self._compute_critics(
            self._critic_networks, *critic_network_inputs, state.critic)

        new_state = new_state._replace(critic=critic_state)
        if self._act_type == ActionType.Discrete:
            alpha = torch.exp(self._log_alpha).detach()
        else:
            alpha = torch.exp(self._log_alpha[0]).detach()
        # p(a|s) = exp(Q(s,a)/alpha) / Z;
        logits = q_values / alpha
        discrete_action_dist = td.Categorical(logits=logits)
        if eps_greedy_sampling:
            discrete_action = dist_utils.epsilon_greedy_sample(
                discrete_action_dist,
                epsilon_greedy,
                # NOTE: This is the only departure from SAC:
                True)
        else:
            discrete_action = dist_utils.sample_action_distribution(
                discrete_action_dist)

        action_dist = discrete_action_dist
        action = discrete_action

        if (self._reproduce_locomotion and rollout
                and not self._training_started):
            # This uniform sampling seems important because for a squashed Gaussian,
            # even with a large scale, a random policy is not nearly uniform.
            action = alf.nest.map_structure(
                lambda spec: spec.sample(outer_dims=observation.shape[:1]),
                self._action_spec)

        return action_dist, action, q_values, new_state

    # Copied and modified from sac_algorithm (discrete actions).
    def rollout_step(self, inputs: TimeStep, state: DqnState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        action_dist, action, _, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            # NOTE: This is the only departure from SAC.
            epsilon_greedy=self._rollout_epsilon_greedy(),
            eps_greedy_sampling=True,
            rollout=True)

        if self.need_full_rollout_state():
            _, critics_state = self._compute_critics(
                self._critic_networks, inputs.observation, action,
                state.critic.critics)
            _, target_critics_state = self._compute_critics(
                self._target_critic_networks, inputs.observation, action,
                state.critic.target_critics)
            critic_state = DqnCriticState(
                critics=critics_state, target_critics=target_critics_state)
            actor_state = ()
        else:
            actor_state = state.actor
            critic_state = state.critic

        new_state = DqnState(
            action=action_state, actor=actor_state, critic=critic_state)
        return AlgStep(
            output=action,
            state=new_state,
            info=DqnInfo(action=action, action_distribution=action_dist))

    # Copied and modified from sac_algorithm (discrete actions).
    def _critic_train_step(self, inputs: TimeStep, state: DqnCriticState,
                           rollout_info: DqnInfo, action, action_distribution):
        critics, critics_state = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_info.action,
            state.critics,
            replica_min=False,
            apply_reward_weights=False)

        target_critics, target_critics_state = self._compute_critics(
            self._target_critic_networks,
            inputs.observation,
            action,
            state.target_critics,
            apply_reward_weights=False)

        critics = self._select_q_value(rollout_info.action, critics)
        # [B, num_actions] -> [B, num_actions, reward_dim]
        probs = common.expand_dims_as(action_distribution.probs,
                                      target_critics)

        # NOTE: This block is the only departure from SAC.
        target_critics_1d_rwd = target_critics
        if self.has_multidim_reward():
            target_critics_1d_rwd = self._apply_reward_weights(
                target_critics_1d_rwd)
        target_action_idx = torch.argmax(target_critics_1d_rwd, dim=1)
        # [B, reward_dim]
        target_critics = target_critics[(torch.arange(action.shape[0]),
                                         target_action_idx)]

        target_critic = target_critics.reshape(inputs.reward.shape)

        target_critic = target_critic.detach()

        state = DqnCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = DqnCriticInfo(critics=critics, target_critic=target_critic)

        return state, info
