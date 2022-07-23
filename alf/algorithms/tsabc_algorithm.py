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
"""Thompson Sampling Actor and Bayesian Critic Algorithm."""

import torch

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.algorithms.oabc_algorithm import OabcActionState, OabcAlgorithm, OabcState
from alf.data_structures import StepType, TimeStep
from alf.optimizers import AdamTF
from alf.networks import ActorDistributionNetwork
from alf.networks.param_networks import CriticDistributionParamNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import dist_utils, summary_utils


@alf.configurable
class TsabcAlgorithm(OabcAlgorithm):
    r"""Soft Actor and Bayesian Critic Algorithm. """
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 explore_network_cls=ActorDistributionNetwork,
                 critic_module_cls=FuncParVIAlgorithm,
                 num_critic_replicas=10,
                 deterministic_actor=True,
                 deterministic_critic=False,
                 reward_weights=None,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 beta_ub=1.,
                 beta_lb=1.,
                 common_td_target: bool = True,
                 random_actor_every_step: bool = True,
                 entropy_regularization_weight=1.,
                 entropy_regularization=None,
                 target_entropy=None,
                 initial_log_alpha=0.0,
                 max_log_alpha=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 explore_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 explore_alpha_optimizer=None,
                 debug_summaries=False,
                 name="TsabcAlgorithm"):
        """
        Args:
            explore_network_cls
            critic_module_cls
            deterministic_actor
            deterministic_critic
            beta_ub (float): parameter for computing the upperbound of Q value:
                :math:`Q_ub(s,a) = \mu_Q(s,a) + \beta_ub * \sigma_Q(s,a)`
            beta_lb
            explore_optimizer
            explore_alpha_optimizer
        """
        self._idx = 0
        # self._cyclic_unroll_steps = 0
        self._random_actor_every_step = random_actor_every_step
        self._num_critic_replicas = num_critic_replicas

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            explore_network_cls=explore_network_cls,
            critic_module_cls=critic_module_cls,
            deterministic_actor=deterministic_actor,
            deterministic_critic=deterministic_critic,
            reward_weights=reward_weights,
            epsilon_greedy=epsilon_greedy,
            use_entropy_reward=use_entropy_reward,
            env=env,
            config=config,
            critic_loss_ctor=critic_loss_ctor,
            beta_ub=beta_ub,
            beta_lb=beta_lb,
            common_td_target=common_td_target,
            entropy_regularization_weight=entropy_regularization_weight,
            entropy_regularization=entropy_regularization,
            target_entropy=target_entropy,
            initial_log_alpha=initial_log_alpha,
            max_log_alpha=max_log_alpha,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            dqda_clipping=dqda_clipping,
            actor_optimizer=actor_optimizer,
            explore_optimizer=explore_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            explore_alpha_optimizer=explore_alpha_optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._explore_networks = self._explore_network

    def _make_modules(self, observation_spec, action_spec, reward_spec,
                      actor_network_cls, explore_network_cls,
                      critic_module_cls, critic_optimizer,
                      deterministic_critic):

        assert actor_network_cls is not None, (
            "ActorNetwork must be provided!")
        actor_network = actor_network_cls(input_tensor_spec=observation_spec,
                                          action_spec=action_spec)

        assert explore_network_cls is not None, (
            "ExploreNetwork must be provided!")
        explore_network = explore_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        # explore_networks = nn.ModuleList()
        # for i in range(num_critic_replicas):
        #     explore_networks.append(explore_network_cls(
        #         input_tensor_spec=observation_spec,
        #         action_spec=action_spec))

        # explore_state_spec = alf.nest.map_structure(
        #     lambda spec: alf.TensorSpec((num_critic_replicas, ) + spec.shape,
        #                                  spec.dtype),
        #     explore_network.state_spec)

        input_tensor_spec = (observation_spec, action_spec)
        critic_network = CriticDistributionParamNetwork(
            input_tensor_spec=input_tensor_spec,
            output_tensor_spec=reward_spec,
            deterministic=deterministic_critic)
        target_critic_network = critic_network.copy(
            name='TargetCriticDistributionParamNetwork')

        if critic_optimizer is None:
            critic_optimizer = AdamTF(lr=3e-4)
        critic_module = critic_module_cls(
            input_tensor_spec=input_tensor_spec,
            param_net=critic_network,
            num_particles=self._num_critic_replicas,
            optimizer=critic_optimizer)

        explore_network = explore_network.make_parallel(
            self._num_critic_replicas)

        return actor_network, explore_network, critic_module, target_critic_network

    def _predict_action(self,
                        observation,
                        state: OabcActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False,
                        train=False):

        new_state = OabcActionState()
        if explore:
            # action_dist, explore_network_state = self._explore_network(
            #     observation, state=state.explore_network)
            # action, explore_network_state = self._explore_network(
            #     observation, state=state.explore_network)
            action_dist = ()
            action, explore_network_state = self._explore_networks(
                observation, state=state.explore_network)
            new_state = new_state._replace(
                explore_network=explore_network_state)
            # if train:
            # output_states = []
            # for i in range(self._num_critic_replicas):
            #     s = alf.nest.map_structure(
            #         lambda x: x[:, i, ...], state.explore_network)
            #     ret = self._explore_networks[i](observation, state=s)
            #     ret = alf.nest.map_structure(lambda x: x.unsqueeze(1), ret)
            #     output_states.append(ret)
            # action, explore_network_state = alf.nest.map_structure(
            #     lambda *tensors: torch.cat(tensors, dim=1), *output_states)
            # new_state = new_state._replace(explore_network=explore_network_state)
            # else:
            if not train:
                # if self._cyclic_unroll_steps == 0:
                if self._random_actor_every_step:
                    self._idx = torch.randint(self._num_critic_replicas, ())
                action = action[:, self._idx, :]
                # import pdb; pdb.set_trace()

                # action, explore_network_state = self._explore_networks[self._idx](
                #     observation, state=state.explore_network)

                # self._cyclic_unroll_steps += 1
                # if self._cyclic_unroll_steps >= 100:
                #     self._cyclic_unroll_steps = 0
        else:
            if self._deterministic_actor:
                action_dist = ()
                action, actor_network_state = self._actor_network(
                    observation, state=state.actor_network)

            else:
                action_dist, actor_network_state = self._actor_network(
                    observation, state=state.actor_network)

                if eps_greedy_sampling:
                    action = dist_utils.epsilon_greedy_sample(
                        action_dist, epsilon_greedy)
                else:
                    action = dist_utils.rsample_action_distribution(
                        action_dist)
            new_state = new_state._replace(actor_network=actor_network_state)

        return action_dist, action, new_state

    def rollout_step(self, inputs: TimeStep, state: OabcState):
        if inputs.step_type == StepType.FIRST:
            self._idx = torch.randint(self._num_critic_replicas, ())
        super().rollout_step(inputs, state)

    def _get_actor_train_q_value(self, critics, explore):
        q_mean = critics.mean(1)
        q_std = critics.std(1)
        if explore:
            q_value = critics
        else:
            q_value = q_mean - self._beta_lb * q_std

        prefix = "explore_" if explore else ""
        with alf.summary.scope(self._name):
            summary_utils.add_mean_hist_summary(prefix + "critics_batch_mean",
                                                q_mean)
            summary_utils.add_mean_hist_summary(prefix + "critics_std", q_std)

        return q_value
