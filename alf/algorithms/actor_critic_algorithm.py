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

import functools
from typing import Callable

import gin.tf

import tensorflow as tf

from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.encoding_network import EncodingNetwork
from tf_agents.networks.network import Network, DistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.trajectories.policy_step import PolicyStep

from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.entropy_target_algorithm import EntropyTargetAlgorithm
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.rl_algorithm import ActionTimeStep, TrainingInfo, LossInfo, namedtuple

import sys

ActorCriticState = namedtuple(
    "ActorCriticPolicyState", ["actor", "value", "icm"], default_value=())

ActorCriticInfo = namedtuple(
    "ActorCriticInfo", ["value", "icm", "entropy_target"], default_value=())

ActorCriticAlgorithmLossInfo = namedtuple(
    "ActorCriticAlgorithmLossInfo", ["ac", "icm", "entropy_target"],
    default_value=())


@gin.configurable
class ActorCriticAlgorithm(OnPolicyAlgorithm):
    def __init__(self,
                 action_spec,
                 actor_network: DistributionNetwork,
                 value_network: Network,
                 encoding_network: Network = None,
                 intrinsic_curiosity_module=None,
                 intrinsic_reward_coef=1.0,
                 extrinsic_reward_coef=1.0,
                 enforce_entropy_target=False,
                 loss=None,
                 loss_class=ActorCriticLoss,
                 optimizer=None,
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 reward_shaping_fn: Callable = None,
                 train_step_counter=None,
                 debug_summaries=False,
                 name="ActorCriticAlgorithm"):
        """Create an ActorCriticAlgorithm

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            actor_network (DistributionNetwork): A network that returns nested
                tensor of action distribution for each observation given observation
                and network state.
            value_network (Network): A function that returns value tensor from neural
                net predictions for each observation given observation and nwtwork
                state.
            encoding_network (Network): A function that encodes the observation
            intrinsic_curiosity_module (Algorithm): an algorithm whose outputs
                is a scalar intrinsid reward
            intrinsic_reward_coef (float): Coefficient for intrinsic reward
            extrinsic_reward_coef (float): Coefficient for extrinsic reward
            enforce_entropy_target (bool): If True, use EntropyTargetAlgorithm
                to dynamically adjust entropy regularization so that entropy is
                not smaller than `entropy_target` supplied for constructing
                EntropyTargetAlgorithm. If this is enabled, make sure you don't
                use entropy_regularization for loss (see ActorCriticLoss or
                PPOLoss).
            loss (None|ActorCriticLoss): an object for calculating loss. If
                None, a default loss of class loss_class will be used.
            loss_class (type): the class of the loss. The signature of its
                constructor: loss_class(action_spec, debug_summaries)
            optimizer (tf.optimizers.Optimizer): The optimizer for training
            gradient_clipping (float): If not None, serve as a positive threshold
                for clipping gradient norms
            clip_by_global_norm (bool): If True, use tf.clip_by_global_norm to
                clip gradient. If False, use tf.clip_by_norm for each grad.
            reward_shaping_fn (Callable): a function that transforms extrinsic
                immediate rewards
            train_step_counter (tf.Variable): An optional counter to increment.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
            """
        optimizers = [optimizer]
        module_sets = [[actor_network, value_network, encoding_network]]

        def _add_algorithm(algorithm: Algorithm):
            if algorithm:
                if algorithm.optimizer:
                    optimizers.append(algorithm.optimizer)
                    module_sets.append([algorithm])
                else:
                    module_sets[0].append(algorithm)

        train_state_spec = ActorCriticState(
            actor=actor_network.state_spec, value=value_network.state_spec)

        predict_state_spec = ActorCriticState(actor=actor_network.state_spec)

        if intrinsic_curiosity_module is not None:
            train_state_spec = train_state_spec._replace(
                icm=intrinsic_curiosity_module.train_state_spec)
            _add_algorithm(intrinsic_curiosity_module)

        entropy_target_algorithm = None
        if enforce_entropy_target:
            entropy_target_algorithm = EntropyTargetAlgorithm(
                action_spec, debug_summaries=debug_summaries)
            _add_algorithm(entropy_target_algorithm)

        def _collect_trainable_variables(modules):
            vars = []
            for module in modules:
                if module is not None:
                    vars = vars + list(module.trainable_variables)
            return vars

        get_trainable_variables_funcs = [
            functools.partial(_collect_trainable_variables, module_set)
            for module_set in module_sets
        ]

        super(ActorCriticAlgorithm, self).__init__(
            action_spec=action_spec,
            predict_state_spec=predict_state_spec,
            train_state_spec=train_state_spec,
            action_distribution_spec=actor_network.output_spec,
            optimizer=optimizers,
            get_trainable_variables_func=get_trainable_variables_funcs,
            gradient_clipping=gradient_clipping,
            clip_by_global_norm=clip_by_global_norm,
            reward_shaping_fn=reward_shaping_fn,
            train_step_counter=train_step_counter,
            debug_summaries=debug_summaries,
            name=name)

        self._entropy_target_algorithm = entropy_target_algorithm
        self._actor_network = actor_network
        self._value_network = value_network
        self._encoding_network = encoding_network
        self._intrinsic_reward_coef = intrinsic_reward_coef
        self._extrinsic_reward_coef = extrinsic_reward_coef
        if loss is None:
            loss = loss_class(action_spec, debug_summaries=debug_summaries)
        self._loss = loss
        self._icm = intrinsic_curiosity_module

    def _encode(self, time_step: ActionTimeStep):
        observation = time_step.observation
        if self._encoding_network is not None:
            observation, _ = self._encoding_network(observation)
        return observation

    def predict(self, time_step: ActionTimeStep, state: ActorCriticState):
        observation = self._encode(time_step)

        new_state = ActorCriticState()

        action_distribution, actor_state = self._actor_network(
            observation,
            step_type=time_step.step_type,
            network_state=state.actor)
        new_state = new_state._replace(actor=actor_state)

        return PolicyStep(action=action_distribution, state=new_state, info=())

    def rollout(self,
                time_step: ActionTimeStep,
                state: ActorCriticState,
                with_experience=False):
        new_state = ActorCriticState()
        info = ActorCriticInfo()
        observation = self._encode(time_step)

        if self._icm is not None:
            icm_step = self._icm.train_step(
                (observation, time_step.prev_action),
                state=state.icm,
                calc_intrinsic_reward=not with_experience)
            info = info._replace(icm=icm_step.info)
            new_state = new_state._replace(icm=icm_step.state)

        value, value_state = self._value_network(
            observation,
            step_type=time_step.step_type,
            network_state=state.value)
        # ValueRnnNetwork will add a time dim to value
        # See value_rnn_network.py L153
        #        if isinstance(self._value_network, ValueRnnNetwork):
        #            value = tf.squeeze(value, axis=1)

        new_state = new_state._replace(value=value_state)
        info = info._replace(value=value)

        action_distribution, actor_state = self._actor_network(
            observation,
            step_type=time_step.step_type,
            network_state=state.actor)
        new_state = new_state._replace(actor=actor_state)

        # TODO: can avoid computing this when collecting exps
        if self._entropy_target_algorithm:
            et_step = self._entropy_target_algorithm.train_step(
                action_distribution)
            info = info._replace(entropy_target=et_step.info)

        return PolicyStep(
            action=action_distribution, state=new_state, info=info)

    def calc_training_reward(self, external_reward, info: ActorCriticInfo):
        """Calculate the reward actually used for training.

        The training_reward includes both intrinsic reward (if there's any) and
        the external reward.
        Args:
            external_reward (Tensor): reward from environment
            info (ActorCriticInfo): (batched) policy_step.info from train_step()
        Returns:
            reward used for training.
        """
        if self._icm is not None:
            self.add_reward_summary("reward/icm", info.icm.reward)
            reward = (self._extrinsic_reward_coef * external_reward +
                      self._intrinsic_reward_coef * info.icm.reward)
            self.add_reward_summary("reward/overall", reward)
        else:
            reward = external_reward

        return reward

    def calc_loss(self, training_info):
        if self._icm is not None and isinstance(training_info.info.icm.reward,
                                                tf.Tensor):
            training_info = training_info._replace(
                reward=self.calc_training_reward(training_info.reward,
                                                 training_info.info))

        def _add(x, y):
            if not isinstance(y, tf.Tensor):
                return x
            elif not isinstance(x, tf.Tensor):
                return y
            else:
                return x + y

        def _update_loss(loss_info, training_info, name, algorithm):
            if algorithm is None:
                return loss_info
            new_loss_info = algorithm.calc_loss(
                getattr(training_info.info, name))
            return LossInfo(
                loss=_add(loss_info.loss, new_loss_info.loss),
                scalar_loss=_add(loss_info.scalar_loss,
                                 new_loss_info.scalar_loss),
                extra=loss_info.extra._replace(**{name: new_loss_info.extra}))

        ac_loss = self._loss(training_info, training_info.info.value)
        loss_info = LossInfo(
            loss=ac_loss.loss,
            scalar_loss=ac_loss.scalar_loss,
            extra=ActorCriticAlgorithmLossInfo(ac=ac_loss.extra))

        loss_info = _update_loss(loss_info, training_info, 'icm', self._icm)
        loss_info = _update_loss(loss_info, training_info, 'entropy_target',
                                 self._entropy_target_algorithm)

        return loss_info
