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

from collections import namedtuple
import functools
from typing import Callable

import gin.tf

import tensorflow as tf

from tf_agents.agents.tf_agent import LossInfo
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
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm, OffPolicyAdapter
from alf.algorithms.rl_algorithm import ActionTimeStep, TrainingInfo

ActorCriticState = namedtuple("ActorCriticPolicyState",
                              ["actor_state", "value_state", "icm_state"])

ActorCriticInfo = namedtuple(
    "ActorCriticInfo",
    ["value", "icm_reward", "icm_info", "entropy_target_info"])

ActorCriticAlgorithmLossInfo = namedtuple("ActorCriticAlgorithmLossInfo",
                                          ["ac", "icm", "entropy_target"])


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

        icm_state_spec = ()
        if intrinsic_curiosity_module is not None:
            icm_state_spec = intrinsic_curiosity_module.train_state_spec
        entropy_target_algorithm = None
        if enforce_entropy_target:
            entropy_target_algorithm = EntropyTargetAlgorithm(
                action_spec, debug_summaries=debug_summaries)

        optimizers = [optimizer]
        module_sets = [[actor_network, value_network, encoding_network]]

        def _add_algorithm(algorithm: Algorithm):
            if algorithm:
                if algorithm.optimizer:
                    optimizers.append(algorithm.optimizer)
                    module_sets.append([algorithm])
                else:
                    module_sets[0].append(algorithm)

        _add_algorithm(intrinsic_curiosity_module)
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
            predict_state_spec=actor_network.state_spec,
            train_state_spec=ActorCriticState(
                actor_state=actor_network.state_spec,
                value_state=value_network.state_spec,
                icm_state=icm_state_spec),
            action_distribution_spec=actor_network.output_spec,
            optimizer=optimizers,
            get_trainable_variables_func=get_trainable_variables_funcs,
            gradient_clipping=gradient_clipping,
            clip_by_global_norm=clip_by_global_norm,
            reward_shaping_fn=reward_shaping_fn,
            train_step_counter=train_step_counter,
            debug_summaries=debug_summaries,
            name=name)

        self._entropy_target_algortihm = entropy_target_algorithm
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

    def predict(self, time_step: ActionTimeStep, state=None):
        observation = self._encode(time_step)
        action_distribution, actor_state = self._actor_network(
            observation, step_type=time_step.step_type, network_state=state)
        return PolicyStep(
            action=action_distribution, state=actor_state, info=())

    def train_step(self, time_step: ActionTimeStep, state=None):
        observation = self._encode(time_step)

        value, value_state = self._value_network(
            observation,
            step_type=time_step.step_type,
            network_state=state.value_state)
        # ValueRnnNetwork will add a time dim to value
        # See value_rnn_network.py L153
        if isinstance(self._value_network, ValueRnnNetwork):
            value = tf.squeeze(value, axis=1)

        action_distribution, actor_state = self._actor_network(
            observation,
            step_type=time_step.step_type,
            network_state=state.actor_state)

        info = ActorCriticInfo(
            value=value, icm_reward=(), icm_info=(), entropy_target_info=())
        if self._icm is not None:
            icm_step = self._icm.train_step(
                (observation, time_step.prev_action), state=state.icm_state)
            info = info._replace(
                icm_reward=icm_step.outputs, icm_info=icm_step.info)
            icm_state = icm_step.state
        else:
            icm_state = ()

        if self._entropy_target_algortihm:
            et_step = self._entropy_target_algortihm.train_step(
                action_distribution)
            info = info._replace(entropy_target_info=et_step.info)

        state = ActorCriticState(
            actor_state=actor_state,
            value_state=value_state,
            icm_state=icm_state)

        return PolicyStep(action=action_distribution, state=state, info=info)

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
            return (self._extrinsic_reward_coef * external_reward +
                    self._intrinsic_reward_coef * info.icm_reward)
        else:
            return external_reward

    def calc_loss(self, training_info):
        if self._icm is not None:
            self.add_reward_summary("reward/intrinsic",
                                    training_info.info.icm_reward)

            training_info = training_info._replace(
                reward=self.calc_training_reward(training_info.reward,
                                                 training_info.info))

            self.add_reward_summary("reward/overall", training_info.reward)

        ac_loss = self._loss(training_info, training_info.info.value)
        loss = ac_loss.loss
        extra = ActorCriticAlgorithmLossInfo(
            ac=ac_loss.extra, icm=(), entropy_target=())

        if self._icm is not None:
            icm_loss = self._icm.calc_loss(training_info.info.icm_info)
            loss += icm_loss.loss
            extra = extra._replace(icm=icm_loss.extra)

        if self._entropy_target_algortihm:
            et_loss = self._entropy_target_algortihm.calc_loss(
                training_info.info.entropy_target_info)
            loss += et_loss.loss
            extra = extra._replace(entropy_target=et_loss.extra)

        return LossInfo(loss=loss, extra=extra)


@gin.configurable
def create_ac_algorithm(env,
                        actor_fc_layers=(200, 100),
                        value_fc_layers=(200, 100),
                        encoding_conv_layers=(),
                        encoding_fc_layers=(),
                        use_rnns=False,
                        use_icm=False,
                        learning_rate=5e-5,
                        off_policy=False,
                        debug_summaries=False):
    """Create a simple ActorCriticAlgorithm.

    Args:
        env (TFEnvironment): A TFEnvironment
        actor_fc_layers (list[int]): list of fc layers parameters for actor network
        value_fc_layers (list[int]): list of fc layers parameters for value network
        encoding_conv_layers (list[int]): list of convolution layers parameters for encoding network
        encoding_fc_layers (list[int]): list of fc layers parameters for encoding network
        use_rnns (bool): True if rnn should be used
        use_icm (bool): True if intrinsic curiosity module should be used
        learning_rate (float) : learning rate
        off_policy (bool) : True if used for off policy training
        debug_summaries (bool): True if debug summaries should be created.
    """
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    if use_rnns:
        actor_net = ActorDistributionRnnNetwork(
            env.observation_spec(),
            env.action_spec(),
            input_fc_layer_params=actor_fc_layers,
            output_fc_layer_params=None)
        value_net = ValueRnnNetwork(
            env.observation_spec(),
            input_fc_layer_params=value_fc_layers,
            output_fc_layer_params=None)
    else:
        actor_net = ActorDistributionNetwork(
            env.observation_spec(),
            env.action_spec(),
            fc_layer_params=actor_fc_layers)
        value_net = ValueNetwork(
            env.observation_spec(), fc_layer_params=value_fc_layers)

    encoding_net = None
    if encoding_fc_layers or encoding_conv_layers:
        encoding_net = EncodingNetwork(
            input_tensor_spec=env.observation_spec(),
            conv_layer_params=encoding_conv_layers,
            fc_layer_params=encoding_fc_layers)

    icm = None
    if use_icm:
        feature_spec = env.observation_spec()
        if encoding_net:
            feature_spec = tf.TensorSpec((encoding_fc_layers[-1], ),
                                         dtype=tf.float32)
        icm = ICMAlgorithm(
            env.action_spec(), feature_spec, encoding_net=encoding_net)

    algorithm = ActorCriticAlgorithm(
        action_spec=env.action_spec(),
        actor_network=actor_net,
        value_network=value_net,
        intrinsic_curiosity_module=icm,
        optimizer=optimizer,
        debug_summaries=debug_summaries)

    if off_policy:
        algorithm = OffPolicyAdapter(algorithm)

    return algorithm
