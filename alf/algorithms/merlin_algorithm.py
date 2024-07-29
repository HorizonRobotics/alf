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
"""Implementation of MERLIN algorithm. See class MerlinAlgorithm for detail."""

from collections import namedtuple
import copy
import functools
import numpy as np
import torch
import torch.nn as nn

import alf
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.actor_critic_algorithm import ActorCriticInfo
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.decoding_algorithm import DecodingAlgorithm
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.vae import VariationalAutoEncoder
from alf.data_structures import TimeStep, AlgStep, LossInfo
from alf.networks import EncodingNetwork, LSTMEncodingNetwork
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.networks.action_encoder import SimpleActionEncoder
from alf.networks.memory import MemoryWithUsage
from alf.nest import flatten, map_structure
from alf.utils import common, dist_utils, math_ops
from alf.tensor_specs import TensorSpec

MBPState = namedtuple(
    "MBPState",
    [
        "latent_vector",
        "mem_readout",
        "rnn_state",
        "memory",  # memory state
    ])

MBPLossInfo = namedtuple("MBPLossInfo", ["decoder", "vae"])


@alf.configurable
class MemoryBasedPredictor(Algorithm):
    """The Memory Based Predictor.

    It's described in:
    Wayne et al "Unsupervised Predictive Memory in a Goal-Directed Agent"
    `arXiv:1803.10760 <https://arxiv.org/abs/1803.10760>`_
    """

    def __init__(self,
                 action_spec,
                 encoders,
                 decoders,
                 num_read_keys=3,
                 lstm_size=(256, 256),
                 latent_dim=200,
                 memory_size=1350,
                 loss_weight=1.0,
                 name="mbp"):
        """
        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            encoders (nested Network): the nest should match observation_spec
            decoders (nested Algorithm): the nest should match observation_spec
            num_read_keys (int): number of keys for reading memory.
            lstm_size (list[int]): size of lstm layers for MBP and MBA
            latent_dim (int): the dimension of the hidden representation of VAE.
            memroy_size (int): number of memory slots
            loss_weight (float): weight for the loss
            name (str): name of the algorithm.
        """
        action_encoder = SimpleActionEncoder(action_spec)

        memory = MemoryWithUsage(
            latent_dim, memory_size, name=name + "/memory")

        rnn_input_size = (latent_dim + num_read_keys * latent_dim +
                          action_encoder.output_spec.shape[0])

        rnn = LSTMEncodingNetwork(
            input_tensor_spec=alf.TensorSpec((rnn_input_size, )),
            hidden_size=lstm_size,
            name=name + "/lstm")

        state_spec = MBPState(
            latent_vector=alf.TensorSpec((latent_dim, )),
            mem_readout=alf.TensorSpec((num_read_keys * latent_dim, )),
            rnn_state=rnn.state_spec,
            memory=memory.state_spec)

        super().__init__(train_state_spec=state_spec, name=name)

        self._encoders = encoders
        self._decoders = decoders
        self._action_encoder = action_encoder

        self._rnn = rnn
        self._memory = memory

        self._key_net = self._memory.create_keynet(rnn.output_spec,
                                                   num_read_keys)

        prior_network = EncodingNetwork(
            input_tensor_spec=(rnn.output_spec, state_spec.mem_readout),
            preprocessing_combiner=alf.nest.utils.NestConcat(),
            fc_layer_params=(2 * latent_dim, 2 * latent_dim),
            activation=torch.tanh,
            last_layer_size=2 * latent_dim,
            last_activation=math_ops.identity,
            name=name + "/prior_network")

        encoder_output_specs = alf.nest.map_structure(
            lambda encoder: encoder.output_spec, self._encoders)
        self._vae = VariationalAutoEncoder(
            latent_dim,
            input_tensor_spec=encoder_output_specs,
            z_prior_network=prior_network,
            name=name + "/vae")

        self._loss_weight = loss_weight

    @property
    def memory(self):
        """Return the external memory of this module."""
        return self._memory

    def encode_step(self, inputs, state: MBPState):
        """Calculate latent vector.

        Args:
            inputs (tuple): a tuple of ``(observation, prev_action)``.
            state (MBPState): RNN state
        Returns:
            AlgStep:
            - output: latent vector
            - state: next_state
            - info (LossInfo): loss
        """
        observation, prev_action = inputs
        self._memory.from_states(state.memory)

        prev_action = self._action_encoder(prev_action)[0]

        prev_rnn_input = torch.cat(
            [state.latent_vector, prev_action, state.mem_readout], dim=-1)

        prev_rnn_output, prev_rnn_state = self._rnn(prev_rnn_input,
                                                    state.rnn_state)

        prev_mem_readout = self._memory.genkey_and_read(
            self._key_net, prev_rnn_output)

        self._memory.write(state.latent_vector.detach())

        prior_input = (prev_rnn_output, prev_mem_readout)

        current_input = map_structure(lambda encoder, obs: encoder(obs)[0],
                                      self._encoders, observation)

        vae_step = self._vae.train_step((prior_input, current_input))

        next_state = MBPState(
            latent_vector=vae_step.output.z,
            mem_readout=prev_mem_readout,
            rnn_state=prev_rnn_state,
            memory=self._memory.states)

        return vae_step._replace(output=vae_step.output.z, state=next_state)

    def decode_step(self, latent_vector, observations):
        """Calculate decoding loss."""
        decoders = flatten(self._decoders)
        observations = flatten(observations)
        decoder_losses = [
            decoder.train_step((latent_vector, obs)).info
            for decoder, obs in zip(decoders, observations)
        ]
        loss = math_ops.add_n(
            [decoder_loss.loss for decoder_loss in decoder_losses])
        decoder_losses = alf.nest.pack_sequence_as(self._decoders,
                                                   decoder_losses)
        return LossInfo(loss=loss, extra=decoder_losses)

    def predict_step(self, inputs, state: MBPState):
        """Train one step.

        Args:
            inputs (tuple): a tuple of ``(observation, action)``.
            state (nested Tensor): RNN state
        Returns:
            AlgStep:
            - output: latent vector
            - state: next state
            - info: empty tuple
        """
        encode_step = self.encode_step(inputs, state)

        return encode_step._replace(info=())

    def train_step(self, inputs, state: MBPState):
        """Train one step.

        Args:
            inputs (tuple): a tuple of ``(observation, action)``.
        Returns:
            AlgStep:
            - output: latent vector
            - state: next state
            - info (LossInfo): loss
        """
        observation, _ = inputs
        encode_step = self.encode_step(inputs, state)

        # TODO: decoder for action
        decoder_loss = self.decode_step(encode_step.output, observation)

        return encode_step._replace(
            info=LossInfo(
                loss=self._loss_weight *
                (decoder_loss.loss + encode_step.info.loss),
                extra=MBPLossInfo(
                    decoder=decoder_loss.extra, vae=encode_step.info.kld)))


@alf.configurable
class MemoryBasedActor(OnPolicyAlgorithm):
    """The policy module for MERLIN model."""

    def __init__(self,
                 observation_spec,
                 action_spec,
                 memory: MemoryWithUsage,
                 reward_spec=TensorSpec(()),
                 epsilon_greedy=None,
                 num_read_keys=1,
                 lstm_size=(256, 256),
                 latent_dim=200,
                 loss=None,
                 loss_class=ActorCriticLoss,
                 loss_weight=1.0,
                 debug_summaries=False,
                 name="mba"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            memory (MemoryWithUsage): the memory module from ``MemoryBasedPredictor``
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            num_read_keys (int): number of keys for reading memory.
            latent_dim (int): the dimension of the hidden representation of VAE.
            lstm_size (list[int]): size of lstm layers
            loss (None|ActorCriticLoss): an object for calculating the loss
                for reinforcement learning. If None, a default ``ActorCriticLoss``
                will be used.
            loss_class (type): the class of the loss. The signature of its
                constructor: loss_class(debug_summaries)
            name (str): name of the algorithm.
        """
        if epsilon_greedy is None:
            # TODO: use ``epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)``
            # once config is passed into __init__.
            epsilon_greedy = alf.get_config_value(
                'TrainerConfig.epsilon_greedy')
        self._epsilon_greedy = epsilon_greedy
        rnn = LSTMEncodingNetwork(
            input_tensor_spec=alf.TensorSpec((latent_dim, )),
            hidden_size=lstm_size,
            name=name + "/lstm")

        actor_input_dim = (
            latent_dim + rnn.output_spec.shape[0] + num_read_keys * memory.dim)

        actor_net = ActorDistributionNetwork(
            input_tensor_spec=alf.TensorSpec((actor_input_dim, ),
                                             dtype=torch.float32),
            action_spec=action_spec,
            fc_layer_params=(200, ),
            activation=torch.tanh,
            name=name + "/actor_net")

        super(MemoryBasedActor, self).__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=rnn.state_spec,
            name=name)

        if loss is None:
            loss = loss_class(debug_summaries=debug_summaries)
        self._loss = loss
        self._loss_weight = loss_weight
        self._memory = memory

        self._key_net = self._memory.create_keynet(rnn.output_spec,
                                                   num_read_keys)

        # TODO: add log p(a_i) as input to value net
        value_input_dim = latent_dim
        self._value_net = ValueNetwork(
            input_tensor_spec=alf.TensorSpec((value_input_dim, )),
            fc_layer_params=(200, ),
            activation=torch.tanh,
            name=name + "/value_net")

        self._rnn = rnn
        self._actor_net = actor_net

        # TODO: add qvalue_net for predicting Q-value

    def _get_action(self, latent_vector, state):
        rnn_output, rnn_state = self._rnn(latent_vector, state)
        mem_readout = self._memory.genkey_and_read(self._key_net, rnn_output)
        policy_input = torch.cat(
            [latent_vector.detach(), rnn_output, mem_readout], dim=-1)
        action_distribution, _ = self._actor_net(policy_input)
        return action_distribution, rnn_state

    def rollout_step(self, time_step: TimeStep, state):
        """Train one step.

        Args:
            time_step (TimeStep): ``time_step.observation`` should be the latent
                vector.
            state (nested Tensor): state of the model
        """
        latent_vector = time_step.observation
        action_distribution, state = self._get_action(latent_vector, state)
        value, _ = self._value_net(latent_vector)
        action = dist_utils.sample_action_distribution(action_distribution)

        info = ActorCriticInfo(
            action=common.detach(action),
            reward=time_step.reward,
            step_type=time_step.step_type,
            discount=time_step.discount,
            action_distribution=action_distribution,
            value=value)
        return AlgStep(output=action, state=state, info=info)

    def predict_step(self, time_step: TimeStep, state):
        action_distribution, state = self._get_action(time_step.observation,
                                                      state)
        action = dist_utils.epsilon_greedy_sample(action_distribution,
                                                  self._epsilon_greedy)
        return AlgStep(output=action, state=state, info=())

    def calc_loss(self, train_info: ActorCriticInfo):
        """Calculate loss."""
        loss = self._loss(train_info)
        return loss._replace(loss=self._loss_weight * loss.loss)


MerlinState = namedtuple("MerlinState", ["mbp_state", "mba_state"])
MerlinLossInfo = namedtuple("MerlinLossInfo", ["mba", "mbp"])
MerlinInfo = namedtuple("MerlinInfo", ["mbp_info", "mba_info"])


@alf.configurable
class MerlinAlgorithm(OnPolicyAlgorithm):
    """MERLIN model.

    This implements the MERLIN model described in
    Wayne et al "Unsupervised Predictive Memory in a Goal-Directed Agent" arXiv:1803.10760

    Current differences:

    * No action encoding and decoding
    * No retroactive memory update
    * No prediction of state-action value
    * Value prediction does not use action distribution as feature.
    * No q-value prediction
    * Image encoding and decoding use batch-norm. The paper didn't use.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 encoders,
                 decoders,
                 reward_spec=TensorSpec(()),
                 env=None,
                 config: TrainerConfig = None,
                 latent_dim=200,
                 lstm_size=(256, 256),
                 memory_size=1350,
                 rl_loss=None,
                 optimizer=None,
                 debug_summaries=False,
                 name="Merlin"):
        """
        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            encoders (nested Network): the nest should match observation_spec
            decoders (nested Algorithm): the nest should match observation_spec
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple
                simulations simultaneously. Running multiple environments in
                parallel is crucial to on-policy algorithms as it increases the
                diversity of data and decreases temporal correlation. ``env`` only
                needs to be provided to the root ``Algorithm``.
            config (TrainerConfig): config for training. ``config`` only needs to
                be provided to the algorithm which performs ``train_iter()`` by
                itself.
            latent_dim (int): the dimension of the hidden representation of VAE.
            lstm_size (list[int]): size of lstm layers for MBP and MBA
            memroy_size (int): number of memory slots
            rl_loss (None|ActorCriticLoss): an object for calculating the loss
                for reinforcement learning. If None, a default ``ActorCriticLoss``
                will be used.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            debug_summaries: True if debug summaries should be created.
            name (str): name of the algorithm.
        """
        mbp = MemoryBasedPredictor(
            action_spec=action_spec,
            encoders=encoders,
            decoders=decoders,
            latent_dim=latent_dim,
            lstm_size=lstm_size,
            memory_size=memory_size)

        mba = MemoryBasedActor(
            observation_spec=observation_spec,
            action_spec=action_spec,
            latent_dim=latent_dim,
            lstm_size=lstm_size,
            loss=rl_loss,
            memory=mbp.memory,
            debug_summaries=debug_summaries)

        super(MerlinAlgorithm, self).__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=MerlinState(
                mbp_state=mbp.train_state_spec,
                mba_state=mba.train_state_spec),
            env=env,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._mbp = mbp
        self._mba = mba

    def rollout_step(self, time_step: TimeStep, state):
        """Train one step."""
        mbp_step = self._mbp.train_step(
            inputs=(time_step.observation, time_step.prev_action),
            state=state.mbp_state)
        mba_step = self._mba.rollout_step(
            time_step=time_step._replace(observation=mbp_step.output),
            state=state.mba_state)

        return AlgStep(
            output=mba_step.output,
            state=MerlinState(
                mbp_state=mbp_step.state, mba_state=mba_step.state),
            info=MerlinInfo(mbp_info=mbp_step.info, mba_info=mba_step.info))

    def predict_step(self, time_step: TimeStep, state):
        mbp_step = self._mbp.predict_step(
            inputs=(time_step.observation, time_step.prev_action),
            state=state.mbp_state)
        mba_step = self._mba.predict_step(
            time_step=time_step._replace(observation=mbp_step.output),
            state=state.mba_state)
        return AlgStep(
            output=mba_step.output,
            state=MerlinState(
                mbp_state=mbp_step.state, mba_state=mba_step.state),
            info=())

    def calc_loss(self, info: MerlinInfo):
        """Calculate loss."""
        self.summarize_reward("reward", info.mba_info.reward)
        mbp_loss_info = self._mbp.calc_loss(info.mbp_info)
        mba_loss_info = self._mba.calc_loss(info.mba_info)

        return LossInfo(
            loss=mbp_loss_info.loss + mba_loss_info.loss,
            extra=MerlinLossInfo(
                mbp=mbp_loss_info.extra, mba=mba_loss_info.extra))


@alf.configurable
class ResnetEncodingNetwork(alf.networks.Network):
    """Image encoding network using ResNet bottleneck blocks.

    This is not a generic network, it implements `ImageEncoder` described in
    2.1.1 of "Unsupervised Predictive Memory in a Goal-Directed Agent"
    """

    def __init__(self,
                 input_tensor_spec,
                 output_size=500,
                 output_activation=torch.tanh,
                 use_fc_bn=False,
                 norm_layer=None,
                 name='ResnetEncodingNetwork'):
        """
        Args:
            input_tensor_spec (nested TensorSpec): input observations spec.
            output_size (int): dimension of the encoding result
            output_activation (Callable): activation for the output
            use_fc_bn (bool): whether to use batch normalization for the final
                ``FC`` layer.
            norm_layer (nn.Module|None): optional additional layer for normalization.
        """
        super().__init__(input_tensor_spec, name=name)

        enc_layers = []

        in_channels = input_tensor_spec.shape[0]
        shape = input_tensor_spec.shape
        for stride in [2, 1, 2, 1, 2, 1]:
            res_block = alf.layers.BottleneckBlock(
                in_channels=in_channels,
                kernel_size=3,
                filters=(64, 32, 64),
                stride=stride)
            shape = res_block.calc_output_shape(shape)
            enc_layers.append(res_block)
            in_channels = 64

        enc_layers.extend([
            nn.Flatten(),
            alf.layers.FC(
                input_size=int(np.prod(shape)),
                output_size=output_size,
                use_bn=use_fc_bn,
                activation=output_activation)
        ])

        if norm_layer:
            enc_layers.append(norm_layer)

        self._model = nn.Sequential(*enc_layers)

    def forward(self, observation, state=()):
        return self._model(observation), ()


@alf.configurable
class ResnetDecodingNetwork(alf.networks.Network):
    """Image decoding network using ResNet bottleneck blocks.

    This is not a generic network, it implements `ImageDecoder` described in
    2.2.1 of "Unsupervised Predictive Memory in a Goal-Directed Agent"
    """

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec=alf.TensorSpec((3, 64, 64)),
                 name='ResnetDecodingNetwork'):
        """

        Args:
             input_tensor_spec (TensorSpec): input latent spec.
             output_tensor_spec (TensorSpec): desired output shape. Height and
                width needs to be divisible by 8.
        """
        super().__init__(input_tensor_spec, name=name)
        c, h, w = output_tensor_spec.shape
        assert h % 8 == 0
        assert w % 8 == 0

        dec_layers = []
        relu = torch.relu_
        dec_layers.extend([
            alf.layers.FC(input_tensor_spec.shape[0], 500, activation=relu),
            alf.layers.FC(500, h * w, activation=relu),
            alf.layers.Reshape((64, h // 8, w // 8))
        ])

        for stride in reversed([2, 1, 2, 1, 2, 1]):
            dec_layers.append(
                alf.layers.BottleneckBlock(
                    in_channels=64,
                    kernel_size=3,
                    filters=(64, 32, 64),
                    stride=stride,
                    transpose=True))

        dec_layers.append(
            alf.layers.ConvTranspose2D(
                in_channels=64,
                out_channels=3,
                kernel_size=1,
                activation=torch.sigmoid))

        self._model = nn.Sequential(*dec_layers)

    def forward(self, observation, state=()):
        return self._model(observation), ()
