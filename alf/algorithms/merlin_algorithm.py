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
import functools

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks.network import Network
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.specs.tensor_spec import TensorSpec
from tf_agents.trajectories.policy_step import PolicyStep

import alf
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.memory import MemoryWithUsage
from alf.algorithms.algorithm import Algorithm, AlgorithmStep
from alf.algorithms.actor_critic_algorithm import ActorCriticInfo
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.on_policy_algorithm import TrainingInfo
from alf.algorithms.on_policy_algorithm import ActionTimeStep
from alf.algorithms.vae import VariationalAutoEncoder
from alf.utils.action_encoder import SimpleActionEncoder

MBPState = namedtuple(
    "MBPState",
    [
        "latent_vector",
        "mem_readout",
        "rnn_state",
        "memory",  # memory state
    ])

MBPLossInfo = namedtuple("MBPLossInfo", ["decoder", "vae"])


def get_rnn_cell_state_spec(cell):
    """Get the state spec for RNN cell."""
    return tf.nest.map_structure(
        lambda size: TensorSpec(size, dtype=tf.float32), cell.state_size)


def make_lstm_cell(lstm_size, name="lstm_cell"):
    """Make a stacked LSTM cell.

    Args:
      lstm_size (int|list[int]): specifying the LSTM cell sizes to use.
      name (str): name of the cell.
      
    Returns:
      cell (LSTMCell|StackedRNNCell): the stacked LSTM cell.

    """
    implementation = 2  # FUSED IMPLEMENTATION
    cells = [
        tf.keras.layers.LSTMCell(
            size, implementation=implementation, name=name + "/" + str(i))
        for i, size in enumerate(lstm_size)
    ]
    if len(cells) == 1:
        return cells[0]
    else:
        return tf.keras.layers.StackedRNNCells(cells)


def _collect_variables(*modules):
    modules = tf.nest.flatten(modules)
    return sum([mod.variables for mod in modules], [])


class MemoryBasedPredictor(Algorithm):
    """The Memroy Based Predictor.
    
    It's described in:
    Wayne et al "Unsupervised Predictive Memory in a Goal-Directed Agent" arXiv:1803.10760
    """

    def __init__(
            self,
            action_spec,
            encoders,
            decoders,
            num_read_keys=3,
            lstm_size=(100, 100),  # not sure what Merlin uses
            latent_dim=20,
            memory_size=100,
            name="mbp"):
        """Create a MemoryBasedPredictor.

        Args:
            encoders (nested Network): the nest should match observation_spec
            decoders (nested Algorithm): the nest should match observation_spec
        """
        rnn = make_lstm_cell(lstm_size, name=name + "/lstm")
        memory = MemoryWithUsage(
            latent_dim, memory_size, name=name + "/memory")

        state_spec = MBPState(
            latent_vector=TensorSpec(shape=(latent_dim, ), dtype=tf.float32),
            mem_readout=TensorSpec(
                shape=(num_read_keys * latent_dim, ), dtype=tf.float32),
            rnn_state=get_rnn_cell_state_spec(rnn),
            memory=memory.state_spec)

        super(MemoryBasedPredictor, self).__init__(train_state_spec=state_spec)

        self._encoders = encoders
        self._decoders = decoders
        self._action_encoder = SimpleActionEncoder(action_spec)

        # This is different from Merlin LSTM. This rnn only uses the output from
        # ths last LSTM layer, while Merlin uses outputs from all LSTM layers
        self._rnn = rnn
        self._memory = memory

        self._key_net = tf.keras.layers.Dense(
            num_read_keys * (self._memory.dim + 1), name=name + "/key_net")

        prior_network = tf.keras.Sequential(
            name=name + "/prior_network",
            layers=[
                tf.keras.layers.Dense(2 * latent_dim, activation='tanh'),
                tf.keras.layers.Dense(2 * latent_dim, activation='tanh'),
                tf.keras.layers.Dense(2 * latent_dim),
                alf.layers.Split(2, axis=-1)
            ])

        self._vae = VariationalAutoEncoder(
            latent_dim, prior_network, name=name + "/vae")

    @property
    def memory(self):
        """Return the external memory of this module."""
        return self._memory

    def encode_step(self, inputs, state: MBPState):
        """Calculate latent vector.

        Args:
            inputs (tuple): a tuple of (observation, prev_action)
            state (MBPState)
        Returns:
            tuple of (latent_vector, kl_divengence, next_state)
        """
        observation, prev_action = inputs
        self._memory.from_states(state.memory)

        prev_action = self._action_encoder(prev_action)

        prev_rnn_input = tf.concat(
            [state.latent_vector, prev_action, state.mem_readout], axis=-1)

        prev_rnn_output, prev_rnn_state = self._rnn(prev_rnn_input,
                                                    state.rnn_state)

        prev_mem_readout = self._memory.genkey_and_read(
            self._key_net, prev_rnn_output)

        self._memory.write(tf.stop_gradient(state.latent_vector))

        prior_input = tf.concat([prev_rnn_output, prev_mem_readout], axis=-1)

        current_input = tf.nest.map_structure(
            lambda encoder, obs: encoder(obs)[0], self._encoders, observation)
        current_input = tf.concat(tf.nest.flatten(current_input), axis=-1)

        latent_vector, kld = self._vae.sampling_forward((prior_input,
                                                         current_input))

        next_state = MBPState(
            latent_vector=latent_vector,
            mem_readout=prev_mem_readout,
            rnn_state=prev_rnn_state,
            memory=self._memory.states)

        return latent_vector, kld, next_state

    def decode_step(self, latent_vector, observations):
        """Calculate decoding loss."""
        decoders = tf.nest.flatten(self._decoders)
        observations = tf.nest.flatten(observations)
        decoder_losses = [
            decoder.train_step((latent_vector, obs)).info
            for decoder, obs in zip(decoders, observations)
        ]
        loss = tf.add_n([decoder_loss.loss for decoder_loss in decoder_losses])
        decoder_losses = tf.nest.pack_sequence_as(self._decoders,
                                                  decoder_losses)
        return LossInfo(loss=loss, extra=decoder_losses)

    def train_step(self, inputs, state: MBPState):
        """Train one step.

        Args:
            inputs (tuple): a tuple of (observation, action)
        """
        observation, _ = inputs
        latent_vector, kld, next_state = self.encode_step(inputs, state)

        # TODO: decoder for action
        decoder_loss = self.decode_step(latent_vector, observation)

        return AlgorithmStep(
            outputs=latent_vector,
            state=next_state,
            info=LossInfo(
                loss=decoder_loss.loss + kld,
                extra=MBPLossInfo(decoder=decoder_loss, vae=kld)))


class MemoryBasedActor(OnPolicyAlgorithm):
    """The policy module for MERLIN model."""

    def __init__(
            self,
            action_spec,
            memory: MemoryWithUsage,
            num_read_keys=1,
            lstm_size=(100, 100),  # not sure what Merlin uses
            latent_dim=20,
            loss=None,
            name="mba"):
        """Create the policy module of MERLIN.

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            memory (MemoryWithUsage): the memory module from MemoryBasedPredictor
            num_read_keys (int): number of keys for reading memory.
            latent_dim (int): the dimension of the hidden represenation of VAE.
            lstm_size (list[int]): size of lstm layers
            loss (None|ActorCriticLoss): an object for calculating the loss
                for reinforcement learning. If None, a default ActorCriticLoss
                will be used.
            name (str): name of the algorithm.
        """
        # This is different from Merlin LSTM. This rnn only uses the output
        # from ths last LSTM layer, while Merlin uses outputs from all LSTM
        # layers
        rnn = make_lstm_cell(lstm_size, name=name + "/lstm")

        actor_input_dim = (
            latent_dim + lstm_size[-1] + num_read_keys * memory.dim)

        actor_net = ActorDistributionNetwork(
            input_tensor_spec=TensorSpec((actor_input_dim, ),
                                         dtype=tf.float32),
            output_tensor_spec=action_spec,
            fc_layer_params=(200, ),
            activation_fn=tf.keras.activations.tanh,
            name=name + "/actor_net")

        super(MemoryBasedActor, self).__init__(
            action_spec=action_spec,
            train_state_spec=get_rnn_cell_state_spec(rnn),
            action_distribution_spec=actor_net.output_spec)

        self._loss = ActorCriticLoss(action_spec) if loss is None else loss
        self._memory = memory

        self._key_net = tf.keras.layers.Dense(
            num_read_keys * (self._memory.dim + 1), name=name + "/key_net")

        # TODO: add log p(a_i) as input to value net
        value_input_dim = latent_dim
        self._value_net = ValueNetwork(
            input_tensor_spec=TensorSpec((value_input_dim, ),
                                         dtype=tf.float32),
            fc_layer_params=(200, ),
            activation_fn=tf.keras.activations.tanh,
            name=name + "/value_net")

        self._rnn = rnn
        self._actor_net = actor_net

        # TODO: add qvalue_net for predicting Q-value

    def train_step(self, time_step: ActionTimeStep, state):
        """Train one step.
        
        Args:
            time_step: time_step.observation should be the latent vector
            state: state of the model
        """
        latent_vector = time_step.observation
        rnn_output, rnn_state = self._rnn(latent_vector, state)
        mem_readout = self._memory.genkey_and_read(self._key_net, rnn_output)
        policy_input = tf.concat(
            [tf.stop_gradient(latent_vector), rnn_output, mem_readout],
            axis=-1)
        action_distribution, _ = self._actor_net(
            policy_input, step_type=time_step.step_type, network_state=None)

        value, _ = self._value_net(
            latent_vector, step_type=time_step.step_type, network_state=None)

        info = ActorCriticInfo(value=value, icm_reward=(), icm_info=())
        return PolicyStep(
            action=action_distribution, state=rnn_state, info=info)

    def calc_loss(self, training_info: TrainingInfo,
                  final_time_step: ActionTimeStep,
                  final_policy_step: PolicyStep):
        """Calculate loss."""
        final_value = final_policy_step.info.value
        return self._loss(training_info, training_info.info.value,
                          final_time_step, final_value)


MerlinState = namedtuple("MerlinState", ["mbp_state", "mba_state"])
MerlinLossInfo = namedtuple("MerlinLossInfo", ["mba", "mbp"])
MerlinInfo = namedtuple("MerlinInfo", ["mbp_info", "mba_info"])


class MerlinAlgorithm(OnPolicyAlgorithm):
    """MERLIN model.
    
    This implements the MERLIN model described in
    Wayne et al "Unsupervised Predictive Memory in a Goal-Directed Agent" arXiv:1803.10760

    Current differences:
    * no action encoding and decoding
    * no retroactive memory update
    * no prediction of state-action value
    * only uses outputs from last layers of stacked LSTMs
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 encoders,
                 decoders,
                 latent_dim=20,
                 lstm_size=(100, 100),
                 memory_size=100,
                 rl_loss=None,
                 optimizer=None,
                 train_step_counter=None,
                 debug_summaries=False,
                 name="MerlinNetwork"):
        """Create MerlinAlgorithm.

        Args:
            observation_spec (nested TensorSpec): spec for observation
            action_spec (nested BoundedTensorSpec): representing the actions.
            encoders (nested Network): the nest should match observation_spec
            decoders (nested Algorithm): the nest should match observation_spec
            latent_dim (int): the dimension of the hidden represenation of VAE.
            lstm_size (list[int]): size of lstm layers for MBP and MBA
            memroy_size (int): number of memory slots
            rl_loss (None|ActorCriticLoss): an object for calculating the loss
                for reinforcement learning. If None, a default ActorCriticLoss
                will be used.
            optimizer (tf.optimizers.Optimizer): The optimizer for training.
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
            action_spec=action_spec,
            latent_dim=latent_dim,
            lstm_size=lstm_size,
            loss=rl_loss,
            memory=mbp.memory)

        super(MerlinAlgorithm, self).__init__(
            action_spec=action_spec,
            train_state_spec=MerlinState(
                mbp_state=mbp.train_state_spec,
                mba_state=mba.train_state_spec),
            action_distribution_spec=mba.action_distribution_spec,
            optimizer=optimizer,
            train_step_counter=train_step_counter,
            debug_summaries=debug_summaries,
            name=name)

        self._mbp = mbp
        self._mba = mba

    def train_step(self, time_step: ActionTimeStep, state):
        """Train one step."""
        mbp_step = self._mbp.train_step(
            inputs=(time_step.observation, time_step.action),
            state=state.mbp_state)
        mba_step = self._mba.train_step(
            time_step=time_step._replace(observation=mbp_step.outputs),
            state=state.mba_state)

        return PolicyStep(
            action=mba_step.action,
            state=MerlinState(
                mbp_state=mbp_step.state, mba_state=mba_step.state),
            info=MerlinInfo(mbp_info=mbp_step.info, mba_info=mba_step.info))

    def calc_loss(self, training_info: TrainingInfo,
                  final_time_step: ActionTimeStep,
                  final_policy_step: PolicyStep):
        """Calculate loss."""
        self.add_reward_summary("reward", training_info.reward)
        mbp_loss_info = self._mbp.calc_loss(training_info.info.mbp_info)
        mba_loss_info = self._mba.calc_loss(
            training_info._replace(info=training_info.info.mba_info),
            final_time_step,
            PolicyStep(
                action=None,
                state=final_policy_step.state.mba_state,
                info=final_policy_step.info.mba_info))

        return LossInfo(
            loss=mbp_loss_info.loss + mba_loss_info.loss,
            extra=MerlinLossInfo(
                mbp=mbp_loss_info.extra, mba=mba_loss_info.extra))
