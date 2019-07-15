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

import tensorflow as tf
import numpy as np
from tf_agents.networks import network
from tf_agents import specs
import gin.tf


@gin.configurable
class TextEncodeNetwork(network.Network):
    def __init__(self, vocab_size, seq_len, embed_size, lstm_size):
        """Create an instance of `TextEncodeNetwork`
        See Methods 2.1.5 of "Unsupervised Predictive Memory in a Goal-Directed
        Agent"

        Args:
            vocab_size (int): vocabulary size
            seq_len (int): sequence length
            embed_size (int): embedding size
            lstm_size (int): lstm size for encoding
        """
        super(TextEncodeNetwork, self).__init__(
            input_tensor_spec=specs.ArraySpec(
                shape=(None, seq_len), dtype=np.int),
            state_spec=(),
            name='TextEncodeNetwork')
        self._vocab_size = vocab_size
        self._seq_len = seq_len
        self._embed_size = embed_size
        self._lstm_size = lstm_size

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Embedding(
                self._vocab_size, self._embed_size, mask_zero=True))
        model.add(tf.keras.layers.LSTM(self._lstm_size))
        self._model = model

    def call(self, inputs, step_type=None, network_state=()):
        """Encode sequences.

                Args:
                    inputs (Tensor): sequences of word ids, shape is (b, seq_len)
                    where b is batch and seq_len is sequence length
                Returns:
                    result Tensor: shape is (b, lstm_size).
        """
        return self._model(inputs), network_state


@gin.configurable
class TextDecodeNetwork(network.Network):
    def __init__(self, vocab_size, code_len, seq_len, lstm_size):
        """Create an instance of `TextDecodeNetwork`
        See Methods 2.2.3 of "Unsupervised Predictive Memory in a Goal-Directed
        Agent"

        Args:
            vocab_size (int): vocabulary size
            code_len (int): encoded length
            seq_len (int): output sequence length
            lstm_size (int): lstm size for decoding
        """
        super(TextDecodeNetwork, self).__init__(
            input_tensor_spec=specs.ArraySpec(
                shape=(None, code_len), dtype=np.float),
            state_spec=(),
            name='TextDecodeNetwork')
        self._vocab_size = vocab_size
        self._seq_len = seq_len
        self._lstm_size = lstm_size
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.RepeatVector(self._seq_len))
        model.add(tf.keras.layers.LSTM(self._lstm_size, return_sequences=True))
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self._vocab_size, activation="linear")))
        model.add(tf.keras.layers.Softmax())
        self._model = model

    def call(self, inputs, step_type=None, network_state=()):
        """Decode to sequences

                Args:
                    inputs (Tensor): shape is (b, dim)  where b is batch
                    and dim is dimension of coding
                Returns:
                    result Tensor: word probabilities, shape is (b, seq_len, vocab_size).

        """
        return self._model(inputs), network_state
