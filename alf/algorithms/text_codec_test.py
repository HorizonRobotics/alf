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
"""Test for alf.algorithms.text_codec."""

import unittest
import tensorflow as tf
import alf.algorithms.text_codec as text_codec


class TestTextEncodeDecodeNetwork(unittest.TestCase):
    def test_encode_decode(self):
        vocab_size = 1000
        seq_len = 5
        embed_size = 50
        encoder_lstm_size = 100
        code_len = encoder_lstm_size
        decoder_lstm_size = 100

        encoder = text_codec.TextEncodeNetwork(vocab_size, seq_len, embed_size,
                                               encoder_lstm_size)
        decoder = text_codec.TextDecodeNetwork(vocab_size, code_len, seq_len,
                                               decoder_lstm_size)

        s0 = tf.constant([1, 3, 2])
        s1 = tf.constant([4])

        batch = [s0, s1]
        batch_size = len(batch)

        input = tf.keras.preprocessing.sequence.pad_sequences(
            batch, padding='post', maxlen=seq_len)

        encoded, _ = encoder(input)
        self.assertEqual((batch_size, encoder_lstm_size), encoded.shape)

        decoded, _ = decoder(encoded)
        self.assertEqual((batch_size, seq_len, vocab_size), decoded.shape)


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    unittest.main()
