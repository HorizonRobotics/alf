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

import os
import tensorflow as tf
import alf.algorithms.vae as vae
import numpy as np

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

INTERACTIVE_MODE = False


class VaeMnistTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        if os.environ.get('SKIP_LONG_TIME_COST_TESTS', False):
            self.skipTest("It takes very long to run this test.")
        # MNIST dataset
        (x_train, self.y_train), (x_test, self.y_test) = mnist.load_data()
        self.image_size = x_train.shape[1]
        self.original_dim = self.image_size * self.image_size
        self.x_train = np.reshape(x_train, [-1, self.original_dim])
        self.x_test = np.reshape(x_test, [-1, self.original_dim])
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        # network parameters
        self.input_shape = (self.original_dim, )
        self.latent_dim = 2
        self.batch_size = 100
        self.epochs = 20

    def show_encoded_images(self, model, with_priors=False):
        # test decoding image qualities
        nrows = 20
        ncols = 2

        if with_priors:
            test_inputs = [
                self.y_test[:nrows].astype('float32'), self.x_test[:nrows]
            ]
        else:
            test_inputs = self.x_test[:nrows]

        decoded_output = model.predict(test_inputs)

        fig = plt.figure()
        idx = 1
        for i in range(nrows):
            fig.add_subplot(nrows, ncols, idx)
            plt.imshow(
                np.reshape(self.x_test[i], (self.image_size, self.image_size)))
            plt.axis('off')
            fig.add_subplot(nrows, ncols, idx + 1)
            plt.imshow(
                np.reshape(decoded_output[i],
                           (self.image_size, self.image_size)))
            plt.axis('off')
            idx += 2
        plt.show()

    def show_sampled_images(self, decoding_function):
        nrows = 10
        eps = tf.random.normal((nrows * nrows, self.latent_dim),
                               dtype=tf.float32,
                               mean=0.,
                               stddev=1.0)
        sampled_outputs = decoding_function(eps)
        fig = plt.figure()
        idx = 0
        for i in range(nrows):
            for j in range(nrows):
                fig.add_subplot(nrows, nrows, idx + 1)
                plt.imshow(
                    np.reshape(sampled_outputs[idx],
                               (self.image_size, self.image_size)))
                plt.axis('off')
                idx += 1

        plt.show()


class VaeTest(VaeMnistTest):
    def runTest(self):
        encoder = vae.VariationalAutoEncoder(
            self.latent_dim,
            preprocess_layers=tf.keras.layers.Dense(512, activation='relu'))

        decoding_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.original_dim, activation='sigmoid')
        ])

        inputs = tf.keras.layers.Input(shape=self.input_shape)
        z, kl_loss = encoder.sampling_forward(inputs)
        outputs = decoding_layers(z)

        loss = tf.reduce_mean(
            tf.keras.losses.mse(inputs, outputs) * self.original_dim + kl_loss)

        model = tf.keras.Model(inputs, outputs, name="vae")
        model.add_loss(loss)
        model.compile(optimizer='adam')
        model.summary()

        hist = model.fit(
            self.x_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.x_test, None))

        last_val_loss = hist.history['val_loss'][-1]
        print("loss: ", last_val_loss)
        self.assertTrue(37.5 < last_val_loss <= 39.0)
        if INTERACTIVE_MODE:
            self.show_encoded_images(model)
            self.show_sampled_images(lambda eps: decoding_layers(eps))


# train conditional vae on mnist
# refer to: https://arxiv.org/pdf/1606.05908.pdf
class CVaeTest(VaeMnistTest):
    def runTest(self):
        encoder = vae.VariationalAutoEncoder(
            self.latent_dim,
            preprocess_layers=tf.keras.layers.Dense(512, activation='relu'))

        decoding_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.original_dim, activation='sigmoid')
        ])

        inputs = tf.keras.layers.Input(shape=self.input_shape)
        prior_inputs = tf.keras.layers.Input(shape=(1, ))
        prior_inputs_one_hot = tf.reshape(
            tf.one_hot(tf.cast(prior_inputs, tf.int32), 10), shape=(-1, 10))

        z, kl_loss = encoder.sampling_forward(
            tf.concat([prior_inputs_one_hot, inputs], -1))
        outputs = decoding_layers(tf.concat([prior_inputs_one_hot, z], -1))

        loss = tf.reduce_mean(
            tf.keras.losses.mse(inputs, outputs) * self.original_dim + kl_loss)

        model = tf.keras.Model(
            inputs=[prior_inputs, inputs], outputs=outputs, name="vae")
        model.add_loss(loss)
        model.compile(optimizer='adam')
        model.summary()

        hist = model.fit([self.y_train, self.x_train],
                         epochs=self.epochs,
                         batch_size=self.batch_size,
                         validation_data=([self.y_test, self.x_test], None))

        last_val_loss = hist.history['val_loss'][-1]
        # cvae seems have the lowest errors with the same settings
        print("loss: ", last_val_loss)
        self.assertTrue(30.0 < last_val_loss < 31.0)

        if INTERACTIVE_MODE:
            self.show_encoded_images(model, with_priors=True)
            nrows = 10
            fig = plt.figure()
            idx = 0
            for i in range(10):
                eps = tf.random.normal((nrows, self.latent_dim),
                                       dtype=tf.float32,
                                       mean=0.,
                                       stddev=1.0)
                conditionals = tf.stack([tf.one_hot(i, 10) for _ in range(10)])
                sampled_outputs = decoding_layers(
                    tf.concat([conditionals, eps], -1))
                # for the same digit i, we sample a bunch of images
                # it actually looks great.
                for j in range(nrows):
                    fig.add_subplot(nrows, nrows, idx + 1)
                    plt.imshow(
                        np.reshape(sampled_outputs[j],
                                   (self.image_size, self.image_size)))
                    plt.axis('off')
                    idx += 1

            plt.show()


class PriorNetwork(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(PriorNetwork, self).__init__()
        self.latent_nn = tf.keras.layers.Dense(2 * hidden_dim)
        self.z_mean = tf.keras.layers.Dense(hidden_dim)
        self.z_log_var = tf.keras.layers.Dense(hidden_dim)

    def call(self, state):
        latent = self.latent_nn(state)
        return self.z_mean(latent), self.z_log_var(latent)


class VaePriorNetworkTest(VaeMnistTest):
    def runTest(self):
        prior_network = PriorNetwork(self.latent_dim)
        encoder = vae.VariationalAutoEncoder(
            self.latent_dim,
            prior_network=prior_network,
            preprocess_layers=tf.keras.layers.Dense(512, activation='relu'))

        decoding_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.original_dim, activation='sigmoid')
        ])

        inputs = tf.keras.layers.Input(shape=self.input_shape)
        prior_inputs = tf.keras.layers.Input(shape=(1, ))
        prior_inputs_one_hot = tf.reshape(
            tf.one_hot(tf.cast(prior_inputs, tf.int32), 10), shape=(-1, 10))
        z, kl_loss = encoder.sampling_forward((prior_inputs_one_hot, inputs))
        outputs = decoding_layers(z)

        loss = tf.reduce_mean(
            tf.keras.losses.mse(inputs, outputs) * self.original_dim + kl_loss)

        model = tf.keras.Model(
            inputs=[prior_inputs, inputs], outputs=outputs, name="vae")
        model.add_loss(loss)
        model.compile(optimizer='adam')
        model.summary()

        hist = model.fit([self.y_train, self.x_train],
                         epochs=self.epochs,
                         batch_size=self.batch_size,
                         validation_data=([self.y_test, self.x_test], None))

        last_val_loss = hist.history['val_loss'][-1]
        # total loss is much smaller with label based prior network.
        print("loss: ", last_val_loss)
        self.assertTrue(34.0 < last_val_loss < 35.5)
        if INTERACTIVE_MODE:
            self.show_encoded_images(model, with_priors=True)

            # with prior network, sampling is more complicated
            nrows = 10
            fig = plt.figure()
            idx = 0
            for i in range(10):
                z_mean_prior, z_log_var_prior = prior_network(
                    tf.stack([tf.one_hot(i, 10) for _ in range(10)]))

                eps = tf.random.normal((nrows, self.latent_dim),
                                       dtype=tf.float32,
                                       mean=0.,
                                       stddev=1.0)
                sampled_outputs = decoding_layers(eps * z_log_var_prior +
                                                  z_mean_prior)
                # for the same digit i, we sample a bunch of images
                # it actually looks great.
                for j in range(nrows):
                    fig.add_subplot(nrows, nrows, idx + 1)
                    plt.imshow(
                        np.reshape(sampled_outputs[j],
                                   (self.image_size, self.image_size)))
                    plt.axis('off')
                    idx += 1

            plt.show()


class SimpleVaeTest(tf.test.TestCase):
    def test_gaussian(self):
        """Test for one dimensional Gaussion."""
        input_shape = (1, )
        epochs = 20
        batch_size = 100
        latent_dim = 1
        loss_f = tf.square

        encoder = vae.VariationalAutoEncoder(latent_dim)
        decoding_layers = tf.keras.layers.Dense(1)

        inputs = tf.keras.layers.Input(shape=input_shape)
        z, kl_loss = encoder.sampling_forward(inputs)
        outputs = decoding_layers(z)

        loss = tf.reduce_mean(100 * loss_f(inputs - outputs) + kl_loss)

        model = tf.keras.Model(inputs, outputs, name="vae")
        model.add_loss(loss)
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1))
        model.summary()

        x_train = np.random.randn(10000, 1)
        x_val = np.random.randn(10000, 1)
        x_test = np.random.randn(10, 1)
        y_test = model(x_test.astype(np.float32))

        hist = model.fit(
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, None))

        y_test = model(x_test.astype(np.float32))
        reconstruction_loss = float(tf.reduce_mean(loss_f(x_test - y_test)))
        print("reconstruction_loss:", reconstruction_loss)
        self.assertLess(reconstruction_loss, 0.05)


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
