import tensorflow as tf
import unittest
import alf.algorithms.vae as vae
import numpy as np

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

class VaeMnistTest(unittest.TestCase):
    def setUp(self):
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
            test_inputs = [self.y_test[:nrows], self.x_test[:nrows]]
        else:
            test_inputs = self.x_test[:nrows]

        decoded_output = model.predict(test_inputs)

        fig = plt.figure()
        idx = 1
        for i in range(nrows):
            fig.add_subplot(nrows, ncols, idx)
            plt.imshow(np.reshape(self.x_test[i], (self.image_size, self.image_size)))
            fig.add_subplot(nrows, ncols, idx + 1)
            plt.imshow(np.reshape(decoded_output[i], (self.image_size, self.image_size)))
            idx += 2
        plt.show()


    def show_sampled_images(self, decoding_function):
        nrows = 10
        eps = tf.random.normal((nrows * nrows, self.latent_dim), dtype=tf.float32, mean=0., stddev=1.0)
        sampled_outputs = decoding_function(eps)
        fig = plt.figure()
        idx = 0
        for i in range(nrows):
            for j in range(nrows):
                fig.add_subplot(nrows, nrows, idx + 1)
                plt.imshow(np.reshape(sampled_outputs[idx], (self.image_size, self.image_size)))
                idx += 1

        plt.show()


class VaeTest(VaeMnistTest):
    def runTest(self):
        encoder = vae.VariationalAutoEncoder(
            self.latent_dim,
            preprocess_layers=tf.keras.layers.Dense(512, activation='relu'))

        decoding_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.original_dim, activation='sigmoid')])

        inputs = tf.keras.layers.Input(shape=self.input_shape)
        z, kl_loss = encoder.sampling_forward(inputs)
        outputs = decoding_layers(z)

        loss = tf.reduce_mean(tf.keras.losses.mse(inputs, outputs) * self.original_dim + kl_loss)

        model = tf.keras.Model(inputs, outputs, name = "vae")
        model.add_loss(loss)
        model.compile(optimizer='adam')
        model.summary()

        hist = model.fit(self.x_train,
          epochs=self.epochs,
          batch_size=self.batch_size,
          validation_data=(self.x_test, None))

        last_val_loss = hist.history['val_loss'][-1]
        self.assertTrue(38.0 < last_val_loss <= 39.0)
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
            tf.keras.layers.Dense(self.original_dim, activation='sigmoid')])

        inputs = tf.keras.layers.Input(shape=self.input_shape)
        prior_inputs = tf.keras.layers.Input(shape=(1,))
        prior_inputs_one_hot = tf.reshape(tf.one_hot(tf.cast(prior_inputs, tf.int32), 10), shape=(-1, 10))

        z, kl_loss = encoder.sampling_forward(tf.concat([prior_inputs_one_hot, inputs], -1))
        outputs = decoding_layers(tf.concat([prior_inputs_one_hot, z], -1))

        loss = tf.reduce_mean(tf.keras.losses.mse(inputs, outputs) * self.original_dim + kl_loss)

        model = tf.keras.Model(inputs=[prior_inputs, inputs], outputs=outputs, name="vae")
        model.add_loss(loss)
        model.compile(optimizer='adam')
        model.summary()

        hist = model.fit([self.y_train, self.x_train],
          epochs=self.epochs,
          batch_size=self.batch_size,
          validation_data=([self.y_test, self.x_test], None))

        last_val_loss = hist.history['val_loss'][-1]
        # cvae seems have the lowest errors with the same settings
        self.assertTrue(30.0 < last_val_loss < 31.0)

        self.show_encoded_images(model, with_priors=True)
        nrows = 10
        fig = plt.figure()
        idx = 0
        for i in range(10):
            eps = tf.random.normal((nrows, self.latent_dim), dtype=tf.float32, mean=0., stddev=1.0)
            conditionals = tf.stack([tf.one_hot(i, 10) for _ in range(10)])
            sampled_outputs = decoding_layers(tf.concat([conditionals, eps], -1))
            # for the same digit i, we sample a bunch of images
            # it actually looks great.
            for j in range(nrows):
                fig.add_subplot(nrows, nrows, idx + 1)
                plt.imshow(np.reshape(sampled_outputs[j], (self.image_size, self.image_size)))
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
            tf.keras.layers.Dense(self.original_dim, activation='sigmoid')])

        inputs = tf.keras.layers.Input(shape=self.input_shape)
        prior_inputs = tf.keras.layers.Input(shape=(1,))
        prior_inputs_one_hot = tf.reshape(tf.one_hot(tf.cast(prior_inputs, tf.int32), 10), shape=(-1, 10))
        z, kl_loss = encoder.sampling_forward((prior_inputs_one_hot, inputs))
        outputs = decoding_layers(z)

        loss = tf.reduce_mean(tf.keras.losses.mse(inputs, outputs) * self.original_dim + kl_loss)

        model = tf.keras.Model(inputs=[prior_inputs, inputs], outputs=outputs, name="vae")
        model.add_loss(loss)
        model.compile(optimizer='adam')
        model.summary()

        hist = model.fit([self.y_train, self.x_train],
          epochs=self.epochs,
          batch_size=self.batch_size,
          validation_data=([self.y_test, self.x_test], None))

        last_val_loss = hist.history['val_loss'][-1]
        # total loss is much smaller with label based prior network.
        self.assertTrue(34.0 < last_val_loss < 35.5)
        self.show_encoded_images(model, with_priors=True)

        # with prior network, sampling is more complicated
        nrows = 10
        fig = plt.figure()
        idx = 0
        for i in range(10):
            z_mean_prior, z_log_var_prior = prior_network(
                tf.stack([tf.one_hot(i, 10) for _ in range(10)]))

            eps = tf.random.normal((nrows, self.latent_dim), dtype=tf.float32, mean=0., stddev=1.0)
            sampled_outputs = decoding_layers(eps * z_log_var_prior + z_mean_prior)
            # for the same digit i, we sample a bunch of images
            # it actually looks great.
            for j in range(nrows):
                fig.add_subplot(nrows, nrows, idx + 1)
                plt.imshow(np.reshape(sampled_outputs[j], (self.image_size, self.image_size)))
                idx += 1

        plt.show()


if __name__ == '__main__':
    unittest.main()
