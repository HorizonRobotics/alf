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


    def showEncodedImages(self, model, withPrior=False):
        # test decoding image qualities
        nrows = 20
        ncols = 2
        test_inputs = self.x_test[:nrows] if not withPrior else (self.y_test[:nrows], self.x_test[:nrows])
        decoded_output = model.predict(test_inputs)

        fig = plt.figure()
        idx = 1
        for i in range(nrows):
            fig.add_subplot(nrows, ncols, idx)
            plt.imshow(np.reshape(self.x_test[i], (self.image_size, self.image_size)))
            fig.add_subplot(nrows, ncols, idx + 1)
            plt.imshow(np.reshape(decoded_output[i], (self.image_size, self.image_size)))
            idx+= 2

        plt.show()


    def showSampledImages(self, decodingFunction):
        nrows = 10
        eps = tf.random.normal((nrows * nrows, self.latent_dim), dtype=tf.float32, mean=0., stddev=1.0)
        sampled_outputs = decodingFunction(eps)
        fig = plt.figure()
        idx = 0
        for i in range(nrows):
            for j in range(nrows):
                fig.add_subplot(nrows, nrows, idx + 1)
                plt.imshow(np.reshape(sampled_outputs[idx], (self.image_size, self.image_size)))
                idx += 1

        plt.show()


class VaeNopriorTest(VaeMnistTest):
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
        self.assertTrue(last_val_loss <= 39.0 and last_val_loss > 38.0)
        self.showEncodedImages(model)
        self.showSampledImages(lambda eps: decoding_layers(eps))


class PriorNetwork(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim):
        super(PriorNetwork, self).__init__()
        self.latent_nn = tf.keras.layers.Dense(2 * hidden_dim)
        self.z_mean = tf.keras.layers.Dense(hidden_dim)
        self.z_log_var = tf.keras.layers.Dense(hidden_dim)

    def call(self, state):
        latent = self.latent_nn(state)
        return self.z_mean(latent), self.z_log_var(latent)

class VaePriorNetworkTest(VaeMnistTest):
    def runTest(self):
        prior_network = PriorNetwork(2, self.latent_dim)
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

        model = tf.keras.Model(inputs=[prior_inputs, inputs], outputs=outputs, name = "vae")
        model.add_loss(loss)
        model.compile(optimizer='adam')
        model.summary()

        hist = model.fit([self.y_train, self.x_train],
          epochs=self.epochs,
          batch_size=self.batch_size,
          validation_data=([self.y_test, self.x_test], None))

        last_val_loss = hist.history['val_loss'][-1]
        # total loss is much smaller with label based prior network.
        self.assertTrue(last_val_loss <= 35.5 and last_val_loss >= 34.5)
        self.showEncodedImages(model, True)

        # with prior network, sampling is more complicated
        nrows = 10
        fig = plt.figure()
        idx = 0
        for i in range(10):
            z_mean_prior, z_log_var_prior = prior_network(
                tf.stack([tf.one_hot(i, 10) for _ in range(10)]))

            eps = tf.random.normal((nrows, self.latent_dim), dtype=tf.float32, mean=0., stddev=0.5)
            sampled_outputs = decoding_layers(eps + z_mean_prior)
            # for the same digit i, we sample a bunch of images
            # it actually looks great.
            for j in range(nrows):
                fig.add_subplot(nrows, nrows, idx + 1)
                plt.imshow(np.reshape(sampled_outputs[j], (self.image_size, self.image_size)))
                idx += 1

        plt.show()


if __name__ == '__main__':
    unittest.main()
