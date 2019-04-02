import tensorflow as tf
import alf.algorithms.vae as vae
import numpy as np

from tensorflow.keras.datasets import mnist

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (original_dim, )
latent_dim = 2
batch_size = 100
epochs = 50

vae = vae.VariationalAutoEncoder(
    latent_dim,
    preprocess_layers=tf.keras.layers.Dense(512, activation='relu'))

decoding_l1 = tf.keras.layers.Dense(512, activation='relu')
decoding_l2 = tf.keras.layers.Dense(original_dim, activation='sigmoid')

inputs = tf.keras.layers.Input(shape=input_shape)
z, kl_loss = vae.sampling_forward(inputs)
outputs = decoding_l2(decoding_l1(z))

loss = tf.reduce_mean(tf.keras.losses.mse(inputs, outputs) * original_dim + kl_loss)

model = tf.keras.Model(inputs, outputs, name = "vae")
model.add_loss(loss)
model.compile(optimizer='adam')
model.summary()

model.fit(x_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(x_test, None))
model.save_weights('/tmp/vae/vae_mlp_mnist')

import matplotlib.pyplot as plt

# test decoding image qualities
nrows = 20
ncols = 2
test_inputs = x_test[:nrows]
decoded_output = model.predict(test_inputs)

fig = plt.figure()
idx = 1
for i in range(nrows):
    fig.add_subplot(nrows, ncols, idx)
    plt.imshow(np.reshape(x_test[i], (image_size, image_size)))
    fig.add_subplot(nrows, ncols, idx + 1)
    plt.imshow(np.reshape(decoded_output[i], (image_size, image_size)))

    idx+= 2
plt.show()


# test sampled image qualities
eps = tf.random.normal((nrows * nrows, latent_dim), dtype=tf.float32, mean=0., stddev=1.0)
sampled_outputs = decoding_l2(decoding_l1(eps))

fig = plt.figure()
#import pdb; pdb.set_trace()
idx = 0
for i in range(nrows):
    for j in range(nrows):
        fig.add_subplot(nrows, nrows, idx + 1)
        plt.imshow(np.reshape(sampled_outputs[idx], (image_size, image_size)))
        idx += 1

plt.show()
