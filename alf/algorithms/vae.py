import tensorflow as tf


class VariationalAutoEncoder:
    """
    VariationalAutoEncoder encodes data into diagonal multivariate gaussian, do sampling with
    reparametrization trick, and return kl divergence between posterior and prior.
    """
    def __init__(self,
                 hidden_dims,
                 prior_network=None,
                 preprocess_layers=None):

        """Create an instance of `VariationalAutoEncoder`

        Args:
             hidden_dims (int): dimension of latent dimension
             prior_network:     network to compute the priors (mean, log_var)
             preprocess_layers: layers to preprocess input data before project into (mean, log_var)

        """
        if preprocess_layers is None:
            # according to appendix in deepmind merlin paper
            self._preprocess_layers = tf.keras.Sequential([
                tf.keras.layers.Dense(2 * hidden_dims, activation='tanh') for i in range(2)])
        else:
            self._preprocess_layers = preprocess_layers

        self._hidden_dims = hidden_dims
        self._prior_network = prior_network
        self._z_mean = tf.keras.layers.Dense(hidden_dims)
        self._z_log_var = tf.keras.layers.Dense(hidden_dims)

    def sampling_forward(self, inputs):
        """Encoding the data into latent space then do sampling.

        Args:
            inputs (Tensor or Tuple of Tensor): data to be encoded. If it has a prior network, then the argument
            is a tuple of (prior_input, new_observation).
        """
        if self._prior_network:
            prior_input, new_obs = inputs
            prior_z_mean, prior_z_log_var = self._prior_network(prior_input)
            inputs = tf.concat([prior_input, new_obs, prior_z_mean, priror_z_log_var], -1)

        latents = self._preprocess_layers(inputs)
        z_mean = self._z_mean(latents)
        z_log_var = self._z_log_var(latents)

        if self._prior_network:
            z_mean += prior_z_mean
            z_log_var += prior_z_log_var

            kl_div_loss = tf.square(z_mean - prior_z_mean) / tf.exp(prior_z_log_var) + \
                          tf.exp(z_log_var - prior_z_log_var) + \
                          prior_z_log_var - z_log_var - 1.0
        else:
            kl_div_loss = tf.square(z_mean) + tf.exp(z_log_var) - 1.0 - z_log_var

        kl_div_loss = 0.5 * tf.reduce_sum(kl_div_loss, axis=-1)
        # reparameterization sampling: z = u + var ** 0.5 * eps
        eps = tf.random.normal(tf.shape(z_mean), dtype=tf.float32, mean=0., stddev=1.0)
        z = z_mean + tf.exp(z_log_var / 2) * eps
        return z, kl_div_loss
