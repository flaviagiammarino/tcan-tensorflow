import tensorflow as tf

@tf.function
def entmax(z, alpha, n_iter=100):

    '''
    Entmax Activations.

    Parameters:
    __________________________________
    z: tf.Tensor
        Logits.

    alpha: float
        Entmax parameter.

    n_iter: int.
        Number of iterations of the bisection algorithm.

    Returns:
    __________________________________
    tf.Tensor
        Probabilities.

    References:
    __________________________________
    Peters, B., Niculae, V., & Martins, A. F. (2019). Sparse sequence-to-sequence models.
    https://arxiv.org/abs/1905.05702.
    '''

    if alpha < 1:
        raise ValueError('The sparsity parameter should greater than or equal to 1.')

    elif alpha == 1:
        # Calculate the softmax probabilities.
        return tf.nn.softmax(tf.cast(z, tf.float32), axis=-1)

    else:
        # Calculate the entmax probabilities.
        z = (alpha - 1) * tf.cast(z, tf.float32)
        z_max = tf.reduce_max(z, axis=-1, keepdims=True)
        tau_min = z_max - 1
        tau_max = z_max - (z.shape[-1]) ** (1 - alpha)

        for _ in tf.range(n_iter):
            tau = (tau_min + tau_max) / 2
            p = tf.maximum(z - tau, 0.0) ** (1 / (alpha - 1))
            Z = tf.reduce_sum(p, axis=-1, keepdims=True)
            tau_min = tf.where(Z >= 1, tau, tau_min)
            tau_max = tf.where(Z < 1, tau, tau_max)

        return p / Z
