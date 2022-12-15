import numpy as np
import tensorflow as tf

def negative_loglikelihood(y_true, y_pred):

    '''
    Negative Log-Likelihood, see Section IV.B in the TCAN paper.

    Parameters:
    __________________________________
    y_true: tf.Tensor.
        Actual values of target time series, a tensor with shape (n_samples, n_targets) where n_samples is the batch
        size and n_targets is the number of target time series.

    y_pred: tf.Tensor.
        Predicted means and standard deviations of target time series, a tensor with shape (n_samples, n_targets, 2)
        where n_samples is the batch size and n_targets is the number of target time series.

    Returns:
    __________________________________
    tf.Tensor.
        Loss value, a scalar tensor.
    '''

    y_true = tf.cast(y_true, dtype=tf.float32)

    mu = tf.cast(y_pred[:, :, 0], dtype=tf.float32)
    sigma = tf.cast(y_pred[:, :, 1], dtype=tf.float32)

    L = 0.5 * tf.math.log(2 * np.pi) + tf.math.log(sigma) + tf.math.divide(tf.math.pow(y_true - mu, 2), 2 * tf.math.pow(sigma, 2))

    return tf.experimental.numpy.nanmean(tf.experimental.numpy.nansum(L, axis=-1))


def mean_absolute_error(y_true, y_pred):

    '''
    Mean Absolute Error, see Section IV.B in the TCAN paper.

    Parameters:
    __________________________________
    y_true: tf.Tensor.
        Actual values of target time series, a tensor with shape (n_samples, n_targets) where n_samples is the batch
        size and n_targets is the number of target time series.

    y_pred: tf.Tensor.
        Predicted means and standard deviations of target time series, a tensor with shape (n_samples, n_targets, 2)
        where n_samples is the batch size and n_targets is the number of target time series. Note that only the
        predicted means are used in this case.

    Returns:
    __________________________________
    tf.Tensor.
        Loss value, a scalar tensor.
    '''

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred[:, :, 0], dtype=tf.float32)

    L = tf.abs(y_true - y_pred)

    return tf.reduce_mean(tf.reduce_sum(L, axis=-1))
