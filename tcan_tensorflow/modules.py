import tensorflow as tf
import tensorflow_addons as tfa

from tcan_tensorflow.layers.layers import SparseAttention

def encoder(encoder_input, filters, kernel_size, dilation_rate, dropout):

    '''
    Encoder module, see Section III.A in the TCAN paper.

    Parameters:
    __________________________________
    encoder_input: tf.Tensor.
        For the first stack, this is a tensor with shape (n_samples, n_lookback, n_features + n_targets) where
        n_samples is the batch size, n_lookback is the encoder length, n_features is the number of features and
        n_targets is the number of targets. For the subsequent stacks, this is a tensor with shape (n_samples,
        n_lookback, filters) where filters is the number of channels of the convolutional layers.

    filters: int.
        Number of filters (or channels) of the convolutional layers.

    kernel_size: int.
        Kernel size of the convolutional layers.

    dilation_rate: int.
        Dilation rate of the convolutional layers.

    dropout: float.
        Dropout rate.

    Returns:
    __________________________________
    encoder_output: tf.Tensor.
        A tensor with shape (n_samples, n_lookback, filters) where n_samples is the batch size, n_lookback is
        the encoder length and filters is the number of channels of the convolutional layers.
    '''

    encoder_output = tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal'), data_init=False)(encoder_input)
    encoder_output = tf.keras.layers.ReLU()(encoder_output)
    encoder_output = tf.keras.layers.Dropout(rate=dropout)(encoder_output)

    encoder_output = tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal'), data_init=False)(encoder_output)
    encoder_output = tf.keras.layers.ReLU()(encoder_output)
    encoder_output = tf.keras.layers.Dropout(rate=dropout)(encoder_output)

    if encoder_input.shape[-1] != encoder_output.shape[-1]:
        encoder_input = tf.keras.layers.Conv1D(filters=filters, kernel_size=1)(encoder_input)

    encoder_output = tf.keras.layers.Add()([encoder_input, encoder_output])

    return encoder_output


def decoder(encoder_output, alpha):

    '''
    Decoder module, see Section IV.B in the TCAN paper.

    Parameters:
    __________________________________
    encoder_output: tf.Tensor.
        A tensor with shape (n_samples, n_lookback, filters) where n_samples is the batch size, n_lookback
        is the encoder length and filters is the number of channels of the convolutional layers.

    alpha: float.
        Entmax parameter.

    Returns:
    __________________________________
    decoder_output: tf.Tensor.
        A tensor with shape (n_samples, 1, 2 * filters) where n_samples is the batch size and filters is the
        number of channels of the convolutional layers. The length of the second dimension (i.e. the decoder
        length) is equal to one given that the model generates one-step-ahead forecasts.
    '''

    # Extract the query matrix (current step).
    q = encoder_output[:, -1:, :]

    # Extract the value matrix (historical steps).
    v = encoder_output[:, :-1, :]

    # Calculate the context vector.
    c = SparseAttention(alpha=alpha)(inputs=[q, v])

    # Concatenate the context vector with the query matrix (current step).
    decoder_output = tf.keras.layers.Concatenate()([c, q])

    return decoder_output
