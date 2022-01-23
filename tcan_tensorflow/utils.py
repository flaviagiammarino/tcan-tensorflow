import numpy as np

def get_training_sequences_with_covariates(y, x, n_samples, n_targets, n_features, n_lookback):

    '''
    Split the time series into input and output sequences, see Section II.B in the TCAN paper.

    Parameters:
    __________________________________
    y: np.array.
        Target time series, array with shape (n_samples, n_targets) where n_samples is the length of the time series
        and n_targets is the number of target time series.

    x: np.array.
        Features time series, array with shape (n_samples, n_features) where n_samples is the length of the time series
        and n_features is the number of features time series.

    n_samples: int.
        Length of the time series.

    n_targets: int.
        Number of target time series.

    n_features: int.
        Number of features time series.

    n_lookback: int.
        Encoder length.

    Returns:
    __________________________________
    x_encoder: np.array.
        Encoder features, array with shape (n_samples - n_lookback, n_lookback, n_features).

    y_encoder: np.array.
        Encoder targets, array with shape (n_samples - n_lookback, n_lookback, n_targets).

    y_decoder: np.array.
        Decoder targets, array with shape (n_samples - n_lookback, n_targets).
    '''

    x_encoder = np.zeros((n_samples, n_lookback, n_features))
    y_encoder = np.zeros((n_samples, n_lookback, n_targets))
    y_decoder = np.zeros((n_samples, n_targets))

    for i in range(n_lookback, n_samples):

        x_encoder[i, :, :] = x[i - n_lookback + 1: i + 1, :]
        y_encoder[i, :, :] = y[i - n_lookback: i, :]
        y_decoder[i, :] = y[i, :]

    x_encoder = x_encoder[n_lookback:, :, :]
    y_encoder = y_encoder[n_lookback:, :, :]
    y_decoder = y_decoder[n_lookback:, :]

    return x_encoder, y_encoder, y_decoder


def get_training_sequences(y, n_samples, n_targets, n_lookback):

    '''
    Split the time series into input and output sequences, see Section II.B in the TCAN paper.

    Parameters:
    __________________________________
    y: np.array
        Target time series, array with shape (n_samples, n_targets) where n_samples is the length of the time series
        and n_targets is the number of target time series.

    n_samples: int
        Length of the time series.

    n_targets: int
        Number of target time series.

    n_lookback: int
        Encoder length.

    Returns:
    __________________________________
    y_encoder: np.array.
        Encoder targets, array with shape (n_samples - n_lookback, n_lookback, n_targets).

    y_decoder: np.array.
        Decoder targets, array with shape (n_samples - n_lookback, n_targets).
    '''

    y_encoder = np.zeros((n_samples, n_lookback, n_targets))
    y_decoder = np.zeros((n_samples, n_targets))

    for i in range(n_lookback, n_samples):

        y_encoder[i, :, :] = y[i - n_lookback: i, :]
        y_decoder[i, :] = y[i, :]

    y_encoder = y_encoder[n_lookback:, :, :]
    y_decoder = y_decoder[n_lookback:, :]

    return y_encoder, y_decoder
