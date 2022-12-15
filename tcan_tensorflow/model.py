import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
pd.options.mode.chained_assignment = None

from tcan_tensorflow.modules import encoder, decoder
from tcan_tensorflow.utils import get_training_sequences_with_covariates, get_training_sequences
from tcan_tensorflow.losses import negative_loglikelihood, mean_absolute_error

class TCAN():

    def __init__(self,
                 y,
                 x=None,
                 forecast_period=1,
                 lookback_period=2,
                 quantiles=[0.1, 0.5, 0.9],
                 filters=32,
                 kernel_size=2,
                 dilation_rates=[1, 2, 4, 8],
                 dropout=0,
                 alpha=1.5):

        '''
        Implementation of multivariate time series forecasting model introduced in Lin, Y., Koprinska, I., & Rana, M.
        (2021). Temporal Convolutional Attention Neural Networks for Time Series Forecasting. In 2021 International
        Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.

        Parameters:
        __________________________________
        y: np.array.
            Target time series, array with shape (n_samples, n_targets) where n_samples is the length of the time series
            and n_targets is the number of target time series.

        x: np.array.
            Features time series, array with shape (n_samples, n_features) where n_samples is the length of the time series
            and n_features is the number of features time series.

        forecast_period: int.
            Number of future time steps to forecast.

        lookback_period: int.
            Number of past time steps to use as input.

        quantiles: list.
            Quantiles of target time series to be predicted.

        filters: int.
            Number of filters (or channels) of the convolutional layers.

        kernel_size: int.
            Kernel size of the convolutional layers.

        dilation_rates: list.
            Dilation rates of the convolutional layers.

        dropout: float.
            Dropout rate.

        alpha: float.
            Entmax parameter.
        '''

        # Extract the quantiles.
        q = np.unique(np.array(quantiles))
        if 0.5 not in q:
            q = np.sort(np.append(0.5, q))

        # Normalize the targets.
        y_min, y_max = np.min(y, axis=0), np.max(y, axis=0)
        y = (y - y_min) / (y_max - y_min)
        self.y_min = y_min
        self.y_max = y_max

        # Normalize the features.
        if x is not None:
            x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
            x = (x - x_min) / (x_max - x_min)
            self.x_min = x_min
            self.x_max = x_max

        # Save the inputs.
        self.y = y
        self.x = x
        self.q = q
        self.n_features = x.shape[1] if x is not None else 0
        self.n_samples = y.shape[0]
        self.n_targets = y.shape[1]
        self.n_quantiles = len(q)
        self.n_lookback = lookback_period
        self.n_forecast = forecast_period

        if x is not None:

            # Extract the input and output sequences.
            self.x_encoder, self.y_encoder, self.y_decoder = get_training_sequences_with_covariates(
                y=y,
                x=x,
                n_samples=self.n_samples,
                n_targets=self.n_targets,
                n_features=self.n_features,
                n_lookback=self.n_lookback,
            )

            # Build the model.
            self.model = build_fn_with_covariates(
                n_targets=self.n_targets,
                n_features=self.n_features,
                n_lookback=self.n_lookback,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rates=dilation_rates,
                dropout=dropout,
                alpha=alpha
            )

        else:

            # Extract the input and output sequences.
            self.y_encoder, self.y_decoder = get_training_sequences(
                y=y,
                n_samples=self.n_samples,
                n_targets=self.n_targets,
                n_lookback=self.n_lookback,
            )

            # Build the model.
            self.model = build_fn(
                n_targets=self.n_targets,
                n_lookback=self.n_lookback,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rates=dilation_rates,
                dropout=dropout,
                alpha=alpha
            )

    def fit(self,
            regularization=0.5,
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            validation_split=0,
            verbose=1):

        '''
        Train the model.

        Parameters:
        __________________________________
        regularization: float.
            Regularization parameter.

        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        validation_split: float.
            Fraction of the training data to be used as validation data, must be between 0 and 1.

        verbose: int.
            Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.
        '''

        # Compile the model.
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=lambda y_true, y_pred: regularization * negative_loglikelihood(y_true, y_pred) + mean_absolute_error(y_true, y_pred),
        )

        # Fit the model.
        if self.x is not None:

            self.model.fit(
                x=[self.x_encoder, self.y_encoder],
                y=self.y_decoder,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                verbose=verbose
            )

        else:

            self.model.fit(
                x=self.y_encoder,
                y=self.y_decoder,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                verbose=verbose
            )

    def forecast(self, y, x=None):
    
        '''
        Generate the forecasts.
        Parameters:
        __________________________________
        y: np.array.
            Past values of target time series, array with shape (n_samples, n_targets) where n_samples is the length
            of the time series and n_targets is the number of target time series. The number of past samples provided
            (n_samples) should not be less than the length of the lookback period.

        x: np.array.
            Past and future values of features time series, array with shape (n_samples + n_forecast, n_features) where
            n_samples is the length of the time series, n_forecast is the decoder length and n_features is the number
            of features time series. The number of past samples provided (n_samples) should not be less than the length
            of the lookback period.
        Returns:
        __________________________________
        forecasts: pd.DataFrame.
            Data frame including the actual values of the time series and the predicted quantiles.
        '''

        # Scale the data.
        y = (y - self.y_min) / (self.y_max - self.y_min)

        if x is not None:
            x = (x - self.x_min) / (self.x_max - self.x_min)

        # Reshape the data.
        y_decoder = np.reshape(y[- self.n_lookback:, :], (1, self.n_lookback, self.n_targets))
        
        if x is not None:
            x_encoder = np.array([x[i - self.n_lookback: i, :] for i in range(x.shape[0] - self.n_forecast, x.shape[0])])
  
        # Generate the forecasts.
        y_pred = []

        for i in range(self.n_forecast):

            # Generate the one-step-ahead forecast.
            if x is not None:
                y_future = self.model([x_encoder[i: i + 1, :, :], y_decoder]).numpy()
            else:
                y_future = self.model(y_decoder).numpy()

            # Feed the mean forecast back to the model as an input.
            y_decoder = np.append(y_decoder[:, 1:, :], y_future[:, :, 0].reshape(1, 1, self.n_targets), axis=1)

            # Save the mean and standard deviation forecasts.
            y_pred.append(y_future[0, :, :])

        y_pred = np.array(y_pred)

        # Organize the forecasts in a data frame.
        columns = ['time_idx']
        columns.extend(['target_' + str(i + 1) for i in range(self.n_targets)])
        columns.extend(['target_' + str(i + 1) + '_' + str(self.q[j]) for i in range(self.n_targets) for j in range(self.n_quantiles)])

        df = pd.DataFrame(columns=columns)
        df['time_idx'] = np.arange(self.n_samples + self.n_forecast)

        for i in range(self.n_targets):
            df['target_' + str(i + 1)].iloc[: - self.n_forecast] = \
                self.y_min[i] + (self.y_max[i] - self.y_min[i]) * self.y[:, i]

            for j in range(self.n_quantiles):
                df['target_' + str(i + 1) + '_' + str(self.q[j])].iloc[- self.n_forecast:] = \
                self.y_min[i] + (self.y_max[i] - self.y_min[i]) * norm_ppf(y_pred[:, i, 0], y_pred[:, i, 1], self.q[j])

        # Return the data frame.
        return df.astype(float)


def build_fn(
        n_targets,
        n_lookback,
        filters,
        kernel_size,
        dilation_rates,
        dropout,
        alpha):

    '''
    Build the model.

    Parameters:
    __________________________________
    n_targets: int.
        Number of target time series.

    n_lookback: int.
        Encoder length.

    filters: int.
        Number of filters (or channels) of the convolutional layers.

    kernel_size: int.
        Kernel size of the convolutional layers.

    dilation_rates: list.
        Dilation rates of the convolutional layers.

    dropout: float.
        Dropout rate.

    alpha: float.
        Entmax parameter.
    '''

    # Define the inputs.
    x = tf.keras.layers.Input(shape=(n_lookback, n_targets))

    # Forward pass the inputs through the encoder module.
    for i in range(len(dilation_rates)):

        if i == 0:

            encoder_output = encoder(
                encoder_input=x,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rates[i],
                dropout=dropout
            )

        else:

            encoder_output = encoder(
                encoder_input=encoder_output,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rates[i],
                dropout=dropout
            )

    # Forward pass the encoder output through the decoder module.
    decoder_output = decoder(encoder_output, alpha)

    # Calculate the means and standard deviations.
    y1 = tf.keras.layers.Dense(units=n_targets)(decoder_output)
    y2 = softplus(tf.keras.layers.Dense(units=n_targets)(decoder_output))

    # Reshape the output to match the shape required by the loss function.
    y1 = tf.keras.layers.Reshape(target_shape=(n_targets, 1))(y1)
    y2 = tf.keras.layers.Reshape(target_shape=(n_targets, 1))(y2)
    y = tf.keras.layers.Concatenate()([y1, y2])

    return tf.keras.models.Model(x, y)


def build_fn_with_covariates(
        n_targets,
        n_features,
        n_lookback,
        filters,
        kernel_size,
        dilation_rates,
        dropout,
        alpha):

    '''
    Build the model with covariates.

    Parameters:
    __________________________________
    n_targets: int.
        Number of target time series.

    n_features: int.
        Number of features time series.

    n_lookback: int.
        Encoder length.

    filters: int.
        Number of filters (or channels) of the convolutional layers.

    kernel_size: int.
        Kernel size of the convolutional layers.

    dilation_rates: list.
        Dilation rates of the convolutional layers.

    dropout: float.
        Dropout rate.

    alpha: float.
        Entmax parameter.
    '''

    # Define the inputs.
    x1 = tf.keras.layers.Input(shape=(n_lookback, n_features))
    x2 = tf.keras.layers.Input(shape=(n_lookback, n_targets))

    # Concatenate the inputs.
    encoder_input = tf.keras.layers.Concatenate()([x1, x2])

    # Forward pass the inputs through the encoder module.
    for i in range(len(dilation_rates)):

        if i == 0:

            encoder_output = encoder(
                encoder_input=encoder_input,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rates[i],
                dropout=dropout
            )

        else:

            encoder_output = encoder(
                encoder_input=encoder_output,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rates[i],
                dropout=dropout
            )

    # Forward pass the encoder output through the decoder module.
    decoder_output = decoder(encoder_output, alpha)

    # Calculate the means and standard deviations.
    y1 = tf.keras.layers.Dense(units=n_targets)(decoder_output)
    y2 = softplus(tf.keras.layers.Dense(units=n_targets)(decoder_output))

    # Reshape the output to match the shape required by the loss function.
    y1 = tf.keras.layers.Reshape(target_shape=(n_targets, 1))(y1)
    y2 = tf.keras.layers.Reshape(target_shape=(n_targets, 1))(y2)
    y = tf.keras.layers.Concatenate()([y1, y2])

    return tf.keras.models.Model([x1, x2], y)


def softplus(x):

    '''
    Softplus activation function, used for ensuring the positivity of the standard deviation of the Normal distribution.
    See Section IV.B in the TCAN paper.
    '''

    return tf.math.log(1.0 + tf.math.exp(x))


def norm_ppf(loc, scale, value):

    '''
    Quantiles of the Normal distribution.
    '''

    return tfp.distributions.Normal(loc, scale).quantile(value).numpy().flatten()
