import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Concatenate, Reshape
pd.options.mode.chained_assignment = None

from tcan_tensorflow.modules import encoder, decoder
from tcan_tensorflow.utils import get_training_sequences_with_covariates, get_training_sequences
from tcan_tensorflow.losses import NLL, MAE
from tcan_tensorflow.plots import plot

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
        Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE. https://doi.org/10.1109/IJCNN52387.2021.9534351.

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

        if type(y) != np.ndarray:
            raise ValueError('The target time series must be provided as a numpy array.')

        elif np.isnan(y).sum() != 0:
            raise ValueError('The target time series cannot contain missing values.')

        if len(y.shape) > 2:
            raise ValueError('The targets array cannot have more than 2 dimensions. Found {} dimensions.'.format(len(y.shape)))

        elif len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

        if x is not None:

            if type(x) != np.ndarray:
                raise ValueError('The features time series must be provided as a numpy array.')

            elif np.isnan(x).sum() != 0:
                raise ValueError('The features time series cannot contain missing values.')

            if len(x.shape) > 2:
                raise ValueError('The features array cannot have more than 2 dimensions. Found {} dimensions.'.format(len(x.shape)))

            elif len(x.shape) == 1:
                x = np.expand_dims(x, axis=1)

            if y.shape[0] != x.shape[0]:
                raise ValueError('The targets and features time series must have the same length.')

        if type(dilation_rates) != list:
            raise ValueError('The dilation rates must be provided as a list.')

        elif len(dilation_rates) == 0:
            raise ValueError('No dilation rates were provided.')

        if type(quantiles) != list:
            raise ValueError('The quantiles must be provided as a list.')

        elif len(quantiles) == 0:
            raise ValueError('No quantiles were provided.')

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
            optimizer=Adam(learning_rate=learning_rate),
            loss=lambda y_true, y_pred: regularization * NLL(y_true, y_pred) + MAE(y_true, y_pred),
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

    def predict(self, index):

        '''
        Extract the in-sample predictions.

        Parameters:
        __________________________________
        index: int.
            The start index of the sequence to predict.

        Returns:
        __________________________________
        predictions: pd.DataFrame.
            Data frame including the actual values of the time series and the predicted quantiles.
        '''

        if index < self.n_lookback:
            raise ValueError('The index must be greater than {}.'.format(self.n_lookback))

        elif index > len(self.y) - self.n_forecast:
            raise ValueError('The index must be less than {}.'.format(self.n_samples - self.n_forecast))

        # Extract the predictions for the selected sequence.
        if self.x is not None:
            y_pred = self.model.predict([self.x_encoder, self.y_encoder])
        else:
            y_pred = self.model.predict(self.y_encoder)

        y_pred = y_pred[index - self.n_lookback: index - self.n_lookback + self.n_forecast, :, :]

        # Organize the predictions in a data frame.
        columns = ['time_idx']
        columns.extend(['target_' + str(i + 1) for i in range(self.n_targets)])
        columns.extend(['target_' + str(i + 1) + '_' + str(self.q[j]) for i in range(self.n_targets) for j in range(self.n_quantiles)])

        predictions = pd.DataFrame(columns=columns)
        predictions['time_idx'] = np.arange(self.n_samples)

        for i in range(self.n_targets):
            predictions['target_' + str(i + 1)] = self.y_min[i] + (self.y_max[i] - self.y_min[i]) * self.y[:, i]

            for j in range(self.n_quantiles):
                predictions['target_' + str(i + 1) + '_' + str(self.q[j])].iloc[index: index + self.n_forecast] = \
                self.y_min[i] + (self.y_max[i] - self.y_min[i]) * norm_ppf(y_pred[:, i, 0], y_pred[:, i, 1], self.q[j])

        predictions = predictions.astype(float)

        # Save the data frame.
        self.predictions = predictions

        # Return the data frame.
        return predictions

    def forecast(self, x=None):

        '''
        Generate the out-of-sample forecasts.

        Parameters:
        __________________________________
        x: np.array.
            Features time series, array with shape (n_forecast, n_features) where n_forecast is the number of future
            time steps to forecast and n_features is the number of features time series.

        Returns:
        __________________________________
        forecasts: pd.DataFrame.
            Data frame including the actual values of the time series and the predicted quantiles.
        '''

        if x is not None:

            if type(x) != np.ndarray:
                raise ValueError('The features time series must be provided as a numpy array.')

            elif np.isnan(x).sum() != 0:
                raise ValueError('The features time series cannot contain missing values.')

            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=1)

            elif len(x.shape) > 2:
                raise ValueError('The features array cannot have more than 2 dimensions. Found {} dimensions.'.format(len(x.shape)))

            if x.shape[0] != self.n_forecast:
                raise ValueError('The length of the features time series must be equal to the length of the forecast period.')

        if self.x is not None:

            # Append the future features values to the past values.
            x = (x - self.x_min) / (self.x_max - self.x_min)
            x = np.vstack([self.x, x])

            # Generate the new features sequences.
            x_encoder = np.zeros((self.n_samples + self.n_forecast, self.n_lookback, self.n_features))

            for i in range(self.n_lookback, x.shape[0]):
                x_encoder[i, :, :] = x[i - self.n_lookback + 1: i + 1, :]

            # Keep only the future features sequences.
            x_encoder = x_encoder[- self.n_forecast:, :, :]

        # Extract the last observed target sequence.
        y_decoder = np.reshape(self.y[- self.n_lookback:, :], (1, self.n_lookback, self.n_targets))

        # Generate the multi-step forecasts.
        y_pred = []

        for i in range(self.n_forecast):

            # Generate the one-step-ahead forecast.
            if self.x is not None:
                y_future = self.model.predict([x_encoder[i: i + 1, :, :], y_decoder])
            else:
                y_future = self.model.predict(y_decoder)

            # Feed the mean forecast back to the model as an input.
            y_decoder = np.append(y_decoder[:, 1:, :], y_future[:, :, 0].reshape(1, 1, self.n_targets), axis=1)

            # Save the mean and standard deviation forecasts.
            y_pred.append(y_future[0, :, :])

        y_pred = np.array(y_pred)

        # Organize the forecasts in a data frame.
        columns = ['time_idx']
        columns.extend(['target_' + str(i + 1) for i in range(self.n_targets)])
        columns.extend(['target_' + str(i + 1) + '_' + str(self.q[j]) for i in range(self.n_targets) for j in range(self.n_quantiles)])

        forecasts = pd.DataFrame(columns=columns)
        forecasts['time_idx'] = np.arange(self.n_samples + self.n_forecast)

        for i in range(self.n_targets):
            forecasts['target_' + str(i + 1)].iloc[: - self.n_forecast] = \
                self.y_min[i] + (self.y_max[i] - self.y_min[i]) * self.y[:, i]

            for j in range(self.n_quantiles):
                forecasts['target_' + str(i + 1) + '_' + str(self.q[j])].iloc[- self.n_forecast:] = \
                self.y_min[i] + (self.y_max[i] - self.y_min[i]) * norm_ppf(y_pred[:, i, 0], y_pred[:, i, 1], self.q[j])

        forecasts = forecasts.astype(float)

        # Save the data frame.
        self.forecasts = forecasts

        # Return the data frame.
        return forecasts

    def plot_predictions(self):

        '''
        Plot the in-sample predictions.

        Returns:
        __________________________________
        go.Figure.
        '''

        return plot(self.predictions, self.q, self.n_targets, self.n_quantiles)

    def plot_forecasts(self):

        '''
        Plot the out-of-sample forecasts.

        Returns:
        __________________________________
        go.Figure.
        '''

        return plot(self.forecasts, self.q, self.n_targets, self.n_quantiles)


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
    x = Input(shape=(n_lookback, n_targets))

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
    y1 = Dense(units=n_targets)(decoder_output)
    y2 = softplus(Dense(units=n_targets)(decoder_output))

    # Reshape the output to match the shape required by the loss function.
    y1 = Reshape(target_shape=(n_targets, 1))(y1)
    y2 = Reshape(target_shape=(n_targets, 1))(y2)
    y = Concatenate()([y1, y2])

    return Model(x, y)


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
    x1 = Input(shape=(n_lookback, n_features))
    x2 = Input(shape=(n_lookback, n_targets))

    # Concatenate the inputs.
    encoder_input = Concatenate()([x1, x2])

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
    y1 = Dense(units=n_targets)(decoder_output)
    y2 = softplus(Dense(units=n_targets)(decoder_output))

    # Reshape the output to match the shape required by the loss function.
    y1 = Reshape(target_shape=(n_targets, 1))(y1)
    y2 = Reshape(target_shape=(n_targets, 1))(y2)
    y = Concatenate()([y1, y2])

    return Model([x1, x2], y)


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
