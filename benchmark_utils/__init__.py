# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

import numpy as np
from sklearn.preprocessing import StandardScaler


def mse(X: np.array, y: np.array):
    """Compute mean square error (MSE).
    The (potentially multivariate) time series have n_features variables.
    The forecasting task is done at the horizon n_horizon.

    Args:
        X (np.array): output of the solver. Shape = (n_features, horizon).
        y (np.array): ground-truth. Shape = (n_features, horizon).

    Returns:
        MSE (float).
    """
    diff = (X - y) ** 2
    return diff.mean()


def mae(X: np.array, y: np.array):
    """Compute mean absolute error (MAE).
    The (potentially multivariate) time series have n_features variables.
    The forecasting task is done at the horizon n_horizon.

    Args:
        X (np.array): output of the solver. Shape = (n_features, n_horizon).
        y (np.array): ground-truth. Shape = (n_features, n_horizon).

    Returns:
        MAE (float).
    """
    diff = np.abs(X - y)
    return diff.mean()


def scale_data(*data: np.array):
    """Standardize features by removing the mean and scaling to unit variance.
    The input data is a tuple of matrices containing multivariate time series. For instance,
    data = [X_train, X_val, X_test] and we fit the standard scaler on X_train and then
    transform X_val and X_test.

    Args:
        *data(tuple of np.array): Tuple of np.array.

    Returns:
        data (List of np.array): List of scaled np.array.
    """
    data = list(data)
    scaler = StandardScaler()
    scaler.fit(data[0])
    for i in range(1, len(data)):
        data[i] = scaler.transform(data[i])

    return data

def data_windowing(
    data: np.ndarray,
    window_size: int,
    step: int,
    horizon: int,
):
    """
    Apply windowing to the data

    Parameters
    ----------
    data : np.ndarray
        Data to be windowed
    window_size : int
        Size of the window
    step : int
        Step size between windows
    horizon : int
        Number of steps to forecast
    """
    X = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=window_size, axis=0
    )[::step]

    labels = np.lib.stride_tricks.sliding_window_view(
        data[window_size:], window_shape=window_size, axis=0
    )[::step][:, :, :horizon]

    # Remove the last windows that do not have labels
    data = X[: -(len(X) - len(labels))]

    return data, labels
