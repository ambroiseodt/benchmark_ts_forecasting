# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from warnings import warn



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


def check_data(data):
    """
    Ensure the given data is a tuple of size 3 (train, val, test)
    and that the dimensions are valid:
    - train.shape = (n_windows_train, n_features, window_size)
    - val.shape = (n_windows_val, n_features, window_size) or val is None
    - test.shape = (n_windows_test, n_features, window_size)

    Parameters:
    -----------
    data (tuple or list): A tuple or list containing train, val, and test datasets.

    Returns:
    --------
    bool: True if the data is valid, raises ValueError otherwise.
    """

    # Convert list to tuple if necessary
    if isinstance(data, list):
        data = tuple(data)

    # Handle case where data is a tuple of size 2 (train, test)
    if isinstance(data, tuple) and len(data) == 2:
        data = (data[0], None, data[1])

    # Ensure data is a tuple of size 3
    if not isinstance(data, tuple) or len(data) != 3:
        raise ValueError("Data must be a tuple of size 3 (train, val, test).")

    train, val, test = data

    # Check if train and test are numpy arrays
    if not isinstance(train, np.ndarray) or not isinstance(test, np.ndarray):
        raise ValueError(
            "Train and test elements of the data tuple must be numpy arrays."
        )

    # Check dimensions for train and test
    train_shape = train.shape
    test_shape = test.shape

    if len(train_shape) != 3 or len(test_shape) != 3:
        raise ValueError(
            "Train and test datasets must have 3 dimensions (n_windows, n_features, window_size)."
        )

    # Check if val is None or a numpy array
    if val is not None:
        if not isinstance(val, np.ndarray):
            raise ValueError(
                "Validation element of the data tuple must be a numpy array or None."
            )

        # Check dimensions for val
        val_shape = val.shape
        if len(val_shape) != 3:
            raise ValueError(
                "Validation dataset must have 3 dimensions (n_windows, n_features, window_size)."
            )

        # Check if the second and third dimensions (n_features, window_size) match across all datasets
        if train_shape[1:] != val_shape[1:] or train_shape[1:] != test_shape[1:]:
            raise ValueError(
                "The second and third dimensions (n_features, window_size) must match across train, val, and test datasets."
            )
    else:
        # Check if the second and third dimensions (n_features, window_size) match between train and test
        if train_shape[1:] != test_shape[1:]:
            raise ValueError(
                "The second and third dimensions (n_features, window_size) must match across train and test datasets."
            )

    return data



def df_fit_predict(X, model, horizon):
    """
    Fit the model and predict the next `horizon` steps.
    Works with skforecast by converting each window (for a specific time step)
    to a DataFrame.

    Parameters:
    -----------
    X (np.array): Array of shape (n_windows, n_features, window_size).
    model (skforecast.ForecasterAutoregMultiSeries): Forecasting model.
    horizon (int): Number of steps to forecast.

    Returns:
    --------
    np.array: Array of shape (n_windows, n_features, horizon).
    """
    
    def reshape_and_fit(X, model):
        n_windows, n_features, window_size = X.shape
        if n_features != 1:
            warn("The model only accepts 1 feature.")
            return
        
        X_train_reshaped = X.transpose(1, 2, 0).reshape(n_features, -1).T
        model.fit(pd.DataFrame(X_train_reshaped))

    output_list = []

    if model.__class__.__name__ == "ForcasterSarimax":
        reshape_and_fit(X, model)
    
    for x in X:
        x_df = pd.DataFrame(x.T)  # x: (n_features, n_obs) -> x_df: (n_obs, n_features)
        
        if model.__class__.__name__ == "ForecasterAutoregMultiOutput":
            model.fit(x_df)
            
        y_pred = model.predict(steps=horizon)
        output_list.append(y_pred.to_numpy().T)
    
    return np.array(output_list)
