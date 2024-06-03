# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

import numpy as np


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
