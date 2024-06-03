# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

from typing import List

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


def scale_data(*data: List):
    data = list(data)
    scaler = StandardScaler()
    scaler.fit(data[0])
    for i in range(1, len(data)):
        data[i] = scaler.transform(data[i])

    return data
