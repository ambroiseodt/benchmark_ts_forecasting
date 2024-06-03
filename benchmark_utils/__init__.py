# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

def gradient_ols(X, y, beta):
    return X.T @ (X @ beta - y)


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
        raise ValueError("Train and test elements of the data tuple must be numpy arrays.")

    # Check dimensions for train and test
    train_shape = train.shape
    test_shape = test.shape

    if len(train_shape) != 3 or len(test_shape) != 3:
        raise ValueError("Train and test datasets must have 3 dimensions (n_windows, n_features, window_size).")

    # Check if val is None or a numpy array
    if val is not None:
        if not isinstance(val, np.ndarray):
            raise ValueError("Validation element of the data tuple must be a numpy array or None.")

        # Check dimensions for val
        val_shape = val.shape
        if len(val_shape) != 3:
            raise ValueError("Validation dataset must have 3 dimensions (n_windows, n_features, window_size).")

        # Check if the second and third dimensions (n_features, window_size) match across all datasets
        if train_shape[1:] != val_shape[1:] or train_shape[1:] != test_shape[1:]:
            raise ValueError("The second and third dimensions (n_features, window_size) must match across train, val, and test datasets.")
    else:
        # Check if the second and third dimensions (n_features, window_size) match between train and test
        if train_shape[1:] != test_shape[1:]:
            raise ValueError("The second and third dimensions (n_features, window_size) must match across train and test datasets.")

    return data


