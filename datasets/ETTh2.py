from benchopt import BaseDataset, safe_import_context
from benchmark_utils import scale_data, data_windowing


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    import os
    import requests


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "ETTh2"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {"window_size": [512], "horizon": [96], "stride": [1]}

    train_ratio = 0.7
    val_ratio = 0.20
    test_ratio = 0.10

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_cmd = "conda"
    requirements = ["os", "requests", "scikit-learn"]

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Load the data
        data_path = os.path.join(os.path.dirname(__file__), "data/")
        os.makedirs(data_path, exist_ok=True)

        # Download the data if it does not exist
        if not os.path.exists(os.path.join(data_path, "ETTh2.csv")):
            url = (
                "https://drive.google.com/uc?&id=1bOcmp9"
                "VAv03d3kUYSrttOFvLZ0keXDC5&export=download"
            )
            response = requests.get(url)
            with open(os.path.join(data_path, "ETTh2.csv"), "wb") as f:
                f.write(response.content)

        data = pd.read_csv(os.path.join(data_path, "ETTh2.csv"))
        data = data.to_numpy()
        data[:, 0] = data[:, 0].astype(np.datetime64).astype(np.float32)

        # Split the data
        n = len(data)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        X_train = data[:n_train]
        X_val = data[n_train: n_train + n_val]  # noqa
        X_test = data[n_train + n_val:]  # noqa

        # Need to scale data first
        X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

        # Data shape is (n_windows, n_features, window_size)
        X_train, y_train = data_windowing(
            X_train, self.window_size, self.stride, self.horizon
        )

        X_val, y_val = data_windowing(
            X_val, self.window_size, self.stride, self.horizon
        )

        X_test, y_test = data_windowing(
            X_test, self.window_size, self.stride, self.horizon
        )

        # The dictionary defines the keyword arguments for `Objective.set_data`

        return dict(X=(X_train, X_val, X_test), y=(y_train, y_val, y_test))
