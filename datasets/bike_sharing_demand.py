from benchopt import BaseDataset, safe_import_context

from sklearn.datasets import fetch_openml

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils import data_windowing, scale_data


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "bike_sharing_demand"

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
    requirements = ["scikit-learn"]

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Fetch dataset from OpenML
        # https://www.openml.org/search?type=data&status=active&id=44063
        bike_sharing = fetch_openml(
            "Bike_Sharing_Demand", version=2, as_frame=True
        )  # noqa
        df = bike_sharing.frame

        # Only keep a subsample of variables, the quantitative ones
        columns = ["temp", "feel_temp", "humidity", "windspeed", "count"]
        data = df[columns]

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

        length = 10
        X_train = X_train[:length]
        y_train = y_train[:length]

        X_val = X_val[:length]
        y_val = y_val[:length]

        X_test = X_test[:length]
        y_test = y_test[:length]

        print(X_train.shape, y_train.shape)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=(X_train, X_val, X_test), y=(y_train, y_val, y_test))
