from benchopt import BaseDataset, safe_import_context

from sklearn.datasets import fetch_openml

with safe_import_context() as import_ctx:
    from benchmark_utils import data_windowing

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Bike_Sharing_Demand"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'train_size': [0.7],
        'window_size': [100],
        'stride': [1]
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = []

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Fetch dataset from OpenML
        # https://www.openml.org/search?type=data&status=active&id=44063
        bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
        df = bike_sharing.frame

        # Only keep a subsample of variables, the quantitative ones
        columns = ['temp', 'feel_temp', 'humidity', 'windspeed', 'count']
        df = df[columns]

        X, y = data_windowing(df)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X, y=y)
