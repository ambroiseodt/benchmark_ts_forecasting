from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "n_features": [5],
        "n_samples": [1_000],
        "random_state": [42],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_cmd = "conda"
    requirements = ["scikit-learn"]

    def __init__(self, n_features=5, n_samples=1000, random_state=42):
        self.n_features = n_features
        self.n_samples = n_samples
        self.random_state = random_state

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        rng = np.random.RandomState(self.random_state)

        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)

        return dict(X=X, y=y)
