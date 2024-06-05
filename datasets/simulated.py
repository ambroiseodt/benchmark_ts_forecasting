from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils import check_data


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
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_cmd = "conda"
    requirements = ["scikit-learn"]

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        rng = np.random.RandomState(self.random_state)

        X = rng.randn(self.n_samples, self.n_features)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X)
