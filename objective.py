from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils import mse, mae
    from sklearn.model_selection import TimeSeriesSplit


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "MSE"

    # URL of the main repo for this benchmark.
    url = "https://github.com/ambroiseodt/benchmark_ts_forecasting"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        # Parameters for cv-split
        "n_splits": [5],  # number of folds in the cross-validation split
        # Parameters for evaluate results
        "eval_window_size": [512],  # size of the initial window for prediction
        "fixed_window_size": [
            True
        ],  # If False, will increase the window by 'horizon' steps after each sub-evaluation
        "horizon": [32],  # number of step to predict
    }

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    install_cmd = "conda"
    requirements = [numpy, sklearn]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, X, y):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape

        # X.shape = (n_samples, n_features)
        # Y is None

        # Specify a cross-validation splitter as the `cv` attribute.
        # This will be automatically used in `self.get_split` to split
        # the arrays provided.
        self.cv = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=None,
            test_size=None,
            gap=0,
        )

        # Compute length of one fold
        self.n_samples_fold = int(self.n_samples / (self.n_splits + 1))

        assert (
            self.eval_window_size + self.horizon <= self.n_samples_fold
        ), "The number of data in each fold is smaller than the needed number of data for evaluation. Decrease the number of splits and/or the size of the evaluation window size."

        # If the cross-validation requires some metadata, it can be
        # provided in the `cv_metadata` attribute. This will be passed
        # to `self.cv.split` and `self.cv.get_n_splits`.

    def evaluate_result(self, model):
        """
        model : instance of ForecastModel that has a .predict() method
        """
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        def get_pred(X):
            """
            Compute the sliding forecast predictions for a given nd.array
            """
            n_pred = int(
                (X.shape[0] - self.eval_window_size) / self.horizon
            )  # number of sub-evaluation possible

            # Initialize the data to compute the prediction from
            new_data = X[: self.eval_window_size]

            predictions = []
            for i in range(n_pred):
                predictions.append(model.predict(new_data, horizon=self.horizon))

                next_data = X[
                    self.eval_window_size
                    + i * s