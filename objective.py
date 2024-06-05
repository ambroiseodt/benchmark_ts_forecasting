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

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = [numpy, sklearn]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, X, y):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y = X, y

        # X.shape = (n_samples, n_features)
        # Y is None

        # Specify a cross-validation splitter as the `cv` attribute.
        # This will be automatically used in `self.get_split` to split
        # the arrays provided.
        self.cv = TimeSeriesSplit(
            n_splits=5,
            max_train_size=None,
            test_size=None,
            gap=0,
        )

        # If the cross-validation requires some metadata, it can be
        # provided in the `cv_metadata` attribute. This will be passed
        # to `self.cv.split` and `self.cv.get_n_splits`.

    def evaluate_result(self, pred):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        Y_train = self.y[0]  # keep only train data

        loss_mse = mse(pred, Y_train)  # ((Y_train - pred) ** 2).mean()

        loss_mae = mae(pred, Y_train)  # np.abs(Y_train - pred).mean()

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=loss_mse,
            loss_mae=loss_mae,
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(pred=np.zeros_like(self.y[0]))

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.

        # Call `self.get_split` with the arrays to split.
        # This method default behavior behave like sklearn's
        # `train_test_split`, splitting the input arrays using
        # the indexes returned by `self.cv.split`.
        self.X_train, self.X_test = self.get_split(self.X)

        # How it works:
        # 1) get_split() returns 1 fold
        # 2) the solver is run on the X_train of the current fold
        # 3) the solver returns the model, and we can cumpute the predictions
        # on X_train and X_test in the evaluate_result() method.
        # 4) the solver is run until there is convergence
        # 5) Once convergence is reached, get_split() is called again, and then
        # returns the next fold.

        # Only the full train data, of shape (n_sample_fold, n_features),
        # is passed to the solver.
        return dict(
            X=self.X_train,
            # Possibly, here we can send X_test to the solver, as follows:
            # X_test=self.X_test
        )
