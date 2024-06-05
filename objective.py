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
    name = "Time Series Forecasting"

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
    requirements = ["scikit-learn"]


    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.6"

    def set_data(self, X, y):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape

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
                    + i * self.horizon : self.eval_window_size
                    + (i + 1) * self.horizon
                ]
                new_data = np.r_[new_data, next_data]

                if self.fixed_window_size:
                    # Remove the first samples to keep new_data at a fixed size
                    new_data = new_data[self.horizon :]

            return np.concatenate(predictions)

        def compute_metrics_from_pred(X, pred):
            true_data = X[self.eval_window_size :]
            return mse(pred, true_data), mae(pred, true_data)

        # Compute the evaluation metrics on the train data
        pred_train = get_pred(self.X_train)
        loss_mse_train, loss_mae_train = compute_metrics_from_pred(
            self.X_train, pred_train
        )

        # Compute the evaluation metrics on the test data
        pred_test = get_pred(self.X_test)
        loss_mse_test, loss_mae_test = compute_metrics_from_pred(self.X_test, pred_test)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=loss_mse_train,
            loss_mae_train=loss_mae_train,
            loss_mse_test=loss_mse_test,
            loss_mae_test=loss_mae_test,
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
            X_train=self.X_train,
            # Possibly, here we can send X_test to the solver, as follows:
            # X_test=self.X_test
        )
