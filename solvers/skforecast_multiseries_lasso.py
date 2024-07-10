from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skforecast.ForecasterAutoregMultiSeries import (
        ForecasterAutoregMultiSeries,
    )
    from sklearn.linear_model import Lasso
    from benchmark_utils import df_fit_predict


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = "skforecast_multiseries_lasso"

    # To run only once the solver
    stopping_criterion = SingleRunCriterion()

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "lags": [5, 10, 20],
        "alpha": [1e-1, 1e0, 1e1],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    install_cmd = "conda"
    requirements = ["scikit-learn", "pip:skforecast"]

    def set_objective(self, X, y):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X, self.y = X, y
        self.forecaster_model = ForecasterAutoregMultiSeries(
            regressor=Lasso(alpha=self.alpha),
            lags=self.lags,  # noqa
        )

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        X_train = self.X[0]
        Y_train = self.y[0]
        horizon = Y_train.shape[-1]

        assert X_train.ndim == 3

        self.pred = df_fit_predict(
            X=X_train, model=self.forecaster_model, horizon=horizon
        )

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(pred=self.pred)
