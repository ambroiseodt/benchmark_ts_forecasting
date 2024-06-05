from benchopt import BaseSolver


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.

# import your reusable functions here


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = "solver_template"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "param1": [1],
    }
    method = None

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    def set_objective(self, X_train):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        self.X_train = X_train  # shape (n_samples_train, n_features)

        # Here, do desire pre-processing of X_train

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        # Fit develop forecast method

        self.method.fit(self.X_train, n_iter)

    class ForecastModel:
        def __init__(self, method):
            self.method = method

        def predict(new_data, horizon):
            """
            Function that, given data, predicts the next steps

            Parameters
            ----------
            new_data : np.ndarray, shape (n_samples, n_features)
                The data to predict from

            horizon : int
                The number of steps to predict

            Returns
            -------
            pred : np.ndarray, shape (horizon, n_features)
                The predicted data
            """
            raise NotImplementedError

    def get_result(self):
        model = self.ForecastModel(self.method)
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=model)
