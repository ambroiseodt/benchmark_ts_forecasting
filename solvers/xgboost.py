from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import xgboost as xgb

# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = "XGBoost"

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = ["xgboost", "scikit-learn"]

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "max_depth": [5, 10, 20],
        "eta": [0.1, 0.01],  # Learning rate
    }

    method = None

    def set_objective(self, X, y):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X, self.y = X, y

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html
        X_train = self.X[0]
        y_train = self.y[0]

        X_test = self.X[-1]
        y_test = self.y[-1]

        length = n_iter
        model = None
        for i in range(length):
            # Discarding the timestamps because of compatibility issues
            x = xgb.DMatrix(X_train[i], label=y_train[i])
            model = xgb.train(
                params={
                    "objective": "reg:squarederror",
                    "max_depth": self.max_depth,
                    "eta": self.eta,
                },
                dtrain=x,
                xgb_model=model,
            )

        testing = False
        if testing:
            Xu = X_test
            yu = y_test
        else:
            Xu = X_train
            yu = y_train

        test_length = 10
        self.y_pred = np.zeros_like(yu[:test_length])
        for i in range(test_length):
            x = xgb.DMatrix(Xu[i])
            self.y_pred[i] = model.predict(x)

    class ForecastModel:
        def __init__(self, method):
            self.method = method

        def predict(self, new_data, horizon):
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
            x = xgb.DMatrix(new_data)
            return self.method.predict(x, ntree_limit=horizon)

    def get_result(self):
        model = self.ForecastModel(self.method)
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(pred=self.y_pred, model=model)
