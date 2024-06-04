from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import xgboost as xgb


class Solver(BaseSolver):
    name = "XGBoost"

    install_cmd = "conda"
    requirements = ["xgboost"]

    parameters = {
        "n_estimators": [10, 100],
        "max_depth": [3, 5],
        "eta": [0.1, 0.01],  # Learning rate
    }

    def set_objective(self, X, y):
        self.X, self.y = X, y

    def run(self):
        X_train = self.X[0]
        Y_train = self.y[0]

        X_test = self.X[-1]
        y_test = self.y[-1]

        length = 10
        model = None
        for i in range(length):
            # Discarding the timestamps because of compatibility issues
            x = xgb.DMatrix(self.X[i, 1:], label=self.y[i, 1:])
            model = xgb.train(
                params={
                    "objective": "reg:squarederror",
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "eta": self.eta,
                },
                dtrain=x,
                xgb_model=model,
            )

        test_length = 10
        y_pred = np.zeros_like(y_test[:test_length, 1:])
        for i in range(test_length):
            x = xgb.DMatrix(X_test[i, 1:])
            y_pred[i] = model.predict(x)

    def get_result(self):
        return dict(pred=self.y_pred)
