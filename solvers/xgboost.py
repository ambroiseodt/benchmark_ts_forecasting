from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    import xgboost as xgb


class Solver(BaseSolver):
    name = "XGBoost"

    install_cmd = "conda"
    requirements = ["xgboost"]

    parameters = {
        "max_depth": [5, 10, 20],
        "eta": [0.1, 0.01],  # Learning rate
    }

    def set_objective(self, X, y):
        self.X, self.y = X, y

    def run(self, _):
        X_train = self.X[0]
        y_train = self.y[0]

        X_test = self.X[-1]
        y_test = self.y[-1]

        length = 20
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

    def get_result(self):
        return dict(pred=self.y_pred)
