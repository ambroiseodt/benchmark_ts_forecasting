from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import xgboost


class Solver(BaseSolver):
    name = "XGBoost"
    sampling_strategy = "tolerance"

    install_cmd = "conda"
    requirements = ["xgboost"]

    def set_objective(self, X, y):
        pass

    def run(self, tolerance):
        pass

    def get_result(self):
        pass
