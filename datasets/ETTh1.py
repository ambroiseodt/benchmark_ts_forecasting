from benchopt import BaseDataset, safe_import_context
from benchmark_utils import scale_data, data_windowing

with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    import os


class Dataset(BaseDataset):
    name = "ETTh1"

    train_ratio = 0.7
    val_ratio = 0.20
    test_ratio = 0.10
    window_size = 100
    step = 10
    horizon = 96

    requirements = ["numpy"]

    def get_data(self):
        # Load the data
        data_path = os.path.join(os.path.dirname(__file__), "data/")
        os.makedirs(data_path, exist_ok=True)
        os.system(
            f"""
            wget -O {data_path}/ETTh1.csv "https://drive.google.com/uc?&id=1vOClm_t4RgUf8nqherpTfNnB8rrYq14Q&export=download"
            """
        )
        data = pd.read_csv(os.join(data_path, "ETTh1.csv"))

        # Split the data
        n = len(data)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        n_test = n - n_train - n_val

        X_train = data[:n_train]
        X_val = data[n_train : n_train + n_val]
        X_test = data[n_train + n_val :]

        # Need to scale data first
        X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

        # Data shape is (n_windows, n_features, window_size)
        X_train, y_train = data_windowing(
            X_train, self.window_size, self.step, self.horizon
        )

        X_val, y_val = data_windowing(X_val, self.window_size, self.step, self.horizon)

        X_test, y_test = data_windowing(
            X_test, self.window_size, self.step, self.horizon
        )

        return dict(X=(X_train, X_val, X_test), y=(y_train, y_val, y_test))
