from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    import os


class Dataset(BaseDataset):
    name = "ETTh1"

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
        X = data.drop("OT", axis=1).values
        X = np.array(X).T

        y = data["OT"].values
        return dict(X=X, y=y)
