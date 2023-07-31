import os
import sys
import requests
import numpy as np
from typing import Optional, Literal, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import KNNImputer, SimpleImputer
import pandas as pd


def CURRENT_DIR():
    cwd = os.getcwd()
    return os.chdir(cwd[: (cwd.index("Eliza") + 5)])


CURRENT_DIR()


class input_preprocess:
    @staticmethod
    def input_data(source: dict):
        if source is None or isinstance(source, dict) is False:
            raise ValueError("Please input your dataset again")

        data_array = np.array(list(source.values()), dtype=np.float32)
        serialized_data = data_array.tobytes()
        posted_data = requests.post(
            url="http://127.0.0.1:8000/items/", data=serialized_data
        )
        return posted_data

    @staticmethod
    def get_data():
        get_data = requests.get(url="http://127.0.0.1:8000/items/").json()
        data_df = pd.DataFrame(get_data)
        data_df = data_df.drop("id", axis=1)
        data_df = data_df.drop_duplicates(keep="first")
        return data_df

    @staticmethod
    def preprocess_new_data(source: pd.DataFrame):
        if source is None or isinstance(source, pd.DataFrame) is False:
            raise ValueError("Source file not recognized")

        preprocessing_pipeline = make_pipeline(
            KNNImputer(missing_values=np.nan, n_neighbors=10),
            StandardScaler(with_std=True),
        )
        data = source.values

        data_preprocessed = preprocessing_pipeline.fit_transform(data)
        data_preprocessed = data_preprocessed[0].reshape(1, -1)

        return data_preprocessed
