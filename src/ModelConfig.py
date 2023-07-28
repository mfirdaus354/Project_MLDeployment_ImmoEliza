import pandas as pd
import seaborn as sns
import numpy as np
import xgboost as xgb
import os
from typing import Optional, Union, Literal
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from pandas.core.frame import DataFrame
from sklearn.impute import KNNImputer, SimpleImputer
import pickle


class ModelConfig:

    @staticmethod
    def load_data(filepath: str, file_type: str, usecols: Optional[list] = None):
        """
        Load data from a specified file path and type.

        Parameters:
        filepath (str): The path to the data file.
        file_type (str): The type of data file ('csv', 'json', 'excel', or 'pickle').
        usecols (list, optional): List of columns to use from the data file.

        Returns:
        pd.DataFrame: The loaded data as a DataFrame.
        """
        if file_type not in ["csv", "json", "excel", "pickle"]:
            raise ValueError(
                "Invalid file_type. Supported types are 'csv', 'json', 'excel', and 'pickle'."
            )

        if file_type == "csv":
            return pd.read_csv(filepath, usecols=usecols)
        elif file_type == "json":
            return pd.read_json(filepath, usecols=usecols)
        elif file_type == "excel":
            return pd.read_excel(filepath, usecols=usecols)
        elif file_type == "pickle":
            return pd.read_pickle(filepath)
        
    @staticmethod
    def feature_target_config(df, target_col: Optional[str] = None):
        """
        Extract features and target from the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing features and target data.
        target_col (str, optional): The name of the target column (default is None).

        Returns:
        tuple: A tuple containing the feature data and target data as numpy arrays.
        """
        if df is None:
            raise ValueError("Please provide a valid DataFrame.")

        if target_col is not None:
            y = df[target_col].values
            X = df.drop(columns=[target_col], axis=1).values
            return X, y
    
    @staticmethod
    def PimpMyPipeline(steps: Union[str, list] = None, poly_degree: Optional[int] = 2):
        if isinstance(steps, list):
            pipeline_steps = steps
            for step in pipeline_steps:
                if step == "knn_imputer":
                    pipeline_steps[pipeline_steps.index("knn_imputer")] = (
                        "knn_imputer",
                        KNNImputer(
                            n_neighbors=np.linspace(
                                1, 10, num=1, dtype=int, endpoint=True
                            )[0],
                            missing_values=np.NaN,
                            weights="distance",
                            keep_empty_features=True,
                        ),
                    )
                elif step == "poly_features":
                    pipeline_steps[pipeline_steps.index("poly_features")] = (
                        "poly_features",
                        ModelConfig.poly_features_config(
                            degree=poly_degree, include_bias=True, ordzzz="F"
                        ),
                    )
                elif step == "std_scaler":
                    pipeline_steps[pipeline_steps.index("std_scaler")] = (
                        "std_scaler",
                        StandardScaler(with_mean=False, with_std=True),
                    )
                else:
                    raise ValueError("Keywords not found")
            if len(pipeline_steps) >= 2:
                pipeline = Pipeline(steps=pipeline_steps)
                return pipeline
            else:
                raise ValueError(
                    "The number of steps are inssufficient to build a pipeline."
                )

        raise TypeError("Please provide a valid list of step keywords.")
    
    @staticmethod
    def poly_features_config(
        degree: Optional[int] = 2,
        interaction_only: Optional[bool] = True,
        include_bias: Optional[bool] = False,
        ordzzz: Optional[str] = "F",
    ):
        """
        Create a configuration for polynomial features.

        Parameters:
        degree (int, optional): The degree of the polynomial features (default is 2).
        interaction_only (bool, optional): If True, only interaction features are produced (default is True).
        include_bias (bool, optional): If True, include a bias column (default is False).
        ordzzz (str, optional): The order of the output array (default is 'F').

        Returns:
        PolynomialFeatures: A configured PolynomialFeatures object.
        """
        poly_config = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            order=ordzzz,
        )
        return poly_config
