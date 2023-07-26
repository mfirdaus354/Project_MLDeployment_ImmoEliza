import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from typing import Optional

class ModelConfig:
    def __init__(self):
        pass

    @staticmethod
    def load_data(filepath: str, file_type: str, usecols: Optional[list] = None):
        if file_type not in ["csv", "json", "excel", "pickle"]:
            raise ValueError("Invalid file_type. Supported types are 'csv', 'json', 'excel', and 'pickle'.")

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
        if df is None:
            raise ValueError("Please provide a valid DataFrame.")
        
        if target_col is not None:
            y = df[target_col].values
            X = df.drop(columns=[target_col], axis=1).values
            return X, y

        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        return X, y

    def column_transformer(df, numeric_features=[], categorical_features=[], ordinal_categorical_features=[]):
        preprocessor = ColumnTransformer(
            transformers=[]
        )
        
        ohe = OneHotEncoder()
        feature_array = ohe.fit_transform(df[[x, y]]).toarray()
        feature_labels = ohe.get_feature_names_out(["subtype", "epc_score"])

        return preprocessor
    
    def custom_scorer(y_true, y_pred):
        return -mean_squared_error(y_true, y_pred)
    
    def huber_loss(y_true, y_pred, delta=1.0):
        residual = y_true - y_pred
        absolute_residual = np.abs(residual)
        quadratic_residual = 0.5 * (residual ** 2)
        is_small_residual = absolute_residual <= delta
        loss = np.where(is_small_residual, quadratic_residual, delta * absolute_residual - 0.5 * delta ** 2)
        return np.mean(loss)
    
    def MAPE(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100