import pandas as pd
import seaborn as sns
import numpy as np
from typing import Optional, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_percentage_error
from sklearn.impute import KNNImputer, SimpleImputer
import xgboost as xgb


class ModelConfig:
    XGB_ParamGrid = {
        'xgbregressor__n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        'xgbregressor__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4],
        'xgbregressor__max_depth': [3, 5, 7, 9, 11, 13],
        'xgbregressor__min_child_weight': [1, 1.5, 2, 2.5, 3],
        'xgbregressor__subsample': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        'xgbregressor__colsample_bytree': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        'xgbregressor__reg_lambda': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        'xgbregressor__gamma': [0],
        'xgbregressor__random_state': [42, 59, 6351],
        'xgbregressor__objective': ['reg:squarederror'],
        'xgbregressor__eval_metric': ['rmse', 'mape', 'explained_variance'],
        'xgbregressor__early_stopping_rounds': [10, 20, 30, 40, 50],
        'xgbregressor__verbose': [True]
    }

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
    
    @staticmethod    
    def column_transformer(to_fill: Optional[Union[str, list]] = None, to_scale: Optional[Union[str, list]] = None, to_encode: Optional[Union[str, list]] = None):
        if isinstance(to_fill, str):
            features_to_fill = [to_fill]
        
        if isinstance(to_scale, str):
            features_to_scale = [to_scale]

        if isinstance(to_encode, str):
            features_to_encode = [to_encode]

        if to_fill is None and to_encode is None and to_scale is None:
            raise ValueError("Invalid parameter input. Please try again.")

        preprocessor = ColumnTransformer(
            transformers=[
                (
            "to_fill", 
            make_pipeline(
                        SimpleImputer(strategy="median")
                        ),to_fill
            ), 
                (
            "to_scale", 
            make_pipeline(
                        StandardScaler(with_mean=False, with_std=True)
                        ),to_scale
            ),
                (
            "to_encode", 
            make_pipeline(
                        OneHotEncoder(
                                        sparse=True,
                                        sparse_output=True, 
                                        handle_unknown="ignore", 
                                        max_categories=99
                                        )
                        ),to_encode
            )
                    ]
                                        )

        return preprocessor
    
    @staticmethod
    def XGBREGRConfig():
        xgb_reg = xgb.XGBRegressor(
            n_estimators=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            learning_rate=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4],
            max_depth=[3, 5, 7, 9, 11, 13],
            min_child_weight=[1, 1.5, 2, 2.5, 3],
            subsample=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            colsample_bytree=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            reg_alpha=0.01,
            reg_lambda=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            gamma=0,
            n_jobs=-1,
            objective='reg:squarederror',
            eval_metric=['rmse', 'mape', 'explained_variance'],  # Evaluation metric on the validation set
            early_stopping_rounds=[10, 20, 30, 40, 50],  # Early stopping after 10 rounds without improvement
            verbose=1
            )
        return xgb_reg
    
    @staticmethod
    def poly_features_config(degree: Optional[int] = 2, interaction_only: Optional[bool] = True, include_bias: Optional[bool] = False):
        poly_config = PolynomialFeatures(
                            degree=degree,
                            interaction_only=interaction_only,
                            include_bias=include_bias
                                )
        return poly_config
    
    @staticmethod
    def XGBGridSearchCV(estimator = str(), 
                        param: Optional[dict] =XGB_ParamGrid, 
                        scoring: Optional[dict] = {'explained_variance': 'explained_variance',
                                                   'mape': make_scorer(mean_absolute_percentage_error),
                                                   'r2': 'r2'},
                        refit=str(), 
                        cv_fold: Optional[int] = 2, 
                        n_jobs: Optional[int]= -1, 
                        verbose: Optional[int]= 2):
 
        xgb_gridsearch = GridSearchCV(estimator=estimator,
            param_grid=param,
            refit=refit,
            cv=cv_fold,
            n_jobs=-1,
            verbose=2
        )

        return xgb_gridsearch
    
    
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



class Config:
    column_list_one = ["price", "habitable_surface", "bedroom_count", "room_count"]

    def __init__(self):
        pass

    @staticmethod
    def expand_display(x, y: Optional[pd.DataFrame] = None, z: Optional[pd.DataFrame] = None,):
        if x is None:
            raise ValueError("Invalid parameter input. Please try again.")
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        pd.set_option('display.float_format', '{:20,.2f}'.format)
        pd.set_option('display.max_colwidth', None)
        return display(x, y, z) 
    
    @staticmethod
    def fill_nan(df: pd.DataFrame, column: Optional[Union[str, list]] = None):
        if isinstance(column, str):
            column = [column]

        for col in column:
            if col not in Config.column_list_one:
                print(f"Invalid column name '{col}'. Please check your parameters.")
                continue

            if col in df.columns:
                if df[col].isnull().any():
                    imputer = KNNImputer(n_neighbors=5)
                    col_idx = df.columns.get_loc(col)
                    df.iloc[:, col_idx:col_idx+1] = imputer.fit_transform(df.iloc[:, col_idx:col_idx+1])

        # Filling any remaining missing values with SimpleImputer
        imputer = SimpleImputer(strategy='median')
        df[Config.column_list_one] = imputer.fit_transform(df[Config.column_list_one])

    @classmethod
    def configure_seaborn(cls):
        # Graph settings for seaborn
        sns.set_theme(style="whitegrid", palette="pastel")
        sns.set(rc={"figure.figsize": (100, 100)})
        sns.set(font_scale=3)







