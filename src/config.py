import pandas as pd
import seaborn as sns
import numpy as np
from typing import Optional, Union, Literal
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV 
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.impute import KNNImputer, SimpleImputer
import xgboost as xgb
from sklearn.pipeline import Pipeline
import random

class ModelConfig:


    @staticmethod
    def XGBParamConfig(Size=int()):
        GB_ParamGrid = {
            'xgbregressor_n_estimator': np.linspace(100, 1000, num=Size, dtype=int, endpoint=True).tolist(),
            'xgbregressor__learning_rate': np.linspace(0.01, 0.1, num=Size, dtype=float, endpoint=True).tolist(),
            'xgbregressor__max_depth': np.linspace(1, 10, num=Size, dtype=int, endpoint=True).tolist(),
            'xgbregressor__min_child_weight': np.linspace(1.0, 5.0, num=Size, dtype=float, endpoint=True).tolist(),
            'xgbregressor__subsample': np.linspace(0.5, 0.75, num=Size, dtype=float, endpoint=True).tolist(),
            'xgbregressor__colsample_bytree':np.linspace(0.5, 0.75, num=Size, dtype=float, endpoint=True).tolist(),
            'xgbregressor__reg_alpha':np.linspace(0.001, 0.1, num=Size, dtype=float, endpoint=True).tolist(),
            'xgbregressor__reg_lambda': np.linspace(0.1, 0.5, num=Size, dtype=float, endpoint=True).tolist(),
            'xgbregressor__gamma': np.linspace(0.1, 0.5, num=Size, dtype=float, endpoint=True).tolist(),
            'xgbregressor__random_state': np.linspace(10, 1000, num=1, dtype=int, endpoint=True).tolist(),
            'xgbregressor__objective': ['reg:pseudohubererror'],
            'xgbregressor__eval_metric': ['mphe','mse','mae','r2'],
            'xgbregressor__early_stopping_rounds': np.linspace(10, 1000, num=Size, dtype=int, endpoint=True).tolist(),
            'xgbregressor__huber_slope': np.linspace(0.01, 0.1, num=Size, dtype=float, endpoint=True).tolist(),
            'xgbregressor__verbose': np.linspace(1, 3, num=1, dtype=int, endpoint=True).tolist()
    }


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
    def PimpMyPipeline(steps: Union[str, list] = None, poly_degree:Optional[int]=2):
        if isinstance(steps, list):
            pipeline_steps= steps
            for step in pipeline_steps:
                if step == 'knn_imputer':
                    pipeline_steps[pipeline_steps.index('knn_imputer')] = ('knn_imputer', KNNImputer(n_neighbors=np.linspace(1, 10, num=1, dtype=int, endpoint=True)[0], missing_values=np.NaN, weights="distance", keep_empty_features=True))
                elif step == "poly_features":
                    pipeline_steps[pipeline_steps.index('poly_features')] = ('poly_features', ModelConfig.poly_features_config(degree=poly_degree))
                elif step == "std_scaler":
                    pipeline_steps[pipeline_steps.index('std_scaler')] = ('std_scaler', StandardScaler(with_mean=False, with_std=True))
                else:
                    raise ValueError("Keywords not found")
            if len(pipeline_steps) >= 2:
                pipeline = Pipeline(steps=pipeline_steps)
                return pipeline
            else:
                raise ValueError("The number of steps are inssufficient to build a pipeline.")

        raise TypeError("Please provide a valid list of step keywords.")
    
    
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
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.4,
            colsample_bytree=0.4,
            gamma=0.1,
            reg_alpha=0.001,
            reg_lambda=0.1,
            objective='reg:pseudohubererror',
            tree_method="hist",
            min_child_weight=0.1,
            base_score=0.4,
            eval_metric="mphe",
            early_stopping_rounds=30,
            random_state=452,
            huber_slope=0.1,
            validate_parameters=True)
        


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
    def GridSearchCV(est=str(), param=dict(), scores=list(), refit=str(), cv_base=list()):
        grid_search = GridSearchCV(
            estimator=est, 
            param_grid=param, 
            scoring=scores, 
            n_jobs=-1, 
            refit=refit, 
            BaseCrossValidator=cv_base, 
            verbose=2)

    @staticmethod
    def XGBGridSearchCV(
        estimator = str(),
        param: Optional[dict] = XGBParamConfig, 
        scores= str(),
        cv_fold: Optional[int] = 2):
        xgb_gridsearch = GridSearchCV(estimator=estimator,
            param_grid=param,
            scoring=scores,
            refit=True,
            return_train_score=True,
            pre_dispatch='2*n_jobs',
            error_score="raise",
            cv=cv_fold,
            n_jobs=-1,
            verbose=2
        )

        return xgb_gridsearch
    
    @staticmethod
    def custom_scorer(y_true, y_pred, keyword: Literal["mse", "mae", "r2", "mape"]):
        if keyword == "mse":
            return np.abs(mean_squared_error(y_true, y_pred, squared=True))
        elif keyword == "mae":
            return np.abs(mean_absolute_error(y_true, y_pred))
        elif keyword == "r2":
            return np.abs(r2_score(y_true, y_pred))
        elif keyword == "mape":
            return np.abs(mean_absolute_percentage_error(y_true, y_pred))
        else:
            raise ValueError("Invalid keyword input. Please try again.")

    @staticmethod
    def custom_pseudo_huber_loss(y_true, y_pred):
        delta = 1.0  # You can adjust the delta value as needed
        residual = y_true - y_pred
        huber_loss = np.abs(delta ** 2 * (np.sqrt(1 + (residual / delta) ** 2) - 1))
        return huber_loss

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







