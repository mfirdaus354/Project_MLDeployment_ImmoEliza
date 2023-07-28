import pandas as pd
import seaborn as sns
import numpy as np
import xgboost as XGB
import os
import pickle
from typing import Optional, Union, Literal
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, validation_curve
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from pandas.core.frame import DataFrame
from sklearn.impute import KNNImputer, SimpleImputer



class ModelConfig:
    @staticmethod
    def DMatrixGenerator(xgb_model: XGB.XGBRegressor, main_array: Optional[Union[np.ndarray, list]], num_array=2, keyword= Literal["train"] | Literal["test"] | Literal["Test_Pred"], ref=Optional[Union[np.ndarray, list]]):
        r"""
        Prepare and return DMatrix objects for XGBoost training and evaluation.

        Parameters:
        main_array --> List object containg multiple np.ndarray.
        num_array = 2 --> can only accepts 2 np.ndarrays at a time
        keyword ---> Literal[str objects] that is either "train", or "test" or "Test_Pred".
        ref --> a np.ndaray object with identical shape dimension to main_array.

        Returns:

        DTrain, XGB.DMatrix: DMatrix object for training.
        Dtest, XGB.DMatrix: DMatrix object for testing
        DTest_Pred, XGB.DMatrix: DMatrix object for Model Evaluation.

        Suggestion:
        
        To generate DTrain use [x_Train, y_train] for both the main_array and ref parameter, as well as the keyword "train".
        To generate DTest use [x_Test, y_test] for both the main_array and ref parameter "test".
        To generate DTest_Pred use [x_Train, y_train] for both the main_array and ref parameter "Test_Pred".

        """

        
        # GENERATING DTrain       
        if keyword == "train" or ref.shape == list(tuple([12936, 32]), ([3234, 32])):
            for array in main_array:    
                sort_indices_one = np.argsort(ref[0][:, 0])#template to begin sorting indices
                array_sorted_one = array[0][sort_indices]
                sort_indices_two = np.argsort(ref[-1][:, 0])#template to begin sorting indices
                array_sorted_two = array[-1][sort_indices]
                array_df_one = pd.DataFrame(array_sorted_one[:, :5], columns=["plot_area", "habitable_surface", "bedroom_count", "land_surface", "room_count"])
                array_df_two = pd.DataFrame(array_sorted_two[:, :1], columns=["price"])
                
                DTrain = XGB.DMatrix(data=pd.concat(objs=[array_df_one, array_df_two], axis=1), label=["X_train", "y_train"], nthread=-1, silent=True)
                return DTrain
        # GENERATING Dtest
        elif keyword == "test" or ref.shape == tuple([3234, 32]):
            for array in main_array:    
                sort_indices_one = np.argsort(ref[0][:, 0])#template to begin sorting indices
                array_sorted_one = array[0][sort_indices]
                sort_indices_two = np.argsort(ref[-1][:, 0])#template to begin sorting indices
                array_sorted_two = array[-1][sort_indices]
                array_df_one = pd.DataFrame(array_sorted_two[:, :5], columns=["plot_area", "habitable_surface", "bedroom_count", "land_surface", "room_count"])
                array_df_two = pd.DataFrame(array_sorted_one[:, :1], columns=["price"])
                Dtest = XGB.DMatrix(pd.concat(objs=[array_df_one, array_df_two], label=["X_test", "Y_test"], axis=1, nthread=-1, silent=True))
                return Dtest
        # GENERATING DTest_Pred
        elif keyword == "Test_Pred" or ref[0].shape == tuple([3234, 32]):
            for array in main_array:    
                sort_indices = np.argsort(ref[0][:, 0])#template to begin sorting indices
                array_sorted_one = array[0][sort_indices]
                array_sorted_two = array[-1][sort_indices]
                array_df_one = pd.DataFrame(array_sorted_one[:, :1], columns=["test_va;ue"])
                array_df_two = pd.DataFrame(array_sorted_two[:, :1], columns=["predicted_value"])
                DTest_Pred = XGB.DMatrix(pd.concat(objs=[array_df_one, array_df_two], axis=1, nthread=-1, silent=True))
                return DTest_Pred
        # Generating DCustom with minimum one main_array
        elif keyword == "test" or keyword == "Test_Pred" and len(main_array)  == 1:
            sort_indices = np.argsort(ref[0][:, 0])#template to begin sorting indices
            array_sorted_one = array[0][sort_indices]
            array_df_one = pd.DataFrame(array_sorted_two[:, :5], columns=["plot_area", "habitable_surface", "bedroom_count", "land_surface", "room_count"])
            DCustom = XGB.DMatrix(data=array_df_one, label=["X_test"])

            
    @staticmethod
    def XGBParamConfig(Size=int()):
        """
        Generate a dictionary of XGBoost hyperparameters with a specified grid size.

        Parameters:
        Size (int): The grid size for hyperparameter search.

        Returns:
        dict: A dictionary containing the grid of XGBoost hyperparameters.
        """
        XGB_ParamGrid = {
            "XGBregressor_booster": ["gbtree", "dart"],
            "XGBregressor_n_estimator": np.linspace(
                100, 1000, num=Size, dtype=int, endpoint=True
            ).tolist(),
            "XGBregressor__learning_rate": np.linspace(
                0.01, 0.1, num=Size, dtype=float, endpoint=True
            ).tolist(),
            "XGBregressor__max_depth": np.linspace(
                1, 10, num=Size, dtype=int, endpoint=True
            ).tolist(),
            "XGBregressor__min_child_weight": np.linspace(
                1.0, 5.0, num=Size, dtype=float, endpoint=True
            ).tolist(),
            "XGBregressor__subsample": np.linspace(
                0.5, 0.75, num=Size, dtype=float, endpoint=True
            ).tolist(),
            "XGBregressor__colsample_bytree": np.linspace(
                0.5, 0.75, num=Size, dtype=float, endpoint=True
            ).tolist(),
            "XGBregressor__sampling_method": ["uniform"],
            "XGBregressor__updater": [
                "refresh,sync,grow_colmaker,prune",
                "refresh,sync,grow_histmaker,prune",
            ],
            "XGBregressor__reg_alpha": np.linspace(
                0.001, 0.1, num=Size, dtype=float, endpoint=True
            ).tolist(),
            "XGBregressor__reg_lambda": np.linspace(
                0.1, 0.5, num=Size, dtype=float, endpoint=True
            ).tolist(),
            "XGBregressor__gamma": np.linspace(
                0.1, 0.5, num=Size, dtype=float, endpoint=True
            ).tolist(),
            "XGBregressor__random_state": np.linspace(
                10, 1000, num=1, dtype=int, endpoint=True
            ).tolist(),
            "XGBregressor__objective": ["reg:pseudohubererror"],
            "XGBregressor__eval_metric": ["mphe", "mse", "mae", "r2"],
            "XGBregressor__early_stopping_rounds": np.linspace(
                10, 1000, num=Size, dtype=int, endpoint=True
            ).tolist(),
            "XGBregressor__huber_slope": np.linspace(
                0.01, 0.1, num=Size, dtype=float, endpoint=True
            ).tolist(),
            "XGBregressor__verbose": np.linspace(
                1, 3, num=1, dtype=int, endpoint=True
            ).tolist(),
        }

        return XGB_ParamGrid

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

        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
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
    def column_transformer(
        to_fill: Optional[Union[str, list]] = None,
        to_scale: Optional[Union[str, list]] = None,
        to_encode: Optional[Union[str, list]] = None,
    ):
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
                ("to_fill", make_pipeline(SimpleImputer(strategy="median")), to_fill),
                (
                    "to_scale",
                    make_pipeline(StandardScaler(with_mean=False, with_std=True)),
                    to_scale,
                ),
                (
                    "to_encode",
                    make_pipeline(
                        OneHotEncoder(
                            sparse=True,
                            sparse_output=True,
                            handle_unknown="ignore",
                            max_categories=99,
                        )
                    ),
                    to_encode,
                ),
            ]
        )

        return preprocessor


    @staticmethod
    def XGBREGRConfig(
        x: np.ndarray, y: np.ndarray, test_size: float = 0.20, random_state: int = 452
    ) -> dict:
        """
        Configure and train an XGBoost regressor.

        Parameters:
        x (numpy.ndarray): The feature data for training.
        y (numpy.ndarray): The target data for training.
        test_size (float): The proportion of test data to split (default is 0.20).
        random_state (int): Random seed for reproducibility (default is 452).

        Returns:
        dict: A dictionary containing information about the training process and model performance.
        """
        steps_taken = []
        parameters=ModelConfig.XGB_param()

        def train_test_split(feature=x, target=y):
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=random_state
            )
            if isinstance(X_train, np.ndarray):
                steps_taken.append("Train-Test Split has been done")
                return X_train, X_test, y_train, y_test, steps_taken 
            raise ValueError("Tran Test Split is failed")
        
        def load_fit_predict_XGBRegressor():
            # calling train_test_split
            X_train, X_test, y_train, y_test, steps_taken = train_test_split()

            # generate DMatrixes
            DTrain = ModelConfig.DMatrixGenerator(xgb_model=XGBRegr_model, keyword="train", main_array=[X_train, y_train], ref=[X_train, y_train])
            if isinstance(DTrain, XGB.DMatrix):
                steps_taken.append("DTrain has ben generated")
            DTest = ModelConfig.DMatrixGenerator(xgb_model=XGBRegr_model, keyword="train", main_array=[X_test, y_test], ref=[X_test, y_test])
            if isinstance(DT, XGB.DMatrix):
                steps_taken.append("DTrain has ben generated")

            # Loading XGBRFRegressor
            XGBRegr_model = XGB.XGBRegressor().set_params(params=parameters)
            if isinstance(XGBRegr_model, XGB.XGBRegressor):
                steps_taken("The XGBRegr_model is now initiated")
                model_params = XGBRegr_model.get_xgb_params()

            
            # Fitting XGBRegressor
            XGBRegr_model.fit(X_train, y_train, sample_weight=np.linspace(start=0.2, stop=2.0, num=5, endpoint=True, dtype=float), verbose=True)

            # predicting y_pred
            DXTest = ModelConfig.DMatrixGenerator(xgb_model=XGBRegr_model,keyword="test", main_array=[X_test], ref=[X_test])
            y_predict= XGBRegr_model.predict(x=DXTest, output_margin=True, training=True, iteration_range=(100, 1000))

            return XGBRegr_model, y_predict
            
        def evaluate_XGBRegr_model():

            # calling train_test_split
            X_train, X_test, y_train, y_test, steps_taken = train_test_split()

            # Calling load_fit_predict_XGBRegressor()
            XGBRegr_model, y_predict = load_fit_predict_XGBRegressor()

            DYTest = ModelConfig.DMatrixGenerator(xgb_model=XGBRegr_model, main_array=[X_test], keyword="test", ref=[X_test])

            best_iter= XGBRegr_model.best_iteration()
            best_score = XGBRegr_model.best_score
            eval_result = XGBRegr_model.evals_result()
            if eval_results:
                steps_taken.append("Evaluation matrices have been generated")
            mse = mean_squared_error(y_true=DYTest, y_pred=y_predict, multioutput={'raw_values', 'uniform_average'}, squared=True)
            rmse = mean_squared_error(y_true=DYTest, y_pred=y_predict, multioutput={'raw_values', 'uniform_average'}, squared=False)
            mae = mean_absolute_error(y_true=DYTest, y_pred=y_predict, multioutput={'raw_values', 'uniform_average'})
            mape= mean_absolute_percentage_error(y_true=DYTest, y_pred=y_predict, multioutput={'raw_values', 'uniform_average'})
            r2_score = r2_score(y_true=y_test, multioutput={'raw_values', 'uniform_average'})

            XGB.to_graphviz(booster=XGBRegr_model)
        

        # Store the results in a dictionary
        results = {
            "Steps Taken": steps_taken,
            "XGBRegressor": {"status" : ["initialized", "fitted", "trained", "predictions generated", "evaluation matrices generated"]},
            "Train Test Split": True,
            "Best N-Iteration": best_iter,
            "Best Scores": best_score,
        }

        if len(results) == 5:
            # Save the model and results to a file
            with open("trained_XGB_model_1.pkl", "wb") as tp:
                pickle.dump(XGB_reg, tp)
            
            XGB_reg.save_model(f"{os.getcwd()}/models/model/trained_XGB_model_1.json")

        if len(results) != 6 :  # Check if all expected keys are present
            raise ValueError(
                "Something went wrong during the process. Please revise the steps."
            )

        return XGB_reg, X_train, X_test, y_train, y_test, y_pred, eval_results, results

    @staticmethod
    def XGB_param():
        param = {
            "booster": "gbtree",
            "tree_method": "hist",
            "interaction_constraints": [[1, 2, 3, 4, 5], [2, 3, 4], [1, 2, 4]],
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.5,
            "sampling_method": "uniform",
            "colsample_bytree": 0.4,
            "gamma": 0.1,
            "reg_alpha": 0.001,
            "reg_lambda": 0.1,
            "objective": "reg:pseudohubererror",
            "min_child_weight": 2,
            "base_score": 0.4,
            "eval_metric": ["rmse", "mae", 'mphe'],
            "early_stopping_rounds": 30,
            "random_state": 452,
            "huber_slope": 0.1,
            "validate_parameters": 1,
        }
        return param

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

    @staticmethod
    def GridSearchCV(
        est=str(), param=dict(), scores=list(), refit=str(), cv_base=list()
    ):
        """
        Perform grid search cross-validation.

        Parameters:
        est (str): The estimator to use for fitting data.
        param (dict): The parameter grid to search.
        scores (list): The scoring method for the evaluation.
        refit (str): The scoring method to use for refitting the best parameters.
        cv_base (list): The cross-validator for the base estimator.

        Returns:
        GridSearchCV: A configured GridSearchCV object.
        """
        grid_search = GridSearchCV(
            estimator=est,
            param_grid=param,
            scoring=scores,
            n_jobs=-1,
            refit=refit,
            BaseCrossValidator=cv_base,
            verbose=2,
        )
        return grid_search

    @staticmethod
    def XGBGridSearchCV(
        estimator=str(),
        param: Optional[dict] = XGBParamConfig,
        scores=str(),
        cv_fold: Optional[int] = 2,
    ):
        """
        Perform grid search cross-validation for XGBoost.

        Parameters:
        estimator (str): The XGBoost estimator to use for fitting data.
        param (dict, optional): The parameter grid to search (default is XGBParamConfig).
        scores (str): The scoring method for the evaluation.
        cv_fold (int, optional): The number of cross-validation folds (default is 2).

        Returns:
        GridSearchCV: A configured GridSearchCV object for XGBoost.
        """
        XGB_gridsearch = GridSearchCV(
            estimator=estimator,
            param_grid=param,
            scoring=scores,
            refit=True,
            return_train_score=True,
            pre_dispatch="2*n_jobs",
            error_score="raise",
            cv=cv_fold,
            n_jobs=-1,
            verbose=2,
        )
        return XGB_gridsearch

    @staticmethod
    def custom_scorer(y_true, y_pred, keyword: Literal["mse", "mae", "r2", "mape"]):
        """
        Calculate a custom evaluation metric for regression.

        Parameters:
        y_true: The true target values.
        y_pred: The predicted target values.
        keyword (Literal["mse", "mae", "r2", "mape"]): The keyword for the evaluation metric.

        Returns:
        float: The calculated evaluation metric value.
        """
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

    def custom_pseudo_huber_loss(y_true, y_pred):
        """
        Calculate the custom Pseudo-Huber loss.

        Pseudo-Huber loss is a smooth approximation of Huber loss that continuously
        differentiable and can be controlled by the 'delta' parameter. As 'delta'
        approaches infinity, the Pseudo-Huber loss approaches the L1 loss (mean absolute
        error). As 'delta' approaches zero, it approaches the L2 loss (mean squared error).

        Parameters:
        y_true: The true target values.
        y_pred: The predicted target values.

        Returns:
        numpy.ndarray: The calculated Pseudo-Huber loss.
        """
        delta = 1.0  # You can adjust the delta value as needed

        # Calculate the residuals
        residual = y_true - y_pred

        # Calculate the Pseudo-Huber loss
        huber_loss = np.abs(delta**2 * (np.sqrt(1 + (residual / delta) ** 2) - 1))

        return huber_loss


class Config:
    """
    A class that provides configuration settings and utility methods for data processing and visualization.
    """

    # List of column names that are considered numeric
    column_numeric = ["price", "habitable_surface", "bedroom_count", "room_count"]

    @staticmethod
    def CURRENT_DIR():
        """
        Get the current working directory and change it to the parent directory named "Eliza".
        """
        cwd = os.getcwd()
        return os.chdir(cwd[: (cwd.index("Eliza") + 5)])

    @staticmethod
    def expand_display(
        x: DataFrame,
        y: Optional[DataFrame] = None,
        z: Optional[DataFrame] = None,
    ):
        """
        Expand the display settings for pandas DataFrames for better visualization.

        Parameters:
        x (DataFrame): The main DataFrame to be displayed.
        y (DataFrame, optional): An additional DataFrame to be displayed side by side (default is None).
        z (DataFrame, optional): Another additional DataFrame to be displayed side by side (default is None).

        Returns:
        None
        """
        if x is None:
            raise ValueError(
                "Invalid parameter input. Please provide a valid DataFrame."
            )

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 2000)
        pd.set_option("display.float_format", "{:20,.2f}".format)
        pd.set_option("display.max_colwidth", None)
        return display(x, y, z)

    @staticmethod
    def fill_nan(df: pd.DataFrame, column: Optional[Union[str, list]] = None):
        """
        Fill missing values in specific columns of a DataFrame using KNNImputer and SimpleImputer.

        Parameters:
        df (pd.DataFrame): The DataFrame to fill missing values in.
        column (str or list, optional): The column name or list of column names to fill missing values (default is None).

        Returns:
        pd.DataFrame: The DataFrame with filled missing values.
        """
        if isinstance(column, str):
            column = [column]

        for col in column:
            if col not in Config.column_numeric:
                print(f"Invalid column name '{col}'. Please check your parameters.")
                continue

            if col in df.columns:
                if df[col].isnull().any():
                    # Use KNNImputer to fill missing values in the specified column
                    imputer = KNNImputer(n_neighbors=5)
                    col_idx = df.columns.get_loc(col)
                    df.iloc[:, col_idx : col_idx + 1] = imputer.fit_transform(
                        df.iloc[:, col_idx : col_idx + 1]
                    )
                    return df

        # Filling any remaining missing values in numeric columns with SimpleImputer using median strategy
        imputer = SimpleImputer(strategy="median")
        df[Config.column_numeric] = imputer.fit_transform(df[Config.column_numeric])
        return df

    @classmethod
    def configure_seaborn(cls):
        """
        Configure the graph settings for seaborn library.

        Returns:
        None
        """
        sns.set_theme(style="whitegrid", palette="pastel")
        sns.set(rc={"figure.figsize": (100, 100)})
        sns.set(font_scale=3)
