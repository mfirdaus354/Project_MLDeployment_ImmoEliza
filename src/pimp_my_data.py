import os
import pandas as pd
from src.config import Config


class PimpMyData:
    """
    PimpMyData is a class with several methods that are being used throughout PROJECT_IMMO_REGRESSION
    a static method _get_file_path is defined to help instantiate a file path to open a certain file in a directory
    This class works upon its instantiation into a variable
    It has several methods
    PimpMyData.dataframe_generator --> to generate a dataframe from csv file
    PimpMyData.present_data --> to summarize a certain column of a dataframe
    PimpMyData.
    """

    def __init__(self):
        self.dtypes = {
            "immocode": "int32",
            "price": "int32",
            "price_per_sqmeter": "float",
            "habitable_surface": "float",
            "plot_area": "float",
            "land_surface": "float",
            "bedroom_count": "int32",
            "room_count": "int32",
        }
        self.option_continuous = [
            "plot_area",
            "habitable_surface",
            "bedroom_count",
            "land_surface",
            "room_count",
        ]
        self.targets = ["price"]
        self.option_categorical = [
            "type",
            "subtype",
            "epc_score",
            "immo_status",
            "postalcode",
            "municipality",
            "district",
            "province",
            "region",
        ]
        self.main_df = self.dataframe_generator("data", "data_forsale_new.csv")

    @staticmethod
    def _get_file_path(folder_one, name, folder2=None, folder3=None):
        current_dir = os.getcwd()
        if folder2 is None and folder3 is None:
            parent_dir = os.path.dirname(current_dir)
            file_dir = os.path.abspath(os.path.join(parent_dir, folder_one, name))

        return file_dir

    def dataframe_generator(self, folder_one, name):
        """ "
        PimpMyData.dataframe_generator --> to generate a dataframe from csv file.
        PARAMETERS
        folder_one first descendent directory below the parent directory
        folder2 = 2nd descendent directory below the parent directory

        """
        data_file_path = f"{Config.CURRENT_DIR()}/data/data_forsale_new.csv"
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError(
                f"File '{data_file_path}' not found. Please check the file path."
            )
        dataframe_main = pd.read_csv(data_file_path)
        return dataframe_main

    def present_data(self, column=None, choice=None, df=None):
        if column is not None and choice is not None and df is not None:
            if choice == "summarize":
                summary = df[column].describe()
                return summary

    def give_continuous(self, df, target):
        if target in self.targets:
            df_price_continuous = df[
                [
                    target,
                    "price_per_sqmeter",
                    "plot_area",
                    "habitable_surface",
                    "bedroom_count",
                    "land_surface",
                    "room_count",
                ]
            ]
            return df_price_continuous
        else:
            raise ValueError(
                f"Invalid target '{target}'. The target must be one of: {', '.join(self.targets)}"
            )

    def give_categorical(self, df=None, option=None, target=None):
        if df is not None and option is not None and target is not None:
            if target in self.targets and len(df[option]) <= 5:
                dummies = pd.get_dummies(df[option])
                output = pd.concat([df[target], dummies], axis=1)
                return output
            else:
                return (
                    "Please check your parameters. Maybe your variable is too complex."
                )

    def clean_dtypes_numerics(self, df=None):
        if df is not None:
            return df.astype(dtype=self.dtypes)

    def give_subset(self, df, subset):
        new_subset = df[subset]
        return new_subset

    def give_nan_columns(self, df):
        nan_columns = df.columns[df.isna().any()].tolist()
        return [f"Column '{col}' has missing values." for col in nan_columns]

    def fill_missing_values(self, df, columns_to_fill=None):
        if columns_to_fill is None:
            columns_to_fill = ["habitable_surface", "land_surface", "plot_area"]
            for col in columns_to_fill:
                df[col] = df[col].fillna(0)
        else:
            for col in columns_to_fill:
                if col in self.option_continuous:
                    df[col] = df[col].fillna(0)
                elif col in self.option_categorical:
                    df[col] = df[col].fillna("UNKNOWN")

        return df
