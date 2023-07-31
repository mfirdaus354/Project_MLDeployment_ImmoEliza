import os

import numpy as np
import pandas as pd
import pickle
import json
import xgboost as xgb


def CURRENT_DIR():
    cwd = os.getcwd()
    return os.chdir(cwd[: (cwd.index("Eliza") + 5)])


CURRENT_DIR()

print(os.getcwd())

################################################################

from preprocessing import input_preprocess
from prediction import prediction

plot_area = input("Please input the plot area of your property in decimals  =  ")
habitable_surface = input(
    "Please input the total area of habitable/living space of your property in decimals  =  "
)
land_surface = input(
    "Please input the total area of non-buildable land area of your property in decimals  =  "
)
room_count = input(
    "Please input the total number of rooms in the property without decimals  =  "
)
bedroom_count = input(
    "Please input the total number of bedroom(s) in the property without decimals  =  "
)


data = {
    "plot_area": plot_area,
    "habitable_surface": habitable_surface,
    "land_surface": land_surface,
    "room_count": room_count,
    "bedroom_count": bedroom_count,
}

# post, get, preprocessed data
posted_data = input_preprocess.input_data(source=data)
get_data = input_preprocess.get_data()
preprocessed_data = input_preprocess.preprocess_new_data(get_data)

# generate prediction
model_file_path = rf"{os.getcwd()}\models\xgb_reg_model.pkl"
predicted_price = prediction.initiate_prediction(
    data=preprocessed_data, model_path=model_file_path
)


print(
    f"Our Machine Learning Model has predicted that the possible price of your property is {predicted_price}"
)

# Load the model using pickle
with open(model_file_path, "rb") as f:
    loaded_model = pickle.load(f)

predictions = loaded_model.predict(preprocessed_data)
