import xgboost as xgb
import os
import pickle
import numpy as np

def CURRENT_DIR():
    cwd = os.getcwd()
    return os.chdir(cwd[:(cwd.index("Eliza")+5)])

CURRENT_DIR()

class prediction:
    
    @staticmethod
    def initiate_prediction(data:np.ndarray, model_path: str = fr"{os.getcwd()}\models\xgb_reg_model.pkl"):
        if data is None:
            raise ValueError("data is unregognized")
        
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        predictions = loaded_model.predict(data)
        prediction= predictions[0]*-1

        return prediction



