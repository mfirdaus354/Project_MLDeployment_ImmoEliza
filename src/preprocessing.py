from src.config import DataProcessor
from typing import Optional, Literal, Union
import requests
import json
import numpy

class input_preprocess:
    
    @staticmethod
    def input_data(source: dict):
        if source is None or isinstance(source, dict) is False:
            raise ValueError("Please input your dataset again")
        
        data = source
        posted_data= requests.post(url="http://127.0.0.1:8000/items/", json=data).content
        retreived_data = requests.get(url="http://127.0.0.1:8000/items/").content
        content_array = numpy.frombuffer(retreived_data)
        return content_array
    
    @staticmethod
    def preprocess_new_data(source: dict):
        if source is None or isinstance(source, dict) is False:
            raise ValueError("Please input your dataset again")
        
        preprocessor = DataProcessor.PimpMyPipeline()




