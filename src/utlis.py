#common funtionalites that can be used any where in the project

import os
import sys
import logging
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill

#this is a common function to save any object in the pickle file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")


    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)