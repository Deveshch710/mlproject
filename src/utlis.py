#common funtionalites that can be used any where in the project

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
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


#this is a common function to load any object from the pickle file
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from {file_path}")
        return obj

    except Exception as e:
        logging.error(f"Error loading object: {e}")
        raise CustomException(e, sys)


#model evaluation function
def evaluate_model(X_train, y_train, X_test, y_test, models,params):
    try:
        reprot={}

        for i in range (len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
           
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            reprot[list(models.keys())[i]] = train_model_score

        return reprot
    
    
    except Exception as e:  
        logging.error(f"Error evaluating models: {e}")
        raise CustomException(e, sys)