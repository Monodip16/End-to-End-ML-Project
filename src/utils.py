import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]

            model.fit(X_train, y_train) #Train model

            y_train_pred = model.predict(X_train) #Predict Training data
            y_test_pred = model.predict(X_test) #Predict Testing data

            train_model_score = r2_score(y_train, y_train_pred) #Train model score
            test_model_score = r2_score(y_test, y_test_pred) #Test model score

            report[list(models.keys())[i]] = test_model_score #Add model score to report
            print(f"Model: {list(models.keys())[i]}, Test Score: {test_model_score}")

        return report
    except Exception as e:
        raise CustomException(e, sys)   
