import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=2)

            # Fit the grid search to the training data
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_) #Set the best parameters to the model
            model.fit(X_train, y_train) #Fit the model to the training data

            y_train_pred = model.predict(X_train) #Predict Training data
            y_test_pred = model.predict(X_test) #Predict Testing data

            train_model_score = r2_score(y_train, y_train_pred) #Train model score
            test_model_score = r2_score(y_test, y_test_pred) #Test model score

            report[list(models.keys())[i]] = test_model_score #Add model score to report
            print(f"Model: {list(models.keys())[i]}, Test Score: {test_model_score}")

        return report
    except Exception as e:
        raise CustomException(e, sys)   
