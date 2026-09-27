import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass
from src.utils import evaluate_models

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):
        logging.info("Model training initiated")

        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso Regression": Lasso(),
                "Ridge Regression": Ridge(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor()

            }

            params = {
                "Decision Tree" : {
                    'criterion' : ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    'max_depth' : [3, 5, 7, 10]
                },
                "Random Forest" : {
                    'n_estimators' : [8,16,32,64,128,256]
                },
                "Gradient Boosting" : {
                    'n_estimators' : [8,16,32,64,128,256],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample' : [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                },
                "Linear Regression": {},
                "Lasso Regression": {}, 
                "Ridge Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators' : [8,16,32,64,128,256]
                },
                "CatBoosting Regressor": {
                    'depth' : [6,8,10],
                    'learning_rate' : [0.01, 0.05, 0.1],
                    'iterations' : [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators' : [8,16,32,64,128,256]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                    
                }   
            }
        

        
            model_report : dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with score greater than 0.6", sys)

            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            best_predictions = best_model.predict(X_test)
            r2_square = r2_score(y_test, best_predictions)



            
        except Exception as e:
            raise CustomException(e, sys)