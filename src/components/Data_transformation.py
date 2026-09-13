import sys
import os
from src.exception import CustomException
from src.logger import logging
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import save_object

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))

@dataclass
class DataTransformtionConfig:
    preprocessor_obj_file_path = os.path.join(PROJECT_ROOT,"artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__ (self):
        self.data_transformation_config = DataTransformtionConfig()

    def get_data_transformer_object(self):
        """ This function is responsible for Data Transformation """
        try:
            numerical_col = ["writing_score","reading_score"]
            categorical_col = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median")),
                    ("Scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("One_hot_encoder",OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ("Scaler",StandardScaler())

                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("Numerial Pipeline", num_pipeline, numerical_col),
                    ("Categorical Pipeline", cat_pipeline, categorical_col)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing obeject on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)


