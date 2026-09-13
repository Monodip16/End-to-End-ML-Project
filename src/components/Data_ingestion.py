import sys
import os
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.Data_transformation import DataTransformtionConfig
from src.components.Data_transformation import DataTransformation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "train.csv")
    test_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "test.csv")
    raw_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or components")

        try:
            data_path = os.path.join(PROJECT_ROOT, "data", "stud.csv")
            df = pd.read_csv(data_path)
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")  

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)   