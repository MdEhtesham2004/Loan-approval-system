import os 
import sys 
import pandas as pd
from src.execption import CustomException
from src.logger import logging 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer


data_transformation = DataTransformation()
model_trainer = ModelTrainer()





@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion method or component")
        try:
            df = pd.read_csv("datasets/loan_approval_dataset.csv").head(1000)
            df = df.drop(0)
            logging.info("Read the dataset successfully as pandas dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split inintiated")
            # X_train, X_test, y_train, y_test = train_test_split(X_transform,y,test_size=0.2,random_state=42)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion Completed Successfully")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
    X_train,y_train,X_test,y_test = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    f1,acc = model_trainer.initiate_model_trainer(X_train,y_train,X_test,y_test)
    logging.info(f"The result evalution matrics are f1: {f1}, acc: {acc}")
    print(f"The result evalution matrics are f1: {f1}, acc: {acc}")

