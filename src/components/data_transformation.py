import numpy as np 
import pandas as pd 
import sys
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
import os 
from src.execption import CustomException
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):
        """  This method is used to get the data transformation object i.e preprocessor object which is used to transform the data"""
        try:
               # Wrap in a list to make it a row
            numeric_features = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']
      
            categorical_features =  ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession',
       'CITY', 'STATE']
            logging.info("Preprocessing for numerical and categorical features initiated")
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Numerical Features: {numeric_features}")
            logging.info(f"Categorical Features: {categorical_features}")

            logging.info("Column Transformer initiated")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numeric_features),
                    ('cat_pipeline',cat_pipeline,categorical_features)
                ]
            )

            logging.info("Preprocessor object created successfully ")

            return preprocessor

        except Exception as e :
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data Transformation initiated")
            logging.info("Read train and test data completed!")

            logging.info("Getting the preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()
            target_column_name="Risk_Flag"  
            cat_features = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession',
                'CITY', 'STATE']
            num_features = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS',]
      
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying the preprocessor object to train and test dataframe")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            # input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

           
            

            logging.info("Data Transformation Completed Successfully and saved the preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessor_obj

            )


            return (input_feature_train_arr,target_feature_train_df,input_feature_test_arr,target_feature_test_df,
                )

        except Exception as e:
            raise CustomException(e,sys)    
        

