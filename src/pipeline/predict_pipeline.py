import sys 
from src.execption import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd 

class PredictPipeline:
    def __init__(self):
        pass 

    def predict(self,features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e: 
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,          
                    Income:int, Age:int, Experience:int, Married_Single:str, House_Ownership:str,
                    Car_Ownership:str, Profession:str, CITY:str , STATE:str, CURRENT_JOB_YRS:int,
                    CURRENT_HOUSE_YRS:int ):
        self.Income = Income
        self.Age = Age
        self.Experience = Experience
        self.Married_Single = Married_Single,
        self.House_Ownership = House_Ownership,
        self.Car_Ownership = Car_Ownership,
        self.Profession = Profession,
        self.City = CITY,
        self.State = STATE,
        self.Current_job_yrs = CURRENT_JOB_YRS, 
        self.Current_house_yrs = CURRENT_HOUSE_YRS

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
            'Income':[self.Income],
            'Age':[self.Age], 
            'Experience':[self.Experience],
                'Married/Single':[self.Married_Single],
                'House_Ownership':[self.House_Ownership],
                'Car_Ownership':[self.Car_Ownership],
                'Profession':[self.Profession],
                    'CITY':[self.City],
                    'STATE':[self.State],
                        'CURRENT_JOB_YRS':[self.Current_job_yrs],
                        'CURRENT_HOUSE_YRS':[self.Current_house_yrs]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)



        