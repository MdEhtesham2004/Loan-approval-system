import os 
from src.execption import CustomException
import dill 
from src.logger import logging 
import sys 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file:
            dill.dump(obj,file)
        logging.info(f"Object saved successfully at {file_path}")
        
    except Exception as e :
        raise CustomException(e,sys)
    


def evaluate_model_score(y_test,y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return accuracy,precision,recall,f1,roc_auc
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
    