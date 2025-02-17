from src.logger import logging 
from src.execption import CustomException
import pandas as pd 
import json 

# models = pd.read_csv("artifacts/results.csv")
'''
  Model  Accuracy  Precision    Recall  F1-Score   ROC-AUC
0      LogisticRegression()     0.700   0.235294  0.190476  0.210526  0.512960
1    KNeighborsClassifier()     0.775   0.285714  0.047619  0.081633  0.507987
2                     SVC()     0.785   0.000000  0.000000  0.000000  0.496835
3  DecisionTreeClassifier()     0.735   0.296296  0.190476  0.231884  0.535112
4  RandomForestClassifier()     0.790   0.000000  0.000000  0.000000  0.500000
'''

# models_dict = {}
# for i in range(0, 5):
#     model = models.iloc[i]
#     model_dict = {
#         'Model': model['Model'],
#         'Accuracy': model['Accuracy'],
#         'Precision': model['Precision'],
#         'Recall': model['Recall'],
#         'F1-Score': model['F1-Score'],
#         'ROC-AUC': model['ROC-AUC']
#     }
#     models_dict.update({i: model_dict})


# print(models_dict)
# with open("artifacts/models.json", "w") as json_file:
#     json.dump(models_dict, json_file, indent=4)

# # Print a message to indicate that the dictionary has been written to the JSON file
# print("Dictionary has been written to artifacts/models.json")

# Print the models_dict to verify the 


class TrainPipeline:
    def __init__(self):
        self.models = pd.read_csv("artifacts/results.csv")
        self.extract_models()

    def extract_models(self):
        self.models_dict = {}
        for i in range(0, 5):
            self.model = self.models.iloc[i]
            self.model_dict = {
                'Model':self.model['Model'],
                'Accuracy': self.model['Accuracy'],
                'Precision': self.model['Precision'],
                'Recall': self.model['Recall'],
                'F1-Score': self.model['F1-Score'],
                'ROC-AUC': self.model['ROC-AUC']
            }
            self.models_dict.update({i: self.model_dict})


        print(self.models_dict)
        with open("artifacts/models.json", "w") as json_file:
            json.dump(self.models_dict, json_file, indent=4)

        # Print a message to indicate that the dictionary has been written to the JSON file
        print("Dictionary has been written to artifacts/models.json")

    def __str__(self):
        pass 
