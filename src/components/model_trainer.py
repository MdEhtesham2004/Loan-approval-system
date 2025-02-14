from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
import os 
import sys
from src.execption import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model_score
from dataclasses import dataclass
import pandas as pd 
from sklearn.metrics import accuracy_score,f1_score



models_list=[]
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
roc_auc_list = []


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts","model.pkl")






class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):
        try:    
                logging.info("Model Training Initialized Successfully !!")
                logging.info("Split train and test data")
                classification_models = {
                        "Logistic Regression": LogisticRegression(),
                        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
                        "Support Vector Machine (SVM)": SVC(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "Random Forest": RandomForestClassifier(),
                        "Gradient Boosting": GradientBoostingClassifier(),
                        "AdaBoost": AdaBoostClassifier(),
                        "Naive Bayes": GaussianNB(),
                        "Ridge Classifier": RidgeClassifier(),
                        "Extra Trees Classifier": ExtraTreesClassifier(),
                        }
                
                # Print the dictionary of classification models
                for model_name, model in classification_models.items():
                        print(f"{model_name}: {model}")
                        logging.info(f"{model_name}: {model}")
                
               

                for model_name,model in classification_models.items():
                        X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
                        X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
                        y_train =  y_train.toarray() if hasattr(y_train, 'toarray') else y_train
                        # X_train_dense =  X_train
                        # X_train_dense =  X_test

                        model.fit(X_train_dense,y_train)

                        y_test_pred = model.predict(X_test_dense)
                        y_train_pred = model.predict(X_train_dense)

                        accuracy,precision,recall,f1,roc_auc = evaluate_model_score(y_test,y_test_pred)
                        accuracy2,precision2,recall2,f12,roc_auc2 = evaluate_model_score(y_train,y_train_pred)

                        global models_list,accuracy_list ,precision_list ,recall_list ,f1_list ,roc_auc_list 
                        models_list.append(model)
                        accuracy_list.append(accuracy)
                        precision_list.append(precision)
                        recall_list.append(recall)
                        f1_list.append(f1)
                        roc_auc_list.append(roc_auc)

                        
                        print(f"Model: {model_name}")
                        print(f"Accuracy: {accuracy}")
                        print(f"Precision: {precision}")
                        print(f"Recall: {recall}")
                        print(f"F1-Score: {f1}")
                        print(f"ROC-AUC: {roc_auc}")
                        print("\n")
                        
                        
                        logging.info(f"Model: {model_name}")
                        logging.info(f"Accuracy: {accuracy}")
                        logging.info(f"Precision: {precision}")
                        logging.info(f"Recall: {recall}")
                        logging.info(f"F1-Score: {f1}")
                        logging.info(f"ROC-AUC: {roc_auc}")
                        logging.info("\n")
                        
                results_df = pd.DataFrame({
                'Model': models_list,
                'Accuracy': accuracy_list,
                'Precision': precision_list,
                'Recall': recall_list,
                'F1-Score': f1_list,
                'ROC-AUC': roc_auc_list
                })

                # Display the results
                print(results_df)
                results_df.to_csv("artifacts/results.csv", index=False)
                logging.info(results_df)
                # results_df = pd.DataFrame(results_df)
                logging.info(f"results_df: we got is {results_df}")
                best_model = results_df.loc[results_df['F1-Score'].idxmax()]
                print(f"Best Model based on F1-Score:\n{best_model}")
                logging.info(f"Best Model based on F1-Score:\n{best_model}")
                print(f"Type Of Best Model{type(best_model)}")


                best_model_obj = best_model['Model']
                best_model_obj.fit(X_train,y_train)
                save_object(self.model_trainer_config.trained_model_path,best_model_obj)


                predictions = best_model_obj.predict(X_test)
                f1 =f1_score(y_test,predictions)
                acc = accuracy_score(y_test,predictions)


                return f1,acc  

        except Exception as e:
            raise CustomException(e,sys)