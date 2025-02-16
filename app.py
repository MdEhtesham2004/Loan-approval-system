from flask import Flask,render_template,request,jsonify
import pandas as pd 
import pickle
from src.components.data_transformation import DataTransformation
from pycaret.classification import *
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.utils import load_object
import json
model_trainer = ModelTrainer()

app = Flask(__name__)

@app.route('/')
def base():
    return render_template('main.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

@app.route('/evaluate_model')
def evaluate_model():
    with open('artifacts/models.json', 'r') as json_file:
            models_dict = json.load(json_file)

    return render_template('model_evaluation.html',models=models_dict)

@app.route('/submit', methods=['GET','POST'])
def submit():
    if request.method == 'GET':
        return render_template('index.html')
    else:
            
        form_data = {
            "income": float(request.form.get("income")),
            "age": int(request.form.get("age")),
            "experience": int(request.form.get("experience")),
            "marital_status": request.form.get("marital_status"),
            "house_ownership": request.form.get("house_ownership"),
            "car_ownership": request.form.get("car_ownership"),
            "profession": request.form.get("profession"),
            "city": request.form.get("city"),
            "state": request.form.get("state"),
            "current_job_yrs": float(request.form.get("current_job_yrs")),
            "current_house_yrs": float(request.form.get("current_house_yrs")),
        }
        data = CustomData(
                    Income=form_data['income'],
                    Age=form_data['age'],
                    Experience=form_data['experience'],
                    Married_Single=form_data['marital_status'],
                    House_Ownership=form_data['house_ownership'],
                    Car_Ownership=form_data['car_ownership'],
                    Profession=form_data['profession'],
                    CITY=form_data['city'],
                    STATE=form_data['state'],
                    CURRENT_JOB_YRS=form_data['current_job_yrs'],
                    CURRENT_HOUSE_YRS=form_data['current_house_yrs']
                )
        model_selection = int(request.form.get("model_selection", 0)) 

        loaded_best_pipeline=load_model("notebooks/pycaret-automation/assets/my_first_pipeline")

        columns = ['Income', 'Age', 'Experience', 'Married/Single', 'House_Ownership',
        'Car_Ownership', 'Profession', 'CITY', 'STATE', 'CURRENT_JOB_YRS',
        'CURRENT_HOUSE_YRS']
        



        


        if model_selection == 0:
            # this is traditional model 
            pred_df = data.get_data_as_data_frame()
            pred_df = pred_df.applymap(lambda x: x[0] if isinstance(x, tuple) else x)
            pred_df["CURRENT_JOB_YRS"] = pred_df["CURRENT_JOB_YRS"].astype(float)
            pred_df["CURRENT_HOUSE_YRS"] = pred_df["CURRENT_HOUSE_YRS"].astype(float)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('index.html',result=results[0])
        
           
        else:
            # this is pycaret model 
            preprocessed_df = loaded_best_pipeline.transform(pd.DataFrame([form_data],columns=columns))
            print(preprocessed_df)
            model=load_model("notebooks/pycaret-automation/assets/lightgbm_model")
            result = model.predict(pd.DataFrame(preprocessed_df))



    if result == 1:
        result=f"Eligible For Loan"
    else:
        result = f"Not Eligible"
    # Print data to console (for debugging)
    print("Received Form Data:", form_data)
    return render_template('index.html',result=result)
    # return f"Form submitted successfully! Received data: {form_data}, result is :{result}"

@app.route('/evaluate_results', methods=['GET', 'POST'])
def evaluate_results():
    # model_ids = ['logistic_regression', 'knn', 'svc', 'decision_tree', 'random_forest', 'compare_models']
    # selected_models = {model_id: request.form.get(model_id) for model_id in model_ids if request.form.get(model_id)}
    if request.method == 'POST':

            # Read the JSON file
            with open('artifacts/models.json', 'r') as json_file:
                models_dict = json.load(json_file)
            
            # Get the respective dictionaries based on selected_models
            # results = {model_id: models_dict.get(model_id, "Model not found") for model_id in selected_models}
            logress =  request.form.get('logistic_regression')
            if logress:
                results = models_dict['0']
                print(results)

            # Return a response with the results
            return render_template('evaluation_results.html', results=results)
   


if __name__ == "__main__":
    app.run()