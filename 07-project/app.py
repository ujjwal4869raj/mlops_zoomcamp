import pickle

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# specify which model to load
model_name = "energy-efficiency-model"
model_version = 1

# load the model from the registry
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
print(f'model loaded: {model}')

# get run_id
for mv in client.search_model_versions("name='energy-efficiency-model'"):
    mv = dict(mv)
    if mv['version'] == str(model_version):
        RUN_ID = mv['run_id']

dv_path = "./models/dv.pkl"

with open(dv_path, 'rb') as f_out:
    dv = pickle.load(f_out)

class InputData(BaseModel):
    relative_compactnes: float
    surface_area: float
    wall_area: float
    roof_area: float
    overall_height: float
    orientation: float
    glazing_area: float
    glazing_area_distribution: float

def preprocess(data):
    """Preprocessing of the data"""
    # turn json input to dataframe
    data_dict = data.dict()
    data_df = pd.DataFrame([data_dict])

    # define numerical and categorical features
    categorical = ["orientation", "glazing_area_distribution"]
    
    numerical = ["relative_compactnes", "surface_area", "wall_area", "roof_area", "overall_height", "glazing_area"]
    
    train_dicts = data_df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(train_dicts)

    return X

def predict(X):
    """make predictions"""
    pred = model.predict(X)
    print('prediction', pred[0])
    return float(pred[0])

app = FastAPI()

@app.post('/predict')
def predict_endpoint(input_data: InputData):
    """request input, preprocess it and make prediction"""
    features = preprocess(input_data)
    pred = predict(features)
    result = {'heat_load': pred, 'model_version': RUN_ID}

    return result

if __name__ == '__main__()':
    uvicorn.run(app, host='127.0.0.1', port=8000)