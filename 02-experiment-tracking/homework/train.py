import os
import pickle
import click

import mlflow

mlflow.autolog()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)


def run_train(data_path: str):

    with mlflow.start_run():

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        params = {"max_depth": 10, "random_state": 0}
        mlflow.log_params(params)

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        #mlflow.log_metric("accuracy", accuracy_score(y_val, y_pred))

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(rmse)
        
        mlflow.sklearn.log_model(rf, artifact_path="models")
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")


if __name__ == '__main__':
    run_train()