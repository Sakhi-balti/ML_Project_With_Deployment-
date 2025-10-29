import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle
import numpy as np

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
port = int(os.getenv("port"))
database = os.getenv("database")
password = os.getenv("password")


def read_sql_data():
    try:
        engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
        df = pd.read_sql("SELECT * FROM studentdata", engine)
        print(df.head())
        return df
    except Exception as ex:
        print("Error:", ex)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)   

    except Exception as e:
        raise CustomException(e, sys)        


def evaluate_model(X_train, y_train, X_test, y_test, models, param_grids):
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = param_grids.get(model_name, {})
            logging.info(f"Training {model_name} with GridSearchCV...")

            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring="r2")
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            report[model_name] = {
                "best_params": gs.best_params_,
                "train_score": train_r2,
                "test_score": test_r2
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)