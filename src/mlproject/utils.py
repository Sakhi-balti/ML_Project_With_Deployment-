import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
from sqlalchemy import create_engine

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