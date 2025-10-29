
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_model

import os, sys
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, train_dir=None),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            param_grids = {
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                },
                "Decision Tree": {
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5, 10],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                },
                "Linear Regression": {
                    "fit_intercept": [True, False],
                },
                "XGBRegressor": {
                    "n_estimators": [200, 500],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                },
                "CatBoosting Regressor": {
                    "iterations": [500],
                    "learning_rate": [0.05, 0.1],
                    "depth": [6, 8],
                },
                "AdaBoost Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                },
            }

            # ✅ Evaluate all models
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models, param_grids)

            # ✅ Find the best model based on test R²
            best_model_name = max(model_report, key=lambda x: model_report[x]["test_score"])
            best_model_data = model_report[best_model_name]
            best_model_score = best_model_data["test_score"]

            logging.info(f"Best model: {best_model_name} with R² = {best_model_score:.4f}")

            # ✅ Get the actual trained model with best params
            best_model = models[best_model_name]
            best_params = best_model_data["best_params"]
            best_model.set_params(**best_params)
            best_model.fit(X_train, y_train)

            # ✅ Save the model
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model,
            )

            return {
                "best_model_name": best_model_name,
                "best_model_score": best_model_score,
                "best_params": best_params,
            }

        except Exception as e:
            raise CustomException(e, sys)
  