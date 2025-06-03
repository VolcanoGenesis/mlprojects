import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomerException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],# all rows except last one
                train_array[:,-1],# all rows and only the last column
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
    "Decision Tree": {
        'criterion':['squared_error']  # Reduced from 4 to 1
    },
    "Random Forest":{
        'n_estimators': [16,32]  # Reduced from 6 to 2
    },
    "Gradient Boosting":{
        'learning_rate':[.1,.01],  # Reduced from 4 to 2
        'subsample':[0.8],         # Reduced from 6 to 1
        'n_estimators': [16,32]    # Reduced from 6 to 2
    },
    "Linear Regression":{},
    "XGBRegressor":{
        'learning_rate':[.1],
        'n_estimators': [16]
    },
    "CatBoosting Regressor":{
        'depth': [6],
        'learning_rate': [0.1],
        'iterations': [30]
    },
    "AdaBoost Regressor":{
        'learning_rate':[.1],
        'n_estimators': [16]
    }
}

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]# list(model_report.values()) gives list of values
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomerException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
                # logging.info(f"Best model found on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            raise CustomerException(e,sys) 