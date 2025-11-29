import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from typing import Dict, Optional, List
import joblib
import warnings

warnings.filterwarnings("ignore")


class CatBoostModel:

    def __init__(self, task: str = "classification", params: Optional[Dict] = None):
        self.task = task
        self.model = None
        if params is None:
            if task == "classification":
                self.params = {
                    "iterations": 1000, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3,
                    "min_data_in_leaf": 1, "border_count": 254, "random_strength": 1,
                    "bootstrap_type": "Bayesian", "bagging_temperature": 1, "random_seed": 42,
                    "verbose": False, "thread_count": -1, "task_type": "CPU",
                    "auto_class_weights": "Balanced", "leaf_estimation_iterations": 10,
                }
            else:
                self.params = {
                    "iterations": 200, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3,
                    "border_count": 128, "random_strength": 1, "bagging_temperature": 1,
                    "random_seed": 42, "verbose": False, "thread_count": -1, "task_type": "CPU",
                }
        else:
            self.params = params

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, cat_features: Optional[List[str]] = None, early_stopping_rounds: int = 50, sample_weight: Optional[np.ndarray] = None) -> "CatBoostModel":
        if self.task == "classification":
            y_train_adjusted = y_train + 1
            if y_val is not None:
                y_val_adjusted = y_val + 1
        else:
            y_train_adjusted = y_train
            y_val_adjusted = y_val if y_val is not None else None
        if cat_features is None:
            cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.task == "classification":
            self.model = CatBoostClassifier(**self.params)
        else:
            self.model = CatBoostRegressor(**self.params)
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train_adjusted, eval_set=(X_val, y_val_adjusted), cat_features=cat_features, early_stopping_rounds=early_stopping_rounds, verbose=False, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train_adjusted, cat_features=cat_features, verbose=False, sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        predictions = self.model.predict(X)
        if self.task == "classification":
            predictions = predictions.astype(int) - 1
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        importance = self.model.get_feature_importance()
        importance_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False).head(top_n)
        return importance_df

    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        self.model.save_model(filepath)
        metadata_path = filepath + ".meta"
        joblib.dump({"task": self.task, "params": self.params}, metadata_path)

    def load_model(self, filepath: str):
        metadata_path = filepath + ".meta"
        metadata = joblib.load(metadata_path)
        self.task = metadata["task"]
        self.params = metadata["params"]
        if self.task == "classification":
            self.model = CatBoostClassifier()
        else:
            self.model = CatBoostRegressor()
        self.model.load_model(filepath)
