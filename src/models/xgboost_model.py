import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Optional
import joblib
import warnings

warnings.filterwarnings("ignore")


class XGBoostModel:

    def __init__(self, task: str = "classification", params: Optional[Dict] = None):
        self.task = task
        self.model = None
        if params is None:
            if task == "classification":
                self.params = {
                    "objective": "multi:softprob", "num_class": 3, "max_depth": 6, "learning_rate": 0.05,
                    "n_estimators": 1000, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0,
                    "reg_lambda": 0.1, "min_child_weight": 1, "gamma": 0.0, "max_delta_step": 1,
                    "scale_pos_weight": None, "random_state": 42, "n_jobs": -1, "tree_method": "hist",
                    "early_stopping_rounds": 50,
                }
            else:
                self.params = {
                    "objective": "reg:squarederror", "max_depth": 6, "learning_rate": 0.1, "n_estimators": 200,
                    "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
                    "min_child_weight": 3, "gamma": 0.1, "random_state": 42, "n_jobs": -1,
                    "tree_method": "hist", "early_stopping_rounds": 50,
                }
        else:
            self.params = params

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, early_stopping_rounds: int = 50) -> "XGBoostModel":
        if self.task == "classification":
            y_train_adjusted = y_train + 1
            if y_val is not None:
                y_val_adjusted = y_val + 1
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train_adjusted)
        else:
            y_train_adjusted = y_train
            y_val_adjusted = y_val if y_val is not None else None
            sample_weights = None
        if self.task == "classification":
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train_adjusted), (X_val, y_val_adjusted)]
            fit_params = {'eval_set': eval_set, 'verbose': False}
            if sample_weights is not None:
                fit_params['sample_weight'] = sample_weights
            self.model.fit(X_train, y_train_adjusted, **fit_params)
        else:
            if sample_weights is not None:
                self.model.fit(X_train, y_train_adjusted, sample_weight=sample_weights, verbose=False)
            else:
                self.model.fit(X_train, y_train_adjusted, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        predictions = self.model.predict(X)
        if self.task == "classification":
            predictions = predictions - 1
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
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False).head(top_n)
        return importance_df

    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        joblib.dump({"model": self.model, "task": self.task, "params": self.params}, filepath)

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.task = data["task"]
        self.params = data["params"]
