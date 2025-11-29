import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Optional
import joblib
import warnings

warnings.filterwarnings("ignore")


class LightGBMModel:

    def __init__(self, task: str = "classification", params: Optional[Dict] = None):
        self.task = task
        self.model = None
        if params is None:
            if task == "classification":
                self.params = {
                    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "boosting_type": "gbdt",
                    "num_leaves": 63, "max_depth": 7, "learning_rate": 0.05, "n_estimators": 1000,
                    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 20,
                    "min_child_weight": 0.001, "reg_alpha": 0.0, "reg_lambda": 0.1, "max_bin": 255,
                    "random_state": 42, "n_jobs": -1, "verbose": -1, "class_weight": "balanced",
                    "min_split_gain": 0.0, "subsample_for_bin": 200000,
                }
            else:
                self.params = {
                    "objective": "regression", "metric": "rmse", "boosting_type": "gbdt", "num_leaves": 31,
                    "learning_rate": 0.05, "n_estimators": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                    "bagging_freq": 5, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 1.0,
                    "random_state": 42, "n_jobs": -1, "verbose": -1,
                }
        else:
            self.params = params

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, early_stopping_rounds: int = 50, sample_weight: Optional[np.ndarray] = None) -> "LightGBMModel":
        if self.task == "classification":
            y_train_adjusted = y_train + 1
            if y_val is not None:
                y_val_adjusted = y_val + 1
        else:
            y_train_adjusted = y_train
            y_val_adjusted = y_val if y_val is not None else None
        self.sample_weight = sample_weight
        if self.task == "classification":
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
        callbacks = []
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            self.model.fit(X_train, y_train_adjusted, eval_set=[(X_val, y_val_adjusted)], callbacks=callbacks, sample_weight=self.sample_weight)
        else:
            self.model.fit(X_train, y_train_adjusted, sample_weight=self.sample_weight)
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
