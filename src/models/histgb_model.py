import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from typing import Dict, Optional
import joblib
import warnings

warnings.filterwarnings("ignore")


class HistGradientBoostingModel:

    def __init__(self, task: str = "classification", params: Optional[Dict] = None):
        if task not in ["classification", "regression"]:
            raise ValueError(f"Task must be 'classification' or 'regression', got '{task}'")
        self.task = task
        self.model = None
        if params is None:
            base_params = {
                "max_iter": 300, "learning_rate": 0.01, "max_depth": 3, "max_leaf_nodes": 15,
                "min_samples_leaf": 50, "l2_regularization": 1.0, "max_bins": 128, "random_state": 42,
                "verbose": 0, "early_stopping": True, "scoring": None, "n_iter_no_change": 10,
                "validation_fraction": 0.1,
            }
            if task == "classification":
                self.params = {**base_params, "loss": "log_loss"}
            else:
                self.params = {**base_params, "loss": "squared_error"}
        else:
            self.params = params

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, early_stopping_rounds: int = 10, sample_weight: Optional[np.ndarray] = None) -> "HistGradientBoostingModel":
        params_copy = self.params.copy()
        params_copy["n_iter_no_change"] = early_stopping_rounds
        if self.task == "classification":
            y_train_adjusted = y_train + 1
            if y_val is not None:
                y_val_adjusted = y_val + 1
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train_adjusted)
            self.model = HistGradientBoostingClassifier(**params_copy)
        else:
            y_train_adjusted = y_train
            y_val_adjusted = y_val if y_val is not None else None
            sample_weights = sample_weight
            self.model = HistGradientBoostingRegressor(**params_copy)
        if sample_weights is not None:
            self.model.fit(X_train, y_train_adjusted, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_train_adjusted)
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
            raise ValueError("predict_proba only available for classification task")
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        joblib.dump({"model": self.model, "task": self.task, "params": self.params}, filepath)

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.task = data["task"]
        self.params = data["params"]
