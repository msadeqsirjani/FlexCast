import numpy as np
import pandas as pd
from typing import Dict, Optional
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel


class CascadeClassifier:

    def __init__(self, stage1_model: str = "xgboost", stage2_model: str = "lightgbm", stage1_params: Optional[Dict] = None, stage2_params: Optional[Dict] = None):
        self.stage1_model_type = stage1_model
        self.stage2_model_type = stage2_model
        self.stage1 = None
        self.stage2 = None
        self.stage1_params = stage1_params
        self.stage2_params = stage2_params

    def _get_model(self, model_type: str, params: Optional[Dict] = None):
        if params is None:
            if model_type == "xgboost":
                params = {
                    "objective": "binary:logistic", "max_depth": 8, "learning_rate": 0.05, "n_estimators": 1000,
                    "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 0.1,
                    "min_child_weight": 1, "gamma": 0.0, "scale_pos_weight": None, "random_state": 42,
                    "n_jobs": -1, "tree_method": "hist", "early_stopping_rounds": 50,
                }
            elif model_type == "lightgbm":
                params = {
                    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt", "num_leaves": 63,
                    "max_depth": 8, "learning_rate": 0.05, "n_estimators": 1000, "feature_fraction": 0.8,
                    "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 20, "min_child_weight": 0.001,
                    "reg_alpha": 0.0, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1, "is_unbalance": True,
                }
            elif model_type == "catboost":
                params = {
                    "iterations": 1000, "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3, "min_data_in_leaf": 1,
                    "border_count": 254, "random_strength": 1, "bootstrap_type": "Bayesian", "bagging_temperature": 1,
                    "random_seed": 42, "verbose": False, "thread_count": -1, "task_type": "CPU", "auto_class_weights": "Balanced",
                }

        if model_type == "xgboost":
            model = XGBoostModel(task="classification", params=params)
            model.params["objective"] = "binary:logistic"
        elif model_type == "lightgbm":
            model = LightGBMModel(task="classification", params=params)
            model.params["objective"] = "binary"
        elif model_type == "catboost":
            model = CatBoostModel(task="classification", params=params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        y_train_stage1 = (y_train != 0).astype(int)
        if y_val is not None:
            y_val_stage1 = (y_val != 0).astype(int)
        else:
            y_val_stage1 = None

        self.stage1 = self._get_model(self.stage1_model_type, self.stage1_params)
        self._train_binary_model(self.stage1, X_train, y_train_stage1, X_val, y_val_stage1, "Stage 1")

        dr_mask_train = y_train != 0
        X_train_stage2 = X_train[dr_mask_train]
        y_train_stage2 = y_train[dr_mask_train]
        y_train_stage2_binary = (y_train_stage2 == 1).astype(int)

        if y_val is not None:
            dr_mask_val = y_val != 0
            X_val_stage2 = X_val[dr_mask_val]
            y_val_stage2 = y_val[dr_mask_val]
            y_val_stage2_binary = (y_val_stage2 == 1).astype(int)
        else:
            X_val_stage2 = None
            y_val_stage2_binary = None

        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights_stage2 = np.ones(len(y_train_stage2_binary))
        sample_weights_stage2[y_train_stage2_binary == 1] = 100.0

        self.stage2 = self._get_model(self.stage2_model_type, self.stage2_params)
        self._train_binary_model(self.stage2, X_train_stage2, y_train_stage2_binary, X_val_stage2, y_val_stage2_binary, "Stage 2", sample_weights=sample_weights_stage2)

    def _train_binary_model(self, model, X_train, y_train, X_val, y_val, stage_name, sample_weights=None):
        if model.__class__.__name__ == "XGBoostModel":
            from sklearn.utils.class_weight import compute_sample_weight
            if sample_weights is None:
                sample_weights = compute_sample_weight('balanced', y_train)
            import xgboost as xgb
            model.model = xgb.XGBClassifier(**model.params)
            if X_val is not None and y_val is not None:
                model.model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
        elif model.__class__.__name__ == "LightGBMModel":
            import lightgbm as lgb
            model.model = lgb.LGBMClassifier(**model.params)
            if X_val is not None and y_val is not None:
                model.model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            else:
                model.model.fit(X_train, y_train, sample_weight=sample_weights)
        elif model.__class__.__name__ == "CatBoostModel":
            from catboost import CatBoostClassifier
            model.model = CatBoostClassifier(**model.params)
            if X_val is not None and y_val is not None:
                model.model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
            else:
                model.model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    def _predict_binary(self, model, X):
        return model.model.predict(X).astype(int)

    def _predict_proba_binary(self, model, X):
        proba = model.model.predict_proba(X)
        return proba[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        stage1_pred = self._predict_binary(self.stage1, X)
        final_pred = np.zeros(len(X), dtype=int)
        dr_mask = stage1_pred == 1
        if np.sum(dr_mask) > 0:
            X_dr = X[dr_mask]
            stage2_pred = self._predict_binary(self.stage2, X_dr)
            stage2_pred_labels = np.where(stage2_pred == 0, -1, 1)
            final_pred[dr_mask] = stage2_pred_labels
        return final_pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_dr_event = self._predict_proba_binary(self.stage1, X)
        p_no_dr = 1 - p_dr_event
        p_increase_given_dr = self._predict_proba_binary(self.stage2, X)
        p_decrease_given_dr = 1 - p_increase_given_dr
        p_class_minus1 = p_dr_event * p_decrease_given_dr
        p_class_0 = p_no_dr
        p_class_plus1 = p_dr_event * p_increase_given_dr
        proba = np.column_stack([p_class_minus1, p_class_0, p_class_plus1])
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    def save_model(self, filepath: str):
        import joblib
        joblib.dump({'stage1': self.stage1, 'stage2': self.stage2, 'stage1_type': self.stage1_model_type, 'stage2_type': self.stage2_model_type}, filepath)


if __name__ == "__main__":
    print("Cascade Classifier for Extreme Class Imbalance")
    print("Two-stage hierarchical classification")
