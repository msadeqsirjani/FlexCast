"""
Cascade/Hierarchical Classifier for Extreme Class Imbalance
Addresses Class 1 (minority) performance issues through staged classification

Strategy:
- Stage 1: Binary classification - Class 0 (No DR) vs Class -1/+1 (DR Event)
- Stage 2: Binary classification - Class -1 (Decrease) vs Class +1 (Increase)

Benefits:
- Stage 1 has better balance (95% vs 5%)
- Stage 2 focuses only on DR events without Class 0 dominance
- Each stage can use specialized parameters and features
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel

logger = logging.getLogger(__name__)


class CascadeClassifier:
    """
    Two-stage cascade classifier for extreme imbalanced multi-class problems
    """

    def __init__(
        self,
        stage1_model: str = "xgboost",
        stage2_model: str = "lightgbm",
        stage1_params: Optional[Dict] = None,
        stage2_params: Optional[Dict] = None,
    ):
        """
        Initialize cascade classifier

        Args:
            stage1_model: Model type for stage 1 ('xgboost', 'lightgbm', 'catboost')
            stage2_model: Model type for stage 2
            stage1_params: Custom parameters for stage 1
            stage2_params: Custom parameters for stage 2
        """
        self.stage1_model_type = stage1_model
        self.stage2_model_type = stage2_model
        self.stage1 = None
        self.stage2 = None
        self.stage1_params = stage1_params
        self.stage2_params = stage2_params

    def _get_model(self, model_type: str, params: Optional[Dict] = None):
        """Get model instance by type"""
        if params is None:
            # Use aggressive parameters optimized for binary classification
            if model_type == "xgboost":
                params = {
                    "objective": "binary:logistic",
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "n_estimators": 1000,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.0,
                    "reg_lambda": 0.1,
                    "min_child_weight": 1,
                    "gamma": 0.0,
                    "scale_pos_weight": None,
                    "random_state": 42,
                    "n_jobs": -1,
                    "tree_method": "hist",
                    "early_stopping_rounds": 50,
                }
            elif model_type == "lightgbm":
                params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",
                    "num_leaves": 63,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "n_estimators": 1000,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "min_child_samples": 20,
                    "min_child_weight": 0.001,
                    "reg_alpha": 0.0,
                    "reg_lambda": 0.1,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1,
                    "is_unbalance": True,
                }
            elif model_type == "catboost":
                params = {
                    "iterations": 1000,
                    "learning_rate": 0.05,
                    "depth": 8,
                    "l2_leaf_reg": 3,
                    "min_data_in_leaf": 1,
                    "border_count": 254,
                    "random_strength": 1,
                    "bootstrap_type": "Bayesian",
                    "bagging_temperature": 1,
                    "random_seed": 42,
                    "verbose": False,
                    "thread_count": -1,
                    "task_type": "CPU",
                    "auto_class_weights": "Balanced",
                }

        # Create model with binary task
        if model_type == "xgboost":
            model = XGBoostModel(task="classification", params=params)
            # Override for binary
            model.params["objective"] = "binary:logistic"
        elif model_type == "lightgbm":
            model = LightGBMModel(task="classification", params=params)
            model.params["objective"] = "binary"
        elif model_type == "catboost":
            model = CatBoostModel(task="classification", params=params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ):
        """
        Train cascade classifier

        Args:
            X_train: Training features
            y_train: Training labels (-1, 0, 1)
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info("="*70)
        logger.info("Training CASCADE CLASSIFIER (Two-Stage)")
        logger.info("="*70)

        # ==================== STAGE 1: Class 0 vs Others ====================
        logger.info("\n[STAGE 1] Binary: Class 0 vs (Class -1 + Class +1)")

        # Create binary labels: 0 -> 0, (-1 and +1) -> 1
        y_train_stage1 = (y_train != 0).astype(int)
        if y_val is not None:
            y_val_stage1 = (y_val != 0).astype(int)
        else:
            y_val_stage1 = None

        # Log distribution
        counts_s1 = np.bincount(y_train_stage1)
        logger.info(f"Stage 1 distribution: No DR (0)={counts_s1[0]} ({counts_s1[0]/len(y_train_stage1)*100:.1f}%), "
                   f"DR Event (1)={counts_s1[1]} ({counts_s1[1]/len(y_train_stage1)*100:.1f}%)")

        # Train Stage 1
        self.stage1 = self._get_model(self.stage1_model_type, self.stage1_params)

        # Custom training for binary (need to handle the model wrapper)
        self._train_binary_model(
            self.stage1, X_train, y_train_stage1, X_val, y_val_stage1, "Stage 1"
        )

        # Evaluate Stage 1
        if y_val is not None:
            stage1_pred_val = self._predict_binary(self.stage1, X_val)
            from sklearn.metrics import f1_score, accuracy_score
            stage1_acc = accuracy_score(y_val_stage1, stage1_pred_val)
            stage1_f1 = f1_score(y_val_stage1, stage1_pred_val)
            logger.info(f"Stage 1 Results: Accuracy={stage1_acc:.4f}, F1={stage1_f1:.4f}")

        # ==================== STAGE 2: Class -1 vs Class +1 ====================
        logger.info("\n[STAGE 2] Binary: Class -1 vs Class +1 (among DR events only)")

        # Filter to only DR events (Class -1 and +1)
        dr_mask_train = y_train != 0
        X_train_stage2 = X_train[dr_mask_train]
        y_train_stage2 = y_train[dr_mask_train]

        # Create binary labels: -1 -> 0, +1 -> 1
        y_train_stage2_binary = (y_train_stage2 == 1).astype(int)

        if y_val is not None:
            dr_mask_val = y_val != 0
            X_val_stage2 = X_val[dr_mask_val]
            y_val_stage2 = y_val[dr_mask_val]
            y_val_stage2_binary = (y_val_stage2 == 1).astype(int)
        else:
            X_val_stage2 = None
            y_val_stage2_binary = None

        # Log distribution
        counts_s2 = np.bincount(y_train_stage2_binary)
        logger.info(f"Stage 2 distribution: Decrease (-1)={counts_s2[0]} ({counts_s2[0]/len(y_train_stage2_binary)*100:.1f}%), "
                   f"Increase (+1)={counts_s2[1]} ({counts_s2[1]/len(y_train_stage2_binary)*100:.1f}%)")

        # Apply extreme class weights for Stage 2 (focus on minority class +1)
        class_ratio = counts_s2[0] / counts_s2[1] if counts_s2[1] > 0 else 1
        logger.info(f"Stage 2 imbalance ratio: {class_ratio:.1f}:1")

        # Train Stage 2 with EXTREME cost-sensitive learning for Class +1
        self.stage2 = self._get_model(self.stage2_model_type, self.stage2_params)

        # Apply 100x penalty for Class +1 misclassification
        from sklearn.utils.class_weight import compute_sample_weight
        # Create custom weights: 100x more weight on Class +1 (minority)
        sample_weights_stage2 = np.ones(len(y_train_stage2_binary))
        sample_weights_stage2[y_train_stage2_binary == 1] = 100.0  # 100x penalty for Class +1

        logger.info(f"Stage 2: Applying 100x penalty for Class +1 misclassification")

        # For Stage 2, apply extreme focus on Class +1
        self._train_binary_model(
            self.stage2, X_train_stage2, y_train_stage2_binary,
            X_val_stage2, y_val_stage2_binary, "Stage 2",
            sample_weights=sample_weights_stage2
        )

        # Evaluate Stage 2
        if y_val_stage2_binary is not None:
            stage2_pred_val = self._predict_binary(self.stage2, X_val_stage2)
            stage2_acc = accuracy_score(y_val_stage2_binary, stage2_pred_val)
            stage2_f1 = f1_score(y_val_stage2_binary, stage2_pred_val)
            logger.info(f"Stage 2 Results: Accuracy={stage2_acc:.4f}, F1={stage2_f1:.4f}")

        logger.info("\n" + "="*70)
        logger.info("CASCADE CLASSIFIER TRAINING COMPLETE")
        logger.info("="*70 + "\n")

    def _train_binary_model(
        self, model, X_train, y_train, X_val, y_val, stage_name, sample_weights=None
    ):
        """Train a binary model with proper handling and optional custom weights"""
        logger.info(f"Training {stage_name} ({model.__class__.__name__})...")

        # Convert to proper format for binary classification
        if model.__class__.__name__ == "XGBoostModel":
            from sklearn.utils.class_weight import compute_sample_weight
            if sample_weights is None:
                sample_weights = compute_sample_weight('balanced', y_train)

            # Create binary XGBoost model
            import xgboost as xgb
            model.model = xgb.XGBClassifier(**model.params)

            if X_val is not None and y_val is not None:
                model.model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        elif model.__class__.__name__ == "LightGBMModel":
            import lightgbm as lgb
            model.model = lgb.LGBMClassifier(**model.params)

            if X_val is not None and y_val is not None:
                model.model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            else:
                model.model.fit(X_train, y_train, sample_weight=sample_weights)

        elif model.__class__.__name__ == "CatBoostModel":
            from catboost import CatBoostClassifier
            model.model = CatBoostClassifier(**model.params)

            if X_val is not None and y_val is not None:
                model.model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    def _predict_binary(self, model, X):
        """Predict binary labels (0 or 1)"""
        return model.model.predict(X).astype(int)

    def _predict_proba_binary(self, model, X):
        """Predict probabilities for binary classification"""
        proba = model.model.predict_proba(X)
        return proba[:, 1]  # Probability of class 1

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict final classes using cascade

        Args:
            X: Features

        Returns:
            Predictions (-1, 0, 1)
        """
        # Stage 1: Predict if DR event or not
        stage1_pred = self._predict_binary(self.stage1, X)

        # Initialize final predictions as Class 0
        final_pred = np.zeros(len(X), dtype=int)

        # For samples predicted as DR events (stage1_pred == 1), use Stage 2
        dr_mask = stage1_pred == 1

        if np.sum(dr_mask) > 0:
            X_dr = X[dr_mask]

            # Stage 2: Predict -1 vs +1
            stage2_pred = self._predict_binary(self.stage2, X_dr)

            # Convert: 0 -> -1, 1 -> +1
            stage2_pred_labels = np.where(stage2_pred == 0, -1, 1)

            # Assign to final predictions
            final_pred[dr_mask] = stage2_pred_labels

        return final_pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using cascade

        Args:
            X: Features

        Returns:
            Probabilities for classes [-1, 0, 1] (shape: n_samples x 3)
        """
        # Stage 1: Get probability of DR event
        p_dr_event = self._predict_proba_binary(self.stage1, X)  # P(DR event)
        p_no_dr = 1 - p_dr_event  # P(Class 0)

        # Stage 2: Get probability of +1 among DR events
        p_increase_given_dr = self._predict_proba_binary(self.stage2, X)  # P(+1 | DR)
        p_decrease_given_dr = 1 - p_increase_given_dr  # P(-1 | DR)

        # Combine using probability chain rule
        # P(Class -1) = P(DR event) * P(-1 | DR)
        # P(Class 0) = P(No DR)
        # P(Class +1) = P(DR event) * P(+1 | DR)
        p_class_minus1 = p_dr_event * p_decrease_given_dr
        p_class_0 = p_no_dr
        p_class_plus1 = p_dr_event * p_increase_given_dr

        # Stack into probability matrix (n_samples x 3)
        proba = np.column_stack([p_class_minus1, p_class_0, p_class_plus1])

        # Normalize to ensure sum = 1 (handle numerical errors)
        proba = proba / proba.sum(axis=1, keepdims=True)

        return proba

    def save_model(self, filepath: str):
        """Save cascade model"""
        import joblib
        joblib.dump({
            'stage1': self.stage1,
            'stage2': self.stage2,
            'stage1_type': self.stage1_model_type,
            'stage2_type': self.stage2_model_type,
        }, filepath)
        logger.debug(f"Cascade model saved to {filepath}")


if __name__ == "__main__":
    print("Cascade Classifier for Extreme Class Imbalance")
    print("Two-stage hierarchical classification")
