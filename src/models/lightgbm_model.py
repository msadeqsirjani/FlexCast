"""
LightGBM Model Implementation
Fast gradient boosting with histogram-based algorithm
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple, Optional
import joblib
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class LightGBMModel:
    """LightGBM model for classification and regression tasks"""

    def __init__(self, task: str = "classification", params: Optional[Dict] = None):
        """
        Initialize LightGBM model

        Args:
            task: 'classification' or 'regression'
            params: Model hyperparameters
        """
        self.task = task
        self.model = None

        # Default parameters
        if params is None:
            if task == "classification":
                self.params = {
                    "objective": "multiclass",
                    "num_class": 3,
                    "metric": "multi_logloss",
                    "boosting_type": "gbdt",
                    "num_leaves": 15,  # Reduced from 31 to 15 - simpler trees
                    "max_depth": 3,  # Added depth constraint - prevent overfitting
                    "learning_rate": 0.01,  # Reduced from 0.05 to 0.01
                    "n_estimators": 500,  # Increased from 200 to compensate for lower LR
                    "feature_fraction": 0.7,  # Reduced from 0.8 for more regularization
                    "bagging_fraction": 0.7,  # Reduced from 0.8
                    "bagging_freq": 5,
                    "min_child_samples": 50,  # Increased from 20 to 50 - more conservative
                    "min_child_weight": 5,  # Added for additional regularization
                    "reg_alpha": 1.0,  # Increased L1 from 0.1 to 1.0
                    "reg_lambda": 5.0,  # Increased L2 from 1.0 to 5.0
                    "max_bin": 128,  # Reduced from default 255 to 128
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1,
                    "class_weight": "balanced",  # Handle class imbalance automatically
                }
            else:  # regression
                self.params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "n_estimators": 200,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "min_child_samples": 20,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1,
                }
        else:
            self.params = params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50,
    ) -> "LightGBMModel":
        """
        Train the LightGBM model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Early stopping rounds

        Returns:
            Self
        """
        logger.info(f"Training LightGBM {self.task} model with {len(X_train)} samples")

        # Adjust target for classification (-1, 0, 1) -> (0, 1, 2)
        if self.task == "classification":
            y_train_adjusted = y_train + 1
            if y_val is not None:
                y_val_adjusted = y_val + 1
        else:
            y_train_adjusted = y_train
            y_val_adjusted = y_val if y_val is not None else None

        if self.task == "classification":
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)

        # Training with or without validation
        callbacks = []
        if X_val is not None and y_val is not None:
            logger.debug(f"Validation: {len(X_val)} samples, early stopping: {early_stopping_rounds} rounds")
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            )
            self.model.fit(
                X_train,
                y_train_adjusted,
                eval_set=[(X_val, y_val_adjusted)],
                callbacks=callbacks,
            )
            logger.debug(f"Best iteration: {self.model.best_iteration_}")
        else:
            self.model.fit(X_train, y_train_adjusted)

        logger.debug("Training completed")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X)

        # Adjust predictions for classification (0, 1, 2) -> (-1, 0, 1)
        if self.task == "classification":
            predictions = predictions - 1

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (classification only)

        Args:
            X: Features to predict on

        Returns:
            Class probabilities
        """
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)

    def get_feature_importance(
        self, feature_names: list, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.feature_importances_
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        return importance_df

    def save_model(self, filepath: str):
        """
        Save model to disk

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        joblib.dump(
            {"model": self.model, "task": self.task, "params": self.params}, filepath
        )
        logger.debug(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model from disk

        Args:
            filepath: Path to load model from
        """
        data = joblib.load(filepath)
        self.model = data["model"]
        self.task = data["task"]
        self.params = data["params"]
        logger.debug(f"Model loaded from {filepath}")

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5, scoring: str = "f1_macro"
    ) -> Dict[str, float]:
        """
        Perform cross-validation

        Args:
            X: Features
            y: Labels
            cv: Number of folds
            scoring: Scoring metric

        Returns:
            Dictionary with CV results
        """
        # Adjust target for classification
        if self.task == "classification":
            y_adjusted = y + 1
        else:
            y_adjusted = y

        if self.task == "classification":
            model = lgb.LGBMClassifier(**self.params)
        else:
            model = lgb.LGBMRegressor(**self.params)

        scores = cross_val_score(model, X, y_adjusted, cv=cv, scoring=scoring)

        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores,
        }


if __name__ == "__main__":
    # Example usage
    print("LightGBM Model Implementation")
    print("Classification and Regression for Demand Response")

    # Create sample data
    X_train = pd.DataFrame(np.random.randn(1000, 10))
    y_train_class = pd.Series(np.random.choice([-1, 0, 1], size=1000))
    y_train_reg = pd.Series(np.random.randn(1000) * 10)

    # Classification example
    clf = LightGBMModel(task="classification")
    clf.train(X_train, y_train_class)
    predictions = clf.predict(X_train[:10])
    print(f"\nClassification predictions: {predictions}")

    # Regression example
    reg = LightGBMModel(task="regression")
    reg.train(X_train, y_train_reg)
    predictions = reg.predict(X_train[:10])
    print(f"\nRegression predictions: {predictions}")
