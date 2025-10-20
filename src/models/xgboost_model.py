"""
XGBoost Model Implementation
Extreme Gradient Boosting for classification and regression
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple, Optional
import joblib
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost model for classification and regression tasks"""

    def __init__(self, task: str = "classification", params: Optional[Dict] = None):
        """
        Initialize XGBoost model

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
                    "objective": "multi:softprob",  # Changed from softmax to softprob for probability outputs
                    "num_class": 3,
                    "max_depth": 3,  # Reduced from 6 to 3 to fight overfitting
                    "learning_rate": 0.01,  # Reduced from 0.1 to 0.01 for better generalization
                    "n_estimators": 500,  # Increased from 200 to compensate for lower learning rate
                    "subsample": 0.7,  # Reduced from 0.8 for more regularization
                    "colsample_bytree": 0.7,  # Reduced from 0.8 for more regularization
                    "reg_alpha": 1.0,  # Increased L1 regularization from 0.1 to 1.0
                    "reg_lambda": 5.0,  # Increased L2 regularization from 1.0 to 5.0
                    "min_child_weight": 5,  # Increased from 3 to 5 to prevent overfitting
                    "gamma": 0.3,  # Increased from 0.1 for more conservative splits
                    "max_delta_step": 5,  # NEW: Helps with imbalanced classes (range 1-10)
                    "random_state": 42,
                    "n_jobs": -1,
                    "tree_method": "hist",
                    "early_stopping_rounds": 50,
                }
            else:  # regression
                self.params = {
                    "objective": "reg:squarederror",
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 200,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "min_child_weight": 3,
                    "gamma": 0.1,
                    "random_state": 42,
                    "n_jobs": -1,
                    "tree_method": "hist",
                    "early_stopping_rounds": 50,
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
    ) -> "XGBoostModel":
        """
        Train the XGBoost model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Early stopping rounds

        Returns:
            Self
        """
        logger.info(f"Training XGBoost {self.task} model with {len(X_train)} samples")

        # Adjust target for classification (-1, 0, 1) -> (0, 1, 2)
        if self.task == "classification":
            y_train_adjusted = y_train + 1  # Convert to 0, 1, 2
            if y_val is not None:
                y_val_adjusted = y_val + 1

            # Compute sample weights for class imbalance
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train_adjusted)
            logger.debug("Applied balanced sample weights for class imbalance")
        else:
            y_train_adjusted = y_train
            y_val_adjusted = y_val if y_val is not None else None
            sample_weights = None

        if self.task == "classification":
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)

        # Training with or without validation
        if X_val is not None and y_val is not None:
            logger.debug(f"Validation: {len(X_val)} samples, early stopping: {self.params.get('early_stopping_rounds', 50)} rounds")
            eval_set = [(X_train, y_train_adjusted), (X_val, y_val_adjusted)]

            fit_params = {
                'eval_set': eval_set,
                'verbose': False
            }
            if sample_weights is not None:
                fit_params['sample_weight'] = sample_weights

            self.model.fit(X_train, y_train_adjusted, **fit_params)
            if hasattr(self.model, 'best_iteration'):
                logger.debug(f"Best iteration: {self.model.best_iteration}")
            if hasattr(self.model, 'best_score'):
                logger.debug(f"Best score: {self.model.best_score:.4f}")
        else:
            if sample_weights is not None:
                self.model.fit(X_train, y_train_adjusted, sample_weight=sample_weights, verbose=False)
            else:
                self.model.fit(X_train, y_train_adjusted, verbose=False)

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
            model = xgb.XGBClassifier(**self.params)
        else:
            model = xgb.XGBRegressor(**self.params)

        scores = cross_val_score(model, X, y_adjusted, cv=cv, scoring=scoring)

        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores,
        }


if __name__ == "__main__":
    # Example usage
    print("XGBoost Model Implementation")
    print("Classification and Regression for Demand Response")

    # Create sample data
    X_train = pd.DataFrame(np.random.randn(1000, 10))
    y_train_class = pd.Series(np.random.choice([-1, 0, 1], size=1000))
    y_train_reg = pd.Series(np.random.randn(1000) * 10)

    # Classification example
    clf = XGBoostModel(task="classification")
    clf.train(X_train, y_train_class)
    predictions = clf.predict(X_train[:10])
    print(f"\nClassification predictions: {predictions}")

    # Regression example
    reg = XGBoostModel(task="regression")
    reg.train(X_train, y_train_reg)
    predictions = reg.predict(X_train[:10])
    print(f"\nRegression predictions: {predictions}")
