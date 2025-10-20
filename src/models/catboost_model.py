"""
CatBoost Model Implementation
Handles categorical features natively with ordered boosting
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple, Optional, List
import joblib
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class CatBoostModel:
    """CatBoost model for classification and regression tasks"""

    def __init__(self, task: str = "classification", params: Optional[Dict] = None):
        """
        Initialize CatBoost model

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
                    "iterations": 500,  # Increased from 200 to compensate for lower LR
                    "learning_rate": 0.01,  # Reduced from 0.1 to 0.01 - fight overfitting
                    "depth": 3,  # Reduced from 6 to 3 - simpler trees
                    "l2_leaf_reg": 10,  # Increased from 3 to 10 - stronger L2 regularization
                    "min_data_in_leaf": 50,  # Added - require more samples per leaf
                    "border_count": 64,  # Reduced from 128 to 64 - less granular splits
                    "random_strength": 2,  # Increased from 1 to 2 - more randomness
                    "bootstrap_type": "Bernoulli",  # Required for subsample to work
                    "subsample": 0.7,  # Use 70% of data per tree (Bernoulli bootstrap)
                    "random_seed": 42,
                    "verbose": False,
                    "thread_count": -1,
                    "task_type": "CPU",
                    "auto_class_weights": "SqrtBalanced",  # Stronger than 'Balanced'
                }
            else:  # regression
                self.params = {
                    "iterations": 200,
                    "learning_rate": 0.1,
                    "depth": 6,
                    "l2_leaf_reg": 3,
                    "border_count": 128,
                    "random_strength": 1,
                    "bagging_temperature": 1,
                    "random_seed": 42,
                    "verbose": False,
                    "thread_count": -1,
                    "task_type": "CPU",
                }
        else:
            self.params = params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_features: Optional[List[str]] = None,
        early_stopping_rounds: int = 50,
    ) -> "CatBoostModel":
        """
        Train the CatBoost model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            cat_features: List of categorical feature names
            early_stopping_rounds: Early stopping rounds

        Returns:
            Self
        """
        logger.info(f"Training CatBoost {self.task} model with {len(X_train)} samples")

        # Adjust target for classification (-1, 0, 1) -> (0, 1, 2)
        if self.task == "classification":
            y_train_adjusted = y_train + 1
            if y_val is not None:
                y_val_adjusted = y_val + 1
        else:
            y_train_adjusted = y_train
            y_val_adjusted = y_val if y_val is not None else None

        # Identify categorical features
        if cat_features is None:
            cat_features = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        if self.task == "classification":
            self.model = CatBoostClassifier(**self.params)
        else:
            self.model = CatBoostRegressor(**self.params)

        # Training with or without validation
        if X_val is not None and y_val is not None:
            logger.debug(f"Validation: {len(X_val)} samples, early stopping: {early_stopping_rounds} rounds")
            self.model.fit(
                X_train,
                y_train_adjusted,
                eval_set=(X_val, y_val_adjusted),
                cat_features=cat_features,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
            logger.debug(f"Best iteration: {self.model.best_iteration_}")
            # best_score_ is a nested dict, get the first metric value
            if hasattr(self.model, 'best_score_') and 'validation' in self.model.best_score_:
                validation_scores = self.model.best_score_['validation']
                if isinstance(validation_scores, dict):
                    # Get the first metric value
                    first_metric = list(validation_scores.values())[0]
                    logger.debug(f"Best score: {first_metric:.4f}")
                else:
                    logger.debug(f"Best score: {validation_scores:.4f}")
        else:
            self.model.fit(
                X_train, y_train_adjusted, cat_features=cat_features, verbose=False
            )

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
            predictions = predictions.astype(int) - 1

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

        importance = self.model.get_feature_importance()
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

        # CatBoost has native save method
        self.model.save_model(filepath)
        # Also save metadata
        metadata_path = filepath + ".meta"
        joblib.dump({"task": self.task, "params": self.params}, metadata_path)
        logger.debug(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model from disk

        Args:
            filepath: Path to load model from
        """
        # Load metadata
        metadata_path = filepath + ".meta"
        metadata = joblib.load(metadata_path)
        self.task = metadata["task"]
        self.params = metadata["params"]

        # Load model
        if self.task == "classification":
            self.model = CatBoostClassifier()
        else:
            self.model = CatBoostRegressor()

        self.model.load_model(filepath)
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
            model = CatBoostClassifier(**self.params)
        else:
            model = CatBoostRegressor(**self.params)

        scores = cross_val_score(model, X, y_adjusted, cv=cv, scoring=scoring)

        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores,
        }


if __name__ == "__main__":
    # Example usage
    print("CatBoost Model Implementation")
    print("Classification and Regression for Demand Response")

    # Create sample data
    X_train = pd.DataFrame(np.random.randn(1000, 10))
    y_train_class = pd.Series(np.random.choice([-1, 0, 1], size=1000))
    y_train_reg = pd.Series(np.random.randn(1000) * 10)

    # Classification example
    clf = CatBoostModel(task="classification")
    clf.train(X_train, y_train_class)
    predictions = clf.predict(X_train[:10])
    print(f"\nClassification predictions: {predictions}")

    # Regression example
    reg = CatBoostModel(task="regression")
    reg.train(X_train, y_train_reg)
    predictions = reg.predict(X_train[:10])
    print(f"\nRegression predictions: {predictions}")
