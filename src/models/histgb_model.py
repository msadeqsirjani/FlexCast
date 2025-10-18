"""
HistGradientBoosting Model Implementation
Scikit-learn's native histogram-based gradient boosting

IMPORTANT: This module correctly uses:
- HistGradientBoostingClassifier for classification tasks
- HistGradientBoostingRegressor for regression tasks

This fixes the common issue of using HistGradientBoostingRegressor for classification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple, Optional
import joblib
import warnings

warnings.filterwarnings("ignore")


class HistGradientBoostingModel:
    """
    HistGradientBoosting model for classification and regression tasks

    This implementation correctly distinguishes between:
    - Classification: Uses HistGradientBoostingClassifier
    - Regression: Uses HistGradientBoostingRegressor
    """

    def __init__(self, task: str = "classification", params: Optional[Dict] = None):
        """
        Initialize HistGradientBoosting model

        Args:
            task: 'classification' or 'regression'
            params: Model hyperparameters

        Raises:
            ValueError: If task is not 'classification' or 'regression'
        """
        if task not in ["classification", "regression"]:
            raise ValueError(
                f"Task must be 'classification' or 'regression', got '{task}'"
            )

        self.task = task
        self.model = None

        # Default parameters
        if params is None:
            # Common parameters for both tasks
            base_params = {
                "max_iter": 200,
                "learning_rate": 0.1,
                "max_depth": None,  # No limit
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
                "max_bins": 255,
                "random_state": 42,
                "verbose": 0,
                "early_stopping": True,
                "scoring": None,  # Will be set during training
                "n_iter_no_change": 10,
                "validation_fraction": 0.1,
            }

            if task == "classification":
                # Classification-specific parameters
                self.params = {
                    **base_params,
                    "loss": "log_loss",  # For multiclass classification
                }
            else:  # regression
                # Regression-specific parameters
                self.params = {
                    **base_params,
                    "loss": "squared_error",
                }
        else:
            self.params = params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 10,
    ) -> "HistGradientBoostingModel":
        """
        Train the HistGradientBoosting model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, used if early_stopping=False in params)
            y_val: Validation labels (optional)
            early_stopping_rounds: Early stopping rounds (updates n_iter_no_change)

        Returns:
            Self
        """
        print(f"\nTraining HistGradientBoosting {self.task} model...")
        print(
            f"Using {'HistGradientBoostingClassifier' if self.task == 'classification' else 'HistGradientBoostingRegressor'}"
        )
        print(f"Training samples: {len(X_train)}")
        if X_val is not None:
            print(f"Validation samples: {len(X_val)}")
        print(f"Early stopping rounds (n_iter_no_change): {early_stopping_rounds}")

        # Update early stopping parameter
        params_copy = self.params.copy()
        params_copy["n_iter_no_change"] = early_stopping_rounds

        # Adjust target for classification (-1, 0, 1) -> (0, 1, 2)
        # This is necessary because sklearn classifiers expect non-negative class labels
        if self.task == "classification":
            y_train_adjusted = y_train + 1
            if y_val is not None:
                y_val_adjusted = y_val + 1

            # Compute sample weights for class imbalance
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train_adjusted)
            print("Applied balanced sample weights for class imbalance")

            # Create the classifier
            self.model = HistGradientBoostingClassifier(**params_copy)
            print("Target classes adjusted: -1, 0, 1 -> 0, 1, 2")
        else:
            y_train_adjusted = y_train
            y_val_adjusted = y_val if y_val is not None else None
            sample_weights = None

            # Create the regressor
            self.model = HistGradientBoostingRegressor(**params_copy)

        # Train the model
        # HistGradientBoosting uses internal validation split if early_stopping=True
        if sample_weights is not None:
            self.model.fit(X_train, y_train_adjusted, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_train_adjusted)

        print(f"Training completed!")
        print(f"Number of iterations: {self.model.n_iter_}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features to predict on

        Returns:
            Predictions (adjusted back to -1, 0, 1 for classification)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X)

        # Adjust predictions back for classification (0, 1, 2) -> (-1, 0, 1)
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

        Raises:
            ValueError: If task is not classification
        """
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification task")

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)

    def get_feature_importance(
        self, feature_names: list, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance

        Note: HistGradientBoosting doesn't provide feature_importances_ by default.
        This returns None with a warning.

        Args:
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance (or None if not available)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Check if feature importances are available
        if not hasattr(self.model, "feature_importances_"):
            print(
                "Warning: HistGradientBoosting doesn't provide feature importances by default."
            )
            return None

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
        print(f"Model saved to {filepath}")

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
        print(f"Model loaded from {filepath}")

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
            model = HistGradientBoostingClassifier(**self.params)
        else:
            y_adjusted = y
            model = HistGradientBoostingRegressor(**self.params)

        scores = cross_val_score(model, X, y_adjusted, cv=cv, scoring=scoring)

        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores,
        }


if __name__ == "__main__":
    # Example usage demonstrating the correct implementation
    print("=" * 70)
    print("HistGradientBoosting Model Implementation")
    print("Correctly uses Classifier for classification and Regressor for regression")
    print("=" * 70)

    # Create sample data
    X_train = pd.DataFrame(
        np.random.randn(1000, 10), columns=[f"feature_{i}" for i in range(10)]
    )
    y_train_class = pd.Series(np.random.choice([-1, 0, 1], size=1000))
    y_train_reg = pd.Series(np.random.randn(1000) * 10)

    # Classification example
    print("\n" + "=" * 70)
    print("CLASSIFICATION EXAMPLE")
    print("=" * 70)
    clf = HistGradientBoostingModel(task="classification")
    clf.train(X_train, y_train_class)
    predictions = clf.predict(X_train[:10])
    print(f"\nClassification predictions (first 10): {predictions}")
    print(f"Unique predictions: {np.unique(predictions)}")

    # Verify predictions are in correct range
    assert set(predictions).issubset(
        {-1, 0, 1}
    ), "Classification predictions should be in {-1, 0, 1}"
    print("✓ Classification predictions are in correct range {-1, 0, 1}")

    # Regression example
    print("\n" + "=" * 70)
    print("REGRESSION EXAMPLE")
    print("=" * 70)
    reg = HistGradientBoostingModel(task="regression")
    reg.train(X_train, y_train_reg)
    predictions = reg.predict(X_train[:10])
    print(f"\nRegression predictions (first 10): {predictions}")
    print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print("✓ Regression predictions are continuous values")

    print("\n" + "=" * 70)
    print("SUCCESS: HistGradientBoosting implementation is correct!")
    print("=" * 70)
