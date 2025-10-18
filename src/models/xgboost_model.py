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

warnings.filterwarnings("ignore")


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
                    "objective": "multi:softmax",
                    "num_class": 3,
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
        print(f"\nTraining XGBoost {self.task} model...")
        print(f"Training samples: {len(X_train)}")

        # Adjust target for classification (-1, 0, 1) -> (0, 1, 2)
        if self.task == "classification":
            y_train_adjusted = y_train + 1  # Convert to 0, 1, 2
            if y_val is not None:
                y_val_adjusted = y_val + 1
        else:
            y_train_adjusted = y_train
            y_val_adjusted = y_val if y_val is not None else None

        if self.task == "classification":
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)

        # Training with or without validation
        if X_val is not None and y_val is not None:
            print(f"Validation samples: {len(X_val)}")
            print(f"Early stopping rounds: {self.params.get('early_stopping_rounds', 50)}")
            eval_set = [(X_train, y_train_adjusted), (X_val, y_val_adjusted)]
            self.model.fit(
                X_train,
                y_train_adjusted,
                eval_set=eval_set,
                verbose=False
            )
            if hasattr(self.model, 'best_iteration'):
                print(f"Best iteration: {self.model.best_iteration}")
            if hasattr(self.model, 'best_score'):
                print(f"Best score: {self.model.best_score:.4f}")
        else:
            self.model.fit(X_train, y_train_adjusted, verbose=False)

        print(f"Training completed!")

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
