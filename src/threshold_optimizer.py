"""
Threshold Optimization for Imbalanced Multi-Class Classification
Optimizes decision thresholds to maximize F1 score for each class
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from scipy.optimize import differential_evolution
from typing import Tuple, Dict
import warnings
warnings.filterwarnings("ignore")


class ThresholdOptimizer:
    """Optimize classification thresholds for imbalanced multi-class problems"""

    def __init__(self, classes: list = [-1, 0, 1]):
        """
        Initialize threshold optimizer

        Args:
            classes: List of class labels
        """
        self.classes = classes
        self.n_classes = len(classes)
        self.best_thresholds = None
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = 'f1_macro'
    ) -> Dict:
        """
        Optimize thresholds using differential evolution

        Args:
            y_true: True labels
            y_proba: Predicted probabilities (n_samples, n_classes)
            metric: Metric to optimize ('f1_macro', 'f1_weighted', 'geometric_mean')

        Returns:
            Dictionary with optimization results
        """
        print(f"\n{'='*80}")
        print("OPTIMIZING CLASSIFICATION THRESHOLDS")
        print(f"{'='*80}")
        print(f"Metric: {metric}")
        print(f"Classes: {self.classes}")

        # Convert y_true to indices
        y_true_idx = np.array([self.class_to_idx[y] for y in y_true])

        def objective(thresholds):
            """Objective function to maximize"""
            predictions = self._predict_with_thresholds(y_proba, thresholds)

            if metric == 'f1_macro':
                score = f1_score(y_true_idx, predictions, average='macro', zero_division=0)
            elif metric == 'f1_weighted':
                score = f1_score(y_true_idx, predictions, average='weighted', zero_division=0)
            elif metric == 'geometric_mean':
                # Geometric mean of per-class recalls
                from sklearn.metrics import recall_score
                recalls = recall_score(y_true_idx, predictions, average=None, zero_division=0)
                score = np.prod(recalls + 1e-10) ** (1.0 / len(recalls))
            else:
                score = f1_score(y_true_idx, predictions, average='macro', zero_division=0)

            return -score  # Negative because we minimize

        # Optimize thresholds using differential evolution
        # bounds[0]: minority class threshold (lower = more sensitive)
        # bounds[1]: majority class threshold (higher = more conservative)
        bounds = [
            (0.05, 0.5),  # Minority class threshold: very low to detect rare events
            (0.3, 0.95)   # Majority class threshold: higher to avoid false positives
        ]

        print("Running threshold optimization...")
        result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=100,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            workers=1,
            updating='deferred',
            polish=True
        )

        self.best_thresholds = np.sort(result.x)  # Ensure thresholds are sorted
        best_score = -result.fun

        # Get predictions with optimized thresholds
        y_pred_optimized = self._predict_with_thresholds(y_proba, self.best_thresholds)

        # Convert back to original labels
        y_pred_original = np.array([self.classes[p] for p in y_pred_optimized])

        # Compute detailed metrics
        f1_macro = f1_score(y_true, y_pred_original, average='macro', zero_division=0)
        f1_per_class = f1_score(y_true, y_pred_original, average=None, zero_division=0, labels=self.classes)

        print(f"\n{'='*80}")
        print("OPTIMIZATION RESULTS")
        print(f"{'='*80}")
        print(f"Optimized Thresholds: {self.best_thresholds}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"\nPer-Class F1 Scores:")
        for cls, f1 in zip(self.classes, f1_per_class):
            print(f"  Class {cls:2d}: {f1:.4f}")

        # Print confusion matrix
        cm = confusion_matrix(y_true, y_pred_original, labels=self.classes)
        print(f"\nConfusion Matrix:")
        print(cm)

        results = {
            'thresholds': self.best_thresholds,
            'f1_macro': f1_macro,
            'f1_per_class': {cls: f1 for cls, f1 in zip(self.classes, f1_per_class)},
            'confusion_matrix': cm,
            'predictions': y_pred_original
        }

        return results

    def _predict_with_thresholds(self, y_proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """
        Make predictions using custom thresholds - AGGRESSIVE minority class detection

        Strategy: Lower thresholds for minority classes to detect them more easily
        - Class 0 (majority): requires high confidence (threshold[1])
        - Class 1,2 (minority): requires lower confidence (threshold[0])

        Args:
            y_proba: Predicted probabilities (n_samples, n_classes)
            thresholds: Decision thresholds [minority_threshold, majority_threshold]

        Returns:
            Class predictions (as indices)
        """
        n_samples = y_proba.shape[0]
        predictions = np.zeros(n_samples, dtype=int)

        # Thresholds: [0] for minority, [1] for majority class confidence
        minority_threshold = thresholds[0]  # Lower threshold for minority
        majority_threshold = thresholds[1]  # Higher threshold for majority

        for i in range(n_samples):
            # Check minority classes first with LOWER thresholds
            # Class -1 (index 0)
            if y_proba[i, 0] >= minority_threshold:
                predictions[i] = 0
            # Class 1 (index 2)
            elif y_proba[i, 2] >= minority_threshold:
                predictions[i] = 2
            # Class 0 (index 1) - majority class requires HIGHER threshold
            elif y_proba[i, 1] >= majority_threshold:
                predictions[i] = 1
            else:
                # If no threshold met, choose highest probability with boost for minority
                adjusted_proba = y_proba[i].copy()
                # Apply 3x boost to minority classes
                adjusted_proba[0] *= 3.0  # Boost class -1
                adjusted_proba[2] *= 3.0  # Boost class 1
                predictions[i] = np.argmax(adjusted_proba)

        return predictions

    def predict(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Make predictions using optimized thresholds

        Args:
            y_proba: Predicted probabilities (n_samples, n_classes)

        Returns:
            Class predictions (original labels)
        """
        if self.best_thresholds is None:
            raise ValueError("Thresholds not optimized. Call optimize_thresholds() first.")

        y_pred_idx = self._predict_with_thresholds(y_proba, self.best_thresholds)
        y_pred = np.array([self.classes[p] for p in y_pred_idx])

        return y_pred

    def evaluate(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """
        Evaluate predictions using optimized thresholds

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(y_proba)

        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=self.classes)
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)

        # Compute geometric mean
        from sklearn.metrics import recall_score
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0, labels=self.classes)
        geometric_mean = np.prod(recalls + 1e-10) ** (1.0 / len(recalls))

        return {
            'f1_macro': f1_macro,
            'f1_per_class': {cls: f1 for cls, f1 in zip(self.classes, f1_per_class)},
            'geometric_mean': geometric_mean,
            'confusion_matrix': cm,
            'predictions': y_pred
        }


if __name__ == "__main__":
    # Example usage
    print("Threshold Optimizer - Example")

    # Create sample imbalanced data
    np.random.seed(42)
    n_samples = 1000

    # Simulated probabilities
    y_proba = np.random.dirichlet(alpha=[10, 1, 1], size=n_samples)  # Imbalanced towards class 0
    y_true = np.random.choice([-1, 0, 1], size=n_samples, p=[0.05, 0.9, 0.05])  # True imbalanced labels

    # Without threshold optimization (using argmax)
    y_pred_default = np.array([-1, 0, 1])[np.argmax(y_proba, axis=1)]
    f1_default = f1_score(y_true, y_pred_default, average='macro', zero_division=0)
    print(f"F1 Score (default): {f1_default:.4f}")

    # With threshold optimization
    optimizer = ThresholdOptimizer(classes=[-1, 0, 1])
    results = optimizer.optimize_thresholds(y_true, y_proba, metric='f1_macro')
    print(f"\nF1 Score (optimized): {results['f1_macro']:.4f}")
    print(f"Improvement: {(results['f1_macro'] - f1_default) / f1_default * 100:.1f}%")
