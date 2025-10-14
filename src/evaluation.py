"""
Evaluation Module for FlexTrack Challenge 2025
Implements competition metrics for classification and regression
"""

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    classification_report,
    confusion_matrix
)
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class Evaluator:
    """Evaluate model performance using competition metrics"""

    def __init__(self):
        """Initialize Evaluator"""
        pass

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        detailed: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate classification performance

        Metrics:
        - F1 Score (macro average)
        - Geometric Mean Score (macro average)

        Args:
            y_true: True labels
            y_pred: Predicted labels
            detailed: Whether to include detailed metrics

        Returns:
            Dictionary with evaluation metrics
        """
        # Calculate primary metrics
        f1 = f1_score(y_true, y_pred, average='macro')
        gm_score = geometric_mean_score(y_true, y_pred, average='macro')

        results = {
            'f1_score_macro': f1,
            'geometric_mean_score': gm_score
        }

        if detailed:
            # Per-class F1 scores
            f1_per_class = f1_score(y_true, y_pred, average=None)
            for i, class_label in enumerate([-1, 0, 1]):
                results[f'f1_score_class_{class_label}'] = f1_per_class[i]

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
            results['confusion_matrix'] = cm

        return results

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        building_power_stats: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Evaluate regression performance

        Metrics:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - NMAE (Normalized MAE by range)
        - NMAE (Normalized MAE by mean)
        - NRMSE (Normalized RMSE by range)
        - NRMSE (Normalized RMSE by mean)
        - CV-RMSE (Coefficient of Variation RMSE)

        Args:
            y_true: True values
            y_pred: Predicted values
            building_power_stats: Dictionary with 'min', 'max', 'mean', 'range'

        Returns:
            Dictionary with evaluation metrics
        """
        # Calculate basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        results = {
            'mae': mae,
            'rmse': rmse
        }

        # Normalized metrics
        if building_power_stats is not None:
            power_range = building_power_stats.get('range')
            power_mean = building_power_stats.get('mean')

            if power_range and power_range > 0:
                results['nmae_range'] = (mae / power_range) * 100
                results['nrmse_range'] = (rmse / power_range) * 100

            if power_mean and power_mean > 0:
                results['nmae_mean'] = (mae / power_mean) * 100
                results['nrmse_mean'] = (rmse / power_mean) * 100

        # CV-RMSE
        y_mean = np.mean(y_true)
        if y_mean > 0:
            results['cv_rmse'] = (rmse / y_mean) * 100

        return results

    def evaluate_combined(
        self,
        y_true_class: np.ndarray,
        y_pred_class: np.ndarray,
        y_true_reg: np.ndarray,
        y_pred_reg: np.ndarray,
        building_power_stats: Dict[str, float] = None
    ) -> Dict[str, Dict]:
        """
        Evaluate both classification and regression

        Args:
            y_true_class: True classification labels
            y_pred_class: Predicted classification labels
            y_true_reg: True regression values
            y_pred_reg: Predicted regression values
            building_power_stats: Building power statistics

        Returns:
            Dictionary with both classification and regression metrics
        """
        class_results = self.evaluate_classification(y_true_class, y_pred_class)
        reg_results = self.evaluate_regression(y_true_reg, y_pred_reg, building_power_stats)

        return {
            'classification': class_results,
            'regression': reg_results
        }

    def print_classification_results(self, results: Dict[str, float]):
        """
        Print classification results in a formatted way

        Args:
            results: Results dictionary from evaluate_classification
        """
        print("\n" + "="*70)
        print("CLASSIFICATION RESULTS")
        print("="*70)
        print(f"F1 Score (macro):          {results['f1_score_macro']:.4f}")
        print(f"Geometric Mean Score:      {results['geometric_mean_score']:.4f}")

        if 'f1_score_class_-1' in results:
            print("\nPer-Class F1 Scores:")
            print(f"  Class -1 (decrease):     {results['f1_score_class_-1']:.4f}")
            print(f"  Class  0 (no change):    {results['f1_score_class_0']:.4f}")
            print(f"  Class +1 (increase):     {results['f1_score_class_1']:.4f}")

        if 'confusion_matrix' in results:
            print("\nConfusion Matrix:")
            print("              Predicted")
            print("              -1    0   +1")
            cm = results['confusion_matrix']
            for i, true_label in enumerate([-1, 0, 1]):
                print(f"True {true_label:2d}  [{cm[i][0]:5d} {cm[i][1]:5d} {cm[i][2]:5d}]")

        print("="*70)

    def print_regression_results(self, results: Dict[str, float]):
        """
        Print regression results in a formatted way

        Args:
            results: Results dictionary from evaluate_regression
        """
        print("\n" + "="*70)
        print("REGRESSION RESULTS")
        print("="*70)
        print(f"MAE:                       {results['mae']:.4f} kW")
        print(f"RMSE:                      {results['rmse']:.4f} kW")

        if 'nmae_range' in results:
            print(f"\nNormalized MAE (range):    {results['nmae_range']:.2f}%")
        if 'nmae_mean' in results:
            print(f"Normalized MAE (mean):     {results['nmae_mean']:.2f}%")
        if 'nrmse_range' in results:
            print(f"Normalized RMSE (range):   {results['nrmse_range']:.2f}%")
        if 'nrmse_mean' in results:
            print(f"Normalized RMSE (mean):    {results['nrmse_mean']:.2f}%")
        if 'cv_rmse' in results:
            print(f"CV-RMSE:                   {results['cv_rmse']:.2f}%")

        print("="*70)

    def print_combined_results(self, results: Dict[str, Dict]):
        """
        Print combined classification and regression results

        Args:
            results: Results dictionary from evaluate_combined
        """
        self.print_classification_results(results['classification'])
        self.print_regression_results(results['regression'])

    def calculate_building_power_stats(self, building_power: pd.Series) -> Dict[str, float]:
        """
        Calculate building power statistics for normalization

        Args:
            building_power: Series of building power values

        Returns:
            Dictionary with statistics
        """
        # Filter non-zero values
        non_zero = building_power[building_power != 0]

        if len(non_zero) == 0:
            return {
                'min': 0,
                'max': 0,
                'mean': 0,
                'range': 0
            }

        min_val = non_zero.min()
        max_val = non_zero.max()

        return {
            'min': min_val,
            'max': max_val,
            'mean': non_zero.mean(),
            'range': max_val - min_val
        }

    def compare_models(
        self,
        model_results: Dict[str, Dict],
        task: str = 'classification'
    ) -> pd.DataFrame:
        """
        Compare multiple model results

        Args:
            model_results: Dictionary mapping model names to their results
            task: 'classification' or 'regression'

        Returns:
            DataFrame with comparison
        """
        comparison_data = []

        for model_name, results in model_results.items():
            if task == 'classification':
                comparison_data.append({
                    'Model': model_name,
                    'F1 Score': results.get('f1_score_macro', np.nan),
                    'Geometric Mean': results.get('geometric_mean_score', np.nan)
                })
            else:  # regression
                comparison_data.append({
                    'Model': model_name,
                    'MAE': results.get('mae', np.nan),
                    'RMSE': results.get('rmse', np.nan),
                    'NMAE (range)': results.get('nmae_range', np.nan),
                    'CV-RMSE': results.get('cv_rmse', np.nan)
                })

        df = pd.DataFrame(comparison_data)

        if task == 'classification':
            # Sort by F1 score (descending)
            df = df.sort_values('F1 Score', ascending=False)
        else:
            # Sort by MAE (ascending - lower is better)
            df = df.sort_values('MAE', ascending=True)

        return df


if __name__ == "__main__":
    # Example usage
    print("Evaluation Module for FlexTrack Challenge 2025")

    # Create sample data
    y_true_class = np.random.choice([-1, 0, 1], size=1000)
    y_pred_class = np.random.choice([-1, 0, 1], size=1000)
    y_true_reg = np.random.randn(1000) * 10
    y_pred_reg = y_true_reg + np.random.randn(1000) * 2

    # Create evaluator
    evaluator = Evaluator()

    # Evaluate classification
    class_results = evaluator.evaluate_classification(y_true_class, y_pred_class)
    evaluator.print_classification_results(class_results)

    # Evaluate regression
    building_power = pd.Series(np.random.uniform(50, 500, size=1000))
    power_stats = evaluator.calculate_building_power_stats(building_power)
    reg_results = evaluator.evaluate_regression(y_true_reg, y_pred_reg, power_stats)
    evaluator.print_regression_results(reg_results)

    # Compare models
    model_results = {
        'XGBoost': class_results,
        'LightGBM': {'f1_score_macro': 0.75, 'geometric_mean_score': 0.73},
        'CatBoost': {'f1_score_macro': 0.78, 'geometric_mean_score': 0.76}
    }
    comparison = evaluator.compare_models(model_results, task='classification')
    print("\nModel Comparison:")
    print(comparison)
